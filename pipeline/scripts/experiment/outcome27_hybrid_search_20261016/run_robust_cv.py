from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "experiment"))
import run_hybrid_search as hybrid  # noqa: E402
import run_outcome27_optimization_20261016 as prior  # noqa: E402

OUT = ROOT / "outputs" / "gru_addgate_outcome27_robust_cv_20261016"
SEED = 20261016


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


@torch.inference_mode()
def gn_embedding(data):
    cfg = prior.Config(
        name="frozen_gn_embedding", task="both", count_loss="huber", ohss_loss="bce",
        pretrained=True, freeze_encoder=True, balanced_batches=False,
    )
    model = prior.OutcomeGRU(data.static.shape[1], data.dynamic.shape[2], cfg)
    info = prior.initialize_pretrained(model)
    model.eval(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    result=[]
    for s in range(0,len(data.static),512):
        sx=torch.tensor(data.static[s:s+512],dtype=torch.float32,device=device)
        dx=torch.tensor(data.dynamic[s:s+512],dtype=torch.float32,device=device)
        mm=torch.tensor(data.mask[s:s+512],dtype=torch.float32,device=device)
        result.append(model.encode(sx,dx,mm).cpu().numpy())
    return np.concatenate(result).astype(np.float32), info


def feature_sets(data):
    summary, flat = hybrid.masked_summary(data); emb, info = gn_embedding(data)
    return {"summary":summary,"summary_emb":np.c_[summary,emb],"flat_emb":np.c_[flat,emb]}, info


def candidates(target, pos_weight):
    all_models = hybrid.oocyte_candidates() if target=="oocyte" else hybrid.ohss_candidates(pos_weight)
    keep=[]
    for name,fs,model in all_models:
        if target=="oocyte":
            ok=(name.startswith("xgb_") or name.startswith("ridge_") or "extra_" in name and "leaf5" in name)
        else:
            ok=(name.startswith("xgb_") or name.startswith("logit_") and ("c0.01" in name or "c0.05" in name) or "extra_" in name and ("leaf5" in name or "leaf10" in name))
        if ok: keep.append((name,fs,model))
    return keep


def cv_stage():
    for s in ("models","tables","logs","predictions","audit"): (OUT/s).mkdir(parents=True,exist_ok=True)
    data,audit=prior.prepare_cycle_data(); split=prior.masks(data); dev=split["train"]|split["valid"]
    xsets,pretrain_info=feature_sets(data); dev_ix=np.flatnonzero(dev)
    yo=data.oocytes[dev]; yh=data.ohss[dev]
    pos_weight=float((len(yh)-yh.sum())/yh.sum())
    rows=[]; pools={"oocyte":{},"ohss":{}}
    for target,y,folds in (
        ("oocyte",yo,list(KFold(5,shuffle=True,random_state=SEED).split(dev_ix))),
        ("ohss",yh,list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(dev_ix,yh))),
    ):
        for name,fs,model in candidates(target,pos_weight):
            p=np.zeros(len(dev_ix),dtype=float)
            for fold,(a,b) in enumerate(folds):
                m=clone(model); m.fit(xsets[fs][dev_ix[a]],y[a])
                p[b]=m.predict(xsets[fs][dev_ix[b]]) if target=="oocyte" else m.predict_proba(xsets[fs][dev_ix[b]])[:,1]
            met=hybrid.regression_metrics(y,p) if target=="oocyte" else hybrid.classification_metrics(y,p)
            rows.append({"target":target,"candidate":name,"feature_set":fs,**met}); pools[target][name]=p
            print("CV",target,name,met,flush=True)
    o_names,o_pred,o_met=hybrid.greedy_regression_ensemble(yo,pools["oocyte"])
    h_names,h_pred,h_met=hybrid.greedy_auc_ensemble(yh,pools["ohss"])
    manifest={}
    all_candidates={"oocyte":{n:(fs,m) for n,fs,m in candidates("oocyte",pos_weight)},"ohss":{n:(fs,m) for n,fs,m in candidates("ohss",pos_weight)}}
    for target,names,y in (("oocyte",o_names,yo),("ohss",h_names,yh)):
        for name in names:
            fs,m=all_candidates[target][name]; m.fit(xsets[fs][dev_ix],y)
            path=OUT/"models"/f"{target}__{name}.joblib"; joblib.dump(m,path)
            manifest[f"{target}::{name}"]={"path":str(path),"feature_set":fs}
    frozen={
        "status":"exploratory redevelopment; test was previously inspected, but this selection used five-fold development CV only",
        "source_variables":27,"representation":"longitudinal summaries plus frozen Gn-pretrained GRU(AddGate) embedding",
        "development_cycles":int(dev.sum()),"development_ohss_positive":int(yh.sum()),"folds":5,"seed":SEED,
        "oocyte":{"members":o_names,"cv_metrics":o_met},"ohss":{"members":h_names,"cv_metrics":h_met},
        "model_manifest":manifest,"pretraining":pretrain_info,
    }
    pd.DataFrame(rows).to_csv(OUT/"tables"/"fivefold_candidate_screen.csv",index=False)
    pd.DataFrame({"cycle_uid":data.meta.loc[dev,"cycle_uid"],"oocytes":yo,"oocyte_oof":o_pred,"ohss":yh,"ohss_oof":h_pred}).to_csv(OUT/"predictions"/"fivefold_selected_oof.csv",index=False)
    write_json(OUT/"logs"/"frozen_cv_selection.json",frozen); write_json(OUT/"audit"/"data_contract.json",audit)
    print(json.dumps({"oocyte":frozen["oocyte"],"ohss":frozen["ohss"]},indent=2),flush=True)


def final_stage():
    frozen=json.loads((OUT/"logs"/"frozen_cv_selection.json").read_text(encoding="utf-8"))
    data,_=prior.prepare_cycle_data(); split=prior.masks(data); te=split["test"]; xsets,_=feature_sets(data)
    def pred(target,name):
        z=frozen["model_manifest"][f"{target}::{name}"]; m=joblib.load(z["path"]); x=xsets[z["feature_set"]][te]
        return m.predict(x) if target=="oocyte" else m.predict_proba(x)[:,1]
    po=np.mean([pred("oocyte",n) for n in frozen["oocyte"]["members"]],axis=0)
    ph=np.mean([pred("ohss",n) for n in frozen["ohss"]["members"]],axis=0)
    om=hybrid.regression_metrics(data.oocytes[te],po); hm=hybrid.classification_metrics(data.ohss[te],ph)
    bt=hybrid.bootstrap(data.oocytes[te],po,data.ohss[te],ph); bt.to_csv(OUT/"tables"/"test_cycle_bootstrap_1000.csv",index=False)
    est={**om,**hm}; ci=[]
    for c in ("r2","mae","rmse","roc_auc"):
        q=bt[c].dropna().quantile([.025,.975]); ci.append({"metric":c,"estimate":est[c],"lower_95ci":q.iloc[0],"upper_95ci":q.iloc[1]})
    pd.DataFrame(ci).to_csv(OUT/"tables"/"exploratory_test_metrics_95ci.csv",index=False)
    pd.DataFrame({"cycle_uid":data.meta.loc[te,"cycle_uid"],"oocytes":data.oocytes[te],"oocyte_prediction":po,"ohss":data.ohss[te],"ohss_probability":ph}).to_csv(OUT/"predictions"/"exploratory_test_predictions.csv",index=False)
    result={"status":frozen["status"],"n_cycles":int(te.sum()),"ohss_positive":int(data.ohss[te].sum()),"oocyte":om,"ohss":hm,"oocyte_members":frozen["oocyte"]["members"],"ohss_members":frozen["ohss"]["members"]}
    write_json(OUT/"tables"/"exploratory_test_metrics.json",result); print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("cv","final"),required=True); a=p.parse_args()
    cv_stage() if a.stage=="cv" else final_stage()
