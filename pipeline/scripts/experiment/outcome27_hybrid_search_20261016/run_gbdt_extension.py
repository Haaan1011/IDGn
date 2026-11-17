from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/"scripts"/"experiment")]
import run_hybrid_search as hybrid  # noqa: E402
import run_robust_cv as robust  # noqa: E402
import run_outcome27_optimization_20261016 as prior  # noqa: E402

OUT=ROOT/"outputs"/"gru_addgate_outcome27_gbdt_extension_20261016"; SEED=20261016

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")

def feature_sets(data):
    summary,flat=hybrid.masked_summary(data); emb,info=robust.gn_embedding(data)
    # summary layout: static9 | first18 | last18 | mean18 | std18 | min18 | max18 | delta18 | slope18 | length1
    static=summary[:,:9]; first=summary[:,9:27]; last=summary[:,27:45]; mean=summary[:,45:63]; minimum=summary[:,81:99]; maximum=summary[:,99:117]; length=summary[:,-1:]
    return {
        "last":np.c_[static,last,length],
        "last_history":np.c_[static,first,last,mean,minimum,maximum,length],
        "summary":summary,
        "summary_emb":np.c_[summary,emb],
        "flat_emb":np.c_[flat,emb],
    },info

def candidates(target):
    out=[]
    for fs in ("last","last_history","summary","summary_emb"):
        if target=="oocyte":
            configs=[(7,40,.025,800,10,.5),(15,40,.02,900,12,.5),(15,60,.04,500,8,.5),(31,50,.02,800,15,1.0)]
            for leaves,child,lr,n,reg,alpha in configs:
                name=f"lgb_{fs}_l{leaves}_c{child}_lr{lr}_r{reg}"
                out.append((name,fs,LGBMRegressor(n_estimators=n,learning_rate=lr,num_leaves=leaves,min_child_samples=child,
                    subsample=.85,colsample_bytree=.8,reg_lambda=reg,reg_alpha=alpha,objective="regression_l1",verbosity=-1,n_jobs=-1,random_state=SEED)))
            for depth,lr,n,l2 in ((4,.025,700,10),(5,.02,800,15)):
                name=f"cat_{fs}_d{depth}_lr{lr}_l{l2}"
                out.append((name,fs,CatBoostRegressor(iterations=n,depth=depth,learning_rate=lr,l2_leaf_reg=l2,loss_function="RMSE",
                    verbose=False,allow_writing_files=False,random_seed=SEED,thread_count=-1)))
        else:
            configs=[(7,20,.02,800,10,.5),(7,40,.03,600,12,.5),(15,40,.02,800,15,.5),(15,60,.04,500,8,.5),(31,60,.02,700,20,1.0)]
            for leaves,child,lr,n,reg,alpha in configs:
                name=f"lgb_{fs}_l{leaves}_c{child}_lr{lr}_r{reg}"
                out.append((name,fs,LGBMClassifier(n_estimators=n,learning_rate=lr,num_leaves=leaves,min_child_samples=child,
                    subsample=.85,colsample_bytree=.8,reg_lambda=reg,reg_alpha=alpha,class_weight="balanced",verbosity=-1,n_jobs=-1,random_state=SEED)))
            for depth,lr,n,l2 in ((3,.025,700,12),(4,.02,800,18)):
                name=f"cat_{fs}_d{depth}_lr{lr}_l{l2}"
                out.append((name,fs,CatBoostClassifier(iterations=n,depth=depth,learning_rate=lr,l2_leaf_reg=l2,
                    auto_class_weights="Balanced",loss_function="Logloss",verbose=False,allow_writing_files=False,random_seed=SEED,thread_count=-1)))
    return out

def previous_predictions(data,xsets,select,target):
    f=json.loads((robust.OUT/"logs"/"frozen_cv_selection.json").read_text(encoding="utf-8")); te=select
    vals=[]
    for name in f[target]["members"]:
        z=f["model_manifest"][f"{target}::{name}"]; m=joblib.load(z["path"]); x=xsets[z["feature_set"]][te]
        vals.append(m.predict(x) if target=="oocyte" else m.predict_proba(x)[:,1])
    return np.mean(vals,axis=0)

def cv_stage():
    for s in ("models","tables","logs","predictions","audit"): (OUT/s).mkdir(parents=True,exist_ok=True)
    data,audit=prior.prepare_cycle_data(); split=prior.masks(data); dev=split["train"]|split["valid"]; ix=np.flatnonzero(dev)
    xsets,info=feature_sets(data); prev=pd.read_csv(robust.OUT/"predictions"/"fivefold_selected_oof.csv")
    if prev["cycle_uid"].astype(str).tolist()!=data.meta.loc[dev,"cycle_uid"].astype(str).tolist(): raise RuntimeError("OOF alignment failed")
    pools={"oocyte":{"previous_ensemble":prev.oocyte_oof.to_numpy()},"ohss":{"previous_ensemble":prev.ohss_oof.to_numpy()}}
    rows=[]; manifest={}
    for target,y,folds in (("oocyte",data.oocytes[dev],KFold(5,shuffle=True,random_state=SEED).split(ix)),
                           ("ohss",data.ohss[dev],StratifiedKFold(5,shuffle=True,random_state=SEED).split(ix,data.ohss[dev]))):
        folds=list(folds)
        for name,fs,model in candidates(target):
            p=np.zeros(len(ix))
            for a,b in folds:
                m=clone(model); m.fit(xsets[fs][ix[a]],y[a]); p[b]=m.predict(xsets[fs][ix[b]]) if target=="oocyte" else m.predict_proba(xsets[fs][ix[b]])[:,1]
            met=hybrid.regression_metrics(y,p) if target=="oocyte" else hybrid.classification_metrics(y,p)
            rows.append({"target":target,"candidate":name,"feature_set":fs,**met}); pools[target][name]=p
            print("CV",target,name,met,flush=True)
        names,p,met=(hybrid.greedy_regression_ensemble(y,pools[target]) if target=="oocyte" else hybrid.greedy_auc_ensemble(y,pools[target]))
        manifest[target]={"members":names,"cv_metrics":met}
        byname={n:(fs,m) for n,fs,m in candidates(target)}
        for name in names:
            if name=="previous_ensemble": continue
            fs,m=byname[name]; m.fit(xsets[fs][ix],y); path=OUT/"models"/f"{target}__{name}.joblib"; joblib.dump(m,path)
            manifest[f"{target}::{name}"]={"path":str(path),"feature_set":fs}
        predcol="oocyte_oof" if target=="oocyte" else "ohss_oof"; prev[predcol+"_gbdt_extension"]=p
    manifest.update({"status":"exploratory; selection uses five-fold development CV only after prior test inspection","seed":SEED,"pretraining":info})
    pd.DataFrame(rows).to_csv(OUT/"tables"/"fivefold_gbdt_candidate_screen.csv",index=False); prev.to_csv(OUT/"predictions"/"fivefold_selected_oof.csv",index=False)
    write_json(OUT/"logs"/"frozen_cv_selection.json",manifest); write_json(OUT/"audit"/"data_contract.json",audit)
    print(json.dumps({"oocyte":manifest["oocyte"],"ohss":manifest["ohss"]},indent=2),flush=True)

def final_stage():
    f=json.loads((OUT/"logs"/"frozen_cv_selection.json").read_text(encoding="utf-8")); data,_=prior.prepare_cycle_data(); split=prior.masks(data); te=split["test"]
    xsets,_=feature_sets(data)
    def one(target,name):
        if name=="previous_ensemble": return previous_predictions(data,xsets,te,target)
        z=f[f"{target}::{name}"]; m=joblib.load(z["path"]); x=xsets[z["feature_set"]][te]
        return m.predict(x) if target=="oocyte" else m.predict_proba(x)[:,1]
    po=np.mean([one("oocyte",n) for n in f["oocyte"]["members"]],axis=0); ph=np.mean([one("ohss",n) for n in f["ohss"]["members"]],axis=0)
    om=hybrid.regression_metrics(data.oocytes[te],po); hm=hybrid.classification_metrics(data.ohss[te],ph); bt=hybrid.bootstrap(data.oocytes[te],po,data.ohss[te],ph)
    bt.to_csv(OUT/"tables"/"test_cycle_bootstrap_1000.csv",index=False); est={**om,**hm}; ci=[]
    for c in ("r2","mae","rmse","roc_auc"):
        q=bt[c].dropna().quantile([.025,.975]); ci.append({"metric":c,"estimate":est[c],"lower_95ci":q.iloc[0],"upper_95ci":q.iloc[1]})
    pd.DataFrame(ci).to_csv(OUT/"tables"/"exploratory_test_metrics_95ci.csv",index=False)
    pd.DataFrame({"cycle_uid":data.meta.loc[te,"cycle_uid"],"oocytes":data.oocytes[te],"oocyte_prediction":po,"ohss":data.ohss[te],"ohss_probability":ph}).to_csv(OUT/"predictions"/"exploratory_test_predictions.csv",index=False)
    result={"status":f["status"],"n_cycles":int(te.sum()),"ohss_positive":int(data.ohss[te].sum()),"oocyte":om,"ohss":hm,"oocyte_members":f["oocyte"]["members"],"ohss_members":f["ohss"]["members"]}
    write_json(OUT/"tables"/"exploratory_test_metrics.json",result); print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("cv","final"),required=True); a=p.parse_args(); cv_stage() if a.stage=="cv" else final_stage()
