from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "experiment"))
import run_outcome27_optimization_20261016 as prior  # noqa: E402


OUT = ROOT / "outputs" / "gru_addgate_outcome27_hybrid_search_20261016"
SEED = 20261016
N_BOOT = 1000


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def regression_metrics(y, p):
    return {
        "r2": float(r2_score(y, p)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
    }


def classification_metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def masked_summary(data):
    x = data.dynamic.astype(np.float64)
    m = data.mask.astype(bool)
    n, t, d = x.shape
    lengths = m.sum(1).astype(int)
    first = x[:, 0, :]
    last = np.stack([x[i, lengths[i] - 1] for i in range(n)])
    means = np.zeros((n, d)); stds = np.zeros((n, d)); mins = np.zeros((n, d)); maxs = np.zeros((n, d)); slopes = np.zeros((n, d))
    for i in range(n):
        z = x[i, : lengths[i]]
        means[i] = z.mean(0); stds[i] = z.std(0); mins[i] = z.min(0); maxs[i] = z.max(0)
        if lengths[i] > 1:
            tt = np.arange(lengths[i], dtype=float)
            tt -= tt.mean()
            slopes[i] = (tt[:, None] * (z - z.mean(0))).sum(0) / max(float((tt * tt).sum()), 1e-12)
    delta = last - first
    summary = np.concatenate([data.static, first, last, means, stds, mins, maxs, delta, slopes, lengths[:, None]], axis=1)
    flat = np.concatenate([data.static, x.reshape(n, -1), data.mask], axis=1)
    return summary.astype(np.float32), flat.astype(np.float32)


@torch.inference_mode()
def checkpoint_features(data, selection):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sx = torch.tensor(data.static, dtype=torch.float32, device=device)
    dx = torch.tensor(data.dynamic, dtype=torch.float32, device=device)
    mm = torch.tensor(data.mask, dtype=torch.float32, device=device)
    result = {}
    for target in ("oocyte", "ohss"):
        embeddings, predictions = [], []
        for path in selection[target]["checkpoints"]:
            model = prior.load_model(Path(path), data, device)
            model.eval()
            hs, ps = [], []
            for start in range(0, len(sx), 512):
                h = model.encode(sx[start:start + 512], dx[start:start + 512], mm[start:start + 512])
                count, _, logit = model(sx[start:start + 512], dx[start:start + 512], mm[start:start + 512])
                hs.append(h.cpu().numpy())
                ps.append((count if target == "oocyte" else torch.sigmoid(logit)).cpu().numpy())
            embeddings.append(np.concatenate(hs)); predictions.append(np.concatenate(ps))
            del model
        result[target + "_embedding"] = np.mean(embeddings, axis=0).astype(np.float32)
        result[target + "_neural"] = np.mean(predictions, axis=0).astype(np.float64)
    return result


def build_features(data, selection):
    summary, flat = masked_summary(data)
    deep = checkpoint_features(data, selection)
    return {
        "summary": summary,
        "flat": flat,
        "summary_emb": np.concatenate([summary, deep["oocyte_embedding"], deep["ohss_embedding"]], axis=1),
        "flat_emb": np.concatenate([flat, deep["oocyte_embedding"], deep["ohss_embedding"]], axis=1),
    }, deep


def oocyte_candidates():
    models = []
    for fs in ("summary", "summary_emb", "flat_emb"):
        models.append((f"ridge_{fs}", fs, make_pipeline(StandardScaler(), Ridge(alpha=20.0))))
        for leaf in (2, 5, 10):
            models.append((f"extra_{fs}_leaf{leaf}", fs, ExtraTreesRegressor(
                n_estimators=700, min_samples_leaf=leaf, max_features=0.8, n_jobs=-1, random_state=SEED)))
        for depth, lr, estimators, child, reg in (
            (2, .025, 800, 8, 8), (3, .02, 900, 10, 10), (3, .03, 650, 15, 15),
            (4, .02, 800, 20, 20), (2, .04, 500, 5, 5),
        ):
            models.append((f"xgb_{fs}_d{depth}_lr{lr}_c{child}_r{reg}", fs, XGBRegressor(
                n_estimators=estimators, learning_rate=lr, max_depth=depth, min_child_weight=child,
                subsample=.85, colsample_bytree=.8, reg_lambda=reg, reg_alpha=.2,
                objective="reg:squarederror", n_jobs=-1, random_state=SEED, tree_method="hist")))
    return models


def ohss_candidates(pos_weight):
    models = []
    for fs in ("summary", "summary_emb", "flat_emb"):
        for c in (.01, .05, .2, 1.0):
            models.append((f"logit_{fs}_c{c}", fs, make_pipeline(
                StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=SEED))))
        for leaf in (2, 5, 10):
            models.append((f"extra_{fs}_leaf{leaf}", fs, ExtraTreesClassifier(
                n_estimators=1000, min_samples_leaf=leaf, max_features=.8, class_weight="balanced",
                n_jobs=-1, random_state=SEED)))
        for depth, lr, estimators, child, reg, spw in (
            (1, .02, 800, 3, 10, math.sqrt(pos_weight)),
            (2, .015, 1000, 5, 15, math.sqrt(pos_weight)),
            (2, .02, 800, 10, 20, pos_weight),
            (3, .015, 900, 10, 25, math.sqrt(pos_weight)),
            (3, .02, 700, 20, 30, pos_weight),
        ):
            models.append((f"xgb_{fs}_d{depth}_lr{lr}_c{child}_r{reg}_w{spw:.1f}", fs, XGBClassifier(
                n_estimators=estimators, learning_rate=lr, max_depth=depth, min_child_weight=child,
                subsample=.85, colsample_bytree=.8, reg_lambda=reg, reg_alpha=.2, gamma=.1,
                scale_pos_weight=spw, objective="binary:logistic", eval_metric="aucpr",
                n_jobs=-1, random_state=SEED, tree_method="hist")))
    return models


def greedy_regression_ensemble(y, pool, max_models=8):
    names = list(pool)
    best = min(names, key=lambda n: (regression_metrics(y, pool[n])["rmse"], regression_metrics(y, pool[n])["mae"]))
    selected = [best]; pred = pool[best].copy(); current = regression_metrics(y, pred)
    while len(selected) < max_models:
        choice = None
        for name in names:
            if name in selected: continue
            q = (pred * len(selected) + pool[name]) / (len(selected) + 1)
            met = regression_metrics(y, q)
            if met["rmse"] < current["rmse"] - 1e-5 and met["mae"] <= current["mae"] + .01:
                key = (met["rmse"], met["mae"])
                if choice is None or key < choice[0]: choice = (key, name, q, met)
        if choice is None: break
        _, name, pred, current = choice; selected.append(name)
    return selected, pred, current


def greedy_auc_ensemble(y, pool, max_models=10):
    names = list(pool)
    best = max(names, key=lambda n: (roc_auc_score(y, pool[n]), average_precision_score(y, pool[n])))
    selected = [best]; pred = pool[best].copy(); current = classification_metrics(y, pred)
    while len(selected) < max_models:
        choice = None
        for name in names:
            if name in selected: continue
            q = (pred * len(selected) + pool[name]) / (len(selected) + 1)
            met = classification_metrics(y, q)
            if met["roc_auc"] > current["roc_auc"] + 1e-5:
                key = (-met["roc_auc"], -met["average_precision"])
                if choice is None or key < choice[0]: choice = (key, name, q, met)
        if choice is None: break
        _, name, pred, current = choice; selected.append(name)
    return selected, pred, current


def validation_stage():
    for sub in ("models", "tables", "logs", "predictions", "audit"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    data, audit = prior.prepare_cycle_data(); split = prior.masks(data)
    selection = json.loads((prior.OUT / "logs" / "frozen_validation_selection.json").read_text(encoding="utf-8"))["selection"]
    features, deep = build_features(data, selection)
    tr, va = split["train"], split["valid"]
    rows = []; oof_o = {"neural": deep["oocyte_neural"][va]}; oof_h = {"neural": deep["ohss_neural"][va]}
    model_manifest = {}
    for name, fs, model in oocyte_candidates():
        model.fit(features[fs][tr], data.oocytes[tr]); p = model.predict(features[fs][va]); met = regression_metrics(data.oocytes[va], p)
        rows.append({"target":"oocyte","candidate":name,"feature_set":fs,**met}); oof_o[name]=p
        path = OUT / "models" / f"oocyte__{name}.joblib"; joblib.dump(model,path); model_manifest[name]={"path":str(path),"feature_set":fs}
        print("OOCYTE", name, met, flush=True)
    pos_weight = float((tr.sum() - data.ohss[tr].sum()) / data.ohss[tr].sum())
    for name, fs, model in ohss_candidates(pos_weight):
        model.fit(features[fs][tr], data.ohss[tr]); p = model.predict_proba(features[fs][va])[:,1]; met = classification_metrics(data.ohss[va], p)
        rows.append({"target":"ohss","candidate":name,"feature_set":fs,**met}); oof_h[name]=p
        path = OUT / "models" / f"ohss__{name}.joblib"; joblib.dump(model,path); model_manifest[name]={"path":str(path),"feature_set":fs}
        print("OHSS", name, met, flush=True)
    o_names, o_pred, o_met = greedy_regression_ensemble(data.oocytes[va], oof_o)
    h_names, h_pred, h_met = greedy_auc_ensemble(data.ohss[va], oof_h)
    frozen = {
        "development_status":"exploratory redevelopment because fixed test had been inspected before this search",
        "selection_data":"validation only", "test_accessed_in_this_stage":False,
        "features":"same 27 source variables; longitudinal summaries and frozen GRU(AddGate) embeddings are derived representations",
        "oocyte":{"members":o_names,"validation_metrics":o_met},
        "ohss":{"members":h_names,"validation_metrics":h_met},
        "model_manifest":model_manifest,"seed":SEED,
    }
    pd.DataFrame(rows).to_csv(OUT/"tables"/"validation_candidate_screen.csv",index=False)
    pd.DataFrame({"cycle_uid":data.meta.loc[va,"cycle_uid"],"oocytes":data.oocytes[va],"oocyte_prediction":o_pred,"ohss":data.ohss[va],"ohss_probability":h_pred}).to_csv(OUT/"predictions"/"validation_selected_predictions.csv",index=False)
    write_json(OUT/"logs"/"frozen_exploratory_selection.json",frozen); write_json(OUT/"audit"/"data_contract.json",audit)
    print(json.dumps({"oocyte":frozen["oocyte"],"ohss":frozen["ohss"]},indent=2),flush=True)


def bootstrap(y_o,p_o,y_h,p_h):
    rng=np.random.default_rng(SEED); rows=[]; n=len(y_o)
    for b in range(N_BOOT):
        ix=rng.integers(0,n,n); yo=y_o[ix]; po=p_o[ix]; yh=y_h[ix]; ph=p_h[ix]
        r=regression_metrics(yo,po); auc=np.nan if len(np.unique(yh))<2 else roc_auc_score(yh,ph)
        rows.append({"replicate":b+1,**r,"roc_auc":auc})
    return pd.DataFrame(rows)


def final_stage():
    frozen=json.loads((OUT/"logs"/"frozen_exploratory_selection.json").read_text(encoding="utf-8"))
    data,_=prior.prepare_cycle_data(); split=prior.masks(data)
    selection=json.loads((prior.OUT/"logs"/"frozen_validation_selection.json").read_text(encoding="utf-8"))["selection"]
    features,deep=build_features(data,selection); te=split["test"]
    def get_pred(target,name):
        if name=="neural": return deep[target+"_neural"][te]
        info=frozen["model_manifest"][name]; model=joblib.load(info["path"]); x=features[info["feature_set"]][te]
        return model.predict(x) if target=="oocyte" else model.predict_proba(x)[:,1]
    po=np.mean([get_pred("oocyte",n) for n in frozen["oocyte"]["members"]],axis=0)
    ph=np.mean([get_pred("ohss",n) for n in frozen["ohss"]["members"]],axis=0)
    om=regression_metrics(data.oocytes[te],po); hm=classification_metrics(data.ohss[te],ph)
    bt=bootstrap(data.oocytes[te],po,data.ohss[te],ph); bt.to_csv(OUT/"tables"/"test_cycle_bootstrap_1000.csv",index=False)
    cis=[]
    for col in ("r2","mae","rmse","roc_auc"):
        q=bt[col].dropna().quantile([.025,.975]); cis.append({"metric":col,"estimate":(om|hm)[col],"lower_95ci":q.iloc[0],"upper_95ci":q.iloc[1]})
    pd.DataFrame(cis).to_csv(OUT/"tables"/"exploratory_test_metrics_95ci.csv",index=False)
    pd.DataFrame({"cycle_uid":data.meta.loc[te,"cycle_uid"],"oocytes":data.oocytes[te],"oocyte_prediction":po,"ohss":data.ohss[te],"ohss_probability":ph}).to_csv(OUT/"predictions"/"exploratory_test_predictions.csv",index=False)
    result={"status":"exploratory; original fixed test had already been inspected in prior experiment","n_cycles":int(te.sum()),"ohss_positive":int(data.ohss[te].sum()),"oocyte_members":frozen["oocyte"]["members"],"ohss_members":frozen["ohss"]["members"],"oocyte":om,"ohss":hm}
    write_json(OUT/"tables"/"exploratory_test_metrics.json",result); print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("validation","final"),required=True); a=p.parse_args()
    validation_stage() if a.stage=="validation" else final_stage()


if __name__=="__main__": main()
