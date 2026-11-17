from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/"scripts"/"experiment"/"outcome27_hybrid_search_20261016"),str(ROOT/"scripts"/"experiment")]
import run_gbdt_extension as ext  # noqa: E402
import run_hybrid_search as hybrid  # noqa: E402
import run_robust_cv as robust  # noqa: E402
import run_outcome27_optimization_20261016 as prior  # noqa: E402

OUT=ROOT/"outputs"/"outcome27_model_family_comparison_20261016"; SEED=20261016; N_BOOT=1000

def reg_metrics(y,p): return {"r2":float(r2_score(y,p)),"mae":float(mean_absolute_error(y,p)),"rmse":float(mean_squared_error(y,p)**.5)}
def cls_metrics(y,p): return {"roc_auc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p))}

def regressors():
    return {
        "Ridge":make_pipeline(StandardScaler(),Ridge(alpha=20.0)),
        "RandomForest":RandomForestRegressor(n_estimators=700,min_samples_leaf=5,max_features=.8,n_jobs=-1,random_state=SEED),
        "ExtraTrees":ExtraTreesRegressor(n_estimators=700,min_samples_leaf=5,max_features=.8,n_jobs=-1,random_state=SEED),
        "GradientBoosting":GradientBoostingRegressor(n_estimators=400,learning_rate=.025,max_depth=2,min_samples_leaf=10,loss="huber",random_state=SEED),
        "HistGradientBoosting":HistGradientBoostingRegressor(max_iter=400,learning_rate=.04,max_leaf_nodes=15,l2_regularization=10,random_state=SEED),
        "XGBoost":XGBRegressor(n_estimators=800,learning_rate=.025,max_depth=2,min_child_weight=8,subsample=.85,colsample_bytree=.8,reg_lambda=8,reg_alpha=.2,objective="reg:squarederror",tree_method="hist",n_jobs=-1,random_state=SEED),
        "LightGBM":LGBMRegressor(n_estimators=800,learning_rate=.025,num_leaves=7,min_child_samples=40,subsample=.85,colsample_bytree=.8,reg_lambda=10,reg_alpha=.5,objective="regression_l1",verbosity=-1,n_jobs=-1,random_state=SEED),
        "CatBoost":CatBoostRegressor(iterations=700,depth=4,learning_rate=.025,l2_leaf_reg=10,loss_function="RMSE",verbose=False,allow_writing_files=False,thread_count=-1,random_seed=SEED),
        "MLP":make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=500,early_stopping=True,validation_fraction=.15,n_iter_no_change=25,random_state=SEED)),
    }

def classifiers(pos_weight):
    return {
        "LogisticRegression":make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight="balanced",max_iter=5000,solver="liblinear",random_state=SEED)),
        "LDA":make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(shrinkage="auto",solver="lsqr")),
        "QDA":make_pipeline(StandardScaler(),QuadraticDiscriminantAnalysis(reg_param=.3)),
        "GaussianNB":GaussianNB(),
        "RandomForest":RandomForestClassifier(n_estimators=900,min_samples_leaf=8,max_features=.8,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED),
        "ExtraTrees":ExtraTreesClassifier(n_estimators=900,min_samples_leaf=10,max_features=.8,class_weight="balanced",n_jobs=-1,random_state=SEED),
        "GradientBoosting":GradientBoostingClassifier(n_estimators=400,learning_rate=.025,max_depth=2,min_samples_leaf=10,random_state=SEED),
        "HistGradientBoosting":HistGradientBoostingClassifier(max_iter=400,learning_rate=.04,max_leaf_nodes=15,l2_regularization=10,class_weight="balanced",random_state=SEED),
        "XGBoost":XGBClassifier(n_estimators=800,learning_rate=.02,max_depth=1,min_child_weight=3,subsample=.85,colsample_bytree=.8,reg_lambda=10,reg_alpha=.2,gamma=.1,scale_pos_weight=np.sqrt(pos_weight),objective="binary:logistic",eval_metric="aucpr",tree_method="hist",n_jobs=-1,random_state=SEED),
        "LightGBM":LGBMClassifier(n_estimators=800,learning_rate=.02,num_leaves=7,min_child_samples=20,subsample=.85,colsample_bytree=.8,reg_lambda=10,reg_alpha=.5,class_weight="balanced",verbosity=-1,n_jobs=-1,random_state=SEED),
        "CatBoost":CatBoostClassifier(iterations=700,depth=3,learning_rate=.025,l2_leaf_reg=12,auto_class_weights="Balanced",loss_function="Logloss",verbose=False,allow_writing_files=False,thread_count=-1,random_seed=SEED),
        "MLP":make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),alpha=.05,learning_rate_init=.001,max_iter=500,early_stopping=True,validation_fraction=.15,n_iter_no_change=25,random_state=SEED)),
    }

def bootstrap_deltas(y_o,gru_o,p_o,y_h,gru_h,p_h):
    rng=np.random.default_rng(SEED); rows=[]; n=len(y_o)
    for b in range(N_BOOT):
        ix=rng.integers(0,n,n); yo=y_o[ix]; po=p_o[ix]; go=gru_o[ix]; yh=y_h[ix]; ph=p_h[ix]; gh=gru_h[ix]
        a=reg_metrics(yo,po); g=reg_metrics(yo,go)
        row={"replicate":b+1,"delta_r2":a["r2"]-g["r2"],"delta_mae":a["mae"]-g["mae"],"delta_rmse":a["rmse"]-g["rmse"]}
        row["delta_roc_auc"]=np.nan if len(np.unique(yh))<2 else roc_auc_score(yh,ph)-roc_auc_score(yh,gh)
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    warnings.filterwarnings("ignore")
    for s in ("models","tables","predictions","bootstrap","audit"): (OUT/s).mkdir(parents=True,exist_ok=True)
    data,audit=prior.prepare_cycle_data(); split=prior.masks(data); tr=split["train"]; te=split["test"]
    xsets,pretrain=ext.feature_sets(data); tab=xsets["last_history"]; hyb=np.c_[xsets["last_history"],xsets["summary_emb"][:,-128:]]
    direct=pd.read_csv(prior.OUT/"predictions"/"test_predictions.csv"); direct["cycle_uid"]=direct.cycle_uid.astype(str)
    order=data.meta.loc[te,"cycle_uid"].astype(str).tolist(); direct=direct.set_index("cycle_uid").loc[order].reset_index()
    gru_o=direct.predicted_oocytes.to_numpy(float); gru_h=direct.predicted_ohss_probability.to_numpy(float)
    yo=data.oocytes[te]; yh=data.ohss[te]; yot=data.oocytes[tr]; yht=data.ohss[tr]
    base_o=reg_metrics(yo,gru_o); base_h=cls_metrics(yh,gru_h)
    rows=[{"target":"Oocytes","representation":"Pure GRU(AddGate)","model":"GRU(AddGate)",**base_o},
          {"target":"OHSS","representation":"Pure GRU(AddGate)","model":"GRU(AddGate)",**base_h}]
    pred_table=pd.DataFrame({"cycle_uid":order,"oocytes":yo,"ohss":yh,"GRU__oocyte":gru_o,"GRU__ohss":gru_h})
    delta_rows=[]; all_boot=[]
    pos_weight=float((tr.sum()-yht.sum())/yht.sum())
    for representation,x in (("Tabular longitudinal summary",tab),("GRU embedding + tabular summary",hyb)):
        for name,model in regressors().items():
            model.fit(x[tr],yot); p=model.predict(x[te]); met=reg_metrics(yo,p); key=("TAB" if representation.startswith("Tabular") else "HYB")+"__"+name
            rows.append({"target":"Oocytes","representation":representation,"model":name,**met,
                         "delta_r2_vs_gru":met["r2"]-base_o["r2"],"delta_mae_vs_gru":met["mae"]-base_o["mae"],"delta_rmse_vs_gru":met["rmse"]-base_o["rmse"]})
            pred_table[key+"__oocyte"]=p; joblib.dump(model,OUT/"models"/(key+"__oocyte.joblib")); print("OOCYTE",key,met,flush=True)
        for name,model in classifiers(pos_weight).items():
            model.fit(x[tr],yht); p=model.predict_proba(x[te])[:,1]; met=cls_metrics(yh,p); key=("TAB" if representation.startswith("Tabular") else "HYB")+"__"+name
            rows.append({"target":"OHSS","representation":representation,"model":name,**met,"delta_roc_auc_vs_gru":met["roc_auc"]-base_h["roc_auc"]})
            pred_table[key+"__ohss"]=p; joblib.dump(model,OUT/"models"/(key+"__ohss.joblib")); print("OHSS",key,met,flush=True)
    metrics=pd.DataFrame(rows); metrics.to_csv(OUT/"tables"/"all_model_metrics_and_deltas_vs_gru.csv",index=False)
    pred_table.to_csv(OUT/"predictions"/"test_predictions_all_models.csv",index=False)
    # Pair each expert's oocyte and OHSS predictions for paired bootstrap deltas against pure GRU.
    for rep in ("TAB","HYB"):
        common=sorted(set(regressors()) & set(classifiers(pos_weight)))
        for name in common:
            po=pred_table[f"{rep}__{name}__oocyte"].to_numpy(); ph=pred_table[f"{rep}__{name}__ohss"].to_numpy()
            bt=bootstrap_deltas(yo,gru_o,po,yh,gru_h,ph); bt.insert(0,"representation",rep); bt.insert(1,"model",name); all_boot.append(bt)
            out={"representation":rep,"model":name}
            for c in ("delta_r2","delta_mae","delta_rmse","delta_roc_auc"):
                q=bt[c].dropna().quantile([.025,.975]); out[c]=float(bt[c].mean()); out[c+"_ci_lower"]=float(q.iloc[0]); out[c+"_ci_upper"]=float(q.iloc[1])
            delta_rows.append(out)
    pd.concat(all_boot,ignore_index=True).to_csv(OUT/"bootstrap"/"paired_delta_bootstrap_1000.csv",index=False)
    pd.DataFrame(delta_rows).to_csv(OUT/"tables"/"paired_delta_vs_pure_gru_95ci.csv",index=False)
    write={"status":"exploratory comparison; fixed test was inspected before this experiment","training_cycles":int(tr.sum()),"test_cycles":int(te.sum()),"test_ohss_positive":int(yh.sum()),"source_features":27,"tabular_representation":"static, first, last, mean, min, max and sequence length","hybrid_representation":"tabular representation plus frozen Gn-pretrained GRU(AddGate) embedding","pure_gru":{**base_o,**base_h},"seed":SEED,"bootstrap_replicates":N_BOOT,"pretraining":pretrain}
    (OUT/"audit"/"comparison_contract.json").write_text(json.dumps(write,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    print(json.dumps(write,indent=2),flush=True)

if __name__=="__main__": main()
