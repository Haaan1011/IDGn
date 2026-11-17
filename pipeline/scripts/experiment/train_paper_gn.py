#!/usr/bin/env python3
"""Train and evaluate the manuscript's 27- or 62-feature dose model.

The clinical target is the post-Monitor-k Gn decision stored in next_*_daily_dose.
The split, target definitions, model configurations, expert family, and fusion weights
are locked from the 27-feature run.  Only the input feature set differs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts/experiment/monitor_level_27feature_current_decision_20261015_run.py"
PARSIMONIOUS_OUT = ROOT / "outputs/monitor_level_shared_multitask_gru_addgate_27features_current_decision_tuned_20261015"
OUT = PARSIMONIOUS_OUT
SEED = 20261015
BOOTSTRAP_N = 1000

# The current-monitor variant replaces the three daily-expanded input fields
# with the three pre-decision drug doses.
STATIC_FEATURES = [
    "age", "bmi", "infertility_duration", "amh", "afc", "basal_fsh", "basal_lh",
    "basal_e2", "basal_p", "male_age", "male_factor_infertility_flag",
]
DYNAMIC_FEATURES = [
    "stim_day", "treatment_count", "fresh_treatment_count", "monitoring_order",
    "current_e2", "current_lh", "current_p", "current_fsh", "current_endometrium",
    "delta_e2", "delta_lh", "delta_p", "delta_fsh", "delta_endometrium", "visits_seen",
    "total_follicle_count", "max_follicle_diameter", "mean_follicle_diameter",
    "mature_follicle_count", "mature_follicle_share", "large_follicle_share",
    "dominant_follicle_count", "follicle_count_lt_10", "follicle_count_10_12",
    "follicle_count_13_15", "follicle_count_16_18", "follicle_count_gt_18",
    "medium_plus_follicle_count", "ohss_follicle_load_score", "follicle_size_weighted_count",
    "follicle_maturity_index", "mid_follicle_share", "large_to_mature_follicle_ratio",
    "e2_per_mature_follicle", "e2_per_weighted_follicle", "p_lh_ratio",
    "delta_p_per_day", "delta_lh_per_day", "previous_fsh_daily_dose",
    "previous_lh_daily_dose", "previous_hmg_daily_dose", "previous_lh_like_hmg_daily_dose",
    "current_fsh_daily_dose", "current_lh_daily_dose",
    "current_hmg_daily_dose", "cycle_day", "gn_day", "days_since_previous_visit",
    "left_follicle_count", "right_follicle_count", "growing_follicle_count",
]
FROZEN_SELECTION = {
    "fsh": {"neural": "attn_latest_adapter_h128", "expert": "rf_unlimited", "weight": 0.775},
    "lh": {"neural": "attn_latest_adapter_smooth", "expert": "xgb_d4", "weight": 0.475},
    "hmg": {"neural": "attn_latest_adapter_smooth", "expert": "et_unlimited_leaf4", "weight": 0.925},
}


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = import_module("full55_current_decision_base", BASE_SCRIPT)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def masked_temporal_features(arrays) -> np.ndarray:
    dynamic = arrays.dynamic.astype(np.float32)
    mask = arrays.mask.astype(bool)
    lengths = mask.sum(axis=1).astype(int).clip(min=1)
    rows = np.arange(len(dynamic))
    latest = dynamic[rows, lengths - 1]
    previous = dynamic[rows, np.maximum(lengths - 2, 0)]
    delta = np.where((lengths > 1)[:, None], latest - previous, 0.0)
    values = np.where(mask[:, :, None], dynamic, np.nan)
    return np.column_stack([
        arrays.static.astype(np.float32), latest, previous, delta,
        np.nanmean(values, axis=1), np.nanstd(values, axis=1),
        np.nanmin(values, axis=1), np.nanmax(values, axis=1), dynamic[:, 0],
        latest - dynamic[:, 0], lengths.astype(np.float32),
    ]).astype(np.float32)


def score(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    labels = list(range(probability.shape[1]))
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, labels=labels, average="weighted", zero_division=0)),
    }


def fit_expert(target: str, features: np.ndarray, labels: dict[str, np.ndarray], masks: dict[str, np.ndarray]):
    train_mask = masks["train"] & (labels[target] >= 0)
    valid_mask = masks["valid"] & (labels[target] >= 0)
    train_y = labels[target][train_mask]
    valid_y = labels[target][valid_mask]
    counts = np.bincount(train_y, minlength=len(base.TARGETS[target]["labels"])).astype(float)
    sample_weight = np.power(counts[train_y] / counts.max(), -0.15)
    seed = SEED + base.TARGETS[target]["offset"]
    selected = FROZEN_SELECTION[target]["expert"]
    if selected == "rf_unlimited":
        model = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3, max_features=0.5,
            n_jobs=12, random_state=seed,
        )
        model.fit(features[train_mask], train_y, sample_weight=sample_weight)
    elif selected == "et_unlimited_leaf4":
        model = ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=4, max_features=1.0,
            n_jobs=12, random_state=seed,
        )
        model.fit(features[train_mask], train_y, sample_weight=sample_weight)
    elif selected == "xgb_d4":
        model = XGBClassifier(
            n_estimators=1600, max_depth=4, learning_rate=0.030, min_child_weight=5,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=2.0,
            objective="multi:softprob", num_class=len(base.TARGETS[target]["labels"]),
            eval_metric="mlogloss", early_stopping_rounds=80, random_state=seed,
            tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu", n_jobs=8,
        )
        model.fit(features[train_mask], train_y, sample_weight=sample_weight,
                  eval_set=[(features[valid_mask], valid_y)], verbose=False)
    else:
        raise ValueError(f"Unsupported fixed expert: {selected}")
    joblib.dump(model, OUT / "checkpoints" / f"{target}_{selected}.joblib")
    return model, score(valid_y, model.predict_proba(features[valid_mask]))


def bootstrap_values(frame: pd.DataFrame, probability: np.ndarray, target: str) -> pd.DataFrame:
    valid = frame["y_true"].to_numpy() >= 0
    work = frame.loc[valid].reset_index(drop=True)
    probs = probability[valid]
    groups = {cycle: indices.to_numpy() for cycle, indices in work.groupby("cycle_uid").groups.items()}
    cycles = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(SEED + base.TARGETS[target]["offset"] + 5000)
    rows = []
    for replicate in range(BOOTSTRAP_N):
        sampled = rng.choice(cycles, size=len(cycles), replace=True)
        indices = np.concatenate([groups[cycle] for cycle in sampled])
        y = work.loc[indices, "y_true"].to_numpy(int)
        pred = probs[indices].argmax(axis=1)
        rows.append({
            "target": target.upper(), "replicate": replicate + 1,
            "accuracy": accuracy_score(y, pred),
            "weighted_f1": f1_score(y, pred, labels=list(range(probs.shape[1])), average="weighted", zero_division=0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, choices=(27, 62), required=True)
    args = parser.parse_args()
    if args.features == 62:
        OUT = ROOT / "outputs/gn_full62"
        base.STATIC_FEATURES = list(STATIC_FEATURES)
        base.DYNAMIC_FEATURES = list(DYNAMIC_FEATURES)
    if len(set(base.STATIC_FEATURES + base.DYNAMIC_FEATURES)) != args.features:
        raise RuntimeError("Feature count does not match the selected configuration.")
    for folder in ("checkpoints", "predictions", "training", "tables", "logs"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    base.OUT = OUT
    base.SEED = SEED

    manifest = pd.read_csv(base.MANIFEST_PATH)
    if manifest["split"].value_counts().to_dict() != {"train": 2849, "valid": 611, "test": 611}:
        raise RuntimeError("Frozen split manifest failed 2849/611/611 verification")
    source = pd.read_csv(base.DATA_PATH, low_memory=False)
    frame, data_audit = base.build_monitor_frame(source)
    raw, labels, metadata = base.prepare_master(frame)
    split_map = manifest.assign(cycle_uid=manifest["cycle_uid"].astype(str)).set_index("cycle_uid")["split"]
    row_splits = raw.row_index["cycle_uid"].astype(str).map(split_map)
    if row_splits.isna().any():
        raise RuntimeError("Rows absent from frozen split manifest")
    masks = {split: row_splits.eq(split).to_numpy(bool) for split in ("train", "valid", "test")}
    arrays = base.phase864.prepare_arrays(raw, fit_mask=masks["train"])
    if int(masks["test"].sum()) != 4291 or int(raw.row_index.loc[masks["test"], "cycle_uid"].nunique()) != 611:
        raise RuntimeError("Test records/cycles do not match the frozen 27-feature evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Supplementary Methods S2 specifies shared AdamW settings.
    candidates = {
        candidate.name: replace(candidate, learning_rate=0.0005, weight_decay=0.0001)
        for candidate in base.CANDIDATES
    }
    neural_names = sorted({item["neural"] for item in FROZEN_SELECTION.values()})
    for name in neural_names:
        base.train_candidate(candidates[name], arrays, labels, masks, device)

    temporal = masked_temporal_features(arrays)
    experts, validation_rows = {}, []
    for target in base.TARGET_ORDER:
        model, validation_metric = fit_expert(target, temporal, labels, masks)
        experts[target] = model
        validation_rows.append({"target": target.upper(), "expert": FROZEN_SELECTION[target]["expert"], **validation_metric})
    pd.DataFrame(validation_rows).to_csv(OUT / "tables" / "validation_fixed_expert_metrics.csv", index=False)

    neural_predictions = {"valid": {}, "test": {}}
    for name in neural_names:
        model = base.load_model(OUT / "checkpoints" / f"{name}.pt", arrays.static.shape[1], arrays.dynamic.shape[2], device)
        for split in neural_predictions:
            neural_predictions[split][name] = base.predict(model, arrays, labels, masks[split], device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_rows, bootstrap_frames = [], []
    for target in base.TARGET_ORDER:
        frozen = FROZEN_SELECTION[target]
        neural_probability = neural_predictions["test"][frozen["neural"]][target]
        expert_probability = experts[target].predict_proba(temporal[masks["test"]])
        probability = (1.0 - frozen["weight"]) * neural_probability + frozen["weight"] * expert_probability
        y = labels[target][masks["test"]]
        test_rows = raw.row_index.loc[masks["test"]].reset_index(drop=True).copy()
        test_rows = pd.concat([test_rows, metadata.loc[masks["test"]].reset_index(drop=True)], axis=1)
        test_rows["target"] = target.upper()
        test_rows["y_true"] = y
        test_rows["y_pred"] = probability.argmax(axis=1)
        for index, label in enumerate(base.TARGETS[target]["labels"]):
            test_rows[f"prob_{label}"] = probability[:, index]
        suffix = "all_expert_hybrid" if args.features == 27 else "full62"
        test_rows.loc[y >= 0].to_csv(OUT / "predictions" / f"{target}_{suffix}_test_predictions.csv", index=False)
        validation = pd.DataFrame({"y_true": labels[target][masks["valid"]]})
        valid_probability = (
            (1.0 - frozen["weight"]) * neural_predictions["valid"][frozen["neural"]][target]
            + frozen["weight"] * experts[target].predict_proba(temporal[masks["valid"]])
        )
        for index, label in enumerate(base.TARGETS[target]["labels"]):
            validation[f"prob_{label}"] = valid_probability[:, index]
        validation.to_csv(OUT / "predictions" / f"{target}_validation_probabilities.csv", index=False)
        point = base.metrics(y, probability)
        boot = bootstrap_values(test_rows, probability, target)
        bootstrap_frames.append(boot)
        result_rows.append({
            "target": target.upper(), "split": "test", "n_cycles": int(test_rows["cycle_uid"].nunique()),
            "neural_member": frozen["neural"], "expert": frozen["expert"], "expert_weight": frozen["weight"],
            **point,
            "accuracy_ci_lower": float(boot["accuracy"].quantile(0.025)),
            "accuracy_ci_upper": float(boot["accuracy"].quantile(0.975)),
            "weighted_f1_ci_lower": float(boot["weighted_f1"].quantile(0.025)),
            "weighted_f1_ci_upper": float(boot["weighted_f1"].quantile(0.975)),
        })
    full_metrics = pd.DataFrame(result_rows)
    full_metrics.to_csv(OUT / "tables" / "test_metrics.csv", index=False)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(OUT / "tables" / "cycle_bootstrap.csv", index=False)

    write_json(OUT / "logs" / "run_manifest.json", {
        "seed": SEED, "device": str(device),
        "clinical_target": "information available through true Monitor k predicts the post-Monitor-k Gn decision",
        "stored_target_columns": {target: base.TARGETS[target]["dose_col"] for target in base.TARGET_ORDER},
        "fixed_split": "2849 train / 611 validation / 611 test cycles", "test_accessed_after_training": True,
        "input_contract": {"feature_count": args.features, "static": base.STATIC_FEATURES, "dynamic": base.DYNAMIC_FEATURES},
        "frozen_architecture": FROZEN_SELECTION,
        "neural_training": [asdict(candidates[name]) for name in neural_names],
        "expert_training": "same target-specific expert hyperparameters and sample-weight rule as the 27-feature run",
        "bootstrap": {"replicates": BOOTSTRAP_N, "unit": "cycle_uid", "interval": "percentile 95%", "seed_rule": "20261015 + task offset + 5000"},
        "data_audit": data_audit,
    })
    write_json(OUT / "logs" / "validation_all_expert_fusion_selection.json", {
        "selection": {target: {**choice, "expert_weight": choice["weight"],
                               "expert_family": {"fsh": "random_forest", "lh": "xgboost", "hmg": "extra_trees"}[target]}
                      for target, choice in FROZEN_SELECTION.items()},
    })
    print(full_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
