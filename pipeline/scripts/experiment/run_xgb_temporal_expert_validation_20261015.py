#!/usr/bin/env python3
"""Validation-only temporal tabular expert using the fixed 27 raw inputs."""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts/experiment/monitor_level_27feature_current_decision_20261015_run.py"
OUT = ROOT / "outputs/monitor_level_shared_multitask_gru_addgate_27features_current_decision_tuned_20261015"


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = import_module("current_decision_xgb_base", BASE_SCRIPT)


CONFIGS = [
    {"name": "xgb_d3", "max_depth": 3, "learning_rate": 0.035, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85},
    {"name": "xgb_d4", "max_depth": 4, "learning_rate": 0.030, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85},
    {"name": "xgb_d5", "max_depth": 5, "learning_rate": 0.025, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.80},
    {"name": "xgb_d6", "max_depth": 6, "learning_rate": 0.020, "min_child_weight": 10, "subsample": 0.80, "colsample_bytree": 0.80},
]


def masked_temporal_features(arrays) -> np.ndarray:
    dynamic = arrays.dynamic.astype(np.float32)
    mask = arrays.mask.astype(bool)
    lengths = mask.sum(axis=1).astype(int).clip(min=1)
    rows = np.arange(len(dynamic))
    latest = dynamic[rows, lengths - 1]
    previous = dynamic[rows, np.maximum(lengths - 2, 0)]
    delta = np.where((lengths > 1)[:, None], latest - previous, 0.0)
    values = np.where(mask[:, :, None], dynamic, np.nan)
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1)
    minimum = np.nanmin(values, axis=1)
    maximum = np.nanmax(values, axis=1)
    first = dynamic[:, 0]
    from_first = latest - first
    return np.column_stack([
        arrays.static.astype(np.float32), latest, previous, delta, mean, std, minimum, maximum,
        first, from_first, lengths.astype(np.float32),
    ]).astype(np.float32)


def metric(y, probability):
    prediction = probability.argmax(axis=1)
    labels = list(range(probability.shape[1]))
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "weighted_f1": float(f1_score(y, prediction, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
    }


def main() -> None:
    manifest = pd.read_csv(base.MANIFEST_PATH)
    source = pd.read_csv(base.DATA_PATH, low_memory=False)
    frame, _ = base.build_monitor_frame(source)
    raw, labels, _ = base.prepare_master(frame)
    split_map = manifest.assign(cycle_uid=manifest["cycle_uid"].astype(str)).set_index("cycle_uid")["split"]
    row_splits = raw.row_index["cycle_uid"].astype(str).map(split_map)
    masks = {split: row_splits.eq(split).to_numpy(bool) for split in ("train", "valid", "test")}
    arrays = base.phase864.prepare_arrays(raw, fit_mask=masks["train"])
    features = masked_temporal_features(arrays)
    rows = []
    probability_store: dict[str, np.ndarray] = {}
    for target in base.TARGET_ORDER:
        train_valid = masks["train"] & (labels[target] >= 0)
        validation_valid = masks["valid"] & (labels[target] >= 0)
        train_y = labels[target][train_valid]
        valid_y = labels[target][validation_valid]
        counts = np.bincount(train_y, minlength=len(base.TARGETS[target]["labels"])).astype(float)
        sample_weight = np.power(counts[train_y] / counts.max(), -0.15)
        for config in CONFIGS:
            model = XGBClassifier(
                n_estimators=1600, max_depth=config["max_depth"], learning_rate=config["learning_rate"],
                min_child_weight=config["min_child_weight"], subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"], reg_alpha=0.05, reg_lambda=2.0,
                objective="multi:softprob", num_class=len(base.TARGETS[target]["labels"]),
                eval_metric="mlogloss", early_stopping_rounds=80, random_state=base.SEED + base.TARGETS[target]["offset"],
                tree_method="hist", device="cuda", n_jobs=8,
            )
            model.fit(
                features[train_valid], train_y, sample_weight=sample_weight,
                eval_set=[(features[validation_valid], valid_y)], verbose=False,
            )
            probability = model.predict_proba(features[validation_valid])
            payload = metric(valid_y, probability)
            key = f"{target}_{config['name']}"
            probability_store[key] = probability
            joblib.dump(model, OUT / "checkpoints" / f"{key}.joblib")
            rows.append({
                "target": target.upper(), "candidate": config["name"], "best_iteration": int(model.best_iteration),
                **payload,
            })
            print(target.upper(), config["name"], payload, flush=True)

    # Per-task validation-only ensemble selection across tree experts and the best shared-GRU candidates.
    neural_names = [candidate.name for candidate in base.CANDIDATES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_probability: dict[str, dict[str, np.ndarray]] = {}
    for name in neural_names:
        model = base.load_model(OUT / "checkpoints" / f"{name}.pt", arrays.static.shape[1], arrays.dynamic.shape[2], device)
        neural_probability[name] = base.predict(model, arrays, labels, masks["valid"], device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selections = {}
    ensemble_rows = []
    for target in base.TARGET_ORDER:
        full_y = labels[target][masks["valid"]]
        valid_target = full_y >= 0
        y = full_y[valid_target]
        expert_names = [config["name"] for config in CONFIGS]
        best = None
        for neural_name, expert_name in itertools.product(neural_names, expert_names):
            neural = neural_probability[neural_name][target][valid_target]
            expert = probability_store[f"{target}_{expert_name}"]
            for weight in np.round(np.arange(0.0, 1.001, 0.025), 3):
                probability = (1.0 - weight) * neural + weight * expert
                payload = metric(y, probability)
                candidate = (payload["weighted_f1"], payload["accuracy"], neural_name, expert_name, float(weight), payload["macro_f1"])
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        weighted_f1, accuracy, neural_name, expert_name, weight, macro_f1 = best
        selections[target] = {"neural": neural_name, "expert": expert_name, "expert_weight": weight}
        ensemble_rows.append({
            "target": target.upper(), "neural": neural_name, "expert": expert_name, "expert_weight": weight,
            "validation_accuracy": accuracy, "validation_macro_f1": macro_f1,
            "validation_weighted_f1": weighted_f1,
        })

    pd.DataFrame(rows).sort_values(["target", "weighted_f1"], ascending=[True, False]).to_csv(
        OUT / "tables" / "validation_xgb_temporal_experts.csv", index=False
    )
    pd.DataFrame(ensemble_rows).to_csv(OUT / "tables" / "validation_neural_xgb_ensemble_selection.csv", index=False)
    (OUT / "logs" / "validation_neural_xgb_ensemble_selection.json").write_text(
        json.dumps({
            "test_accessed": False, "selection": selections,
            "raw_feature_count": 27,
            "derived_temporal_views": ["latest", "previous", "delta", "mean", "sd", "min", "max", "first", "latest-minus-first", "sequence-length"],
        }, indent=2) + "\n", encoding="utf-8"
    )
    print("\nENSEMBLE")
    print(pd.DataFrame(ensemble_rows).to_string(index=False))


if __name__ == "__main__":
    main()
