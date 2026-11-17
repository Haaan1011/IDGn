#!/usr/bin/env python3
"""Compare task-specific expert families on the fixed validation split only."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts/experiment/monitor_level_27feature_current_decision_20261015_run.py"
EXPERT_SCRIPT = ROOT / "scripts/experiment/run_xgb_temporal_expert_validation_20261015.py"
OUT = ROOT / "outputs/monitor_level_shared_multitask_gru_addgate_27features_current_decision_tuned_20261015"


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = import_module("expert_compare_base", BASE_SCRIPT)
feature_module = import_module("expert_compare_features", EXPERT_SCRIPT)


def metric(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    labels = list(range(probability.shape[1]))
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, labels=labels, average="weighted", zero_division=0)),
    }


def candidates(seed: int):
    return [
        ("logistic", "logistic_c02", LogisticRegression(C=0.2, max_iter=2500, solver="lbfgs", random_state=seed)),
        ("logistic", "logistic_c1", LogisticRegression(C=1.0, max_iter=2500, solver="lbfgs", random_state=seed)),
        ("logistic", "logistic_c5", LogisticRegression(C=5.0, max_iter=2500, solver="lbfgs", random_state=seed)),
        ("mlp", "mlp_128", MLPClassifier(
            hidden_layer_sizes=(128,), activation="relu", alpha=2e-4, batch_size=512,
            learning_rate_init=5e-4, max_iter=250, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=18, random_state=seed,
        )),
        ("mlp", "mlp_128_64", MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", alpha=5e-4, batch_size=512,
            learning_rate_init=3e-4, max_iter=250, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=18, random_state=seed,
        )),
        ("random_forest", "rf_depth14", RandomForestClassifier(
            n_estimators=500, max_depth=14, min_samples_leaf=3, max_features="sqrt",
            n_jobs=12, random_state=seed,
        )),
        ("random_forest", "rf_unlimited", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3, max_features=0.5,
            n_jobs=12, random_state=seed,
        )),
        ("extra_trees", "et_depth16", ExtraTreesClassifier(
            n_estimators=500, max_depth=16, min_samples_leaf=2, max_features=0.7,
            n_jobs=12, random_state=seed,
        )),
        ("extra_trees", "et_unlimited_leaf2", ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2, max_features=0.7,
            n_jobs=12, random_state=seed,
        )),
        ("extra_trees", "et_unlimited_leaf4", ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=4, max_features=1.0,
            n_jobs=12, random_state=seed,
        )),
        ("hist_gradient_boosting", "hgb_leaf31", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=0.5, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=seed,
        )),
        ("hist_gradient_boosting", "hgb_leaf63", HistGradientBoostingClassifier(
            learning_rate=0.035, max_iter=650, max_leaf_nodes=63, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=35, random_state=seed,
        )),
        ("hist_gradient_boosting", "hgb_leaf127", HistGradientBoostingClassifier(
            learning_rate=0.025, max_iter=750, max_leaf_nodes=127, min_samples_leaf=25,
            l2_regularization=1.5, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=40, random_state=seed,
        )),
    ]


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    manifest = pd.read_csv(base.MANIFEST_PATH)
    source = pd.read_csv(base.DATA_PATH, low_memory=False)
    frame, _ = base.build_monitor_frame(source)
    raw, labels, _ = base.prepare_master(frame)
    split_map = manifest.assign(cycle_uid=manifest["cycle_uid"].astype(str)).set_index("cycle_uid")["split"]
    row_splits = raw.row_index["cycle_uid"].astype(str).map(split_map)
    masks = {split: row_splits.eq(split).to_numpy(bool) for split in ("train", "valid", "test")}
    arrays = base.phase864.prepare_arrays(raw, fit_mask=masks["train"])
    features = feature_module.masked_temporal_features(arrays)

    rows: list[dict[str, object]] = []
    validation_probabilities: dict[str, dict[str, np.ndarray]] = {target: {} for target in base.TARGET_ORDER}
    for target in base.TARGET_ORDER:
        train_mask = masks["train"] & (labels[target] >= 0)
        valid_mask = masks["valid"] & (labels[target] >= 0)
        train_y = labels[target][train_mask]
        valid_y = labels[target][valid_mask]
        counts = np.bincount(train_y, minlength=len(base.TARGETS[target]["labels"])).astype(float)
        mild_weight = np.power(counts[train_y] / counts.max(), -0.15)
        for family, name, model in candidates(base.SEED + base.TARGETS[target]["offset"]):
            started = time.time()
            fit_kwargs = {}
            if family != "mlp":
                fit_kwargs["sample_weight"] = mild_weight
            model.fit(features[train_mask], train_y, **fit_kwargs)
            probability = model.predict_proba(features[valid_mask])
            payload = metric(valid_y, probability)
            validation_probabilities[target][name] = probability
            joblib.dump(model, OUT / "checkpoints" / f"{target}_{name}.joblib")
            rows.append({
                "target": target.upper(), "family": family, "candidate": name,
                "fit_seconds": float(time.time() - started), **payload,
            })
            print(target.upper(), family, name, payload, flush=True)

        # Reuse the four already fitted XGBoost experts under the same validation protocol.
        for xgb_name in ("xgb_d3", "xgb_d4", "xgb_d5", "xgb_d6"):
            model = joblib.load(OUT / "checkpoints" / f"{target}_{xgb_name}.joblib")
            model.set_params(device="cpu")
            probability = model.predict_proba(features[valid_mask])
            payload = metric(valid_y, probability)
            validation_probabilities[target][xgb_name] = probability
            rows.append({
                "target": target.upper(), "family": "xgboost", "candidate": xgb_name,
                "fit_seconds": np.nan, **payload,
            })

    expert_table = pd.DataFrame(rows).sort_values(["target", "weighted_f1"], ascending=[True, False])
    expert_table.to_csv(OUT / "tables" / "validation_all_expert_families.csv", index=False)

    # Compare all experts in probability fusion with every validation-trained shared-GRU candidate.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_probability: dict[str, dict[str, np.ndarray]] = {}
    for candidate in base.CANDIDATES:
        model = base.load_model(
            OUT / "checkpoints" / f"{candidate.name}.pt",
            arrays.static.shape[1], arrays.dynamic.shape[2], device,
        )
        neural_probability[candidate.name] = base.predict(model, arrays, labels, masks["valid"], device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selections: dict[str, dict[str, object]] = {}
    fusion_rows = []
    for target in base.TARGET_ORDER:
        full_y = labels[target][masks["valid"]]
        valid_target = full_y >= 0
        y = full_y[valid_target]
        best = None
        for neural_name, target_payload in neural_probability.items():
            neural = target_payload[target][valid_target]
            for expert_name, expert_probability in validation_probabilities[target].items():
                for weight in np.round(np.arange(0.0, 1.001, 0.025), 3):
                    probability = (1.0 - weight) * neural + weight * expert_probability
                    payload = metric(y, probability)
                    candidate = (
                        payload["weighted_f1"], payload["accuracy"], payload["macro_f1"],
                        neural_name, expert_name, float(weight),
                    )
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
        weighted_f1, accuracy, macro_f1, neural_name, expert_name, weight = best
        expert_family = expert_table.loc[
            expert_table["candidate"].eq(expert_name) & expert_table["target"].eq(target.upper()), "family"
        ].iloc[0]
        selections[target] = {
            "neural": neural_name, "expert": expert_name, "expert_family": expert_family,
            "expert_weight": weight,
        }
        fusion_rows.append({
            "target": target.upper(), "neural": neural_name, "expert_family": expert_family,
            "expert": expert_name, "expert_weight": weight,
            "validation_accuracy": accuracy, "validation_macro_f1": macro_f1,
            "validation_weighted_f1": weighted_f1,
        })
    fusion_table = pd.DataFrame(fusion_rows)
    fusion_table.to_csv(OUT / "tables" / "validation_all_expert_fusion_selection.csv", index=False)
    (OUT / "logs" / "validation_all_expert_fusion_selection.json").write_text(json.dumps({
        "test_accessed": False,
        "raw_feature_count": 27,
        "selection_metric": "validation Weighted-F1; accuracy and Macro-F1 used only as tie-breakers",
        "selection": selections,
        "families": sorted(expert_table["family"].unique().tolist()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nBEST EXPERT-FUSION SELECTION")
    print(fusion_table.to_string(index=False))


if __name__ == "__main__":
    main()
