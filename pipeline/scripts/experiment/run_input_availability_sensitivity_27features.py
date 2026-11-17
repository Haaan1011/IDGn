#!/usr/bin/env python3
"""Fixed-model input-availability sensitivity analysis for the 27-feature model.

This is an inference-time deployment stress test.  The frozen test split,
validation-selected neural member, expert family, expert weight, and every
checkpoint remain unchanged.  Unavailable inputs are replaced only for test
records by their training-fitted median-imputation values.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts/experiment/monitor_level_27feature_current_decision_20261015_run.py"
EXPERT_FEATURE_SCRIPT = ROOT / "scripts/experiment/run_xgb_temporal_expert_validation_20261015.py"
MODEL_OUT = ROOT / "outputs/monitor_level_shared_multitask_gru_addgate_27features_current_decision_tuned_20261015"
OUT = ROOT / "outputs/monitor_level_shared_multitask_gru_addgate_27features_input_availability_sensitivity_20261016"
BOOTSTRAP_N = 1000
SEED = 20261015

SCENARIOS = [
    {"key": "all_available", "display": "All parsimonious features available", "static": [], "dynamic": []},
    {"key": "amh_unavailable", "display": "AMH unavailable", "static": ["amh"], "dynamic": []},
    {"key": "afc_unavailable", "display": "AFC unavailable", "static": ["afc"], "dynamic": []},
    {"key": "amh_afc_unavailable", "display": "AMH and AFC unavailable", "static": ["amh", "afc"], "dynamic": []},
    {"key": "current_hormones_endometrium_unavailable", "display": "Current hormones/endometrium unavailable", "static": [],
     "dynamic": ["current_e2", "current_lh", "current_p", "current_fsh", "current_endometrium"]},
    {"key": "current_follicle_profile_unavailable", "display": "Current follicle profile unavailable", "static": [],
     "dynamic": ["total_follicle_count", "max_follicle_diameter", "mean_follicle_diameter", "follicle_count_lt_10",
                 "follicle_count_10_12", "follicle_count_13_15", "follicle_count_16_18", "follicle_count_gt_18"]},
]


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = import_module("input_sensitivity_base", BASE_SCRIPT)
feature_module = import_module("input_sensitivity_feature_module", EXPERT_FEATURE_SCRIPT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def transformed_median_values(raw: Any, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training-fitted median values expressed on the model's standardized scale."""
    static_imputer = SimpleImputer(strategy="median").fit(raw.static[train_mask])
    static_fitted = static_imputer.transform(raw.static[train_mask])
    static_mean = static_fitted.mean(axis=0)
    static_sd = static_fitted.std(axis=0)
    static_sd[static_sd == 0] = 1.0
    static_fill = ((static_imputer.statistics_ - static_mean) / static_sd).astype(np.float32)

    observed = raw.dynamic[train_mask][raw.mask[train_mask].astype(bool)]
    dynamic_imputer = SimpleImputer(strategy="median").fit(observed)
    dynamic_fitted = dynamic_imputer.transform(observed)
    dynamic_mean = dynamic_fitted.mean(axis=0)
    dynamic_sd = dynamic_fitted.std(axis=0)
    dynamic_sd[dynamic_sd == 0] = 1.0
    dynamic_fill = ((dynamic_imputer.statistics_ - dynamic_mean) / dynamic_sd).astype(np.float32)
    return static_fill, dynamic_fill


def apply_unavailability(data: Any, test_mask: np.ndarray, static_fill: np.ndarray, dynamic_fill: np.ndarray,
                         static_features: list[str], dynamic_features: list[str]):
    static = data.static.copy()
    dynamic = data.dynamic.copy()
    rows = np.flatnonzero(test_mask)
    for feature in static_features:
        index = data.static_features.index(feature)
        static[rows, index] = static_fill[index]
    observed_steps = data.mask[rows].astype(bool)
    for feature in dynamic_features:
        index = data.dynamic_features.index(feature)
        values = dynamic[rows, :, index]
        values[observed_steps] = dynamic_fill[index]
        dynamic[rows, :, index] = values
    return SimpleNamespace(static=static, dynamic=dynamic, mask=data.mask, y=data.y,
                           row_index=data.row_index, static_features=data.static_features,
                           dynamic_features=data.dynamic_features)


def weighted_f1(y: np.ndarray, prediction: np.ndarray, n_classes: int) -> float:
    matrix = np.bincount(n_classes * y.astype(int) + prediction.astype(int), minlength=n_classes ** 2).reshape(n_classes, n_classes)
    support = matrix.sum(axis=1)
    tp = np.diag(matrix).astype(float)
    denom = 2.0 * tp + matrix.sum(axis=0) - tp + support - tp
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(n_classes, dtype=float), where=denom > 0)
    return float(np.sum(f1 * support) / support.sum())


def bootstrap_indices(cycles: np.ndarray, group_map: dict[str, np.ndarray], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.concatenate([group_map[cycle] for cycle in rng.choice(cycles, size=len(cycles), replace=True)]) for _ in range(BOOTSTRAP_N)]


def format_ci(value: float, lower: float, upper: float) -> str:
    return f"{value:.3f} ({lower:.3f}–{upper:.3f})"


def main() -> None:
    for folder in ("tables", "predictions", "logs"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    selection_path = MODEL_OUT / "logs" / "validation_all_expert_fusion_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selection"]
    manifest = pd.read_csv(base.MANIFEST_PATH)
    if manifest["split"].value_counts().to_dict() != {"train": 2849, "valid": 611, "test": 611}:
        raise RuntimeError("Frozen split manifest failed verification")
    source = pd.read_csv(base.DATA_PATH, low_memory=False)
    frame, data_audit = base.build_monitor_frame(source)
    raw, labels, metadata = base.prepare_master(frame)
    split_map = manifest.assign(cycle_uid=manifest["cycle_uid"].astype(str)).set_index("cycle_uid")["split"]
    row_split = raw.row_index["cycle_uid"].astype(str).map(split_map)
    masks = {name: row_split.eq(name).to_numpy(bool) for name in ("train", "valid", "test")}
    data = base.phase864.prepare_arrays(raw, fit_mask=masks["train"])
    if int(masks["test"].sum()) != 4291:
        raise RuntimeError("Expected 4,291 frozen test Monitor records")
    static_fill, dynamic_fill = transformed_median_values(raw, masks["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neural_models: dict[str, Any] = {}
    expert_models: dict[str, Any] = {}
    for target in base.TARGET_ORDER:
        choice = selection[target]
        neural_name = choice["neural"]
        if neural_name not in neural_models:
            neural_models[neural_name] = base.load_model(
                MODEL_OUT / "checkpoints" / f"{neural_name}.pt", data.static.shape[1], data.dynamic.shape[2], device
            )
        expert = joblib.load(MODEL_OUT / "checkpoints" / f"{target}_{choice['expert']}.joblib")
        if choice["expert_family"] == "xgboost":
            expert.set_params(device="cpu")
        expert_models[target] = expert

    all_rows, all_bootstrap = [], []
    for target in base.TARGET_ORDER:
        choice = selection[target]
        test_mask = masks["test"] & (labels[target] >= 0)
        test_y = labels[target][test_mask]
        n_classes = len(base.TARGETS[target]["labels"])
        test_index = raw.row_index.loc[test_mask].reset_index(drop=True)
        group_map = {str(cycle): indices.to_numpy() for cycle, indices in test_index.groupby("cycle_uid").groups.items()}
        cycles = np.array(list(group_map), dtype=object)
        # Match the fixed cycle-bootstrap stream used for the reported baseline.
        sampled_indices = bootstrap_indices(cycles, group_map, SEED + base.TARGETS[target]["offset"] + 5000)
        scenario_predictions: dict[str, np.ndarray] = {}
        for scenario in SCENARIOS:
            scenario_data = apply_unavailability(data, test_mask, static_fill, dynamic_fill,
                                                 scenario["static"], scenario["dynamic"])
            neural_probability = base.predict(neural_models[choice["neural"]], scenario_data, labels, test_mask, device)[target]
            temporal = feature_module.masked_temporal_features(scenario_data)
            expert_probability = expert_models[target].predict_proba(temporal[test_mask])
            probability = (1.0 - float(choice["expert_weight"])) * neural_probability + float(choice["expert_weight"]) * expert_probability
            prediction = probability.argmax(axis=1)
            scenario_predictions[scenario["key"]] = prediction
            pd.DataFrame({
                "cycle_uid": test_index["cycle_uid"], "visit_uid": test_index["visit_uid"], "y_true": test_y,
                "y_pred": prediction, **{f"prob_{label}": probability[:, i] for i, label in enumerate(base.TARGETS[target]["labels"])},
            }).to_csv(OUT / "predictions" / f"{target}_{scenario['key']}_test_predictions.csv", index=False)

        reference_bootstrap = None
        for scenario in SCENARIOS:
            prediction = scenario_predictions[scenario["key"]]
            point = weighted_f1(test_y, prediction, n_classes)
            values = np.array([weighted_f1(test_y[index], prediction[index], n_classes) for index in sampled_indices])
            if scenario["key"] == "all_available":
                reference_bootstrap = values
            if reference_bootstrap is None:
                delta = np.nan
                delta_ci_lower, delta_ci_upper = np.nan, np.nan
            else:
                delta_values = values - reference_bootstrap
                delta = point - weighted_f1(test_y, scenario_predictions["all_available"], n_classes)
                delta_ci_lower, delta_ci_upper = np.quantile(delta_values, [0.025, 0.975])
            lower, upper = np.quantile(values, [0.025, 0.975])
            display_delta = (
                "Reference" if scenario["key"] == "all_available"
                else ("0.000" if abs(delta) < 0.0005 else f"{delta:+.3f}")
            )
            all_rows.append({
                "target": target.upper(), "scenario_key": scenario["key"], "input_availability_scenario": scenario["display"],
                "n_cycles": len(cycles), "n_records": len(test_y), "weighted_f1": point,
                "weighted_f1_ci_lower": lower, "weighted_f1_ci_upper": upper,
                "delta_weighted_f1": delta, "delta_weighted_f1_ci_lower": delta_ci_lower,
                "delta_weighted_f1_ci_upper": delta_ci_upper, "weighted_f1_95ci": format_ci(point, lower, upper),
                "delta_weighted_f1_display": display_delta,
                "unavailable_static_features": "; ".join(scenario["static"]),
                "unavailable_dynamic_features": "; ".join(scenario["dynamic"]),
            })
            for iteration, value in enumerate(values, start=1):
                all_bootstrap.append({"target": target.upper(), "scenario_key": scenario["key"], "replicate": iteration,
                                      "weighted_f1": value, "delta_weighted_f1_vs_reference": value - reference_bootstrap[iteration - 1]})

    result = pd.DataFrame(all_rows)
    result.to_csv(OUT / "tables" / "input_availability_sensitivity_detailed.csv", index=False)
    pd.DataFrame(all_bootstrap).to_csv(OUT / "tables" / "input_availability_sensitivity_bootstrap_replicates.csv", index=False)
    manuscript = result[["target", "input_availability_scenario", "weighted_f1_95ci", "delta_weighted_f1_display"]].copy()
    manuscript.columns = ["Target", "Input-availability scenario", "Weighted-F1 (95% CI)", "ΔWeighted-F1"]
    manuscript.to_csv(OUT / "tables" / "table4_input_availability_sensitivity.csv", index=False)
    manuscript.to_csv(OUT / "tables" / "table4_input_availability_sensitivity.tsv", index=False, sep="\t")
    write_json(OUT / "logs" / "run_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "purpose": "fixed-model inference-time input-availability sensitivity analysis",
        "frozen_split": "2849 train / 611 validation / 611 test cycles",
        "test_records": 4291, "feature_count": 27, "scenarios": SCENARIOS,
        "unavailability_handling": "training-fitted median imputation values on standardized model scale; only observed test time steps replaced",
        "models": selection, "selection_source": str(selection_path), "selection_source_sha256": sha256(selection_path),
        "bootstrap": {"unit": "cycle_uid", "replicates": BOOTSTRAP_N, "interval": "percentile 95%", "shared_within_target_across_scenarios": True},
        "data_audit": data_audit,
    })
    print(manuscript.to_string(index=False))


if __name__ == "__main__":
    main()
