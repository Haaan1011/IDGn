#!/usr/bin/env python3
"""Train and evaluate GRU(AddGate) on daily-expanded dose groups."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PHASE866_PATH = REPO_ROOT / "scripts" / "experiment" / "phase866_daily_expanded_model_comparison" / "run_phase866_daily_expanded_model_comparison.py"
spec = importlib.util.spec_from_file_location("phase866_runner", PHASE866_PATH)
phase866 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = phase866
assert spec.loader is not None
spec.loader.exec_module(phase866)
phase864 = phase866.phase864

FSH_LABELS = ["stop", "low_dose", "medium_low", "medium_high", "high_dose"]
LH3_LABELS = ["dose_0", "dose_75", "dose_ge150"]
HMG4_LABELS = ["dose_0", "dose_75", "dose_150", "dose_ge225"]
TARGETS = {
    "fsh": {"name": "FSH", "label_col": "daily_fsh_dose_bin", "dose_col": "daily_fsh_daily_dose", "labels": FSH_LABELS},
    "lh": {"name": "LH", "label_col": "daily_lh_dose_bin_grouped", "dose_col": "daily_lh_daily_dose", "labels": LH3_LABELS},
    "hmg": {"name": "HMG", "label_col": "daily_hmg_dose_bin_grouped", "dose_col": "daily_hmg_daily_dose", "labels": HMG4_LABELS},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/layer1_strategy_dataset.csv")
    parser.add_argument("--baseline-input", default="data/processed/baseline_cycle_dataset.csv")
    parser.add_argument("--output-root", default="results/result1_deep_temporal_gn_dose_gru_addgate_final_fsh_lh3_hmg4")
    parser.add_argument("--paper-table-dir", default="paper/tables/paper_claim_validation/phase866_gru_addgate_final_fsh_lh3_hmg4")
    parser.add_argument("--min-day", type=int, default=1)
    parser.add_argument("--max-day", type=int, default=20)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--max-history-days", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--selection-metric", choices=["accuracy", "macro_f1", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--knn-grid", type=int, nargs="+", default=list(phase866.DEFAULT_KNN_GRID))
    parser.add_argument("--reuse-fsh-predictions", default="results/result1_deep_temporal_gn_dose_daily_expanded_model_comparison/phase866_gru_tcn_gate_mag_standard_ce/predictions/gru_addgate_oof_predictions.csv")
    return parser.parse_args()


def lh3_label(dose: Any) -> str | None:
    if pd.isna(dose):
        return None
    d = float(dose)
    if d == 0:
        return "dose_0"
    if d == 75:
        return "dose_75"
    if d >= 150:
        return "dose_ge150"
    return None


def hmg4_label(dose: Any) -> str | None:
    if pd.isna(dose):
        return None
    d = float(dose)
    if d == 0:
        return "dose_0"
    if d == 75:
        return "dose_75"
    if d == 150:
        return "dose_150"
    if d >= 225:
        return "dose_ge225"
    return None


def add_grouped_labels(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["daily_lh_dose_bin_grouped"] = panel["daily_lh_daily_dose"].map(lh3_label)
    panel["daily_hmg_dose_bin_grouped"] = panel["daily_hmg_daily_dose"].map(hmg4_label)
    return panel


def set_target_globals(labels: list[str]) -> None:
    phase866.LABELS = list(labels)
    phase866.NUM_CLASSES = len(labels)
    phase866.MODEL_DISPLAY["gru_addgate"] = "GRU(AddGate)"


def prepare_target_raw(panel: pd.DataFrame, feature_groups: dict[str, list[str]], target: str, args: argparse.Namespace) -> Any:
    spec = TARGETS[target]
    labels = spec["labels"]
    label_col = spec["label_col"]
    frame = panel[panel[label_col].isin(labels)].copy()
    if frame.empty:
        raise ValueError(f"No rows for target {target} after label filtering")
    frame["ctfe_truth"] = frame[label_col].astype(str)
    frame["ctfe_truth_id"] = frame["ctfe_truth"].map({label: i for i, label in enumerate(labels)}).astype(int)
    frame["source_visit_uid"] = frame.get("visit_uid", pd.Series([pd.NA] * len(frame), index=frame.index)).astype(str)
    frame["Day"] = pd.to_numeric(frame["Day"], errors="coerce").astype(int)
    frame["evaluation_day"] = frame["Day"]
    frame["monitoring_order"] = frame["Day"]
    frame["gn_day"] = frame["Day"].astype(float)
    frame["visit_uid"] = frame["cycle_uid"].astype(str) + f"__daily_{target}_day" + frame["Day"].astype(str)
    frame["baseline_only_feature_row"] = frame["baseline_only_feature_row"].astype(bool).astype(int)
    return phase864.build_daily_sequence_arrays(
        frame.sort_values(["cycle_uid", "Day", "visit_uid"], kind="mergesort").reset_index(drop=True),
        static_features=feature_groups["static_features"],
        dynamic_features=feature_groups["dynamic_features"],
        max_history_days=int(args.max_history_days),
    )


def metrics_from_prediction_frame(pred: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    y_true = pd.Categorical(pred["truth"], categories=labels).codes
    y_pred = pd.Categorical(pred["prediction"], categories=labels).codes
    valid = (y_true >= 0) & (y_pred >= 0)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    return {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(labels))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(len(labels))), average="weighted", zero_division=0)),
    }


def class_distribution(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, spec in TARGETS.items():
        counts = panel[spec["label_col"]].value_counts(dropna=False).to_dict()
        for label in spec["labels"]:
            rows.append({"target": target, "Target": spec["name"], "label": label, "count": int(counts.get(label, 0))})
        missing = int(panel[spec["label_col"]].isna().sum())
        if missing:
            rows.append({"target": target, "Target": spec["name"], "label": "unmapped_or_missing", "count": missing})
    return pd.DataFrame(rows)


def confusion_frame(pred: pd.DataFrame, labels: list[str], target: str) -> pd.DataFrame:
    y_true = pd.Categorical(pred["truth"], categories=labels).codes
    y_pred = pd.Categorical(pred["prediction"], categories=labels).codes
    valid = (y_true >= 0) & (y_pred >= 0)
    cm = confusion_matrix(y_true[valid], y_pred[valid], labels=list(range(len(labels))), normalize="true")
    return pd.DataFrame(cm, index=labels, columns=labels).rename_axis("truth").reset_index().assign(target=target)


def confusion_counts(pred: pd.DataFrame, labels: list[str], target: str) -> pd.DataFrame:
    y_true = pd.Categorical(pred["truth"], categories=labels).codes
    y_pred = pd.Categorical(pred["prediction"], categories=labels).codes
    valid = (y_true >= 0) & (y_pred >= 0)
    cm = confusion_matrix(y_true[valid], y_pred[valid], labels=list(range(len(labels))), normalize=None)
    return pd.DataFrame(cm, index=labels, columns=labels).rename_axis("truth").reset_index().assign(target=target)


def first_adjustment_audit(panel: pd.DataFrame) -> pd.DataFrame:
    dose_cols = ["daily_fsh_daily_dose", "daily_lh_daily_dose", "daily_hmg_daily_dose"]
    p = panel[["cycle_uid", "Day", *dose_cols]].copy().sort_values(["cycle_uid", "Day"])
    for c in dose_cols:
        p[c] = pd.to_numeric(p[c], errors="coerce")
        p[f"prev_day_{c}"] = p.groupby("cycle_uid")[c].shift(1)
        p[f"changed_{c}"] = (p[c] != p[f"prev_day_{c}"]) & p[f"prev_day_{c}"].notna()
    p["changed_any"] = p[[f"changed_{c}" for c in dose_cols]].any(axis=1)
    first = p[p["changed_any"]].groupby("cycle_uid", as_index=False)["Day"].min().rename(columns={"Day": "first_adjustment_day"})
    daily = p.groupby("Day").agg(
        daily_count=("cycle_uid", "nunique"),
        any_changed_cycles=("changed_any", "sum"),
        fsh_changed_cycles=("changed_daily_fsh_daily_dose", "sum"),
        lh_changed_cycles=("changed_daily_lh_daily_dose", "sum"),
        hmg_changed_cycles=("changed_daily_hmg_daily_dose", "sum"),
    ).reset_index()
    first_counts = first["first_adjustment_day"].value_counts().rename_axis("Day").reset_index(name="first_adjustment_cycles")
    daily = daily.merge(first_counts, on="Day", how="left").fillna({"first_adjustment_cycles": 0})
    for c in ["any_changed_cycles", "fsh_changed_cycles", "lh_changed_cycles", "hmg_changed_cycles", "first_adjustment_cycles"]:
        daily[c] = daily[c].astype(int)
        daily[f"{c}_rate"] = daily[c] / daily["daily_count"].replace(0, np.nan)
    return daily


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    table_dir = out_root / "tables"
    pred_dir = out_root / "predictions"
    model_dir = out_root / "models"
    for d in [table_dir, pred_dir, model_dir, Path(args.paper_table_dir), out_root / "logs"]:
        d.mkdir(parents=True, exist_ok=True)

    monitoring = pd.read_csv(args.input, low_memory=False)
    baseline = pd.read_csv(args.baseline_input, low_memory=False) if args.baseline_input else None
    panel = phase864.build_daily_no_sw_panel(monitoring, baseline, min_day=int(args.min_day), max_day=int(args.max_day))
    panel = add_grouped_labels(panel)
    feature_candidates = [
        c for c in phase864.select_daily_no_sw_feature_columns(panel, targets=["daily_fsh_dose_bin", "daily_lh_dose_bin", "daily_hmg_dose_bin", "daily_lh_dose_bin_grouped", "daily_hmg_dose_bin_grouped", "ctfe_truth", "ctfe_truth_id"])
        if c not in phase864.DAILY_CTFE_EXTRA_LEAKAGE_COLUMNS
    ]
    for c in feature_candidates:
        if c in panel.columns and not pd.api.types.is_numeric_dtype(panel[c]):
            converted = pd.to_numeric(panel[c], errors="coerce")
            if converted.notna().any():
                panel[c] = converted
    numeric_features = [c for c in feature_candidates if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]
    static_features = [c for c in phase864.CTFE_STATIC_FEATURES if c in numeric_features]
    dynamic_features = [c for c in numeric_features if c not in static_features]
    if "Day" in numeric_features:
        dynamic_features = ["Day"] + [c for c in dynamic_features if c != "Day"]
    feature_groups = {"static_features": static_features, "dynamic_features": dynamic_features}

    class_distribution(panel).to_csv(table_dir / "gru_addgate_fsh_lh3_hmg4_class_distribution.csv", index=False)
    class_distribution(panel).to_csv(Path(args.paper_table_dir) / "gru_addgate_fsh_lh3_hmg4_class_distribution.csv", index=False)
    first_adjustment_audit(panel).to_csv(table_dir / "daily_first_adjustment_audit.csv", index=False)
    first_adjustment_audit(panel).to_csv(Path(args.paper_table_dir) / "daily_first_adjustment_audit.csv", index=False)

    run_args = SimpleNamespace(**vars(args))
    run_args.output_root = str(out_root)
    run_args.focal_gamma = 2.0
    run_args.class_weight_power = 0.5
    import torch
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    metrics_rows: list[dict[str, Any]] = []
    confusions: dict[str, pd.DataFrame] = {}
    for target, spec in TARGETS.items():
        set_target_globals(spec["labels"])
        pred_path = pred_dir / f"gru_addgate_{target}_oof_predictions.csv"
        if target == "fsh" and args.reuse_fsh_predictions and Path(args.reuse_fsh_predictions).exists():
            pred_df = pd.read_csv(args.reuse_fsh_predictions)
            pred_df.to_csv(pred_path, index=False)
        elif pred_path.exists():
            pred_df = pd.read_csv(pred_path)
        else:
            raw = prepare_target_raw(panel, feature_groups, target, args)
            target_out = out_root / f"train_{target}"
            summary, fold_df, cls_rows, cm_rows, curve_df = phase866.run_neural_model(raw, "gru_addgate", run_args, target_out, device)
            src = target_out / "predictions" / "gru_addgate_oof_predictions.csv"
            pred_df = pd.read_csv(src)
            pred_df.to_csv(pred_path, index=False)
            fold_df.to_csv(table_dir / f"gru_addgate_{target}_fold_metrics.csv", index=False)
            pd.DataFrame(cls_rows).to_csv(table_dir / f"gru_addgate_{target}_classwise_metrics.csv", index=False)
            pd.DataFrame(cm_rows).to_csv(table_dir / f"gru_addgate_{target}_confusion_long.csv", index=False)
            if not curve_df.empty:
                curve_df.to_csv(table_dir / f"gru_addgate_{target}_training_curves.csv", index=False)
        m = metrics_from_prediction_frame(pred_df, spec["labels"])
        metrics_rows.append({"target": target, "Target": spec["name"], "model": "GRU(AddGate)", **m})
        conf = confusion_frame(pred_df, spec["labels"], target)
        conf_counts = confusion_counts(pred_df, spec["labels"], target)
        conf.to_csv(table_dir / f"gru_addgate_{target}_confusion_matrix_normalized.csv", index=False)
        conf_counts.to_csv(table_dir / f"gru_addgate_{target}_confusion_matrix_counts.csv", index=False)
        conf.to_csv(Path(args.paper_table_dir) / f"gru_addgate_{target}_confusion_matrix_normalized.csv", index=False)
        conf_counts.to_csv(Path(args.paper_table_dir) / f"gru_addgate_{target}_confusion_matrix_counts.csv", index=False)
        confusions[target] = conf

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(table_dir / "gru_addgate_fsh_lh3_hmg4_overall_metrics.csv", index=False)
    metrics.to_csv(Path(args.paper_table_dir) / "gru_addgate_fsh_lh3_hmg4_overall_metrics.csv", index=False)
    config = {
        "model": "GRU(AddGate)",
        "task": "daily-expanded Day 1-20 dose category prediction, cycle-grouped 5-fold OOF",
        "labels": {k: v["labels"] for k, v in TARGETS.items()},
        "training_parameters": {k: getattr(args, k) for k in ["oof_folds", "max_history_days", "hidden_dim", "dropout", "epochs", "patience", "batch_size", "lr", "weight_decay", "seed", "selection_metric", "knn_grid", "device"]},
        "feature_groups": {k: len(v) for k, v in feature_groups.items()},
        "metrics": metrics_rows,
    }
    (out_root / "logs" / "run_summary.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(args.paper_table_dir) / "gru_addgate_fsh_lh3_hmg4_training_parameters.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
