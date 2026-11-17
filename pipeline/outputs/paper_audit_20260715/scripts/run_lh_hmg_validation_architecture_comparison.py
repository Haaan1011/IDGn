#!/usr/bin/env python3
"""Fixed validation-set architecture comparisons for FSH, LH, and HMG.

The model family and hyperparameters mirror the Phase 8.66 FSH comparison,
while the evaluation contract is deliberately changed to the frozen
train/validation split. The test split is never materialized or predicted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/paper_audit_20260715"
SCRIPT = OUT / "scripts/run_holdout_gru_addgate_audit.py"

import importlib.util
import sys


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = import_module("holdout_audit_for_validation_comparison", SCRIPT)

ARCH_ROWS = [
    ("ET", "D-TDNN", "dtdnn"),
    ("ET", "LSTM", "lstm"),
    ("ET", "GRU", "gru"),
    ("ET", "TCN / 1D-CNN", "tcn"),
    ("ETS & ET", "LSTM(MAG)", "lstm_mag"),
    ("ETS & ET", "LSTM(AddGate)", "lstm_addgate"),
    ("ETS & ET", "LSTM(FC Layer)", "lstm_fc"),
    ("ETS & ET", "TCN / 1D-CNN(MAG)", "tcn_mag"),
    ("ETS & ET", "TCN / 1D-CNN(AddGate)", "tcn_addgate"),
    ("ETS & ET", "TCN / 1D-CNN(FC Layer)", "tcn_fc"),
    ("ETS & ET", "D-TDNN(MAG)", "dtdnn_mag"),
    ("ETS & ET", "D-TDNN(AddGate)", "dtdnn_addgate"),
    ("ETS & ET", "D-TDNN(FC Layer)", "dtdnn_fc"),
    ("ETS & ET", "GRU(MAG)", "gru_mag"),
    ("ETS & ET", "GRU(FC Layer)", "gru_fc"),
    ("ETS & ET", "GRU(AddGate)", "gru_addgate"),
]

HYPERPARAMETERS = {
    "max_history_days": 16,
    "hidden_dim": 96,
    "dropout": 0.15,
    "epochs_max": 70,
    "patience": 12,
    "batch_size": 1024,
    "lr": 7e-4,
    "weight_decay": 2e-4,
    "gradient_clip_norm": 3.0,
    "loss": "cross_entropy",
    "selection_split": "valid",
    "selection_metric": "weighted_f1",
}


def fit_one(
    target: str,
    model_key: str,
    data: Any,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    labels = base.TARGETS[target]["labels"]
    seed = base.SEED + {"fsh": 11, "lh": 23, "hmg": 37}[target]
    base.set_seed(seed)
    args = SimpleNamespace(hidden_dim=96, dropout=0.15)
    model = base.phase866.make_neural_model(model_key, data, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_epoch = 0
    best_score = -np.inf
    wait = 0
    curve: list[dict[str, Any]] = []

    for epoch in range(1, 71):
        model.train()
        losses = []
        for static_x, dynamic_x, obs_mask, y in base.loader(data, train_mask, 1024, True):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(static_x.to(device), dynamic_x.to(device), obs_mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        prob, _ = base.predict(model, data, valid_mask, device)
        pred = prob.argmax(axis=1)
        metrics = base.multiclass_metrics(data.y[valid_mask], pred, prob, len(labels))
        score = float(metrics["weighted_f1"])
        curve.append({
            "task": target.upper(),
            "model_key": model_key,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
        })
        print(
            f"[{target.upper()}:{model_key}] epoch={epoch:02d} "
            f"loss={np.mean(losses):.5f} valid_wf1={score:.5f}",
            flush=True,
        )
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 12:
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {target}/{model_key}")
    model.load_state_dict(best_state)
    prob, _ = base.predict(model, data, valid_mask, device)
    pred = prob.argmax(axis=1)
    metrics = base.multiclass_metrics(data.y[valid_mask], pred, prob, len(labels))
    metrics.update({
        "task": target.upper(),
        "model_key": model_key,
        "best_epoch": best_epoch,
        "seed": seed,
        "selection_split": "valid",
        "test_accessed": False,
    })

    checkpoint_dir = OUT / "checkpoints/validation_architecture_comparison"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "target": target,
        "model_key": model_key,
        "labels": labels,
        "static_features": list(data.static_features),
        "dynamic_features": list(data.dynamic_features),
        "best_epoch": best_epoch,
        "best_validation_weighted_f1": best_score,
        "hyperparameters": {**HYPERPARAMETERS, "seed": seed},
        "manifest_sha256": base.sha256(base.MANIFEST),
        "contract_sha256": base.sha256(base.CONTRACT),
        "test_accessed": False,
    }, checkpoint_dir / f"{target}_{model_key}.pt")

    frame = data.row_index.loc[valid_mask].reset_index(drop=True).copy()
    frame["task"] = target.upper()
    frame["model_key"] = model_key
    frame["y_true"] = data.y[valid_mask]
    frame["y_pred"] = pred
    for idx, label in enumerate(labels):
        frame[f"prob_{label}"] = prob[:, idx]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, pd.DataFrame(curve), frame


def rounded_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    table = raw_table[["Model group", "Model", "accuracy", "macro_f1", "weighted_f1"]].copy()
    table.columns = ["Model group", "Model", "Accuracy", "Macro-F1", "Weighted-F1"]
    for column in ["Accuracy", "Macro-F1", "Weighted-F1"]:
        table[column] = table[column].round(3)
    return table


def write_markdown(target: str, architecture: pd.DataFrame, baseline: pd.DataFrame) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        columns = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            values = [str(value).replace("|", "\\|") for value in row]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    path = OUT / "tables" / f"validation_model_comparison_{target}.md"
    text = [
        f"# {target.upper()} fixed validation-set model comparison",
        "",
        "Protocol: fixed train split for fitting and fixed validation split for early stopping/evaluation; test set not accessed.",
        "",
        "## Architecture comparison",
        "",
        markdown_table(architecture),
        "",
        "## Baseline comparison",
        "",
        markdown_table(baseline),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def prepare_data(target: str, panel: pd.DataFrame, contract: dict[str, Any], manifest: pd.DataFrame):
    labels = base.TARGETS[target]["labels"]
    base.phase866.LABELS = labels
    base.phase866.NUM_CLASSES = len(labels)
    feature_groups = {
        "static_features": contract["static_features"],
        "dynamic_features": contract["dynamic_features"],
    }
    raw = base.phase866_final.prepare_target_raw(
        panel, feature_groups, target, SimpleNamespace(max_history_days=16)
    )
    split_map = manifest.set_index("cycle_uid")["split"]
    split_values = raw.row_index["cycle_uid"].map(split_map)
    if split_values.isna().any():
        raise RuntimeError(f"{target}: samples absent from split manifest")
    train_mask = split_values.eq("train").to_numpy()
    valid_mask = split_values.eq("valid").to_numpy()
    data = base.phase864.prepare_arrays(raw, fit_mask=train_mask)
    return raw, data, train_mask, valid_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["lh", "hmg"],
        choices=["fsh", "lh", "hmg"],
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for directory in ["tables", "training", "predictions", "checkpoints", "logs"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    contract = json.loads(base.CONTRACT.read_text(encoding="utf-8"))
    manifest = pd.read_csv(base.MANIFEST)
    expected = {"train": 2849, "valid": 611, "test": 611}
    if manifest["cycle_uid"].nunique() != 4071 or manifest["split"].value_counts().to_dict() != expected:
        raise RuntimeError("Frozen manifest does not match 4071 / 2849 / 611 / 611 contract")
    monitoring = pd.read_csv(base.MONITORING, low_memory=False)
    baseline_data = pd.read_csv(base.BASELINE, low_memory=False)
    panel = base.phase864.build_daily_no_sw_panel(monitoring, baseline_data, min_day=1, max_day=20)
    panel = base.phase866_final.add_grouped_labels(panel)
    for feature in contract["static_features"] + contract["dynamic_features"]:
        if feature not in panel.columns:
            raise RuntimeError(f"UI-reduced feature absent: {feature}")
        panel[feature] = pd.to_numeric(panel[feature], errors="coerce")

    run_manifest = {
        "comparison_source": "Phase 8.66 FSH architecture family and hyperparameters",
        "evaluation_protocol": "fixed train/validation holdout; no OOF; test not accessed",
        "cohort_cycles": 4071,
        "split_cycles": expected,
        "models": [{"model_group": g, "model": m, "model_key": k} for g, m, k in ARCH_ROWS],
        "hyperparameters": HYPERPARAMETERS,
        "targets": args.targets,
        "device": str(device),
        "manifest_sha256": base.sha256(base.MANIFEST),
        "contract_sha256": base.sha256(base.CONTRACT),
    }
    target_slug = "_".join(args.targets)
    (OUT / f"logs/validation_architecture_comparison_{target_slug}_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for target in args.targets:
        raw, data, train_mask, valid_mask = prepare_data(target, panel, contract, manifest)
        raw_path = OUT / "tables" / f"table_validation_architecture_comparison_{target}_raw.csv"
        existing = pd.read_csv(raw_path) if args.resume and raw_path.exists() else pd.DataFrame()
        rows = existing.to_dict("records")
        finished = set(existing["model_key"].astype(str)) if not existing.empty else set()
        for group, display_name, model_key in ARCH_ROWS:
            if model_key in finished:
                print(f"[{target.upper()}:{model_key}] resume skip", flush=True)
                continue
            metrics, curve, predictions = fit_one(target, model_key, data, train_mask, valid_mask, device)
            rows.append({"Model group": group, "Model": display_name, **metrics})
            order = {key: idx for idx, (_g, _m, key) in enumerate(ARCH_ROWS)}
            raw_result = pd.DataFrame(rows)
            raw_result["_order"] = raw_result["model_key"].map(order)
            raw_result = raw_result.sort_values("_order").drop(columns="_order")
            raw_result.to_csv(raw_path, index=False)
            curve.to_csv(OUT / "training" / f"validation_architecture_{target}_{model_key}.csv", index=False)
            predictions.to_csv(OUT / "predictions" / f"validation_architecture_{target}_{model_key}.csv", index=False)

        raw_result = pd.read_csv(raw_path)
        order = {key: idx for idx, (_g, _m, key) in enumerate(ARCH_ROWS)}
        raw_result["_order"] = raw_result["model_key"].map(order)
        raw_result = raw_result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        architecture = rounded_table(raw_result)
        architecture.to_csv(OUT / "tables" / f"table_validation_architecture_comparison_{target}.csv", index=False)

        baseline_rows = base.validation_baselines(target, raw, train_mask, valid_mask)
        baseline = pd.DataFrame(baseline_rows)[["model", "accuracy", "macro_f1", "weighted_f1"]]
        baseline.columns = ["Model", "Accuracy", "Macro-F1", "Weighted-F1"]
        gru = raw_result.loc[raw_result["model_key"].eq("gru_addgate")].iloc[0]
        baseline.loc[len(baseline)] = ["GRU(AddGate)", gru["accuracy"], gru["macro_f1"], gru["weighted_f1"]]
        for column in ["Accuracy", "Macro-F1", "Weighted-F1"]:
            baseline[column] = baseline[column].astype(float).round(3)
        baseline.to_csv(OUT / "tables" / f"table_validation_baseline_comparison_{target}.csv", index=False)
        write_markdown(target, architecture, baseline)
        print(f"[{target.upper()}] completed {len(architecture)} architectures", flush=True)

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
