#!/usr/bin/env python3
"""Calculate Gn discrimination, accuracy and treatment-cycle bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


TARGETS = ("FSH", "LH", "HMG")
FILES = {
    "FSH": "fsh_all_expert_hybrid_test_predictions.csv",
    "LH": "lh_all_expert_hybrid_test_predictions.csv",
    "HMG": "hmg_all_expert_hybrid_test_predictions.csv",
}
PROBABILITY_COLUMNS = {
    "FSH": ["prob_low_or_none", "prob_moderate", "prob_high"],
    "LH": ["prob_dose_0", "prob_dose_75", "prob_dose_ge150"],
    "HMG": ["prob_dose_0", "prob_dose_75", "prob_dose_ge150"],
}
CLASSES = [0, 1, 2]
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261016


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(prediction_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for target in TARGETS:
        path = prediction_dir / FILES[target]
        frame = pd.read_csv(path)
        required = {
            "cycle_uid",
            "monitoring_order",
            "y_true",
            "y_pred",
            *PROBABILITY_COLUMNS[target],
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"Missing evaluable values in {path.name}")
        if frame.duplicated(["cycle_uid", "monitoring_order"]).any():
            raise ValueError(f"Duplicated cycle x Monitor records in {path.name}")
        if "monitoring_order.1" in frame.columns and not frame["monitoring_order"].equals(
            frame["monitoring_order.1"]
        ):
            raise ValueError(f"Monitor columns disagree in {path.name}")
        frames[target] = frame
        hashes[target] = sha256(path)
    return frames, hashes


def calculate(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    monitor_rows = []
    bootstrap_rows = []
    monitor_bootstrap_rows = []
    for target_index, target in enumerate(TARGETS):
        frame = frames[target]
        y_true = frame["y_true"].to_numpy(dtype=int)
        y_pred = frame["y_pred"].to_numpy(dtype=int)
        probabilities = frame[PROBABILITY_COLUMNS[target]].to_numpy(dtype=float)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise AssertionError(f"{target} probabilities do not sum to one")

        y_binary = label_binarize(y_true, classes=CLASSES)
        macro_auc = float(roc_auc_score(y_binary, probabilities, average="macro"))
        macro_ap = float(average_precision_score(y_binary, probabilities, average="macro"))
        accuracy = float(accuracy_score(y_true, y_pred))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        cycle_groups = {
            cycle_uid: indices.to_numpy(dtype=int)
            for cycle_uid, indices in frame.groupby("cycle_uid", sort=True).groups.items()
        }
        cycle_ids = np.asarray(list(cycle_groups), dtype=object)
        rng = np.random.default_rng(BOOTSTRAP_SEED + target_index)
        boot_accuracy = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        boot_weighted_f1 = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for replicate in range(BOOTSTRAP_REPLICATES):
            sampled_cycles = rng.choice(cycle_ids, size=len(cycle_ids), replace=True)
            sampled_indices = np.concatenate([cycle_groups[cycle] for cycle in sampled_cycles])
            sampled_true = y_true[sampled_indices]
            sampled_pred = y_pred[sampled_indices]
            boot_accuracy[replicate] = accuracy_score(sampled_true, sampled_pred)
            boot_weighted_f1[replicate] = f1_score(
                sampled_true, sampled_pred, average="weighted", zero_division=0
            )
            bootstrap_rows.append(
                {
                    "Target": target,
                    "Replicate": replicate + 1,
                    "Accuracy": boot_accuracy[replicate],
                    "Weighted-F1": boot_weighted_f1[replicate],
                }
            )
        accuracy_ci = np.percentile(boot_accuracy, [2.5, 97.5])
        weighted_f1_ci = np.percentile(boot_weighted_f1, [2.5, 97.5])
        metric_rows.append(
            {
                "Target": target,
                "Evaluable records": len(frame),
                "Treatment cycles": frame["cycle_uid"].nunique(),
                "Accuracy": accuracy,
                "Accuracy CI lower": float(accuracy_ci[0]),
                "Accuracy CI upper": float(accuracy_ci[1]),
                "Weighted-F1": weighted_f1,
                "Weighted-F1 CI lower": float(weighted_f1_ci[0]),
                "Weighted-F1 CI upper": float(weighted_f1_ci[1]),
                "Macro-AUC": macro_auc,
                "Average precision": macro_ap,
            }
        )

        for monitor, group in frame.groupby("monitoring_order", sort=True):
            if int(monitor) > 9:
                continue
            monitor_true = group["y_true"].to_numpy(dtype=int)
            monitor_pred = group["y_pred"].to_numpy(dtype=int)
            monitor_accuracy = float(accuracy_score(monitor_true, monitor_pred))
            monitor_rng = np.random.default_rng(
                BOOTSTRAP_SEED + 1000 + target_index * 100 + int(monitor)
            )
            monitor_boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for replicate in range(BOOTSTRAP_REPLICATES):
                sampled = monitor_rng.integers(0, len(group), size=len(group))
                monitor_boot[replicate] = accuracy_score(
                    monitor_true[sampled], monitor_pred[sampled]
                )
                monitor_bootstrap_rows.append(
                    {
                        "Target": target,
                        "Monitor": int(monitor),
                        "Replicate": replicate + 1,
                        "Accuracy": monitor_boot[replicate],
                    }
                )
            monitor_ci = np.percentile(monitor_boot, [2.5, 97.5])
            monitor_rows.append(
                {
                    "Target": target,
                    "Monitor": int(monitor),
                    "Evaluable treatment cycles": group["cycle_uid"].nunique(),
                    "Accuracy": monitor_accuracy,
                    "Accuracy CI lower": float(monitor_ci[0]),
                    "Accuracy CI upper": float(monitor_ci[1]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    monitor_table = pd.DataFrame(monitor_rows)
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    monitor_bootstrap_table = pd.DataFrame(monitor_bootstrap_rows)
    if len(monitor_table) != 27:
        raise AssertionError("Expected Monitor 1-9 for all three targets")
    return (
        metrics,
        monitor_table,
        bootstrap_table,
        monitor_bootstrap_table,
    )


def main() -> None:
    args = parse_args()
    source_dir = args.output_dir / "source_data"
    audit_dir = args.output_dir / "audit"
    for folder in (source_dir, audit_dir):
        folder.mkdir(parents=True, exist_ok=True)

    frames, hashes = load_predictions(args.prediction_dir)
    (
        metrics,
        monitor_table,
        bootstrap_table,
        monitor_bootstrap_table,
    ) = calculate(frames)
    metrics.to_csv(source_dir / "gn_test_metrics.csv", index=False, float_format="%.10f")
    confusion_rows = []
    for target, frame in frames.items():
        matrix = confusion_matrix(frame.y_true, frame.y_pred, labels=CLASSES)
        doses = [column.removeprefix("prob_") for column in PROBABILITY_COLUMNS[target]]
        for i, observed in enumerate(doses):
            total = matrix[i].sum()
            for j, predicted in enumerate(doses):
                count = int(matrix[i, j])
                confusion_rows.append({"Target": target, "Observed dose class": observed,
                                       "Predicted dose class": predicted, "Count": count,
                                       "Row proportion": count / total if total else np.nan})
    pd.DataFrame(confusion_rows).to_csv(source_dir / "gn_confusion_tables.csv", index=False)
    monitor_table.to_csv(source_dir / "monitor_accuracy_1to9.csv", index=False, float_format="%.10f")
    bootstrap_table.to_csv(
        source_dir / "gn_cycle_bootstrap_replicates.csv.gz",
        index=False,
        float_format="%.10f",
        compression="gzip",
    )
    monitor_bootstrap_table.to_csv(
        source_dir / "monitor_cycle_bootstrap_replicates.csv.gz",
        index=False,
        float_format="%.10f",
        compression="gzip",
    )

    manifest = {
        "model_source": "final task-specific hybrid models",
        "test_cycles": int(frames["FSH"]["cycle_uid"].nunique()),
        "test_records_by_task": {target: len(frame) for target, frame in frames.items()},
        "reported_monitor_range": [1, 9],
        "overall_bootstrap": {
            "unit": "treatment cycle",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "95% percentile",
        },
        "monitor_bootstrap": {
            "unit": "treatment cycle within Monitor",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_base": BOOTSTRAP_SEED + 1000,
            "interval": "95% percentile",
        },
        "metrics": metrics.to_dict(orient="records"),
        "input_sha256": hashes,
    }
    with (audit_dir / "metric_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
