#!/usr/bin/env python3
"""Evaluate calibration of the frozen GRU + GradientBoosting OHSS-risk model.

Bootstrap resampling is at the cycle level. The input contains one fixed
test prediction for each treatment cycle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


SEED = 20261027
N_BOOTSTRAP = 1000
N_BINS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def roc_auc(y: np.ndarray, probability: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with midranks for tied probabilities."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(len(probability), dtype=float)
    start = 0
    while start < len(probability):
        end = start + 1
        while end < len(probability) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def calibration_coefficients(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood calibration intercept and slope by IRLS."""
    y = np.asarray(y, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(probability / (1.0 - probability))
    design = np.column_stack([np.ones(len(logit)), logit])
    beta = np.array([0.0, 1.0])
    for _ in range(100):
        linear = np.clip(design @ beta, -35.0, 35.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weight = np.maximum(fitted * (1.0 - fitted), 1e-10)
        hessian = design.T @ (design * weight[:, None])
        gradient = design.T @ (y - fitted)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def brier_score(y: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, dtype=float) - np.asarray(probability, dtype=float)) ** 2))


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_coefficients(y, probability)
    return {
        "roc_auc": roc_auc(y, probability),
        "brier_score": brier_score(y, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def cycle_bootstrap(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int]] = []
    for replicate in range(1, N_BOOTSTRAP + 1):
        index = rng.integers(0, len(y), size=len(y))
        sampled_y, sampled_probability = y[index], probability[index]
        if sampled_y.min() == sampled_y.max():
            continue
        try:
            rows.append({"replicate": replicate, **metric_values(sampled_y, sampled_probability)})
        except np.linalg.LinAlgError:
            continue
    result = pd.DataFrame(rows)
    if len(result) < 0.98 * N_BOOTSTRAP:
        raise RuntimeError(f"Only {len(result)} bootstrap replicates were estimable")
    return result


def wilson_interval(events: np.ndarray, n: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = events / n
    denominator = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2.0 * n)) / denominator
    half_width = z * np.sqrt((p * (1.0 - p) + z ** 2 / (4.0 * n)) / n) / denominator
    return np.clip(centre - half_width, 0.0, 1.0), np.clip(centre + half_width, 0.0, 1.0)


def calibration_bins(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"outcome": y, "predicted_probability": probability})
    frame["risk_group"] = pd.qcut(frame["predicted_probability"], q=N_BINS, duplicates="drop")
    summary = frame.groupby("risk_group", observed=True).agg(
        n=("outcome", "size"),
        events=("outcome", "sum"),
        mean_predicted_risk=("predicted_probability", "mean"),
        observed_risk=("outcome", "mean"),
        lower_bound=("predicted_probability", "min"),
        upper_bound=("predicted_probability", "max"),
    ).reset_index(drop=True)
    lower, upper = wilson_interval(summary["events"].to_numpy(float), summary["n"].to_numpy(float))
    summary["observed_risk_ci_lower"] = lower
    summary["observed_risk_ci_upper"] = upper
    summary.insert(0, "risk_group", np.arange(1, len(summary) + 1))
    return summary


def percentile_interval(values: pd.Series) -> tuple[float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    for directory in (output_dir / "source_data", output_dir / "tables", output_dir / "audit"):
        directory.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.predictions)
    required = {"cycle_uid", "ohss", "HYB__GradientBoosting__ohss"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    frame = raw.loc[:, ["cycle_uid", "ohss", "HYB__GradientBoosting__ohss"]].dropna().copy()
    if frame["cycle_uid"].duplicated().any():
        raise ValueError("Expected one final prediction per treatment cycle")
    y = frame["ohss"].astype(int).to_numpy()
    probability = frame["HYB__GradientBoosting__ohss"].astype(float).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("OHSS endpoint must be binary")
    metrics = metric_values(y, probability)
    bootstrap = cycle_bootstrap(y, probability)
    bootstrap.to_csv(output_dir / "source_data" / "ohss_calibration_cycle_bootstrap_1000.csv", index=False)
    curve = calibration_bins(y, probability)
    curve.to_csv(output_dir / "source_data" / "ohss_calibration_curve_5_quantile_groups.csv", index=False)
    intervals = {column: percentile_interval(bootstrap[column]) for column in ["roc_auc", "brier_score", "calibration_intercept", "calibration_slope"]}
    auc_lower, auc_upper = intervals["roc_auc"]
    metrics_table = pd.DataFrame([
        {"metric": "ROC-AUC", "estimate": metrics["roc_auc"], "ci_lower": auc_lower, "ci_upper": auc_upper},
        {"metric": "Brier score", "estimate": metrics["brier_score"], "ci_lower": intervals["brier_score"][0], "ci_upper": intervals["brier_score"][1]},
        {"metric": "Calibration intercept", "estimate": metrics["calibration_intercept"], "ci_lower": intervals["calibration_intercept"][0], "ci_upper": intervals["calibration_intercept"][1]},
        {"metric": "Calibration slope", "estimate": metrics["calibration_slope"], "ci_lower": intervals["calibration_slope"][0], "ci_upper": intervals["calibration_slope"][1]},
    ])
    metrics_table.to_csv(output_dir / "tables" / "ohss_calibration_metrics.csv", index=False)
    (output_dir / "audit" / "calibration_manifest.json").write_text(json.dumps({
        "model": "GRU embedding + GradientBoosting OHSS risk model",
        "prediction_column": "HYB__GradientBoosting__ohss",
        "data_split": "fixed independent test set",
        "unit": "treatment cycle; one final pre-retrieval monitoring record per cycle",
        "n_cycles": int(len(frame)), "ohss_events": int(y.sum()),
        "calibration_curve": "five quantile-defined risk groups with Wilson 95% binomial intervals",
        "metric_intervals": "1,000 cycle-level percentile bootstrap replicates",
        "seed": SEED,
        "model_refit": False,
        "test_predictions_modified": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
