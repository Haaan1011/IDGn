from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "brier": float(brier_score_loss(y_true, y_score if y_score is not None else y_pred)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score if y_score is not None else y_pred))
    except Exception:
        metrics["roc_auc"] = None
    try:
        metrics["average_precision"] = float(
            average_precision_score(y_true, y_score if y_score is not None else y_pred)
        )
    except Exception:
        metrics["average_precision"] = None
    return metrics


def primary_metric_value(task_type: str, metrics: dict[str, Any], primary_metric: str) -> float | None:
    value = metrics.get(primary_metric)
    if value is None:
        return None
    return float(value)
