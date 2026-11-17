from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

from .artifacts import save_dataframe, write_json


TREE_MODEL_MARKERS = ("xgb", "lgbm", "catboost", "randomforest")


def _is_tree_like(model_name: str) -> bool:
    name = model_name.lower()
    return any(marker in name for marker in TREE_MODEL_MARKERS)


def _select_positive_class_shap_values(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if isinstance(values, list):
        if not values:
            return np.asarray(values)
        if len(values) > 1:
            array = np.asarray(values[1])
        else:
            array = np.asarray(values[0])
    if array.ndim == 3:
        if array.shape[2] > 1:
            return array[:, :, 1]
        return array[:, :, 0]
    return array


def explain_bundle(
    bundle: dict[str, Any],
    x_sample: pd.DataFrame,
    y_sample: pd.Series,
    output_dir: Path,
    task_name: str,
    model_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = bundle["estimator"]
    preprocessor = bundle["preprocessor"]
    feature_names = bundle["feature_names"]
    transformed = preprocessor.transform(x_sample)

    summary_rows: list[dict[str, Any]] = []
    method = "shap"
    try:
        if shap is None:
            raise ImportError("shap is not installed in this environment.")
        if _is_tree_like(model_name):
            explainer = shap.TreeExplainer(estimator)
            values = _select_positive_class_shap_values(explainer.shap_values(transformed))
            mean_abs = np.abs(values).mean(axis=0)
        else:
            raise ValueError("SHAP limited to tree-like models in this phase.")
    except Exception as exc:
        method = f"permutation_fallback:{exc.__class__.__name__}:{str(exc).splitlines()[0][:120]}"
        perm = permutation_importance(estimator, transformed, y_sample, n_repeats=5, random_state=42)
        mean_abs = perm.importances_mean

    order = np.argsort(mean_abs)[::-1]
    for rank, idx in enumerate(order[:20], start=1):
        summary_rows.append(
            {
                "task_name": task_name,
                "model_name": model_name,
                "rank": rank,
                "feature": feature_names[idx],
                "importance": float(mean_abs[idx]),
                "method": method,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(output_dir / f"{task_name}_{model_name}_shap_summary.csv", summary_df)
    write_json(
        output_dir / f"{task_name}_{model_name}_shap_summary.json",
        {
            "task_name": task_name,
            "model_name": model_name,
            "method": method,
            "top_features": summary_rows[:10],
        },
    )
    return {
        "method": method,
        "top_features": summary_rows[:10],
        "summary_path": str(output_dir / f"{task_name}_{model_name}_shap_summary.csv"),
    }
