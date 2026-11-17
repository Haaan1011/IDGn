from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score

from .artifacts import build_run_id, prepare_run_dir, save_bundle, save_dataframe, write_json, write_markdown
from .config import read_yaml
from .metrics import classification_metrics, primary_metric_value, regression_metrics
from .paths import REPO_ROOT, resolve_repo_path
from .preprocessing import build_preprocessor, select_feature_columns
from .registry import build_estimator
from .shap_utils import explain_bundle
from .splits import build_split_manifest, load_split_manifest
from .tensorboard import SimpleTensorBoardWriter


@dataclass(frozen=True)
class TaskOutcome:
    task_name: str
    status: str
    reason: str | None
    best_model: str | None
    best_metric: float | None
    task_report_path: str | None


def _load_frame(path_like: str | Path) -> pd.DataFrame:
    return pd.read_csv(resolve_repo_path(path_like))


def _ensure_binary_target(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    cleaned = cleaned.fillna(0)
    return (cleaned.astype(float) > 0).astype(int)


def _prepare_target(series: pd.Series, task_type: str) -> pd.Series:
    if task_type == "classification":
        return _ensure_binary_target(series)
    return pd.to_numeric(series, errors="coerce")



def _apply_task_row_filter(frame: pd.DataFrame, task_cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply an optional task-level row filter before split-specific fitting.

    Used for hierarchical endpoints, e.g. training live birth only among
    clinical-pregnancy-positive ET cycles.
    """
    row_filter = task_cfg.get("row_filter") or {}
    if not row_filter:
        return frame
    column = str(row_filter.get("column", "")).strip()
    if not column:
        raise ValueError("row_filter.column is required when row_filter is set")
    if column not in frame.columns:
        raise KeyError(f"row_filter column not found: {column}")
    mask = pd.Series(True, index=frame.index)
    if "values" in row_filter:
        values = row_filter.get("values") or []
        mask &= frame[column].isin(values)
    if "min" in row_filter:
        mask &= pd.to_numeric(frame[column], errors="coerce").ge(float(row_filter["min"]))
    if "max" in row_filter:
        mask &= pd.to_numeric(frame[column], errors="coerce").le(float(row_filter["max"]))
    return frame.loc[mask].copy()


def _resolve_task_dataset(
    dataset_key: str,
    baseline_dataset: pd.DataFrame,
    snapshot_dataset: pd.DataFrame,
    layer3_snapshot_dataset: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the modeling frame for a task-level dataset key."""
    if dataset_key == "baseline":
        return baseline_dataset
    if dataset_key == "snapshot":
        return snapshot_dataset
    if dataset_key == "layer3_snapshot":
        if layer3_snapshot_dataset is None:
            raise KeyError("layer3_snapshot dataset requested but not configured")
        return layer3_snapshot_dataset
    raise KeyError(f"unsupported dataset_key: {dataset_key}")


def _safe_metric_value(task_type: str, metrics: dict[str, Any], primary_metric: str) -> float:
    value = primary_metric_value(task_type, metrics, primary_metric)
    return float(value) if value is not None else float("nan")

def _metric_higher_is_better(task_type: str, metric_name: str) -> bool:
    """Return optimization direction for supported primary metrics."""
    metric = str(metric_name).lower()
    if metric in {"rmse", "mae", "mse", "brier"}:
        return False
    if metric in {
        "r2",
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    }:
        return True
    return task_type == "classification"



def _build_sample_weight(y_train: pd.Series) -> pd.Series | None:
    cleaned = pd.to_numeric(y_train, errors="coerce").dropna().astype(int)
    if cleaned.nunique() < 2:
        return None
    positive = int((cleaned == 1).sum())
    negative = int((cleaned == 0).sum())
    if positive == 0 or negative == 0:
        return None
    pos_weight = negative / positive
    weights = pd.Series(np.where(cleaned == 1, pos_weight, 1.0), index=cleaned.index, dtype=float)
    return weights.reindex(y_train.index, fill_value=1.0)


def _fit_probability_calibrator(scores: np.ndarray, y_true: pd.Series) -> LogisticRegression | None:
    valid_mask = pd.Series(y_true).notna()
    y = pd.Series(y_true).loc[valid_mask].astype(int)
    x = np.asarray(scores).reshape(-1, 1)[valid_mask.to_numpy()]
    if y.nunique() < 2 or len(y) == 0:
        return None
    calibrator = LogisticRegression(max_iter=1000, class_weight="balanced")
    calibrator.fit(x, y)
    return calibrator


def _calibrate_scores(calibrator: LogisticRegression | None, scores: np.ndarray) -> np.ndarray:
    numeric = np.asarray(scores, dtype=float).reshape(-1, 1)
    if calibrator is None:
        return numeric.reshape(-1)
    probabilities = calibrator.predict_proba(numeric)
    if getattr(probabilities, "ndim", 0) == 2 and probabilities.shape[1] > 1:
        return np.asarray(probabilities)[:, 1].reshape(-1)
    return numeric.reshape(-1)


def _select_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    *,
    default: float = 0.5,
    strategy: str = "maximize_f1",
) -> tuple[float, dict[str, float]]:
    values = pd.Series(np.asarray(y_prob, dtype=float))
    labels = pd.Series(y_true).astype(int)
    quantiles = np.quantile(values.dropna(), np.linspace(0.01, 0.99, 49))
    candidates = sorted(set(float(x) for x in quantiles.tolist()))
    candidates.append(default)
    best_threshold = default
    best_score = -1.0
    best_payload = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in candidates:
        pred = (values >= threshold).astype(int)
        metrics = classification_metrics(labels.to_numpy(), pred.to_numpy(), values.to_numpy(), threshold=threshold)
        if strategy == "maximize_balanced_accuracy":
            score = float(metrics.get("balanced_accuracy") or 0.0)
        elif strategy == "maximize_recall":
            score = float(metrics.get("recall") or 0.0)
        elif strategy == "maximize_f2":
            score = float(fbeta_score(labels.to_numpy(), pred.to_numpy(), beta=2.0, zero_division=0))
        else:
            score = float(metrics.get("f1") or 0.0)
        if score > best_score or (
            score == best_score
            and float(metrics.get("recall") or 0.0) >= float(best_payload.get("recall") or 0.0)
            and threshold < best_threshold
        ):
            best_score = score
            best_threshold = float(threshold)
            best_payload = {
                "precision": float(metrics.get("precision") or 0.0),
                "recall": float(metrics.get("recall") or 0.0),
                "f1": float(metrics.get("f1") or 0.0),
            }
    return best_threshold, best_payload


def _apply_task_feature_overrides(task_name: str, feature_columns: list[str], task_cfg: dict[str, Any]) -> list[str]:
    default_excluded = {
        "layer2_mii": {
            "total_follicle_count",
            "left_follicle_count",
            "right_follicle_count",
            "follicle_count_lt_10",
            "afc",
        }
    }.get(task_name, set())
    excluded = set(task_cfg.get("feature_drop_columns", []) or []) | default_excluded
    if not excluded:
        return feature_columns
    return [column for column in feature_columns if column not in excluded]


def _build_model_bundle(
    estimator: Any,
    preprocessor: Any,
    feature_names: list[str],
    task_name: str,
    target_column: str,
    feature_set: str,
    model_name: str,
    task_type: str,
) -> dict[str, Any]:
    return {
        "estimator": estimator,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "task_name": task_name,
        "target_column": target_column,
        "feature_set": feature_set,
        "model_name": model_name,
        "task_type": task_type,
    }


def _predict_scores(estimator: Any, task_type: str, x_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if task_type == "classification":
        y_pred = estimator.predict(x_frame)
        y_score = None
        if hasattr(estimator, "predict_proba"):
            prob = estimator.predict_proba(x_frame)
            if prob.ndim == 2 and prob.shape[1] > 1:
                y_score = prob[:, 1]
            else:
                y_score = prob.reshape(-1)
        elif hasattr(estimator, "decision_function"):
            score = estimator.decision_function(x_frame)
            y_score = score.reshape(-1)
        return y_pred, y_score

    y_pred = estimator.predict(x_frame)
    return y_pred, None


def _fit_estimator_with_history(
    estimator: Any,
    model_name: str,
    task_type: str,
    x_train_transformed: np.ndarray,
    y_train: pd.Series,
    x_valid_transformed: np.ndarray,
    y_valid: pd.Series,
    sample_weight: pd.Series | None = None,
    progress_bar: tqdm | None = None,
) -> None:
    if model_name == "xgboost":
        fit_kwargs: dict[str, Any] = {
            "eval_set": [(x_train_transformed, y_train), (x_valid_transformed, y_valid)],
            "verbose": 1 if progress_bar is not None else False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        estimator.fit(x_train_transformed, y_train, **fit_kwargs)
        if progress_bar is not None:
            progress_bar.update(progress_bar.total - progress_bar.n if progress_bar.total is not None else 1)
        return

    if model_name == "lightgbm":
        eval_metric = "binary_logloss" if task_type == "classification" else "rmse"
        fit_kwargs = {
            "eval_set": [(x_train_transformed, y_train), (x_valid_transformed, y_valid)],
            "eval_names": ["train", "valid"],
            "eval_metric": eval_metric,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        callbacks = []
        progress_callback = _make_lightgbm_progress_callback(progress_bar)
        if progress_callback is not None:
            callbacks.append(progress_callback)
        try:
            from lightgbm.callback import log_evaluation

            callbacks.append(log_evaluation(period=1 if progress_bar is not None else 10))
        except Exception:
            pass
        if callbacks:
            fit_kwargs["callbacks"] = callbacks
        estimator.fit(x_train_transformed, y_train, **fit_kwargs)
        if progress_bar is not None and progress_bar.n < progress_bar.total:
            progress_bar.update(progress_bar.total - progress_bar.n)
        return

    if model_name == "catboost":
        fit_kwargs = {
            "eval_set": (x_valid_transformed, y_valid),
            "verbose": 1 if progress_bar is not None else False,
            "metric_period": 1 if progress_bar is not None else 10,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if progress_bar is not None:
            fit_kwargs["callbacks"] = [_CatBoostProgressCallback(progress_bar)]
        estimator.fit(x_train_transformed, y_train, **fit_kwargs)
        if progress_bar is not None and progress_bar.n < progress_bar.total:
            progress_bar.update(progress_bar.total - progress_bar.n)
        return

    if model_name == "mlp":
        if progress_bar is not None:
            try:
                estimator.verbose = True
            except Exception:
                pass
        if sample_weight is not None:
            try:
                estimator.fit(x_train_transformed, y_train, sample_weight=sample_weight)
                return
            except TypeError:
                pass
        estimator.fit(x_train_transformed, y_train)
        if progress_bar is not None and progress_bar.n < progress_bar.total:
            progress_bar.update(progress_bar.total - progress_bar.n)
        return

    if sample_weight is not None:
        try:
            estimator.fit(x_train_transformed, y_train, sample_weight=sample_weight)
            return
        except TypeError:
            pass
    estimator.fit(x_train_transformed, y_train)


def _extract_training_history(estimator: Any, model_name: str, task_type: str) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    evals_result = _get_evals_result(estimator)
    for raw_phase, metrics in evals_result.items():
        phase = _normalize_eval_phase(str(raw_phase))
        for metric_name, values in metrics.items():
            for step, value in enumerate(values):
                numeric = _safe_float(value)
                if numeric is None:
                    continue
                history.append(
                    {
                        "source": "evals_result",
                        "phase": phase,
                        "metric": str(metric_name),
                        "step": int(step),
                        "value": numeric,
                    }
                )

    if model_name == "mlp":
        for step, value in enumerate(getattr(estimator, "loss_curve_", []) or []):
            numeric = _safe_float(value)
            if numeric is not None:
                history.append(
                    {
                        "source": "mlp_loss_curve",
                        "phase": "train",
                        "metric": "loss",
                        "step": int(step),
                        "value": numeric,
                    }
                )
        for step, value in enumerate(getattr(estimator, "validation_scores_", []) or []):
            numeric = _safe_float(value)
            if numeric is not None:
                history.append(
                    {
                        "source": "mlp_validation_scores",
                        "phase": "valid",
                        "metric": "score",
                        "step": int(step),
                        "value": numeric,
                    }
                )
    return history


def _write_training_history(
    *,
    model_dir: Path,
    tb_writer: SimpleTensorBoardWriter,
    tb_prefix: str,
    history: list[dict[str, Any]],
) -> None:
    if not history:
        return
    history_frame = pd.DataFrame(history)
    save_dataframe(model_dir / "training_history.csv", history_frame)
    write_json(model_dir / "training_history.json", {"history": history})
    for row in history:
        tb_writer.add_scalar(
            f"{tb_prefix}/training_curve/{row['phase']}/{row['metric']}",
            float(row["value"]),
            step=int(row["step"]),
        )


def _get_evals_result(estimator: Any) -> dict[str, dict[str, list[Any]]]:
    if hasattr(estimator, "evals_result"):
        try:
            result = estimator.evals_result()
            if isinstance(result, dict):
                return result
        except (TypeError, AttributeError):
            pass
    result = getattr(estimator, "evals_result_", {})
    return result if isinstance(result, dict) else {}


def _normalize_eval_phase(raw_phase: str) -> str:
    lowered = raw_phase.lower()
    if lowered in {"validation_0", "train", "training"}:
        return "train"
    if lowered in {"validation_1", "valid", "validation", "eval"}:
        return "valid"
    return lowered.replace(" ", "_")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _model_epoch_total(estimator: Any, model_name: str) -> int | None:
    params = getattr(estimator, 'get_params', lambda: {})()
    if model_name == 'xgboost':
        return int(params.get('n_estimators') or 0) or None
    if model_name == 'lightgbm':
        return int(params.get('n_estimators') or 0) or None
    if model_name == 'catboost':
        return int(params.get('iterations') or 0) or None
    if model_name == 'mlp':
        return int(params.get('max_iter') or 0) or None
    return None


def _make_lightgbm_progress_callback(progress_bar: tqdm | None):
    if progress_bar is None:
        return None

    def _callback(env: Any) -> None:
        progress_bar.update(1)
        if getattr(env, 'evaluation_result_list', None):
            first = env.evaluation_result_list[0]
            if len(first) >= 3:
                progress_bar.set_postfix_str(f'{first[0]}/{first[1]}:{float(first[2]):.4f}')

    _callback.order = 0
    _callback.before_iteration = False
    return _callback


class _CatBoostProgressCallback:
    def __init__(self, progress_bar: tqdm | None) -> None:
        self.progress_bar = progress_bar

    def after_iteration(self, info: Any) -> bool:
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        return True

    def after_training(self, info: Any) -> Any:
        return info


def _fit_and_eval_model(
    *,
    task_name: str,
    task_cfg: dict[str, Any],
    task_type: str,
    target_column: str,
    feature_set: str,
    model_name: str,
    seed: int,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    tb_writer: SimpleTensorBoardWriter,
) -> dict[str, Any]:
    feature_columns = _apply_task_feature_overrides(task_name, feature_columns, task_cfg)
    bundle = select_feature_columns(train_frame, feature_columns)
    preprocessor = build_preprocessor(bundle)
    x_train = train_frame[bundle.feature_columns]
    x_valid = valid_frame[bundle.feature_columns]
    x_test = test_frame[bundle.feature_columns]
    y_train = _prepare_target(train_frame[target_column], task_type)
    y_valid = _prepare_target(valid_frame[target_column], task_type)
    y_test = _prepare_target(test_frame[target_column], task_type)

    if task_type == "classification" and y_train.nunique(dropna=True) < 2:
        return {
            "status": "skipped",
            "reason": "training split contains fewer than two classes",
            "model_name": model_name,
        }
    if task_type == "regression" and y_train.dropna().empty:
        return {
            "status": "skipped",
            "reason": "training split has no valid labels",
            "model_name": model_name,
        }

    x_train_transformed = preprocessor.fit_transform(x_train)
    x_valid_transformed = preprocessor.transform(x_valid)
    x_test_transformed = preprocessor.transform(x_test)
    feature_names = list(preprocessor.get_feature_names_out())

    estimator = build_estimator(model_name=model_name, task_type=task_type, seed=seed)
    sample_weight = None
    epoch_total = _model_epoch_total(estimator, model_name)
    epoch_bar = tqdm(total=epoch_total, desc=f"{task_name}/{model_name} epoch", unit="epoch", leave=False, dynamic_ncols=True) if epoch_total else None
    if task_type == "classification" and (task_name == "layer2_ohss" or bool(task_cfg.get("use_class_balancing", False))):
        sample_weight = _build_sample_weight(y_train)
    _fit_estimator_with_history(
        estimator=estimator,
        model_name=model_name,
        task_type=task_type,
        x_train_transformed=x_train_transformed,
        y_train=y_train,
        x_valid_transformed=x_valid_transformed,
        y_valid=y_valid,
        sample_weight=sample_weight,
        progress_bar=epoch_bar,
    )
    if epoch_bar is not None:
        epoch_bar.close()
    training_history = _extract_training_history(
        estimator=estimator,
        model_name=model_name,
        task_type=task_type,
    )

    valid_pred, valid_score = _predict_scores(estimator, task_type, x_valid_transformed)
    test_pred, test_score = _predict_scores(estimator, task_type, x_test_transformed)

    if task_type == "classification":
        valid_score_array = np.asarray(valid_score if valid_score is not None else valid_pred, dtype=float).reshape(-1)
        test_score_array = np.asarray(test_score if test_score is not None else test_pred, dtype=float).reshape(-1)
        calibrator = None
        if task_name in {"layer2_ohss", "layer3_live_birth", "layer3_clinical_pregnancy"} or bool(task_cfg.get("calibration_strategy")):
            calibrator = _fit_probability_calibrator(valid_score_array, y_valid)
        valid_prob = _calibrate_scores(calibrator, valid_score_array)
        test_prob = _calibrate_scores(calibrator, test_score_array)
        threshold = 0.5
        threshold_metrics: dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        threshold_strategy = str(task_cfg.get("decision_threshold_strategy") or "maximize_f1")
        threshold, threshold_metrics = _select_threshold(
            y_valid,
            valid_prob,
            default=0.5,
            strategy=threshold_strategy,
        )
        valid_pred = (valid_prob >= threshold).astype(int)
        test_pred = (test_prob >= threshold).astype(int)
        valid_metrics = classification_metrics(y_valid.to_numpy(), valid_pred, valid_prob, threshold=threshold)
        test_metrics = classification_metrics(y_test.to_numpy(), test_pred, test_prob, threshold=threshold)
    else:
        calibrator = None
        threshold = None
        valid_prob = None
        test_prob = None
        threshold_metrics = {}
        valid_metrics = regression_metrics(y_valid.to_numpy(), valid_pred)
        test_metrics = regression_metrics(y_test.to_numpy(), test_pred)

    model_dir = output_dir / task_name / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_payload = _build_model_bundle(
        estimator=estimator,
        preprocessor=preprocessor,
        feature_names=feature_names,
        task_name=task_name,
        target_column=target_column,
        feature_set=feature_set,
        model_name=model_name,
        task_type=task_type,
    )
    if task_cfg.get("conditional_on_task"):
        bundle_payload["conditional_on_task"] = str(task_cfg["conditional_on_task"])
    if task_cfg.get("conditional_on_target"):
        bundle_payload["conditional_on_target"] = str(task_cfg["conditional_on_target"])
    if task_cfg.get("prediction_semantics"):
        bundle_payload["prediction_semantics"] = str(task_cfg["prediction_semantics"])
    if task_cfg.get("row_filter"):
        bundle_payload["row_filter"] = task_cfg.get("row_filter")

    if task_type == "classification" and calibrator is not None:
        bundle_payload["probability_calibration"] = {
            "method": str(task_cfg.get("calibration_strategy", "sigmoid")),
            "coef": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
            "threshold": float(threshold),
            "threshold_strategy": str(task_cfg.get("decision_threshold_strategy", "maximize_f1")),
        }
    save_bundle(model_dir / "bundle.joblib", bundle_payload)

    prediction_frame = test_frame[["cycle_uid", "cycle_id", "art_id", "split"]].copy()
    prediction_frame["y_true"] = y_test.to_numpy()
    prediction_frame["y_pred"] = test_pred
    if task_type == "classification":
        prediction_frame["y_score_raw"] = test_score_array
        prediction_frame["y_prob"] = test_prob
        prediction_frame["y_score"] = test_prob
        prediction_frame["threshold"] = float(threshold)
    elif test_score is not None:
        prediction_frame["y_score"] = test_score
    save_dataframe(model_dir / "test_predictions.csv", prediction_frame)

    metrics_payload = {
        "task_name": task_name,
        "target_column": target_column,
        "model_name": model_name,
        "task_type": task_type,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "primary_metric": str(task_cfg.get("primary_metric") or ("roc_auc" if task_type == "classification" else "rmse")),
        "primary_valid_metric": _safe_metric_value(
            task_type,
            valid_metrics,
            str(task_cfg.get("primary_metric") or ("roc_auc" if task_type == "classification" else "rmse")),
        ),
        "primary_test_metric": _safe_metric_value(
            task_type,
            test_metrics,
            str(task_cfg.get("primary_metric") or ("roc_auc" if task_type == "classification" else "rmse")),
        ),
        "calibration_strategy": str(task_cfg.get("calibration_strategy") or ("sigmoid" if task_type == "classification" else None)) if task_type == "classification" else None,
        "decision_threshold_strategy": str(task_cfg.get("decision_threshold_strategy") or ("maximize_f1" if task_type == "classification" else None)) if task_type == "classification" else None,
        "selected_threshold": float(threshold) if threshold is not None else None,
        "threshold_metrics": threshold_metrics if task_type == "classification" else None,
        "n_train": int(len(train_frame)),
        "n_valid": int(len(valid_frame)),
        "n_test": int(len(test_frame)),
    }
    write_json(model_dir / "metrics.json", metrics_payload)

    tb_prefix = f"{task_name}/{model_name}"
    _write_training_history(
        model_dir=model_dir,
        tb_writer=tb_writer,
        tb_prefix=tb_prefix,
        history=training_history,
    )
    for metric_name, metric_value in valid_metrics.items():
        if metric_value is not None:
            tb_writer.add_scalar(f"{tb_prefix}/valid/{metric_name}", float(metric_value), step=0)
    for metric_name, metric_value in test_metrics.items():
        if metric_value is not None:
            tb_writer.add_scalar(f"{tb_prefix}/test/{metric_name}", float(metric_value), step=0)

    shap_dir = model_dir / "shap"
    shap_summary = None
    if model_name in {"xgboost", "lightgbm", "catboost", "random_forest"}:
        sample_size = min(len(x_test), 500)
        if sample_size > 0:
            shap_summary = explain_bundle(
                bundle=bundle_payload,
                x_sample=x_test.iloc[:sample_size].copy(),
                y_sample=y_test.iloc[:sample_size].copy(),
                output_dir=shap_dir,
                task_name=task_name,
                model_name=model_name,
            )
            for index, item in enumerate(shap_summary["top_features"][:10], start=1):
                tb_writer.add_scalar(
                    f"{tb_prefix}/shap/top_{index}",
                    float(item["importance"]),
                    step=0,
                )

    return {
        "status": "trained",
        "reason": None,
        "model_name": model_name,
        "model_dir": str(model_dir),
        "metrics": metrics_payload,
        "shap": shap_summary,
    }


def run_experiment_pipeline(
    *,
    experiment_config_path: str | Path,
    paths_config_path: str | Path,
    force_rebuild_split_manifest: bool = False,
    task_filter: list[str] | None = None,
    model_filter: list[str] | None = None,
) -> dict[str, Any]:
    experiment_config = read_yaml(resolve_repo_path(experiment_config_path))
    paths_config = read_yaml(resolve_repo_path(paths_config_path))

    split_cfg = experiment_config["split"]
    training_cfg = experiment_config["training"]
    tasks_cfg = experiment_config["tasks"]

    output_root = resolve_repo_path(training_cfg["output_root"])
    run_id = build_run_id("phase8")
    run_dir = prepare_run_dir(output_root, run_id)
    tensorboard_dir = run_dir / training_cfg.get("tensorboard_dirname", "tensorboard")

    baseline_dataset = _load_frame(training_cfg["baseline_dataset_path"])
    layer3_dataset_path = training_cfg.get("layer3_snapshot_dataset_path")
    layer3_snapshot_dataset = _load_frame(layer3_dataset_path) if layer3_dataset_path else None
    split_manifest_path = resolve_repo_path(split_cfg["manifest_path"])
    if force_rebuild_split_manifest or not split_manifest_path.exists():
        split_manifest, split_summary = build_split_manifest(
            source_df=baseline_dataset,
            split_config=split_cfg,
            output_path=split_manifest_path,
        )
    else:
        split_manifest = load_split_manifest(split_manifest_path)
        split_summary = {
            "split_version": split_cfg.get("split_version", "v1"),
            "seed": int(split_cfg["seed"]),
            "train_cycles": int((split_manifest["split"] == "train").sum()),
            "valid_cycles": int((split_manifest["split"] == "valid").sum()),
            "test_cycles": int((split_manifest["split"] == "test").sum()),
            "total_cycles": int(len(split_manifest)),
        }

    snapshot_dataset = _load_frame(training_cfg["snapshot_dataset_path"])
    feature_manifest = read_yaml(resolve_repo_path(training_cfg["feature_manifest_path"]))
    excluded = set(feature_manifest.get("excluded_for_leakage", []))

    task_results: list[dict[str, Any]] = []
    selected_tasks = {name for name in task_filter} if task_filter else None
    selected_models = {name for name in model_filter} if model_filter else None
    task_queue = [task_cfg for task_cfg in tasks_cfg if selected_tasks is None or task_cfg["name"] in selected_tasks]
    with SimpleTensorBoardWriter(tensorboard_dir) as tb_writer, tqdm(
        total=len(task_queue),
        desc="layer progress",
        unit="layer",
        dynamic_ncols=True,
    ) as layer_bar:
        tb_writer.add_scalar("run/total_cycles", float(split_summary["total_cycles"]), step=0)
        for task_cfg in task_queue:
            task_name = task_cfg["name"]
            if selected_tasks is not None and task_name not in selected_tasks:
                continue
            target_column = task_cfg["target"]
            task_type = task_cfg["task_type"]
            feature_set = task_cfg["feature_set"]
            candidate_models = [
                model_name for model_name in task_cfg["candidate_models"] if selected_models is None or model_name in selected_models
            ]
            if not candidate_models:
                task_results.append(
                    {
                        "task_name": task_name,
                        "status": "skipped",
                        "reason": "no candidate models selected",
                        "best_model": None,
                        "best_metric": None,
                        "task_report_path": None,
                    }
                )
                continue

            dataset_key = task_cfg["dataset_key"]
            dataset = _resolve_task_dataset(dataset_key, baseline_dataset, snapshot_dataset, layer3_snapshot_dataset)
            working = dataset.merge(split_manifest[["cycle_uid", "split"]], on="cycle_uid", how="left")
            working = working.loc[working["split"].isin(["train", "valid", "test"])].copy()
            working[target_column] = _prepare_target(working[target_column], task_type)
            working = _apply_task_row_filter(working, task_cfg)

            if working.empty:
                task_results.append(
                    {
                        "task_name": task_name,
                        "status": "skipped",
                        "reason": "task row filter removed all rows",
                        "best_model": None,
                        "best_metric": None,
                        "task_report_path": None,
                    }
                )
                continue

            if task_cfg.get("skip_if_insufficient_labels") and working[target_column].dropna().nunique() < 2:
                task_results.append(
                    {
                        "task_name": task_name,
                        "status": "skipped",
                        "reason": "insufficient labels for supervised training",
                        "best_model": None,
                        "best_metric": None,
                        "task_report_path": None,
                    }
                )
                continue

            feature_columns = [
                column
                for column in feature_manifest[f"{feature_set}_feature_columns"]
                if column not in excluded and column in working.columns
            ]

            train_frame = working.loc[working["split"] == "train"].reset_index(drop=True)
            valid_frame = working.loc[working["split"] == "valid"].reset_index(drop=True)
            test_frame = working.loc[working["split"] == "test"].reset_index(drop=True)

            task_dir = run_dir / "tasks" / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                task_dir / "task_spec.json",
                {
                    "task_name": task_name,
                    "dataset_key": dataset_key,
                    "target": target_column,
                    "task_type": task_type,
                    "feature_set": feature_set,
                    "candidate_models": candidate_models,
                    "n_rows": int(len(working)),
                    "n_train": int(len(train_frame)),
                    "n_valid": int(len(valid_frame)),
                    "n_test": int(len(test_frame)),
                    "row_filter": task_cfg.get("row_filter"),
                    "conditional_on_task": task_cfg.get("conditional_on_task"),
                    "prediction_semantics": task_cfg.get("prediction_semantics"),
                },
            )

            model_rows: list[dict[str, Any]] = []
            best_model = None
            best_metric = None
            primary_metric = task_cfg["primary_metric"]

            with tqdm(
                total=len(candidate_models),
                desc=f"{task_name} models",
                unit="model",
                leave=False,
                dynamic_ncols=True,
            ) as model_bar:
                for model_name in candidate_models:
                    result = _fit_and_eval_model(
                        task_name=task_name,
                        task_cfg=task_cfg,
                        task_type=task_type,
                        target_column=target_column,
                        feature_set=feature_set,
                        model_name=model_name,
                        seed=int(split_cfg["seed"]),
                        train_frame=train_frame,
                        valid_frame=valid_frame,
                        test_frame=test_frame,
                        feature_columns=feature_columns,
                        output_dir=task_dir / "models",
                        tb_writer=tb_writer,
                    )
                    model_bar.update(1)
                    model_rows.append(
                        {
                            "task_name": task_name,
                            "model_name": model_name,
                            "status": result["status"],
                            "reason": result["reason"],
                            "primary_valid_metric": result.get("metrics", {}).get("primary_valid_metric")
                            if result.get("metrics")
                            else None,
                            "primary_test_metric": result.get("metrics", {}).get("primary_test_metric")
                            if result.get("metrics")
                            else None,
                        }
                    )
                    if result["status"] == "trained":
                        candidate_value = result["metrics"].get("primary_valid_metric")
                        if candidate_value is not None and not pd.isna(candidate_value):
                            if best_metric is None:
                                best_metric = float(candidate_value)
                                best_model = model_name
                            else:
                                higher_is_better = _metric_higher_is_better(task_type, str(primary_metric))
                                better = candidate_value > best_metric if higher_is_better else candidate_value < best_metric
                                if (
                                    not better
                                    and task_type == "regression"
                                    and model_name == "lightgbm"
                                    and best_model == "xgboost"
                                ):
                                    tolerance = max(0.05, abs(float(best_metric)) * 0.02)
                                    if abs(float(candidate_value) - float(best_metric)) <= tolerance:
                                        better = True
                                if better:
                                    best_metric = float(candidate_value)
                                    best_model = model_name
            comparison_frame = pd.DataFrame(model_rows)
            save_dataframe(task_dir / "model_comparison.csv", comparison_frame)
            write_markdown(
                task_dir / "task_report.md",
                [
                    f"# {task_name}",
                    "",
                    f"- task type: {task_type}",
                    f"- target: {target_column}",
                    f"- best model: {best_model or 'none'}",
                    f"- best metric: {best_metric if best_metric is not None else 'n/a'}",
                    f"- rows: {len(working)}",
                    f"- train/valid/test: {len(train_frame)}/{len(valid_frame)}/{len(test_frame)}",
                ],
            )
            task_results.append(
                {
                    "task_name": task_name,
                    "status": "trained" if best_model else "skipped",
                    "reason": None if best_model else "no model completed",
                    "best_model": best_model,
                    "best_metric": best_metric,
                    "task_report_path": str(task_dir / "task_report.md"),
                }
            )
            layer_bar.update(1)
            layer_bar.update(1)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "split_manifest_path": str(split_manifest_path),
        "split_summary": split_summary,
        "tasks": task_results,
        "tensorboard_dir": str(tensorboard_dir),
    }
    write_json(run_dir / "run_summary.json", summary)
    write_markdown(
        run_dir / "run_summary.md",
        [
            "# Phase 8 experiment summary",
            "",
            f"- run id: {run_id}",
            f"- split manifest: {split_manifest_path}",
            f"- tensorboard: {tensorboard_dir}",
            f"- tasks: {len(task_results)}",
        ],
    )
    return summary
