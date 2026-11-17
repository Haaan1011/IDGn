from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR
from xgboost import XGBClassifier, XGBRegressor

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
    CatBoostRegressor = None


def build_estimator(model_name: str, task_type: str, seed: int) -> Any:
    model_name = model_name.lower()
    task_type = task_type.lower()

    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task type: {task_type}")

    if model_name == "baseline":
        return DummyClassifier(strategy="most_frequent") if task_type == "classification" else DummyRegressor(strategy="mean")

    if model_name == "xgboost":
        return (
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                tree_method="hist",
                eval_metric="logloss",
            )
            if task_type == "classification"
            else XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                tree_method="hist",
            )
        )

    if model_name == "lightgbm":
        return (
            LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
            )
            if task_type == "classification"
            else LGBMRegressor(
                n_estimators=250,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
            )
        )

    if model_name == "catboost":
        if CatBoostClassifier is None or CatBoostRegressor is None:
            raise ImportError("catboost is not installed in this environment.")
        return (
            CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=False, random_seed=seed)
            if task_type == "classification"
            else CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, verbose=False, random_seed=seed)
        )

    if model_name == "random_forest":
        return (
            RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)
            if task_type == "classification"
            else RandomForestRegressor(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)
        )

    if model_name == "svm":
        return (
            LinearSVC(C=1.0, random_state=seed, max_iter=5000)
            if task_type == "classification"
            else LinearSVR(C=1.0, random_state=seed, max_iter=5000)
        )

    if model_name == "mlp":
        return (
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", early_stopping=True, max_iter=300, random_state=seed)
            if task_type == "classification"
            else MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", early_stopping=True, max_iter=300, random_state=seed)
        )

    if model_name == "elastic_net":
        return (
            LogisticRegression(max_iter=500, n_jobs=-1)
            if task_type == "classification"
            else ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed)
        )

    raise ValueError(f"Unsupported model name: {model_name}")
