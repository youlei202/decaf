"""CPU classifier adapters for the five registered Covertype families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

MODEL_FAMILIES = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "mlp",
)


def build_model(family: str, seed: int, model_config: Mapping[str, Any]) -> Any:
    """Build one registered classifier with deterministic single-model settings."""

    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown Covertype model family: {family}")
    options = dict(model_config.get(family, {}))
    if family == "logistic_regression":
        if options.get("penalty") == "l2":
            options.pop("penalty")
        return LogisticRegression(random_state=seed, **options)
    if family == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **options)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **options)
    if family == "mlp":
        hidden = tuple(int(value) for value in options.pop("hidden_layer_sizes", (128, 64)))
        return MLPClassifier(hidden_layer_sizes=hidden, random_state=seed, **options)
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError(
            "paper-profile xgboost models require the optional xgboost dependency"
        ) from error
    return XGBClassifier(
        random_state=seed,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        **options,
    )


def fit_model(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Fit an sklearn-compatible binary probability model."""

    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.int8)
    if features.ndim != 2 or target.shape != (len(features),):
        raise ValueError("training features and targets must align")
    model.fit(features, target)
    return model


def predict_positive(model: Any, X: np.ndarray) -> np.ndarray:
    """Return validated positive-class probabilities."""

    probability = np.asarray(model.predict_proba(np.asarray(X, dtype=np.float64)))
    if probability.shape != (len(X), 2):
        raise ValueError(f"predict_proba returned unexpected shape {probability.shape}")
    result = np.asarray(probability[:, 1], dtype=np.float64)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("predict_proba returned invalid probabilities")
    return result


def implementation_name(model: Any) -> str:
    """Return a stable estimator implementation label."""

    return f"{type(model).__module__}.{type(model).__name__}"


__all__ = [
    "MODEL_FAMILIES",
    "build_model",
    "fit_model",
    "implementation_name",
    "predict_positive",
]
