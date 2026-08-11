"""Portable model-agnostic baselines for the Covertype smoke and paper runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from decaf.experiments.covertype.models import predict_positive


def classification_accuracy(probability: np.ndarray, y: np.ndarray) -> float:
    """Return binary accuracy at the registered probability threshold."""

    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    if score.shape != target.shape:
        raise ValueError("probabilities and targets must align")
    return float(np.mean((score >= 0.5) == target))


def paired_trajectory_baselines(
    endpoint: np.ndarray,
    alternate: np.ndarray,
    decaf_scores: dict[str, Any],
) -> dict[str, float]:
    """Summarize endpoint magnitude, sign flips, and opposed trajectory mass."""

    anchor = np.asarray(endpoint, dtype=np.float64).reshape(-1)
    other = np.asarray(alternate, dtype=np.float64).reshape(-1)
    if anchor.shape != other.shape:
        raise ValueError("endpoint and alternate responses must align")
    active = np.asarray(decaf_scores["endpoint_active"], dtype=bool).reshape(-1)
    flip = active & (np.sign(anchor) * np.sign(other) < 0.0)
    return {
        "baseline_endpoint_M": float(np.mean(np.abs(anchor))),
        "baseline_Abs": float(np.mean(np.asarray(decaf_scores["Abs"]))),
        "baseline_Net": float(np.mean(np.asarray(decaf_scores["Net"]))),
        "baseline_SignFlip": float(np.mean(flip)),
        "baseline_OppMass": float(np.mean(np.asarray(decaf_scores["C"]))),
        "baseline_endpoint_active_rate": float(np.mean(active)),
        "baseline_endpoint_null_rate": float(np.mean(~active)),
    }


def permutation_factor_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Measure held-out accuracy loss after permuting only the synthetic factor."""

    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] < 2:
        raise ValueError("augmented features must be a two-dimensional matrix")
    factual = predict_positive(model, features)
    permuted = np.array(features, copy=True)
    order = np.random.default_rng(seed).permutation(len(permuted))
    permuted[:, -1] = permuted[order, -1]
    shuffled = predict_positive(model, permuted)
    factual_accuracy = classification_accuracy(factual, y)
    shuffled_accuracy = classification_accuracy(shuffled, y)
    return {
        "baseline_permutation_factor_importance": factual_accuracy - shuffled_accuracy,
        "baseline_factual_accuracy": factual_accuracy,
        "baseline_permuted_accuracy": shuffled_accuracy,
        "baseline_prediction_rows": int(2 * len(features)),
    }


__all__ = [
    "classification_accuracy",
    "paired_trajectory_baselines",
    "permutation_factor_importance",
]
