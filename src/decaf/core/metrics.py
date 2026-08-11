"""Small model-agnostic metrics used by DECAF analyses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from decaf.core.decomposition import (
    COMPONENT_NAMES,
    PRIMARY_EPSILON,
    audit_conservation,
    endpoint_orientation,
    float64_array,
)


def endpoint_magnitude(endpoint: Any) -> np.ndarray:
    """Return Endpoint M in float64."""

    return np.abs(float64_array(endpoint, name="endpoint"))


def signed_evidence(
    evidence: Any,
    endpoint: Any,
    epsilon: float = PRIMARY_EPSILON,
) -> np.ndarray:
    """Orient non-negative evidence by the active endpoint direction."""

    values = float64_array(evidence, name="evidence")
    if np.any(values < 0.0):
        raise ValueError("evidence must be non-negative")
    _, orientation = endpoint_orientation(endpoint, epsilon)
    try:
        direction = np.broadcast_to(orientation, values.shape)
    except ValueError as error:
        raise ValueError("endpoint cannot be broadcast to the evidence shape") from error
    return np.asarray(direction * values, dtype=np.float64)


def safe_ratio(
    numerator: Any,
    denominator: Any,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Divide finite arrays and use a finite fill value at zero denominators."""

    left = float64_array(numerator, name="numerator")
    right = float64_array(denominator, name="denominator")
    try:
        left, right = np.broadcast_arrays(left, right)
    except ValueError as error:
        raise ValueError("numerator and denominator cannot be broadcast together") from error
    fill = float(fill_value)
    if not np.isfinite(fill):
        raise ValueError("fill_value must be finite")
    result = np.full(left.shape, fill, dtype=np.float64)
    return np.divide(left, right, out=result, where=right != 0.0)


def component_fractions(components: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return each routed component as a fraction of absolute response."""

    audit_conservation(components, raise_on_error=True)
    absolute = float64_array(components["Abs"], name="Abs")
    return {
        f"{name}_fraction": safe_ratio(components[name], absolute) for name in COMPONENT_NAMES[:3]
    }


def pearson_correlation(left: Any, right: Any) -> float:
    """Return the finite-sample Pearson correlation."""

    x = float64_array(left, name="left").reshape(-1)
    y = float64_array(right, name="right").reshape(-1)
    if x.shape != y.shape:
        raise ValueError("left and right must have identical shapes")
    if x.size < 2:
        raise ValueError("correlation requires at least two paired values")
    x_centered = x - np.mean(x, dtype=np.float64)
    y_centered = y - np.mean(y, dtype=np.float64)
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator == 0.0:
        raise ValueError("correlation is undefined for a constant input")
    return float(np.dot(x_centered, y_centered) / denominator)


def mean_absolute_error(left: Any, right: Any) -> float:
    """Return mean absolute paired error in float64."""

    x = float64_array(left, name="left")
    y = float64_array(right, name="right")
    if x.shape != y.shape:
        raise ValueError("left and right must have identical shapes")
    return float(np.mean(np.abs(x - y), dtype=np.float64))
