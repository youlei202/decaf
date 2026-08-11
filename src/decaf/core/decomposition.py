"""Authoritative NumPy implementation of the DECAF pointwise decomposition.

Model code may produce scores in any real numeric precision. This module
promotes the small score-response arrays to float64 before endpoint
classification, orientation, routing, or identity checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any

import numpy as np

COMPONENT_NAMES = ("E", "C", "F", "Abs")
PRIMARY_EPSILON = 0.02
POINTWISE_ATOL = 1.0e-12
POINTWISE_RTOL = 1.0e-10


def float64_array(value: Any, *, name: str) -> np.ndarray:
    """Return a non-empty, finite, real numeric float64 array."""

    source = np.asarray(value)
    if source.dtype.kind == "b" or not np.issubdtype(source.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    if np.iscomplexobj(source):
        raise TypeError(f"{name} cannot be complex-valued")
    result = np.asarray(source, dtype=np.float64)
    if result.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return result


def validate_epsilon(epsilon: float) -> float:
    """Return a finite, strictly positive endpoint threshold."""

    if isinstance(epsilon, (bool, np.bool_)):
        raise TypeError("epsilon must be a real number")
    try:
        result = float(epsilon)
    except (TypeError, ValueError) as error:
        raise TypeError("epsilon must be a real number") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    return result


def normalize_axis(axis: int, ndim: int) -> int:
    """Normalize an axis without relying on NumPy private helpers."""

    if isinstance(axis, (bool, np.bool_)):
        raise TypeError("axis must be an integer")
    try:
        normalized = index(axis)
    except TypeError as error:
        raise TypeError("axis must be an integer") from error
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array with {ndim} dimensions")
    return normalized


def endpoint_effect(positive_scores: Any, negative_scores: Any) -> np.ndarray:
    """Return the signed positive-minus-negative score response."""

    positive = float64_array(positive_scores, name="positive_scores")
    negative = float64_array(negative_scores, name="negative_scores")
    if positive.shape != negative.shape:
        raise ValueError("positive_scores and negative_scores must have identical shapes")
    return np.asarray(positive - negative, dtype=np.float64)


stage_effects = endpoint_effect


def endpoint_orientation(
    endpoint: Any,
    epsilon: float = PRIMARY_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint activity and its gated orientation.

    Equality is active: an endpoint is active exactly when
    abs(endpoint) >= epsilon.
    """

    values = float64_array(endpoint, name="endpoint")
    threshold = validate_epsilon(epsilon)
    active = np.abs(values) >= threshold
    orientation = np.where(active, np.sign(values), 0.0).astype(np.float64, copy=False)
    return active, orientation


def _route_arrays(
    response: np.ndarray,
    endpoint: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    active, orientation = endpoint_orientation(endpoint, epsilon)
    signed = orientation * response
    absolute = np.abs(response)
    evidence = np.where(active, np.maximum(signed, 0.0), 0.0)
    contradiction = np.where(active, np.maximum(-signed, 0.0), 0.0)
    fragility = np.where(active, 0.0, absolute)
    result: dict[str, Any] = {
        "E": np.asarray(evidence, dtype=np.float64),
        "C": np.asarray(contradiction, dtype=np.float64),
        "F": np.asarray(fragility, dtype=np.float64),
        "Abs": np.asarray(absolute, dtype=np.float64),
        "Net": np.asarray(evidence - contradiction, dtype=np.float64),
        "endpoint_active": np.asarray(active, dtype=bool),
        "endpoint_sign": np.asarray(orientation, dtype=np.float64),
        "epsilon": float(epsilon),
    }
    result["identity_audit"] = audit_conservation(result, raise_on_error=True)
    return result


def route_response(
    response: Any,
    endpoint: Any,
    epsilon: float = PRIMARY_EPSILON,
) -> dict[str, Any]:
    """Route a pointwise response against a broadcastable endpoint."""

    values = float64_array(response, name="response")
    anchor = float64_array(endpoint, name="endpoint")
    try:
        broadcast_anchor = np.broadcast_to(anchor, values.shape)
    except ValueError as error:
        raise ValueError("endpoint cannot be broadcast to the response shape") from error
    return _route_arrays(values, broadcast_anchor, validate_epsilon(epsilon))


def _endpoint_for_trajectory(
    response: np.ndarray,
    endpoint: Any | None,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_shape = response.shape[:axis] + response.shape[axis + 1 :]
    if endpoint is None:
        sample_endpoint = np.take(response, -1, axis=axis)
    else:
        candidate = float64_array(endpoint, name="endpoint")
        if candidate.shape == response.shape:
            moved = np.moveaxis(candidate, axis, -1)
            reference = moved[..., :1]
            if not np.array_equal(moved, np.broadcast_to(reference, moved.shape)):
                raise ValueError("a full-shape endpoint must be constant across trajectory stages")
            candidate = np.take(candidate, 0, axis=axis)
        elif candidate.ndim == response.ndim and candidate.shape[axis] == 1:
            candidate = np.squeeze(candidate, axis=axis)
        try:
            sample_endpoint = np.broadcast_to(candidate, sample_shape)
        except ValueError as error:
            raise ValueError(
                "endpoint cannot be broadcast to the trajectory sample shape"
            ) from error
    expanded = np.expand_dims(sample_endpoint, axis=axis)
    return (
        np.asarray(sample_endpoint, dtype=np.float64),
        np.broadcast_to(expanded, response.shape),
    )


def decompose(
    response: Any,
    endpoint: Any | None = None,
    epsilon: float = PRIMARY_EPSILON,
    *,
    axis: int = -1,
) -> dict[str, Any]:
    """Decompose a response trajectory into pointwise E/C/F/Abs.

    If endpoint is omitted, the final value along axis is the endpoint.
    A supplied endpoint may be scalar, have the trajectory's sample shape, or
    have the full response shape when it is constant across stages.
    """

    values = float64_array(response, name="response")
    threshold = validate_epsilon(epsilon)
    if values.ndim == 0:
        anchor = values if endpoint is None else float64_array(endpoint, name="endpoint")
        try:
            broadcast_anchor = np.broadcast_to(anchor, values.shape)
        except ValueError as error:
            raise ValueError("endpoint cannot be broadcast to a scalar response") from error
        result = _route_arrays(values, broadcast_anchor, threshold)
        result["endpoint_value"] = np.asarray(broadcast_anchor, dtype=np.float64)
        result["endpoint_active_sample"] = np.asarray(result["endpoint_active"], dtype=bool)
        return result

    normalized_axis = normalize_axis(axis, values.ndim)
    sample_endpoint, broadcast_endpoint = _endpoint_for_trajectory(
        values,
        endpoint,
        normalized_axis,
    )
    result = _route_arrays(values, broadcast_endpoint, threshold)
    sample_active, _ = endpoint_orientation(sample_endpoint, threshold)
    result["endpoint_value"] = sample_endpoint
    result["endpoint_active_sample"] = sample_active
    result["axis"] = normalized_axis
    return result


compute_components = decompose
decompose_response = decompose


def audit_conservation(
    components: Mapping[str, Any],
    *,
    atol: float = POINTWISE_ATOL,
    rtol: float = POINTWISE_RTOL,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Audit non-negativity and the DECAF partition identities."""

    for name in COMPONENT_NAMES:
        if name not in components:
            raise ValueError(f"components is missing {name}")
    try:
        absolute_tolerance = float(atol)
        relative_tolerance = float(rtol)
    except (TypeError, ValueError) as error:
        raise TypeError("audit tolerances must be real numbers") from error
    if (
        not np.isfinite(absolute_tolerance)
        or absolute_tolerance < 0.0
        or not np.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
    ):
        raise ValueError("audit tolerances must be finite and non-negative")

    arrays: dict[str, np.ndarray] = {}
    for name in COMPONENT_NAMES:
        value = np.asarray(components[name])
        if value.dtype.kind == "b" or not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"component {name} must contain real numeric values")
        if np.iscomplexobj(value):
            raise TypeError(f"component {name} cannot be complex-valued")
        arrays[name] = np.asarray(value, dtype=np.float64)
    if len({array.shape for array in arrays.values()}) != 1 or arrays["Abs"].size == 0:
        raise ValueError("E, C, F, and Abs must have one identical non-empty shape")

    finite = bool(all(np.isfinite(array).all() for array in arrays.values()))
    nonnegative = bool(all(np.all(array >= -absolute_tolerance) for array in arrays.values()))
    residual = arrays["Abs"] - arrays["E"] - arrays["C"] - arrays["F"]
    absolute_residual = np.abs(residual)
    tolerance = absolute_tolerance + relative_tolerance * np.abs(arrays["Abs"])
    partition_violations = (~np.isfinite(residual)) | (absolute_residual > tolerance)
    scale = np.maximum(np.abs(arrays["Abs"]), np.finfo(np.float64).tiny)
    relative_residual = absolute_residual / scale

    net_identity = True
    net_error = 0.0
    if "Net" in components:
        net = np.asarray(components["Net"])
        if net.dtype.kind == "b" or not np.issubdtype(net.dtype, np.number):
            raise TypeError("component Net must contain real numeric values")
        if np.iscomplexobj(net):
            raise TypeError("component Net cannot be complex-valued")
        net = np.asarray(net, dtype=np.float64)
        if net.shape != arrays["Abs"].shape:
            raise ValueError("Net must have the same shape as E, C, F, and Abs")
        finite = bool(finite and np.isfinite(net).all())
        net_residual = net - arrays["E"] + arrays["C"]
        net_error = float(np.max(np.abs(net_residual), initial=0.0))
        net_tolerance = absolute_tolerance + relative_tolerance * np.abs(net)
        net_identity = bool(
            np.isfinite(net_residual).all() and np.all(np.abs(net_residual) <= net_tolerance)
        )

    identity = bool(not np.any(partition_violations))
    payload = {
        "passed": bool(finite and nonnegative and identity and net_identity),
        "finite": finite,
        "nonnegative": nonnegative,
        "identity": identity,
        "partition_identity": identity,
        "net_identity": net_identity,
        "n_values": int(arrays["Abs"].size),
        "n_identity_violations": int(np.count_nonzero(partition_violations)),
        "absolute_residual": float(np.max(absolute_residual, initial=0.0)),
        "relative_residual": float(np.max(relative_residual, initial=0.0)),
        "max_abs_error": float(np.max(absolute_residual, initial=0.0)),
        "max_net_error": net_error,
        "atol": absolute_tolerance,
        "rtol": relative_tolerance,
    }
    if raise_on_error and not payload["passed"]:
        raise AssertionError(f"DECAF conservation audit failed: {payload}")
    return payload


audit_identity = audit_conservation
