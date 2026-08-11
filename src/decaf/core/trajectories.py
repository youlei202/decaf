"""Trajectory-level DECAF scores and streaming accumulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from decaf.core.decomposition import (
    COMPONENT_NAMES,
    PRIMARY_EPSILON,
    decompose,
    endpoint_orientation,
    float64_array,
    normalize_axis,
    route_response,
    validate_epsilon,
)
from decaf.core.quadrature import StreamingTrapezoid, integrate_components


def trajectory_scores(
    grid: Any,
    response: Any,
    endpoint: Any | None = None,
    epsilon: float = PRIMARY_EPSILON,
    *,
    axis: int = -1,
    require_unit_interval: bool = True,
) -> dict[str, Any]:
    """Return endpoint magnitude and integrated DECAF trajectory scores."""

    values = float64_array(response, name="response")
    if values.ndim == 0:
        raise ValueError("response must include a trajectory axis")
    normalized_axis = normalize_axis(axis, values.ndim)
    pointwise = decompose(
        values,
        endpoint,
        epsilon,
        axis=normalized_axis,
    )
    integrated = integrate_components(
        grid,
        pointwise,
        axis=normalized_axis,
        require_unit_interval=require_unit_interval,
    )
    endpoint_value = np.asarray(pointwise["endpoint_value"], dtype=np.float64)
    endpoint_active = np.asarray(pointwise["endpoint_active_sample"], dtype=bool)
    result: dict[str, Any] = {
        "M": np.abs(endpoint_value),
        "E": integrated["E"],
        "C": integrated["C"],
        "F": integrated["F"],
        "Abs": integrated["Abs"],
        "Net": integrated["Net"],
        "signed_E": np.sign(endpoint_value) * integrated["E"],
        "endpoint_delta": endpoint_value,
        "endpoint_active": endpoint_active,
        "pointwise_components": pointwise,
        "numeric_audit": {
            "passed": bool(
                pointwise["identity_audit"]["passed"] and integrated["identity_audit"]["passed"]
            ),
            "pointwise": pointwise["identity_audit"],
            "integrated": integrated["identity_audit"],
        },
    }
    return result


compute_sample_scores = trajectory_scores
decompose_trajectory = trajectory_scores


class StreamingDECAFAccumulator:
    """Accumulate DECAF AUCs stage by stage for a known endpoint."""

    def __init__(
        self,
        endpoint: Any,
        epsilon: float = PRIMARY_EPSILON,
        *,
        require_unit_interval: bool = True,
    ) -> None:
        self._endpoint = float64_array(endpoint, name="endpoint")
        self._epsilon = validate_epsilon(epsilon)
        self._integrator = StreamingTrapezoid(
            (*COMPONENT_NAMES, "Net"),
            require_unit_interval=require_unit_interval,
        )
        self._sample_shape: tuple[int, ...] | None = None
        self._finalized = False

    @property
    def stage_count(self) -> int:
        return self._integrator.stage_count

    def update(self, position: float, response: Any) -> None:
        """Consume one response stage."""

        if self._finalized:
            raise RuntimeError("cannot update a finalized accumulator")
        values = float64_array(response, name="response")
        if self._sample_shape is not None and values.shape != self._sample_shape:
            raise ValueError("response shape changed during streaming accumulation")
        components = route_response(values, self._endpoint, self._epsilon)
        stage_values = {name: components[name] for name in (*COMPONENT_NAMES, "Net")}
        self._integrator.update(position, stage_values)
        self._sample_shape = values.shape

    def finalize(self) -> dict[str, Any]:
        """Return endpoint and integrated component scores."""

        if self._finalized:
            raise RuntimeError("accumulator has already been finalized")
        integrated = self._integrator.finalize()
        assert self._sample_shape is not None
        endpoint = np.broadcast_to(self._endpoint, self._sample_shape)
        active, _ = endpoint_orientation(endpoint, self._epsilon)
        self._finalized = True
        return {
            "M": np.abs(endpoint),
            "E": integrated["E"],
            "C": integrated["C"],
            "F": integrated["F"],
            "Abs": integrated["Abs"],
            "Net": integrated["Net"],
            "signed_E": np.sign(endpoint) * integrated["E"],
            "endpoint_delta": np.asarray(endpoint, dtype=np.float64),
            "endpoint_active": active,
            "numeric_audit": {
                "passed": bool(integrated["identity_audit"]["passed"]),
                "integrated": integrated["identity_audit"],
            },
        }


TrajectoryAccumulator = StreamingDECAFAccumulator
