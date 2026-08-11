"""Finite-grid and streaming trapezoidal integration in float64."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from decaf.core.decomposition import (
    COMPONENT_NAMES,
    audit_conservation,
    float64_array,
    normalize_axis,
)

INTEGRATED_ATOL = 2.0e-12
INTEGRATED_RTOL = 2.0e-10
GRID_ATOL = 1.0e-12


def validate_grid(
    grid: Any,
    *,
    expected_size: int | None = None,
    require_unit_interval: bool = True,
) -> np.ndarray:
    """Validate a one-dimensional, strictly increasing quadrature grid."""

    values = float64_array(grid, name="grid")
    if values.ndim != 1:
        raise ValueError("grid must be one-dimensional")
    if values.size < 2:
        raise ValueError("grid must contain at least two stages")
    if expected_size is not None:
        if isinstance(expected_size, (bool, np.bool_)) or not isinstance(
            expected_size, (int, np.integer)
        ):
            raise TypeError("expected_size must be an integer")
        if values.size != int(expected_size):
            raise ValueError("grid length does not match the trajectory axis")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("grid must be strictly increasing")
    if require_unit_interval and (
        not np.isclose(values[0], 0.0, atol=GRID_ATOL, rtol=0.0)
        or not np.isclose(values[-1], 1.0, atol=GRID_ATOL, rtol=0.0)
    ):
        raise ValueError("grid must span [0, 1]")
    return values


def trapezoid(
    values: Any,
    grid: Any,
    *,
    axis: int = -1,
    require_unit_interval: bool = True,
) -> np.ndarray:
    """Integrate finite values on their observed grid without interpolation."""

    array = float64_array(values, name="values")
    if array.ndim == 0:
        raise ValueError("values must include a trajectory axis")
    normalized_axis = normalize_axis(axis, array.ndim)
    stages = validate_grid(
        grid,
        expected_size=array.shape[normalized_axis],
        require_unit_interval=require_unit_interval,
    )
    return np.asarray(
        np.trapezoid(array, x=stages, axis=normalized_axis),
        dtype=np.float64,
    )


def integrate_components(
    grid: Any,
    components: Mapping[str, Any],
    *,
    axis: int = -1,
    require_unit_interval: bool = True,
) -> dict[str, Any]:
    """Trapezoid-integrate pointwise DECAF components and audit conservation."""

    arrays: dict[str, np.ndarray] = {}
    for name in COMPONENT_NAMES:
        if name not in components:
            raise ValueError(f"components is missing {name}")
        arrays[name] = float64_array(components[name], name=name)
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("E, C, F, and Abs must have identical shapes")

    integrated = {
        name: trapezoid(
            array,
            grid,
            axis=axis,
            require_unit_interval=require_unit_interval,
        )
        for name, array in arrays.items()
    }
    if "Net" in components:
        net = float64_array(components["Net"], name="Net")
        if net.shape != arrays["Abs"].shape:
            raise ValueError("Net must have the same shape as E, C, F, and Abs")
        integrated["Net"] = trapezoid(
            net,
            grid,
            axis=axis,
            require_unit_interval=require_unit_interval,
        )
    else:
        integrated["Net"] = integrated["E"] - integrated["C"]

    result: dict[str, Any] = {
        **integrated,
        **{f"{name}_auc": value for name, value in integrated.items()},
    }
    result["identity_audit"] = audit_conservation(
        integrated,
        atol=INTEGRATED_ATOL,
        rtol=INTEGRATED_RTOL,
        raise_on_error=True,
    )
    return result


compute_sample_aucs = integrate_components


class StreamingTrapezoid:
    """Incrementally integrate named stage values with constant memory."""

    def __init__(
        self,
        component_names: Sequence[str] = COMPONENT_NAMES,
        *,
        require_unit_interval: bool = True,
    ) -> None:
        names = tuple(component_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("component_names must contain non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("component_names must be unique")
        self._names = names
        self._require_unit_interval = bool(require_unit_interval)
        self._previous_position: float | None = None
        self._previous: dict[str, np.ndarray] | None = None
        self._total: dict[str, np.ndarray] | None = None
        self._shape: tuple[int, ...] | None = None
        self._stage_count = 0
        self._finalized = False

    @property
    def stage_count(self) -> int:
        return self._stage_count

    def update(self, position: float, values: Mapping[str, Any]) -> None:
        """Consume one strictly later grid position."""

        if self._finalized:
            raise RuntimeError("cannot update a finalized accumulator")
        if isinstance(position, (bool, np.bool_)):
            raise TypeError("position must be a real number")
        try:
            current_position = float(position)
        except (TypeError, ValueError) as error:
            raise TypeError("position must be a real number") from error
        if not np.isfinite(current_position):
            raise ValueError("position must be finite")

        current: dict[str, np.ndarray] = {}
        for name in self._names:
            if name not in values:
                raise ValueError(f"stage values are missing {name}")
            current[name] = float64_array(values[name], name=name)
        shapes = {array.shape for array in current.values()}
        if len(shapes) != 1:
            raise ValueError("all stage values must have one identical shape")
        current_shape = next(iter(shapes))
        if self._shape is not None and current_shape != self._shape:
            raise ValueError("stage value shape changed during streaming integration")

        if self._previous_position is None:
            if self._require_unit_interval and not np.isclose(
                current_position,
                0.0,
                atol=GRID_ATOL,
                rtol=0.0,
            ):
                raise ValueError("the first streaming position must be 0")
            self._shape = current_shape
            self._previous_position = current_position
            self._previous = {name: value.copy() for name, value in current.items()}
            self._total = {name: np.zeros_like(value) for name, value in current.items()}
            self._stage_count = 1
            return

        if current_position <= self._previous_position:
            raise ValueError("streaming positions must be strictly increasing")
        if self._require_unit_interval and current_position > 1.0 + GRID_ATOL:
            raise ValueError("streaming positions cannot exceed 1")
        assert self._previous is not None
        assert self._total is not None
        width = current_position - self._previous_position
        for name in self._names:
            self._total[name] += 0.5 * width * (self._previous[name] + current[name])
        self._previous_position = current_position
        self._previous = {name: value.copy() for name, value in current.items()}
        self._stage_count += 1

    def finalize(self) -> dict[str, Any]:
        """Return integrated values after validating the terminal stage."""

        if self._stage_count < 2 or self._total is None or self._previous_position is None:
            raise ValueError("streaming integration requires at least two stages")
        if self._require_unit_interval and not np.isclose(
            self._previous_position,
            1.0,
            atol=GRID_ATOL,
            rtol=0.0,
        ):
            raise ValueError("the final streaming position must be 1")
        self._finalized = True
        result: dict[str, Any] = {
            name: np.asarray(value, dtype=np.float64).copy() for name, value in self._total.items()
        }
        if set(COMPONENT_NAMES).issubset(result):
            result.setdefault("Net", result["E"] - result["C"])
            result["identity_audit"] = audit_conservation(
                result,
                atol=INTEGRATED_ATOL,
                rtol=INTEGRATED_RTOL,
                raise_on_error=True,
            )
        return result
