"""Registered controlled perturbation grids and score-only protocol oracles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from decaf.core.decomposition import PRIMARY_EPSILON, route_response
from decaf.core.trajectories import trajectory_scores

PRIMARY_PROTOCOL = "cmmr"
REGISTERED_GEOMETRIES = (
    "cmmr",
    "pixel_trace_matched",
    "diagonal",
    "power_beta_0.25",
    "power_beta_0.50",
    "power_beta_0.75",
)


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """One deterministic trajectory geometry."""

    name: str
    alpha: tuple[float, ...]
    primary: bool = False
    beta: float | None = None


def validate_alpha_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = np.asarray(tuple(values), dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all():
        raise ValueError("alpha grid must contain at least two finite values")
    if not np.all(np.diff(grid) > 0.0) or grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("alpha grid must be strictly increasing and span [0, 1]")
    return tuple(float(value) for value in grid)


def geometry_specs(section: Mapping[str, Any]) -> tuple[ProtocolSpec, ...]:
    """Expand CMMR, pixel, diagonal, and three power geometries."""

    alpha = validate_alpha_grid(section["alpha_grid"])
    requested = tuple(map(str, section.get("geometries", ("cmmr",))))
    specs: list[ProtocolSpec] = []
    for name in requested:
        if name == "power":
            for beta in section.get("power_betas", (0.25, 0.50, 0.75)):
                value = float(beta)
                if not 0.0 < value < 1.0:
                    raise ValueError("power beta must lie strictly between zero and one")
                specs.append(ProtocolSpec(f"power_beta_{value:.2f}", alpha, beta=value))
        elif name == "pixel":
            specs.append(ProtocolSpec("pixel_trace_matched", alpha))
        elif name in {"cmmr", "diagonal"}:
            specs.append(ProtocolSpec(name, alpha, primary=name == "cmmr"))
        else:
            raise ValueError(f"unknown controlled geometry: {name}")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("geometry expansion produced duplicate names")
    return tuple(specs)


def shared_gaussian_increments(
    shape: Sequence[int],
    *,
    seed: int,
    covariance: Any | None = None,
) -> np.ndarray:
    """Generate one reusable Gaussian draw for factual/counterfactual branches."""

    dimensions = tuple(int(value) for value in shape)
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("noise shape must contain positive dimensions")
    generator = np.random.default_rng(int(seed))
    draw = generator.standard_normal(dimensions, dtype=np.float64)
    if covariance is None:
        return draw
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != dimensions[-1]:
        raise ValueError("covariance must be square and match the final noise dimension")
    if not np.allclose(matrix, matrix.T, atol=1.0e-12, rtol=0.0):
        raise ValueError("covariance must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.any(eigenvalues < -1.0e-10):
        raise ValueError("covariance must be positive semidefinite")
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    return np.asarray(draw @ root.T, dtype=np.float64)


def transform_increments(
    increments: Any, geometry: str, *, beta: float | None = None
) -> np.ndarray:
    """Apply a deterministic trace-normalized geometry transform."""

    values = np.asarray(increments, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 1 or not np.isfinite(values).all():
        raise ValueError("increments must be a finite array with a feature dimension")
    if geometry in {"cmmr", "pixel_trace_matched"}:
        transformed = values.copy()
    elif geometry == "diagonal":
        scale = np.sqrt(np.mean(values * values, axis=-1, keepdims=True))
        transformed = np.sign(values) * scale
    elif geometry.startswith("power_beta_"):
        selected = float(beta if beta is not None else geometry.rsplit("_", 1)[-1])
        transformed = np.sign(values) * np.abs(values) ** selected
    else:
        raise ValueError(f"unknown geometry: {geometry}")
    source_energy = np.sum(values * values, axis=-1, keepdims=True)
    target_energy = np.sum(transformed * transformed, axis=-1, keepdims=True)
    scale = np.sqrt(
        np.divide(
            source_energy, target_energy, out=np.ones_like(source_energy), where=target_energy > 0.0
        )
    )
    return np.asarray(transformed * scale, dtype=np.float64)


def decompose_score_trajectory(
    alpha: Sequence[float],
    response: Any,
    *,
    endpoint: Any | None = None,
    epsilon: float = PRIMARY_EPSILON,
    axis: int = -1,
) -> dict[str, Any]:
    """Authoritative controlled-family entry point into the core DECAF math."""

    return trajectory_scores(alpha, response, endpoint, epsilon, axis=axis)


def analytic_context_mixture(
    endpoint_delta: Any,
    swapped_delta: Any,
    mismatch_grid: Sequence[float],
    *,
    endpoint_epsilon: float = PRIMARY_EPSILON,
) -> dict[str, np.ndarray]:
    """Evaluate the exact C2 mixture using the shared core routing rules."""

    endpoint = np.asarray(endpoint_delta, dtype=np.float64)
    swapped = np.asarray(swapped_delta, dtype=np.float64)
    if endpoint.shape != swapped.shape or endpoint.size < 1:
        raise ValueError("endpoint and swapped responses must have one non-empty shape")
    epsilon = np.asarray(tuple(mismatch_grid), dtype=np.float64)
    if epsilon.ndim != 1 or epsilon.size < 1 or np.any((epsilon < 0.0) | (epsilon > 0.5)):
        raise ValueError("context mismatch grid must lie in [0, 0.5]")
    correct = route_response(endpoint, endpoint, endpoint_epsilon)
    changed = route_response(swapped, endpoint, endpoint_epsilon)
    output: dict[str, np.ndarray] = {}
    for name in ("E", "C", "F", "Abs", "Net"):
        left = np.mean(correct[name], dtype=np.float64)
        right = np.mean(changed[name], dtype=np.float64)
        output[name] = np.asarray((1.0 - epsilon) * left + epsilon * right, dtype=np.float64)
    output["phi_C"] = np.divide(
        output["C"],
        output["E"] + output["C"],
        out=np.zeros_like(output["C"]),
        where=(output["E"] + output["C"]) > 0.0,
    )
    if not np.allclose(
        output["Abs"], output["E"] + output["C"] + output["F"], atol=1.0e-12, rtol=0.0
    ):
        raise AssertionError("context mixture violates DECAF conservation")
    return output


__all__ = [
    "PRIMARY_PROTOCOL",
    "REGISTERED_GEOMETRIES",
    "ProtocolSpec",
    "analytic_context_mixture",
    "decompose_score_trajectory",
    "geometry_specs",
    "shared_gaussian_increments",
    "transform_increments",
    "validate_alpha_grid",
]
