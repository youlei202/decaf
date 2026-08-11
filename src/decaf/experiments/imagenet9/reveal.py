"""Blend and patch reveal paths plus the shared DECAF score adapter."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from decaf.core.trajectories import trajectory_scores


def _images(original: Any, counterfactual: Any) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(original, dtype=np.float64)
    right = np.asarray(counterfactual, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired images must have identical shapes")
    if left.ndim < 2:
        raise ValueError("paired images must include spatial dimensions")
    return left, right


def blend_path(
    original: Any,
    counterfactual: Any,
    alpha: Iterable[float],
) -> np.ndarray:
    """Construct a linear pixel-space reveal trajectory."""

    left, right = _images(original, counterfactual)
    grid = np.asarray(tuple(alpha), dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("alpha must be a strictly increasing one-dimensional grid")
    if grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("alpha must span the closed unit interval")
    reshape = (len(grid),) + (1,) * left.ndim
    weights = grid.reshape(reshape)
    return (1.0 - weights) * left + weights * right


def patch_path(
    original: Any,
    counterfactual: Any,
    order: Any,
    alpha: Iterable[float],
) -> np.ndarray:
    """Reveal spatial sites in a fixed rank order, preserving legal pixels."""

    left, right = _images(original, counterfactual)
    ranks = np.asarray(order)
    if ranks.shape != left.shape[:2]:
        raise ValueError("patch order must match the first two image dimensions")
    flat = ranks.reshape(-1)
    if len(set(map(int, flat))) != len(flat):
        raise ValueError("patch order ranks must be unique")
    grid = np.asarray(tuple(alpha), dtype=np.float64)
    if grid[0] != 0.0 or grid[-1] != 1.0 or np.any(np.diff(grid) <= 0):
        raise ValueError("alpha must strictly increase from zero to one")
    ordinal = np.argsort(np.argsort(flat, kind="stable"), kind="stable").reshape(ranks.shape)
    stages: list[np.ndarray] = []
    count = flat.size
    for value in grid:
        revealed = int(round(float(value) * count))
        spatial_mask = ordinal < revealed
        mask = spatial_mask
        while mask.ndim < left.ndim:
            mask = mask[..., None]
        stages.append(np.where(mask, right, left))
    return np.asarray(stages, dtype=np.float64)


def decompose_score_path(
    alpha: Iterable[float],
    response: Any,
    *,
    endpoint: Any | None = None,
    epsilon: float = 0.02,
) -> dict[str, Any]:
    """Route score responses through the sole model-agnostic DECAF core."""

    return trajectory_scores(
        tuple(alpha),
        response,
        endpoint=endpoint,
        epsilon=epsilon,
        axis=-1,
    )


__all__ = ["blend_path", "decompose_score_path", "patch_path"]
