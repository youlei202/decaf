"""Deterministic nonparametric bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np

from decaf.core.decomposition import float64_array, normalize_axis


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Percentile interval and bootstrap standard error."""

    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    standard_error: np.ndarray
    confidence_level: float
    n_resamples: int
    n_observations: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        def serializable(value: np.ndarray) -> float | list[Any]:
            if value.ndim == 0:
                return float(value)
            return value.tolist()

        return {
            "estimate": serializable(self.estimate),
            "lower": serializable(self.lower),
            "upper": serializable(self.upper),
            "standard_error": serializable(self.standard_error),
            "confidence_level": self.confidence_level,
            "n_resamples": self.n_resamples,
            "n_observations": self.n_observations,
            "seed": self.seed,
        }


def _positive_integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def bootstrap_mean(
    values: Any,
    *,
    axis: int = 0,
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    batch_size: int = 256,
) -> BootstrapResult:
    """Bootstrap a mean along one observation axis using bounded batches."""

    array = float64_array(values, name="values")
    if array.ndim == 0:
        raise ValueError("values must include an observation axis")
    normalized_axis = normalize_axis(axis, array.ndim)
    moved = np.moveaxis(array, normalized_axis, 0)
    observation_count = moved.shape[0]
    if observation_count < 2:
        raise ValueError("bootstrap requires at least two observations")
    resample_count = _positive_integer(n_resamples, name="n_resamples", minimum=2)
    chunk_size = _positive_integer(batch_size, name="batch_size", minimum=1)
    if isinstance(confidence_level, (bool, np.bool_)):
        raise TypeError("confidence_level must be a real number")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    random_seed = _positive_integer(seed, name="seed", minimum=0)

    generator = np.random.default_rng(random_seed)
    replicates = np.empty((resample_count, *moved.shape[1:]), dtype=np.float64)
    for start in range(0, resample_count, chunk_size):
        stop = min(start + chunk_size, resample_count)
        indices = generator.integers(
            0,
            observation_count,
            size=(stop - start, observation_count),
        )
        replicates[start:stop] = np.mean(moved[indices], axis=1, dtype=np.float64)

    tail = (1.0 - confidence) / 2.0
    return BootstrapResult(
        estimate=np.asarray(np.mean(moved, axis=0, dtype=np.float64), dtype=np.float64),
        lower=np.asarray(np.quantile(replicates, tail, axis=0), dtype=np.float64),
        upper=np.asarray(np.quantile(replicates, 1.0 - tail, axis=0), dtype=np.float64),
        standard_error=np.asarray(np.std(replicates, axis=0, ddof=1), dtype=np.float64),
        confidence_level=confidence,
        n_resamples=resample_count,
        n_observations=observation_count,
        seed=random_seed,
    )


def paired_bootstrap_mean_difference(
    left: Any,
    right: Any,
    **kwargs: Any,
) -> BootstrapResult:
    """Bootstrap the paired mean difference left minus right."""

    x = float64_array(left, name="left")
    y = float64_array(right, name="right")
    if x.shape != y.shape:
        raise ValueError("left and right must have identical shapes")
    return bootstrap_mean(x - y, **kwargs)
