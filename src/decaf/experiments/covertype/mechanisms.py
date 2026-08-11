"""Deterministic contextual mechanisms and legal binary query construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

C_MECHANISMS = ("direct", "gate", "invert")
F_REGIMES = ("robust", "mild", "fragile")
F_ALTERNATE_ALIGNMENT = {"robust": 0.50, "mild": 0.70, "fragile": 0.95}


@dataclass(frozen=True)
class MechanismRealization:
    """Factual contexts/factors and shared endpoint/alternate realizations."""

    context: np.ndarray
    factor: np.ndarray
    endpoint_factor: np.ndarray
    alternate_factor: np.ndarray


def stream_seed(seed: int, split: str, stream: str, namespace: str) -> int:
    """Derive an independent deterministic random stream."""

    message = f"{namespace}|{int(seed)}|{split}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little")


def _uniform(size: int, seed: int, split: str, stream: str, namespace: str) -> np.ndarray:
    return np.random.default_rng(stream_seed(seed, split, stream, namespace)).random(size)


def _aligned(labels: np.ndarray, probability: float, uniform: np.ndarray) -> np.ndarray:
    signed = np.where(np.asarray(labels, dtype=np.int8) > 0, 1, -1)
    return np.asarray(np.where(uniform < float(probability), signed, -signed), dtype=np.int8)


def realize_module_c(
    y: np.ndarray,
    *,
    strength: float,
    mechanism: str,
    seed: int,
    split: str,
) -> MechanismRealization:
    """Generate Direct/Gate/Invert with an exactly shared positive endpoint."""

    if mechanism not in C_MECHANISMS:
        raise ValueError(f"unknown Module C mechanism: {mechanism}")
    if not 0.5 <= float(strength) <= 1.0:
        raise ValueError("Module C strength must lie in [0.5, 1]")
    size = len(y)
    namespace = "decaf_covertype_module_c_v1"
    context = np.where(
        _uniform(size, seed, split, "context", namespace) < 0.5,
        -1,
        1,
    ).astype(np.int8)
    endpoint_uniform = _uniform(size, seed, split, "endpoint_factor", namespace)
    alternate_uniform = _uniform(size, seed, split, "alternate_factor", namespace)
    endpoint = _aligned(y, strength, endpoint_uniform)
    if mechanism == "direct":
        alternate = _aligned(y, strength, alternate_uniform)
    elif mechanism == "gate":
        alternate = _aligned(y, 0.5, alternate_uniform)
    else:
        alternate = -_aligned(y, strength, alternate_uniform)
    factor = np.where(context > 0, endpoint, alternate).astype(np.int8)
    return MechanismRealization(context, factor, endpoint, alternate)


def realize_module_f(
    y: np.ndarray,
    *,
    regime: str,
    seed: int,
    split: str,
) -> MechanismRealization:
    """Generate endpoint-null robust/mild/fragile mechanisms."""

    if regime not in F_REGIMES:
        raise ValueError(f"unknown Module F regime: {regime}")
    size = len(y)
    namespace = "decaf_covertype_module_f_v1"
    context = np.where(
        _uniform(size, seed, split, "context", namespace) < 0.5,
        -1,
        1,
    ).astype(np.int8)
    endpoint = _aligned(
        y,
        0.5,
        _uniform(size, seed, split, "endpoint_factor", namespace),
    )
    alternate = _aligned(
        y,
        F_ALTERNATE_ALIGNMENT[regime],
        _uniform(size, seed, split, "alternate_factor", namespace),
    )
    factor = np.where(context > 0, endpoint, alternate).astype(np.int8)
    return MechanismRealization(context, factor, endpoint, alternate)


def augmented_features(X: np.ndarray, realization: MechanismRealization) -> np.ndarray:
    """Append factual context and factor to all natural features."""

    if len(X) != len(realization.context):
        raise ValueError("natural rows and mechanism rows must match")
    return np.asarray(
        np.column_stack((X, realization.context, realization.factor)),
        dtype=np.float64,
    )


def legal_query_features(
    X: np.ndarray,
    *,
    context: int,
    factor: int,
) -> np.ndarray:
    """Construct a legal binary endpoint query without interpolating inputs."""

    if context not in (-1, 1) or factor not in (-1, 1):
        raise ValueError("context and factor queries must be -1 or +1")
    return np.asarray(
        np.column_stack(
            (
                X,
                np.full(len(X), context, dtype=np.int8),
                np.full(len(X), factor, dtype=np.int8),
            )
        ),
        dtype=np.float64,
    )


__all__ = [
    "C_MECHANISMS",
    "F_ALTERNATE_ALIGNMENT",
    "F_REGIMES",
    "MechanismRealization",
    "augmented_features",
    "legal_query_features",
    "realize_module_c",
    "realize_module_f",
    "stream_seed",
]
