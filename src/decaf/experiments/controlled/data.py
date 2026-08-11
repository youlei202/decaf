"""Portable Shapes3D asset resolution and exact controlled mappings.

This module deliberately contains no model or reporting code.  The paper-scale
dataset is user supplied; paths are resolved from ``DECAF_DATA_ROOT`` and only
logical, public names are written to run manifests.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decaf.core.manifests import sha256_file

FACTOR_NAMES = (
    "floor_color",
    "wall_color",
    "object_color",
    "object_size",
    "object_shape",
    "object_orientation",
)
FACTOR_ALIASES = {"orientation": "object_orientation"}
FACTOR_CARDINALITIES = (10, 10, 10, 8, 4, 15)
SHAPES3D_ROWS = 480_000
SHAPES3D_FILENAME = "3dshapes.h5"
SHAPES3D_SHA256 = "0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785"
SHAPES3D_BYTES = 267_573_662


def canonical_factor(name: str) -> str:
    """Return the public factor name, accepting the historical alias."""

    value = FACTOR_ALIASES.get(str(name), str(name))
    if value not in FACTOR_NAMES:
        raise ValueError(f"unknown Shapes3D factor: {name!r}")
    return value


def resolve_shapes3d_root(
    configured: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the dataset directory without searching private locations."""

    env = os.environ if environment is None else environment
    if configured is None:
        base = env.get("DECAF_DATA_ROOT")
        if not base:
            raise RuntimeError("DECAF_DATA_ROOT is required for Shapes3D compute")
        candidate = Path(base).expanduser() / "3d_shapes"
    else:
        text = os.fspath(configured)
        marker = "${DECAF_DATA_ROOT}"
        if marker in text:
            base = env.get("DECAF_DATA_ROOT")
            if not base:
                raise RuntimeError("DECAF_DATA_ROOT is required to expand the dataset path")
            text = text.replace(marker, base)
        if "$" in text:
            raise ValueError("dataset path contains an unresolved environment variable")
        candidate = Path(text).expanduser()
    return candidate.resolve()


@dataclass(frozen=True, slots=True)
class Shapes3DAsset:
    """Validated identity of a local Shapes3D source file."""

    root: Path
    source: Path
    size: int
    sha256: str

    def public_record(self) -> dict[str, Any]:
        """Return a manifest record without leaking the resolved host path."""

        return {
            "id": "3d_shapes",
            "logical_root": "${DECAF_DATA_ROOT}/3d_shapes",
            "path": SHAPES3D_FILENAME,
            "bytes": self.size,
            "sha256": self.sha256,
            "rows": SHAPES3D_ROWS,
            "factors": list(FACTOR_NAMES),
        }


def validate_shapes3d_asset(
    root: str | os.PathLike[str],
    *,
    expected_sha256: str = SHAPES3D_SHA256,
    expected_bytes: int = SHAPES3D_BYTES,
) -> Shapes3DAsset:
    """Validate the official HDF5 byte identity before model compute."""

    directory = Path(root).resolve()
    source = directory / SHAPES3D_FILENAME
    if not source.is_file():
        raise FileNotFoundError(f"Shapes3D source file is missing: {source}")
    size = source.stat().st_size
    if size != int(expected_bytes):
        raise ValueError(f"Shapes3D byte count mismatch: expected {expected_bytes}, found {size}")
    digest = sha256_file(source)
    if digest != str(expected_sha256).lower():
        raise ValueError("Shapes3D SHA256 mismatch")
    return Shapes3DAsset(directory, source, size, digest)


def validate_factor_table(
    factors: Any,
    *,
    expected_rows: int | None = None,
) -> np.ndarray:
    """Validate an integer ``rows x 6`` factor-index table."""

    values = np.asarray(factors)
    if values.ndim != 2 or values.shape[1] != len(FACTOR_NAMES):
        raise ValueError("factor table must have shape (rows, 6)")
    if values.dtype.kind not in "iu":
        raise TypeError("factor table must contain integers")
    result = np.asarray(values, dtype=np.int64)
    if expected_rows is not None and result.shape[0] != int(expected_rows):
        raise ValueError("factor table row count does not match the registered contract")
    for column, cardinality in enumerate(FACTOR_CARDINALITIES):
        if np.any((result[:, column] < 0) | (result[:, column] >= cardinality)):
            raise ValueError(f"factor column {FACTOR_NAMES[column]} is out of range")
    return np.ascontiguousarray(result)


def deterministic_sample_ids(
    population_size: int,
    count: int,
    *,
    seed: int,
    excluded: Sequence[int] = (),
) -> np.ndarray:
    """Select sorted sample IDs deterministically without replacement."""

    total = int(population_size)
    selected = int(count)
    if total < 1 or selected < 1:
        raise ValueError("population_size and count must be positive")
    blocked = np.asarray(tuple(excluded), dtype=np.int64)
    if blocked.size and (np.any(blocked < 0) or np.any(blocked >= total)):
        raise ValueError("excluded sample ID is outside the population")
    available = np.setdiff1d(np.arange(total, dtype=np.int64), blocked, assume_unique=False)
    if selected > available.size:
        raise ValueError("not enough unblocked samples")
    generator = np.random.default_rng(int(seed))
    return np.sort(generator.choice(available, size=selected, replace=False)).astype(np.int64)


def exact_counterfactual_rows(
    factors: Any,
    sample_ids: Sequence[int],
    factor: str,
    *,
    seed: int,
) -> np.ndarray:
    """Return exact-support rows that change one factor to a legal alternative.

    The same input row and seed always select the same alternative.  No other
    factor is changed, which is the invariant needed by C0/C1 pair audits.
    """

    table = validate_factor_table(factors)
    ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if ids.size < 1 or np.any(ids < 0) or np.any(ids >= table.shape[0]):
        raise ValueError("sample IDs must select at least one valid row")
    name = canonical_factor(factor)
    column = FACTOR_NAMES.index(name)
    cardinality = FACTOR_CARDINALITIES[column]
    output = table[ids].copy()
    generator = np.random.default_rng(int(seed))
    offset = generator.integers(1, cardinality, size=ids.size, dtype=np.int64)
    output[:, column] = (output[:, column] + offset) % cardinality
    changed = output != table[ids]
    if not np.all(changed.sum(axis=1) == 1) or not np.all(changed[:, column]):
        raise AssertionError("counterfactual map changed the wrong factor")
    return output


def object_color_map(values: Any, map_index: int) -> np.ndarray:
    """Apply a registered bin-flipping object-colour involution."""

    return _color_involution(values, map_index, label="object")


def wall_color_map(values: Any, map_index: int) -> np.ndarray:
    """Apply a registered bin-flipping wall-colour involution."""

    return _color_involution(values, map_index, label="wall")


def _color_involution(values: Any, map_index: int, *, label: str) -> np.ndarray:
    color = np.asarray(values, dtype=np.int64)
    if np.any((color < 0) | (color >= 10)):
        raise ValueError(f"{label} colours must lie in [0, 10)")
    if int(map_index) == 1:
        result = np.where(color < 5, color + 5, color - 5)
    elif int(map_index) == 2:
        result = 9 - color
    else:
        raise ValueError(f"{label} map index must be 1 or 2")
    if not np.all((color < 5) != (result < 5)):
        raise AssertionError(f"{label} map did not flip its binary bin")
    restored = np.where(result < 5, result + 5, result - 5) if map_index == 1 else 9 - result
    if not np.array_equal(restored, color):
        raise AssertionError(f"{label} map is not an involution")
    return np.asarray(result, dtype=np.int64)


__all__ = [
    "FACTOR_ALIASES",
    "FACTOR_CARDINALITIES",
    "FACTOR_NAMES",
    "SHAPES3D_BYTES",
    "SHAPES3D_FILENAME",
    "SHAPES3D_ROWS",
    "SHAPES3D_SHA256",
    "Shapes3DAsset",
    "canonical_factor",
    "deterministic_sample_ids",
    "exact_counterfactual_rows",
    "object_color_map",
    "resolve_shapes3d_root",
    "validate_factor_table",
    "validate_shapes3d_asset",
    "wall_color_map",
]
