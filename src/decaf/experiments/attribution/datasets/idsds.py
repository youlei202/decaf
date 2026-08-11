"""CPU-safe ImageNet-1k IDSDS manifests and the registered 4x4 partition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GRID_ROWS = 4
GRID_COLUMNS = 4
PATCH_COUNT = GRID_ROWS * GRID_COLUMNS
PRIMARY_IMAGES = 10_000
FULL_IMAGES = 50_000

MANIFEST_COLUMNS = (
    "image_id",
    "label",
    "source_shard",
    "row_index",
    "image_filename",
    "wnid",
)
RESULT_COLUMNS = (
    "dataset",
    "model",
    "method",
    "image_id",
    "spearman",
    "effects",
    "finite_complete",
)


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"unsupported IDSDS table type: {source.suffix}")


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{label} is empty")


def validate_manifest(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Validate a frozen ImageNet manifest without loading image bytes."""

    _require_columns(frame, MANIFEST_COLUMNS, "IDSDS manifest")
    result = frame.copy()
    result["image_id"] = result["image_id"].astype(str)
    if result["image_id"].duplicated().any():
        raise ValueError("IDSDS manifest contains duplicate image IDs")
    labels = pd.to_numeric(result["label"], errors="coerce")
    if labels.isna().any() or not bool((labels == np.floor(labels)).all()):
        raise ValueError("IDSDS labels must be integer class indices")
    if not bool(((labels >= 0) & (labels < 1_000)).all()):
        raise ValueError("IDSDS labels must lie in [0, 1000)")
    result["label"] = labels.astype(np.int64)
    if expected_rows is not None and len(result) != expected_rows:
        raise ValueError(f"IDSDS manifest rows differ: {len(result)} != {expected_rows}")
    return result


def load_manifest(path: str | Path, *, expected_rows: int | None = None) -> pd.DataFrame:
    """Load and validate a frozen ImageNet manifest."""

    return validate_manifest(_read_table(path), expected_rows=expected_rows)


def grid_patch_slices(height: int, width: int) -> tuple[tuple[slice, slice], ...]:
    """Return the registered sixteen row-major equal-area patches."""

    if height <= 0 or width <= 0 or height % GRID_ROWS or width % GRID_COLUMNS:
        raise ValueError("image dimensions must be positive and divisible by four")
    patch_height = height // GRID_ROWS
    patch_width = width // GRID_COLUMNS
    return tuple(
        (
            slice(row * patch_height, (row + 1) * patch_height),
            slice(column * patch_width, (column + 1) * patch_width),
        )
        for row in range(GRID_ROWS)
        for column in range(GRID_COLUMNS)
    )


def grid_patch_masks(height: int, width: int) -> np.ndarray:
    """Return disjoint float64 patch indicators with shape ``[16,H,W]``."""

    masks = np.zeros((PATCH_COUNT, height, width), dtype=np.float64)
    for index, (rows, columns) in enumerate(grid_patch_slices(height, width)):
        masks[index, rows, columns] = 1.0
    if not np.array_equal(masks.sum(axis=0), np.ones((height, width))):
        raise AssertionError("IDSDS patch masks do not partition the image")
    return masks


def aggregate_patch_scores(attribution: Any, *, absolute: bool = False) -> np.ndarray:
    """Aggregate dense attribution into sixteen raw signed patch scores."""

    values = np.asarray(attribution)
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[None]
    if values.ndim != 4 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("attribution must have shape HW, CHW, or BCHW")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("attribution contains a non-finite value")
    height, width = values.shape[-2:]
    grid_patch_slices(height, width)
    selected = np.abs(values) if absolute else values
    reshaped = selected.reshape(
        selected.shape[0],
        selected.shape[1],
        GRID_ROWS,
        height // GRID_ROWS,
        GRID_COLUMNS,
        width // GRID_COLUMNS,
    )
    return reshaped.sum(axis=(1, 3, 5), dtype=np.float64).reshape(-1, PATCH_COUNT)


def validate_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the portable per-image result schema used by analysis."""

    _require_columns(frame, RESULT_COLUMNS, "IDSDS result")
    result = frame.copy()
    if result.duplicated(["dataset", "model", "method", "image_id"]).any():
        raise ValueError("IDSDS result contains duplicate members")
    quality = pd.to_numeric(result["spearman"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(quality).all():
        raise ValueError("IDSDS result contains non-finite quality")
    for value in result["effects"]:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != PATCH_COUNT or not np.isfinite(array).all():
            raise ValueError("every IDSDS effects vector must contain sixteen finite values")
    return result


__all__ = [
    "FULL_IMAGES",
    "GRID_COLUMNS",
    "GRID_ROWS",
    "MANIFEST_COLUMNS",
    "PATCH_COUNT",
    "PRIMARY_IMAGES",
    "RESULT_COLUMNS",
    "aggregate_patch_scores",
    "grid_patch_masks",
    "grid_patch_slices",
    "load_manifest",
    "validate_manifest",
    "validate_result_frame",
]
