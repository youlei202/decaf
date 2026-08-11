"""Portable timing, memory, and query-count result schemas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TIMING_COLUMNS = (
    "dataset",
    "model",
    "method",
    "repeat",
    "wall_seconds_per_image",
    "peak_allocated_bytes",
    "forward_rows_per_image",
    "backward_calls_per_image",
)


def validate_timing(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw repeat-level timing without importing a GPU framework."""

    missing = sorted(set(TIMING_COLUMNS) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"timing table is invalid; missing={missing}")
    result = frame.copy()
    if result.duplicated(["dataset", "model", "method", "repeat"]).any():
        raise ValueError("timing table contains duplicate repeats")
    numeric = TIMING_COLUMNS[3:]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    values = result.loc[:, numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("timing measurements must be finite and non-negative")
    return result


def summarize_timing(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the registered median across independent timing repeats."""

    valid = validate_timing(frame)
    value_columns = list(TIMING_COLUMNS[4:])
    summary = (
        valid.groupby(["dataset", "model", "method"], sort=True, observed=True)
        .agg(
            repeats=("repeat", "nunique"),
            **{column: (column, "median") for column in value_columns},
        )
        .reset_index()
    )
    summary["aggregation"] = "median_across_repeats"
    return summary


def timing_ratio(
    summary: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    column: str,
    model: str,
) -> float:
    """Return one registered method-to-method timing or memory ratio."""

    if column not in summary.columns:
        raise ValueError(f"timing summary has no column: {column}")
    selected = summary.loc[summary["model"].astype(str) == model]
    indexed = selected.set_index("method")
    if numerator not in indexed.index or denominator not in indexed.index:
        raise ValueError("timing ratio methods are absent")
    top = float(indexed.loc[numerator, column])
    bottom = float(indexed.loc[denominator, column])
    if not np.isfinite(top) or not np.isfinite(bottom) or bottom <= 0.0:
        raise ValueError("timing ratio requires finite values and a positive denominator")
    return top / bottom


def require_gpu_runtime() -> Any:
    """Import torch lazily and require CUDA only when compute is requested."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("GPU attribution requires an environment with PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("GPU attribution was requested but CUDA is unavailable")
    return torch


__all__ = [
    "TIMING_COLUMNS",
    "require_gpu_runtime",
    "summarize_timing",
    "timing_ratio",
    "validate_timing",
]
