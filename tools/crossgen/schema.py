"""Experiment-independent neutral trajectory records for cross-generation checks.

Legacy exporters are intentionally limited to factual/counterfactual scores and
identity metadata. The current repository consumes those scores and performs
all DECAF routing and integration itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

NEUTRAL_COLUMNS = (
    "experiment_family",
    "reference_run",
    "unit_id",
    "model_id",
    "checkpoint_sha256",
    "sample_or_pair_id",
    "factor_or_part_id",
    "counterfactual_map",
    "protocol",
    "protocol_seed",
    "stage_index",
    "stage_t",
    "quadrature_weight",
    "endpoint_epsilon",
    "endpoint_score_plus",
    "endpoint_score_minus",
    "endpoint_d",
    "stage_score_plus",
    "stage_score_minus",
    "stage_r",
    "historical_M",
    "historical_E",
    "historical_C",
    "historical_F",
    "historical_Abs",
    "metadata_json",
)

IDENTITY_COLUMNS = (
    "experiment_family",
    "reference_run",
    "unit_id",
    "model_id",
    "checkpoint_sha256",
    "sample_or_pair_id",
    "factor_or_part_id",
    "counterfactual_map",
    "protocol",
    "protocol_seed",
)

NUMERIC_COLUMNS = (
    "protocol_seed",
    "stage_index",
    "stage_t",
    "quadrature_weight",
    "endpoint_epsilon",
    "endpoint_score_plus",
    "endpoint_score_minus",
    "endpoint_d",
    "stage_score_plus",
    "stage_score_minus",
    "stage_r",
    "historical_M",
    "historical_E",
    "historical_C",
    "historical_F",
    "historical_Abs",
)

REQUIRED_TEXT_COLUMNS = (
    "experiment_family",
    "reference_run",
    "unit_id",
    "model_id",
    "sample_or_pair_id",
    "protocol",
)

REQUIRED_NUMERIC_COLUMNS = ("stage_index", "stage_t", "endpoint_epsilon")


def trapezoid_weights(stage_t: Any) -> np.ndarray:
    """Return finite-grid trapezoidal weights for a one-dimensional grid."""

    grid = np.asarray(stage_t, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all():
        raise ValueError("stage_t must be a finite one-dimensional grid with at least two stages")
    widths = np.diff(grid)
    if not np.all(widths > 0.0):
        raise ValueError("stage_t must be strictly increasing")
    weights = np.empty_like(grid)
    weights[0] = widths[0] / 2.0
    weights[-1] = widths[-1] / 2.0
    if grid.size > 2:
        weights[1:-1] = (widths[:-1] + widths[1:]) / 2.0
    return weights


def _one_value(frame: pd.DataFrame, column: str, *, required: bool) -> Any:
    values = frame[column].dropna()
    if values.empty:
        if required:
            raise ValueError(f"{column} is missing for unit {frame['unit_id'].iloc[0]!r}")
        return None
    unique = pd.unique(values)
    if len(unique) != 1:
        raise ValueError(f"{column} changes within unit {frame['unit_id'].iloc[0]!r}")
    return unique[0]


def _numeric_response(
    frame: pd.DataFrame,
    direct: str,
    positive: str,
    negative: str,
    *,
    scalar: bool,
) -> np.ndarray | float:
    direct_values = pd.to_numeric(frame[direct], errors="coerce").to_numpy(dtype=np.float64)
    positive_values = pd.to_numeric(frame[positive], errors="coerce").to_numpy(dtype=np.float64)
    negative_values = pd.to_numeric(frame[negative], errors="coerce").to_numpy(dtype=np.float64)
    derived = positive_values - negative_values
    direct_present = np.isfinite(direct_values)
    derived_present = np.isfinite(derived)
    if np.any(direct_present & derived_present):
        disagreement = np.abs(direct_values - derived)
        if np.any(disagreement[direct_present & derived_present] > 1.0e-10):
            raise ValueError(
                f"{direct} disagrees with {positive} - {negative} in "
                f"unit {frame['unit_id'].iloc[0]!r}"
            )
    resolved = np.where(direct_present, direct_values, derived)
    if scalar:
        finite = resolved[np.isfinite(resolved)]
        if finite.size == 0:
            raise ValueError(f"{direct} (or score pair) is missing")
        if not np.allclose(finite, finite[0], atol=1.0e-12, rtol=1.0e-10):
            raise ValueError(f"{direct} changes within unit {frame['unit_id'].iloc[0]!r}")
        return float(finite[0])
    if not np.isfinite(resolved).all():
        raise ValueError(f"{direct} (or score pair) is missing at one or more stages")
    return resolved


def resolve_endpoint_d(frame: pd.DataFrame) -> float:
    """Resolve one endpoint response from a unit's direct or paired scores."""

    return float(
        _numeric_response(
            frame,
            "endpoint_d",
            "endpoint_score_plus",
            "endpoint_score_minus",
            scalar=True,
        )
    )


def resolve_stage_r(frame: pd.DataFrame) -> np.ndarray:
    """Resolve stage responses from direct or paired scores."""

    return np.asarray(
        _numeric_response(
            frame,
            "stage_r",
            "stage_score_plus",
            "stage_score_minus",
            scalar=False,
        ),
        dtype=np.float64,
    )


def validate_trajectory_record(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonically order a neutral trajectory frame."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("trajectory record must be a non-empty pandas DataFrame")
    missing = sorted(set(NEUTRAL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"trajectory record is missing columns: {missing}")

    result = frame.copy()
    for column in REQUIRED_TEXT_COLUMNS:
        values = result[column]
        invalid = values.isna() | values.astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"{column} must be explicit and non-empty")

    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in REQUIRED_NUMERIC_COLUMNS:
        if not np.isfinite(result[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{column} must be finite")
    if (result["endpoint_epsilon"] <= 0.0).any():
        raise ValueError("endpoint_epsilon must be strictly positive")
    stage_indices = result["stage_index"].to_numpy(dtype=np.float64)
    if (stage_indices < 0.0).any() or not np.equal(stage_indices, np.floor(stage_indices)).all():
        raise ValueError("stage_index must contain non-negative integers")
    result["stage_index"] = result["stage_index"].astype(np.int64)

    metadata: list[str] = []
    for row_index, raw in enumerate(result["metadata_json"]):
        if pd.isna(raw) or str(raw).strip() == "":
            payload: dict[str, Any] = {}
        else:
            try:
                parsed = json.loads(str(raw))
            except json.JSONDecodeError as error:
                raise ValueError(f"metadata_json row {row_index} is not valid JSON") from error
            if not isinstance(parsed, dict):
                raise ValueError(f"metadata_json row {row_index} must encode a JSON object")
            payload = parsed
        metadata.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    result["metadata_json"] = metadata

    ordered_units: list[pd.DataFrame] = []
    for unit_id, unit in result.groupby("unit_id", sort=True, dropna=False):
        for column in IDENTITY_COLUMNS:
            _one_value(unit, column, required=column in REQUIRED_TEXT_COLUMNS)
        if unit["metadata_json"].nunique(dropna=False) != 1:
            raise ValueError(f"metadata_json changes within unit {unit_id!r}")
        epsilon = float(_one_value(unit, "endpoint_epsilon", required=True))
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"invalid endpoint_epsilon for unit {unit_id!r}")
        unit = unit.sort_values(["stage_index", "stage_t"], kind="stable").copy()
        if unit["stage_index"].duplicated().any():
            raise ValueError(f"duplicate stage_index in unit {unit_id!r}")
        expected_indices = np.arange(len(unit), dtype=np.int64)
        if not np.array_equal(unit["stage_index"].to_numpy(dtype=np.int64), expected_indices):
            raise ValueError(f"stage_index must be contiguous from zero in unit {unit_id!r}")
        grid = unit["stage_t"].to_numpy(dtype=np.float64)
        trapezoid_weights(grid)
        resolve_endpoint_d(unit)
        resolve_stage_r(unit)
        supplied = unit["quadrature_weight"].to_numpy(dtype=np.float64)
        finite_weights = np.isfinite(supplied)
        if finite_weights.any() and not finite_weights.all():
            raise ValueError(f"quadrature_weight must be complete or empty in unit {unit_id!r}")
        if finite_weights.all() and (supplied < 0.0).any():
            raise ValueError(f"quadrature_weight must be non-negative in unit {unit_id!r}")
        if finite_weights.all():
            expected_weights = trapezoid_weights(grid)
            if not np.allclose(supplied, expected_weights, atol=1.0e-12, rtol=1.0e-10):
                raise ValueError(f"quadrature_weight differs from the grid in unit {unit_id!r}")
        ordered_units.append(unit)

    canonical = pd.concat(ordered_units, ignore_index=True)
    extras = sorted(set(canonical.columns) - set(NEUTRAL_COLUMNS))
    return canonical.loc[:, [*NEUTRAL_COLUMNS, *extras]]


def read_trajectory_record(path: str | Path) -> pd.DataFrame:
    """Read and validate a parquet, CSV, or JSON-lines neutral record."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(source)
    elif suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(source, lines=True)
    else:
        raise ValueError(f"unsupported trajectory-record extension: {source.suffix}")
    return validate_trajectory_record(frame)


def write_trajectory_record(frame: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write a neutral record in a supported format."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canonical = validate_trajectory_record(frame)
    suffix = destination.suffix.lower()
    if suffix == ".parquet":
        canonical.to_parquet(destination, index=False)
    elif suffix == ".csv":
        canonical.to_csv(destination, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        canonical.to_json(destination, orient="records", lines=True)
    else:
        raise ValueError(f"unsupported trajectory-record extension: {destination.suffix}")
    return destination


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
