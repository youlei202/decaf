"""Endpoint-M analysis derived from persisted per-image endpoint quantities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from decaf.core.metrics import endpoint_magnitude


def _matrix(value: Any, *, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(value, dtype=np.float64)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape [B,K] with K >= 2")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array, single


def row_spearman(left: Any, right: Any) -> np.ndarray:
    """Return frozen safe-zero average-rank Spearman for every finite row."""

    first, _ = _matrix(left, name="left")
    second, _ = _matrix(right, name="right")
    if first.shape != second.shape:
        raise ValueError("left and right must have identical shapes")
    left_rank = rankdata(first, axis=1, method="average")
    right_rank = rankdata(second, axis=1, method="average")
    left_rank -= left_rank.mean(axis=1, keepdims=True)
    right_rank -= right_rank.mean(axis=1, keepdims=True)
    numerator = np.sum(left_rank * right_rank, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left_rank * left_rank, axis=1, dtype=np.float64)
        * np.sum(right_rank * right_rank, axis=1, dtype=np.float64)
    )
    result = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def endpoint_m_quality(endpoint_m: Any, target_effects: Any) -> np.ndarray:
    """Score persisted Endpoint M against held-out target effects."""

    magnitude, _ = _matrix(endpoint_m, name="endpoint_m")
    effects, _ = _matrix(target_effects, name="target_effects")
    return row_spearman(magnitude, effects)


def audit_endpoint_identity(
    endpoint_m: Any,
    endpoint_effects: Any,
    *,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Audit persisted M against the common core's endpoint magnitude."""

    magnitude, _ = _matrix(endpoint_m, name="endpoint_m")
    effects, _ = _matrix(endpoint_effects, name="endpoint_effects")
    if magnitude.shape != effects.shape:
        raise ValueError("endpoint_m and endpoint_effects must have identical shapes")
    expected = endpoint_magnitude(effects)
    difference = np.abs(magnitude - expected)
    tolerance = float(atol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be finite and non-negative")
    return {
        "passed": bool(np.all(difference <= tolerance)),
        "units": int(difference.size),
        "max_abs_difference": float(difference.max(initial=0.0)),
        "mean_abs_difference": float(difference.mean(dtype=np.float64)),
        "atol": tolerance,
    }


def append_endpoint_m(
    frame: pd.DataFrame,
    *,
    anchor_method: str = "decaf_5",
    require_identity: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append Endpoint M as an ordinary analysis-derived method.

    The compute stage persists the endpoint effect and DECAF ``M`` once.  This
    function performs no model inference and is intentionally called by the
    normal ``analyze`` stage.
    """

    required = {
        "scope",
        "dataset",
        "model",
        "method",
        "image_id",
        "endpoint_effects",
        "quality_target_effects",
        "decaf_M",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"per-image results cannot generate Endpoint M; missing={missing}")
    if bool((frame["method"].astype(str) == "endpoint_m").any()):
        raise ValueError("Endpoint M already exists in the per-image results")
    anchor = frame.loc[frame["method"].astype(str) == anchor_method].copy()
    if anchor.empty:
        raise ValueError(f"Endpoint M anchor method is absent: {anchor_method}")
    source_keys = ["scope", "dataset", "model", "image_id"]
    if anchor.duplicated(source_keys).any():
        raise ValueError("Endpoint M anchor contains duplicate dataset/model/image rows")
    magnitudes = [np.asarray(value, dtype=np.float64).reshape(-1) for value in anchor["decaf_M"]]
    endpoint_effects = [
        np.asarray(value, dtype=np.float64).reshape(-1) for value in anchor["endpoint_effects"]
    ]
    quality_targets = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in anchor["quality_target_effects"]
    ]
    row_audits = [
        audit_endpoint_identity(magnitude, effect)
        for magnitude, effect in zip(magnitudes, endpoint_effects, strict=True)
    ]
    units = sum(int(row["units"]) for row in row_audits)
    audit = {
        "passed": all(bool(row["passed"]) for row in row_audits),
        "units": units,
        "max_abs_difference": max(float(row["max_abs_difference"]) for row in row_audits),
        "mean_abs_difference": (
            sum(float(row["mean_abs_difference"]) * int(row["units"]) for row in row_audits) / units
        ),
        "atol": float(row_audits[0]["atol"]),
        "variable_length_rows": len({len(value) for value in magnitudes}) > 1,
        "endpoint_effects_column": "endpoint_effects",
        "quality_target_effects_column": "quality_target_effects",
        "roles_separated": True,
    }
    if require_identity and not audit["passed"]:
        raise AssertionError(f"Endpoint M identity audit failed: {audit}")
    endpoint = anchor.copy()
    endpoint["source_method"] = anchor_method
    endpoint["method"] = "endpoint_m"
    endpoint["patch_scores"] = [row.copy() for row in magnitudes]
    endpoint_quality: list[float] = []
    background_quality: list[float] = []
    telea_quality: list[float] = []
    has_heldout_columns = {
        "quality_aggregation",
        "heldout_background_texture_effects",
        "heldout_telea_dilate3_effects",
    }.issubset(anchor.columns)
    for index, (magnitude, target) in enumerate(zip(magnitudes, quality_targets, strict=True)):
        aggregation = str(anchor.iloc[index]["quality_aggregation"]) if has_heldout_columns else ""
        if aggregation == "equal_mean_of_operator_spearman":
            background = np.asarray(
                anchor.iloc[index]["heldout_background_texture_effects"],
                dtype=np.float64,
            ).reshape(-1)
            telea = np.asarray(
                anchor.iloc[index]["heldout_telea_dilate3_effects"],
                dtype=np.float64,
            ).reshape(-1)
            first = float(endpoint_m_quality(magnitude, background)[0])
            second = float(endpoint_m_quality(magnitude, telea)[0])
            background_quality.append(first)
            telea_quality.append(second)
            endpoint_quality.append(0.5 * (first + second))
        else:
            endpoint_quality.append(float(endpoint_m_quality(magnitude, target)[0]))
            background_quality.append(float("nan"))
            telea_quality.append(float("nan"))
    endpoint["spearman"] = endpoint_quality
    if has_heldout_columns:
        endpoint["heldout_background_texture_spearman"] = background_quality
        endpoint["heldout_telea_dilate3_spearman"] = telea_quality
    combined = pd.concat([frame, endpoint], ignore_index=True, sort=False)
    result_keys = ["scope", "dataset", "model", "method", "image_id"]
    if combined.duplicated(result_keys).any():
        raise AssertionError("Endpoint M analysis created duplicate result keys")
    return combined, {**audit, "anchor_method": anchor_method, "rows": len(endpoint)}


__all__ = [
    "append_endpoint_m",
    "audit_endpoint_identity",
    "endpoint_m_quality",
    "row_spearman",
]
