"""Recompute neutral trajectories with the current DECAF core and compare them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.core.decomposition import decompose, endpoint_orientation
from decaf.core.quadrature import integrate_components
from decaf.core.trajectories import StreamingDECAFAccumulator
from tools.crossgen.schema import (
    IDENTITY_COLUMNS,
    read_trajectory_record,
    resolve_endpoint_d,
    resolve_stage_r,
    sha256_file,
    trapezoid_weights,
)

TIER_A_ATOL = 5.0e-4
TIER_A_RTOL = 5.0e-3
TIER_B_ABS = 2.0e-3
HARD_MISMATCH_ABS = 1.0e-2
BOUNDARY_ABS = 2.0e-3
SUMMARY_NAMES = ("M", "E", "C", "F", "Abs")
MECHANISM_NAMES = ("E", "C", "F")


def _scalar(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = values[np.isfinite(values.to_numpy(dtype=np.float64))].to_numpy(dtype=np.float64)
    if finite.size == 0:
        raise ValueError(f"{column} is missing in unit {frame['unit_id'].iloc[0]!r}")
    if not np.allclose(finite, finite[0], atol=1.0e-12, rtol=1.0e-10):
        raise ValueError(f"{column} changes in unit {frame['unit_id'].iloc[0]!r}")
    return float(finite[0])


def _metadata(frame: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw in frame["metadata_json"]:
        payload.update(json.loads(str(raw)))
    return payload


def _dominant(values: dict[str, float]) -> str:
    scores = np.asarray([values[name] for name in MECHANISM_NAMES], dtype=np.float64)
    maximum = float(scores.max())
    winners = [
        name for name, score in zip(MECHANISM_NAMES, scores, strict=True) if score == maximum
    ]
    return "|".join(winners)


def _historical_semantics(
    endpoint_d: float,
    epsilon: float,
    historical: dict[str, float],
    metadata: dict[str, Any],
) -> tuple[bool, int, str]:
    gate = bool(metadata.get("historical_gate", abs(endpoint_d) >= epsilon))
    orientation = int(
        metadata.get("historical_orientation", int(np.sign(endpoint_d)) if gate else 0)
    )
    dominant = str(metadata.get("historical_dominant", _dominant(historical)))
    return gate, orientation, dominant


def _identity_match(frame: pd.DataFrame, metadata: dict[str, Any]) -> bool:
    expected_pairs = {
        "model_id": "current_model_id",
        "checkpoint_sha256": "current_checkpoint_sha256",
        "sample_or_pair_id": "current_sample_or_pair_id",
        "factor_or_part_id": "current_factor_or_part_id",
        "counterfactual_map": "current_counterfactual_map",
        "protocol": "current_protocol",
    }
    required_metadata = {"identity_match", *expected_pairs.values()}
    if not required_metadata.issubset(metadata):
        return False
    matched = metadata["identity_match"] is True
    for historical_column, current_key in expected_pairs.items():
        if current_key not in metadata:
            continue
        historical = frame[historical_column].iloc[0]
        matched = matched and str(historical) == str(metadata[current_key])
    return bool(matched)


def _unit_comparison(unit: pd.DataFrame) -> dict[str, Any]:
    unit = unit.sort_values(["stage_index", "stage_t"], kind="stable")
    grid = unit["stage_t"].to_numpy(dtype=np.float64)
    response = resolve_stage_r(unit)
    endpoint_d = resolve_endpoint_d(unit)
    epsilon = _scalar(unit, "endpoint_epsilon")

    pointwise = decompose(response, endpoint_d, epsilon, axis=0)
    integrated = integrate_components(grid, pointwise, axis=0)
    current = {
        "M": abs(endpoint_d),
        **{name: float(np.asarray(integrated[name])) for name in ("E", "C", "F", "Abs")},
    }

    streaming = StreamingDECAFAccumulator(endpoint_d, epsilon)
    for position, stage_response in zip(grid, response, strict=True):
        streaming.update(float(position), float(stage_response))
    streamed = streaming.finalize()
    streaming_error = max(
        abs(current[name] - float(np.asarray(streamed[name]))) for name in SUMMARY_NAMES
    )
    if streaming_error > 2.0e-12:
        raise AssertionError(
            f"batch/streaming current-core disagreement in {unit['unit_id'].iloc[0]!r}: "
            f"{streaming_error}"
        )

    historical = {name: _scalar(unit, f"historical_{name}") for name in SUMMARY_NAMES}
    absolute_errors = {name: abs(current[name] - historical[name]) for name in SUMMARY_NAMES}
    signed_errors = {name: current[name] - historical[name] for name in SUMMARY_NAMES}
    metadata = _metadata(unit)

    current_gate_array, current_orientation_array = endpoint_orientation(endpoint_d, epsilon)
    current_gate = bool(np.asarray(current_gate_array))
    current_orientation = int(np.asarray(current_orientation_array))
    current_dominant = _dominant(current)
    historical_gate, historical_orientation, historical_dominant = _historical_semantics(
        endpoint_d, epsilon, historical, metadata
    )
    boundary = abs(abs(endpoint_d) - epsilon) <= BOUNDARY_ABS
    gate_match = current_gate == historical_gate
    orientation_match = current_orientation == historical_orientation
    dominant_match = current_dominant == historical_dominant
    identity_match = _identity_match(unit, metadata)

    tier_a_pass = all(
        np.isclose(current[name], historical[name], atol=TIER_A_ATOL, rtol=TIER_A_RTOL)
        for name in SUMMARY_NAMES
    )
    tier_b_pass = (
        max(absolute_errors.values()) <= TIER_B_ABS
        and (boundary or (gate_match and orientation_match))
        and dominant_match
    )
    tier = "A" if tier_a_pass else ("B" if tier_b_pass else "FAIL")
    hard_mismatch = (
        max(absolute_errors.values()) > HARD_MISMATCH_ABS
        or (not boundary and (not gate_match or not orientation_match))
        or not dominant_match
        or not identity_match
    )

    supplied_weights = unit["quadrature_weight"].to_numpy(dtype=np.float64)
    expected_weights = trapezoid_weights(grid)
    weight_match = bool(
        not np.isfinite(supplied_weights).any()
        or np.allclose(supplied_weights, expected_weights, atol=1.0e-12, rtol=1.0e-10)
    )
    identity = {name: unit[name].iloc[0] for name in IDENTITY_COLUMNS}
    return {
        **identity,
        "stage_count": int(len(unit)),
        "endpoint_d": endpoint_d,
        "endpoint_epsilon": epsilon,
        "boundary": boundary,
        "current_gate": current_gate,
        "historical_gate": historical_gate,
        "gate_match": gate_match,
        "current_orientation": current_orientation,
        "historical_orientation": historical_orientation,
        "orientation_match": orientation_match,
        "current_dominant": current_dominant,
        "historical_dominant": historical_dominant,
        "dominant_match": dominant_match,
        "identity_match": identity_match,
        "quadrature_weight_match": weight_match,
        "numeric_identity_pass": bool(
            pointwise["identity_audit"]["passed"]
            and integrated["identity_audit"]["passed"]
            and streamed["numeric_audit"]["passed"]
        ),
        "streaming_max_abs_error": streaming_error,
        "tier_a_pass": tier_a_pass,
        "tier_b_pass": tier_b_pass,
        "tier": tier,
        "hard_mismatch": hard_mismatch,
        **{f"current_{name}": current[name] for name in SUMMARY_NAMES},
        **{f"historical_{name}": historical[name] for name in SUMMARY_NAMES},
        **{f"abs_error_{name}": absolute_errors[name] for name in SUMMARY_NAMES},
        **{f"signed_error_{name}": signed_errors[name] for name in SUMMARY_NAMES},
        "stage_e_json": json.dumps(np.asarray(pointwise["E"]).tolist(), separators=(",", ":")),
        "stage_c_json": json.dumps(np.asarray(pointwise["C"]).tolist(), separators=(",", ":")),
        "stage_f_json": json.dumps(np.asarray(pointwise["F"]).tolist(), separators=(",", ":")),
    }


def _fraction(values: pd.Series) -> float:
    return float(values.astype(bool).mean()) if len(values) else float("nan")


def summarize(comparison: pd.DataFrame, source: Path) -> dict[str, Any]:
    """Build aggregate numerical and semantic agreement statistics."""

    error_columns = [f"abs_error_{name}" for name in SUMMARY_NAMES]
    all_errors = comparison.loc[:, error_columns].to_numpy(dtype=np.float64).ravel()
    non_boundary = comparison.loc[~comparison["boundary"].astype(bool)]
    tier_a = comparison["tier"].eq("A")
    tier_b = comparison["tier"].eq("B")
    metric_summaries = {}
    signed_bias = {}
    for name in SUMMARY_NAMES:
        errors = comparison[f"abs_error_{name}"].to_numpy(dtype=np.float64)
        metric_summaries[name] = {
            "median_absolute_error": float(np.median(errors)),
            "p95_absolute_error": float(np.percentile(errors, 95)),
            "maximum_absolute_error": float(np.max(errors)),
        }
        signed_bias[name] = float(comparison[f"signed_error_{name}"].mean())
    return {
        "schema_version": 1,
        "trajectory_record": str(source.resolve()),
        "trajectory_record_sha256": sha256_file(source),
        "unit_count": int(len(comparison)),
        "non_boundary_unit_count": int(len(non_boundary)),
        "tolerances": {
            "tier_a_atol": TIER_A_ATOL,
            "tier_a_rtol": TIER_A_RTOL,
            "tier_b_absolute": TIER_B_ABS,
            "hard_mismatch_absolute": HARD_MISMATCH_ABS,
            "boundary_absolute": BOUNDARY_ABS,
        },
        "median_absolute_error": float(np.median(all_errors)),
        "p95_absolute_error": float(np.percentile(all_errors, 95)),
        "maximum_absolute_error": float(np.max(all_errors)),
        "metric_summaries": metric_summaries,
        "mean_signed_error": signed_bias,
        "tier_a_fraction": _fraction(tier_a),
        "tier_b_fraction": _fraction(tier_b),
        "tier_a_or_b_fraction": _fraction(tier_a | tier_b),
        "non_boundary_tier_a_or_b_fraction": _fraction(non_boundary["tier"].isin(["A", "B"])),
        "hard_mismatch_fraction": _fraction(comparison["hard_mismatch"]),
        "gate_agreement": _fraction(non_boundary["gate_match"]),
        "orientation_agreement": _fraction(non_boundary["orientation_match"]),
        "dominant_mechanism_agreement": _fraction(comparison["dominant_match"]),
        "identity_agreement": _fraction(comparison["identity_match"]),
        "quadrature_weight_agreement": _fraction(comparison["quadrature_weight_match"]),
        "numeric_identity_fraction": _fraction(comparison["numeric_identity_pass"]),
    }


def _write_comparison(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"unsupported comparison extension: {path.suffix}")


def compare_record(
    trajectory_record: str | Path,
    output: str | Path,
    *,
    summary_output: str | Path | None = None,
) -> dict[str, Any]:
    """Compare every unit and write unit-level plus aggregate outputs."""

    source = Path(trajectory_record)
    destination = Path(output)
    frame = read_trajectory_record(source)
    rows = [_unit_comparison(unit) for _, unit in frame.groupby("unit_id", sort=True)]
    comparison = pd.DataFrame(rows)
    _write_comparison(comparison, destination)
    summary = summarize(comparison, source)
    summary_path = (
        Path(summary_output)
        if summary_output is not None
        else destination.with_name(f"{destination.stem}_summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"comparison": comparison, "summary": summary, "summary_path": summary_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compare_record(
        args.trajectory_record,
        args.output,
        summary_output=args.summary_output,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
