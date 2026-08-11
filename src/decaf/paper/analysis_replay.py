"""CPU-only replay of paper inputs and numerical provenance assertions."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .manifest import load_representative_cases, load_visual_manifest, repository_root
from .reference import (
    ReferenceError,
    discover_archive,
    load_reference_runs,
    materialize_inputs,
    receipt_dict,
    reference_roots,
    verify_archive,
)


def _frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def _finite_number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, received {value!r}")
    return result


def select_figure_02(frame: pd.DataFrame) -> dict[str, Any]:
    """Resolve the paper's fixed matched-magnitude factor pair."""

    required = {
        "model_id",
        "task",
        "architecture",
        "model_seed",
        "factor",
        "mean_auc_abs",
        "endpoint_class",
        "is_intended",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Figure 2 input is missing columns: {sorted(missing)}")
    factors = ("object_shape", "floor_color")
    rows = frame.loc[frame["task"].eq("object_shape") & frame["factor"].isin(factors)].copy()
    rows["mean_auc_abs"] = pd.to_numeric(rows["mean_auc_abs"], errors="coerce")
    rows["model_seed"] = pd.to_numeric(rows["model_seed"], errors="raise").astype(int)
    rows = rows.dropna(subset=["mean_auc_abs"])
    rows = rows.sort_values(
        ["factor", "architecture", "model_seed", "model_id"],
        kind="mergesort",
    )

    duplicate = rows.duplicated(["factor", "architecture", "model_seed"], keep=False)
    if duplicate.any():
        keys = rows.loc[duplicate, ["factor", "architecture", "model_seed"]]
        raise ValueError(f"Figure 2 matched grid has duplicate rows: {keys.to_dict('records')}")

    grids: dict[str, set[tuple[str, int]]] = {}
    for factor in factors:
        local = rows.loc[rows["factor"].eq(factor)]
        grids[factor] = set(
            zip(local["architecture"].astype(str), local["model_seed"], strict=True)
        )
    if not grids[factors[0]] or grids[factors[0]] != grids[factors[1]]:
        raise ValueError(f"Figure 2 matched factors do not share a complete grid: {grids}")

    def summarize(factor: str, endpoint_class: str, intended: bool) -> dict[str, Any]:
        local = rows.loc[rows["factor"].eq(factor)].copy()
        classes = set(local["endpoint_class"].astype(str))
        intentions = set(_truthy(local["is_intended"]).tolist())
        if classes != {endpoint_class} or intentions != {intended}:
            raise ValueError(
                f"Figure 2 factor {factor} changed class: "
                f"endpoint_class={classes}, is_intended={intentions}"
            )
        return {
            "factor": factor,
            "endpoint_class": endpoint_class,
            "is_intended": intended,
            "mean_auc_abs": float(local["mean_auc_abs"].mean()),
            "source_rows": [
                {
                    "model_id": str(row.model_id),
                    "mean_auc_abs": _finite_number(row.mean_auc_abs),
                }
                for row in local.itertuples(index=False)
            ],
        }

    grid = sorted(grids[factors[0]])
    return {
        "task": "object_shape",
        "aggregation": "arithmetic_mean",
        "architecture_scope": sorted({architecture for architecture, _ in grid}),
        "model_seeds": sorted({seed for _, seed in grid}),
        "rows_per_factor": len(grid),
        "intended_factor": summarize("object_shape", "endpoint_supported", True),
        "endpoint_null_factor": summarize("floor_color", "endpoint_null", False),
    }


def select_figure_03(frame: pd.DataFrame) -> dict[str, Any]:
    """Select the Module-E trajectory with the largest observed reliance range."""

    required = {
        "module",
        "primary_geometry",
        "architecture",
        "p_train",
        "seed",
        "trajectory_id",
        "epoch",
        "V_rev",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Figure 3 input is missing columns: {sorted(missing)}")
    rows = frame.loc[frame["module"].eq("E") & _truthy(frame["primary_geometry"])].copy()
    rows["epoch"] = pd.to_numeric(rows["epoch"], errors="coerce")
    rows["V_rev"] = pd.to_numeric(rows["V_rev"], errors="coerce")
    rows["p_train"] = pd.to_numeric(rows["p_train"], errors="coerce")
    rows["seed"] = pd.to_numeric(rows["seed"], errors="coerce")
    keys = ["architecture", "p_train", "seed", "trajectory_id"]
    rows = rows.dropna(subset=[*keys, "epoch", "V_rev"])
    checkpoints = rows.groupby([*keys, "epoch"], as_index=False, sort=True)["V_rev"].first()
    summaries: list[dict[str, Any]] = []
    for group_key, group in checkpoints.groupby(keys, sort=True):
        if group["epoch"].nunique() < 2:
            continue
        minimum = float(group["V_rev"].min())
        maximum = float(group["V_rev"].max())
        summaries.append(
            {
                "architecture": str(group_key[0]),
                "training_correlation": float(group_key[1]),
                "seed": int(group_key[2]),
                "trajectory_id": str(group_key[3]),
                "checkpoint_epochs": sorted(int(value) for value in group["epoch"].unique()),
                "V_rev_min": minimum,
                "V_rev_max": maximum,
                "V_rev_range": maximum - minimum,
            }
        )
    if not summaries:
        raise ValueError("Figure 3 rule found no multi-checkpoint trajectories")
    summaries.sort(
        key=lambda row: (
            -row["V_rev_range"],
            row["architecture"],
            row["training_correlation"],
            row["seed"],
            row["trajectory_id"],
        )
    )
    return summaries[0]


def select_figure_04(frame: pd.DataFrame) -> dict[str, Any]:
    """Select the primary-geometry Module-F model with maximum fragility mass."""

    required = {
        "model_id",
        "module",
        "primary_geometry",
        "architecture",
        "seed",
        "variant",
        "geometry",
        "epoch",
        "F",
        "null_prediction_change_rate",
        "confidence_fragility",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Figure 4 input is missing columns: {sorted(missing)}")
    rows = frame.loc[frame["module"].eq("F") & _truthy(frame["primary_geometry"])].copy()
    rows["F"] = pd.to_numeric(rows["F"], errors="coerce")
    rows["seed"] = pd.to_numeric(rows["seed"], errors="raise").astype(int)
    rows = rows.dropna(subset=["F"])
    rows = rows.sort_values(
        ["F", "architecture", "seed", "variant", "geometry"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    if rows.empty:
        raise ValueError("Figure 4 rule selected no rows")
    row = rows.iloc[0]
    return {
        "model_id": str(row["model_id"]),
        "architecture": str(row["architecture"]),
        "seed": int(row["seed"]),
        "variant": str(row["variant"]),
        "geometry": str(row["geometry"]),
        "epoch": int(row["epoch"]),
        "F": _finite_number(row["F"]),
        "null_prediction_change_rate": _finite_number(row["null_prediction_change_rate"]),
        "confidence_fragility": _finite_number(row["confidence_fragility"]),
    }


def _equivalent(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return set(expected) == set(actual) and all(
            _equivalent(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _equivalent(left, right)
            for left, right in zip(expected, actual, strict=True)
        )
    return expected == actual


def _verify_resolution(
    case_id: str, expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> None:
    failures = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if key not in actual or not _equivalent(value, actual[key])
    }
    if failures:
        raise ValueError(f"representative case {case_id} changed: {failures}")


def replay_representative_cases(repo: Path, paper_data: Path) -> dict[str, Any]:
    """Resolve Figures 2--4 and fail if sealed data no longer match frozen metadata."""

    payload = load_representative_cases(repo / "paper" / "representative_cases.yaml")
    selectors = {
        "figure_02": select_figure_02,
        "figure_03": select_figure_03,
        "figure_04": select_figure_04,
    }
    results: dict[str, Any] = {}
    for case_id, case in payload["cases"].items():
        path = paper_data / str(case["run_id"]) / str(case["input"])
        actual = selectors[case_id](_frame(path))
        _verify_resolution(case_id, case["resolved"], actual)
        results[case_id] = {"status": "verified", "resolved": actual}
    return results


def _input_path(paper_data: Path, spec: Mapping[str, Any]) -> Path:
    return paper_data / str(spec["run_id"]) / str(spec["member"])


def _apply_filters(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    selected = frame
    for column, value in filters.items():
        if column not in selected:
            raise ValueError(f"filter column is missing: {column}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = pd.to_numeric(selected[column], errors="coerce")
            mask = (numeric - float(value)).abs() <= 1e-12
        else:
            mask = selected[column].astype(str).eq(str(value))
        selected = selected.loc[mask]
    return selected


def _assert_spearman(assertion: Mapping[str, Any], paper_data: Path) -> tuple[float, int]:
    frame = _frame(_input_path(paper_data, assertion["input"]))
    left, right = (str(value) for value in assertion["columns"])
    values = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    return float(values[left].corr(values[right], method="spearman")), len(values)


def _assert_joined_spearman(assertion: Mapping[str, Any], paper_data: Path) -> tuple[float, int]:
    inputs = assertion["inputs"]
    left_frame = _frame(_input_path(paper_data, inputs[0]))
    right_frame = _frame(_input_path(paper_data, inputs[1]))
    keys = [str(value) for value in assertion["join_columns"]]
    predictor, target = (str(value) for value in assertion["columns"])
    joined = left_frame[keys + [predictor]].merge(
        right_frame[keys + [target]], on=keys, how="inner", validate="one_to_one"
    )
    values = joined[[predictor, target]].apply(pd.to_numeric, errors="coerce").dropna()
    return float(values[predictor].corr(values[target], method="spearman")), len(values)


def _assert_filtered_mean(assertion: Mapping[str, Any], paper_data: Path) -> tuple[float, int]:
    frame = _frame(_input_path(paper_data, assertion["input"]))
    selected = _apply_filters(frame, assertion.get("filters", {}))
    values = pd.to_numeric(selected[str(assertion["column"])], errors="coerce").dropna()
    return float(values.mean()), len(values)


def _assert_filtered_mean_ratio(
    assertion: Mapping[str, Any], paper_data: Path
) -> tuple[float, int]:
    frame = _frame(_input_path(paper_data, assertion["input"]))
    common = _apply_filters(frame, assertion.get("common_filters", {}))
    numerator = _apply_filters(common, assertion.get("numerator_filters", {}))
    denominator = _apply_filters(common, assertion.get("denominator_filters", {}))
    column = str(assertion["column"])
    numerator_values = pd.to_numeric(numerator[column], errors="coerce").dropna()
    denominator_values = pd.to_numeric(denominator[column], errors="coerce").dropna()
    value = float(numerator_values.mean() / denominator_values.mean())
    return value, min(len(numerator_values), len(denominator_values))


def _filtered_component(spec: Mapping[str, Any], paper_data: Path) -> tuple[float, int]:
    frame = _frame(_input_path(paper_data, spec["input"]))
    selected = _apply_filters(frame, spec.get("filters", {}))
    if spec.get("operation") == "row_count":
        return float(len(selected)), len(selected)
    values = pd.to_numeric(selected[str(spec["column"])], errors="coerce").dropna()
    return float(values.mean()), len(values)


def _assert_per_unit_filtered_ratio(
    assertion: Mapping[str, Any], paper_data: Path
) -> tuple[float, int]:
    numerator, numerator_rows = _filtered_component(assertion["numerator"], paper_data)
    denominator, denominator_rows = _filtered_component(assertion["denominator"], paper_data)
    numerator_units, numerator_unit_rows = _filtered_component(
        assertion["numerator_units"], paper_data
    )
    denominator_units, denominator_unit_rows = _filtered_component(
        assertion["denominator_units"], paper_data
    )
    if denominator == 0.0 or numerator_units == 0.0 or denominator_units == 0.0:
        raise ValueError("per-unit ratio has a zero denominator")
    value = (numerator / numerator_units) / (denominator / denominator_units)
    rows = min(
        numerator_rows,
        denominator_rows,
        numerator_unit_rows,
        denominator_unit_rows,
    )
    return float(value), rows


_ASSERTION_EVALUATORS = {
    "spearman": _assert_spearman,
    "joined_spearman": _assert_joined_spearman,
    "filtered_mean": _assert_filtered_mean,
    "filtered_mean_ratio": _assert_filtered_mean_ratio,
    "per_unit_filtered_ratio": _assert_per_unit_filtered_ratio,
}


def _assertion_input_labels(assertion: Mapping[str, Any]) -> list[str]:
    labels: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if "run_id" in value and "member" in value:
                labels.add(f"{value['run_id']}:{value['member']}")
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(assertion)
    return sorted(labels)


def _confidence_intervals(
    assertion: Mapping[str, Any], paper_data: Path
) -> dict[str, dict[str, float]]:
    if "input" not in assertion:
        return {}
    frame = _frame(_input_path(paper_data, assertion["input"]))
    selected = _apply_filters(frame, assertion.get("filters", {}))
    intervals: dict[str, dict[str, float]] = {}
    for level in ("90", "95"):
        columns = assertion.get(f"ci{level}_columns")
        if not columns:
            continue
        low_column, high_column = (str(value) for value in columns)
        low = pd.to_numeric(selected[low_column], errors="coerce").dropna()
        high = pd.to_numeric(selected[high_column], errors="coerce").dropna()
        if len(low) != len(high) or low.empty:
            raise ValueError(f"invalid CI{level} columns for assertion {assertion['id']}")
        intervals[level] = {"low": float(low.mean()), "high": float(high.mean())}
    return intervals


def replay_headline_assertions(repo: Path, paper_data: Path) -> dict[str, Any]:
    """Evaluate every fully specified numerical headline assertion."""

    manifest = load_visual_manifest(repo / "paper" / "visual_manifest.yaml")
    assertion_count = sum(
        1 for asset in manifest.assets.values() for assertion in asset.headline_assertions
    )
    if assertion_count != 27:
        raise ValueError(
            f"paper numerical assertion contract requires 27 values, found {assertion_count}"
        )
    results: dict[str, Any] = {}
    for asset in manifest.assets.values():
        for assertion in asset.headline_assertions:
            assertion_id = str(assertion["id"])
            operation = str(assertion.get("operation", ""))
            evaluator = _ASSERTION_EVALUATORS.get(operation)
            if evaluator is None:
                raise ValueError(
                    f"unsupported numerical operation for {asset.asset_id}/"
                    f"{assertion_id}: {operation}"
                )
            actual, rows = evaluator(assertion, paper_data)
            expected = float(assertion["expected"])
            tolerance = float(assertion.get("tolerance", 0.0))
            expected_rows = assertion.get("expected_rows")
            row_count_matches = expected_rows is None or rows == int(expected_rows)
            passed = (
                math.isfinite(actual) and abs(actual - expected) <= tolerance and row_count_matches
            )
            result = {
                "asset_id": asset.asset_id,
                "status": "verified" if passed else "mismatch",
                "operation": operation,
                "actual": actual,
                "expected": expected,
                "tolerance": tolerance,
                "rows": rows,
                "inputs": _assertion_input_labels(assertion),
                "confidence_intervals": _confidence_intervals(assertion, paper_data),
            }
            if expected_rows is not None:
                result["expected_rows"] = int(expected_rows)
            results[assertion_id] = result
            if not passed:
                raise ValueError(
                    "headline assertion failed: "
                    f"artifact={asset.asset_id}; assertion={assertion_id}; "
                    f"inputs={result['inputs']}; expected={expected}; actual={actual}; "
                    f"tolerance={tolerance}; rows={rows}"
                )
    return results


def _asset_summaries(
    repo: Path,
    input_receipts: Iterable[Mapping[str, Any]],
    assertions: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = load_visual_manifest(repo / "paper" / "visual_manifest.yaml")
    indexed = {
        (str(item["run_id"]), str(item["requested_suffix"])): item for item in input_receipts
    }
    summaries: dict[str, Any] = {}
    for asset in manifest.assets.values():
        inputs = [
            indexed[(raw.run_id, raw.member)]
            for raw in asset.raw_inputs
            if (raw.run_id, raw.member) in indexed
        ]
        assertion_results = {
            str(spec["id"]): assertions[str(spec["id"])] for spec in asset.headline_assertions
        }
        summaries[asset.asset_id] = {
            "kind": asset.kind,
            "number": asset.number,
            "title": asset.title,
            "manifest_status": asset.status,
            "generation_contract": dict(asset.generation_contract),
            "tex_target": asset.tex_target,
            "input_count": len(inputs),
            "inputs": inputs,
            "headline_assertions": assertion_results,
            "historical_gap": asset.source_note,
        }
    return summaries


def _required_inputs(repo: Path) -> tuple[dict[str, Any], dict[str, set[str]]]:
    runs = load_reference_runs(repo / "manifests" / "reference_runs")
    visual = load_visual_manifest(repo / "paper" / "visual_manifest.yaml")
    requested: dict[str, set[str]] = defaultdict(set)
    for run in runs.values():
        requested[run.run_id].update(run.analysis_inputs)
    for asset in visual.assets.values():
        for item in asset.raw_inputs:
            if item.run_id not in runs:
                raise ReferenceError(f"unknown run in visual manifest: {item.run_id}")
            requested[item.run_id].add(item.member)
    return runs, requested


def replay_paper_data(
    output_root: str | Path,
    *,
    reference_root: str | Path | Iterable[str | Path] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify sealed archives, materialize paper data, and replay frozen assertions."""

    repo = Path(repo_root).resolve() if repo_root else repository_root()
    output = Path(output_root).resolve()
    paper_data = output / "paper_data"
    paper_data.mkdir(parents=True, exist_ok=True)
    roots = reference_roots(reference_root)
    runs, requested = _required_inputs(repo)
    run_receipts: list[dict[str, Any]] = []
    input_receipts: list[dict[str, Any]] = []
    for run_id in sorted(runs):
        run = runs[run_id]
        archive = discover_archive(run, roots)
        verify_archive(archive, run)
        materialized = materialize_inputs(run, archive, requested[run_id], paper_data)
        run_receipts.append(
            {
                "run_id": run.run_id,
                "family": run.family,
                "scientific_status": run.scientific_status,
                "archive_filename": run.archive_filename,
                "archive_sha256": run.archive_sha256,
                "archive_size_bytes": run.archive_size_bytes,
                "archive_member_count": run.archive_member_count,
                "materialized_input_count": len(materialized),
            }
        )
        input_receipts.extend(receipt_dict(item) for item in materialized)

    representatives = replay_representative_cases(repo, paper_data)
    assertions = replay_headline_assertions(repo, paper_data)
    assets = _asset_summaries(repo, input_receipts, assertions)
    receipt = {
        "schema_version": 1,
        "paper_data_directory": "paper_data",
        "runs": run_receipts,
        "inputs": input_receipts,
        "representative_cases": representatives,
        "headline_assertions": assertions,
        "assets": assets,
    }
    receipt_path = output / "replay_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "reference_runs_verified": len(run_receipts),
        "machine_readable_inputs_materialized": len(input_receipts),
        "representative_cases": representatives,
        "headline_assertions": assertions,
    }
    (output / "analysis_summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    verification = output / "verification"
    verification.mkdir(parents=True, exist_ok=True)
    (verification / "analysis_replay.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (verification / "headline_assertions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "verified",
                "assertion_count": len(assertions),
                "verified_count": sum(item["status"] == "verified" for item in assertions.values()),
                "assertions": assertions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


__all__ = [
    "replay_headline_assertions",
    "replay_paper_data",
    "replay_representative_cases",
    "select_figure_02",
    "select_figure_03",
    "select_figure_04",
]
