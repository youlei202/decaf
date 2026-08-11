"""Run every family-local sealed-reference analysis and paper adapter on CPU."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from decaf.experiments.common import (
    RunContext,
    atomic_json,
    bounded_workers,
    load_profile,
    utc_now,
)

from .reference import (
    REFERENCE_ROOT_ENV,
    sha256_file,
)
from .reference import (
    reference_roots as resolve_reference_roots,
)


class FamilyReplayError(RuntimeError):
    """Raised when a family does not reproduce its registered analysis outputs."""


FAMILY_ORDER = ("controlled", "imagenet9", "attribution", "covertype")
COPIED_PAPER_DIRECTORIES = frozenset({"reference", "reference_inputs", "source_assets"})

EXPECTED_ARTIFACTS: dict[str, frozenset[str]] = {
    "controlled": frozenset(
        {
            "metrics/controlled_assertions.json",
            "metrics/controlled_headlines.json",
            "metrics/controlled_metrics.csv",
            "paper_data/controlled/controlled_receipt.json",
            "paper_data/controlled/figure_02_false_null.csv",
            "paper_data/controlled/figure_02_matched_abs.csv",
            "paper_data/controlled/figure_03_correlations.csv",
            "paper_data/controlled/figure_03_selection.json",
            "paper_data/controlled/figure_03_trajectory.csv",
            "paper_data/controlled/figure_04_bootstrap.csv",
            "paper_data/controlled/figure_04_regimes.csv",
            "paper_data/controlled/figure_05_behavior.csv",
            "paper_data/controlled/figure_05_bootstrap.csv",
            "paper_data/controlled/figure_05_epsilon.csv",
            "paper_data/controlled/figure_05_seed.csv",
            "paper_data/controlled/figure_08_atlas.csv",
            "paper_data/controlled/figure_09_all_evidence_trajectories.csv",
            "paper_data/controlled/figure_10_geometry_transfer.csv",
            "paper_data/controlled/figure_11_calibration_transfer.csv",
        }
    ),
    "imagenet9": frozenset(
        {
            "metrics/decaf_scores.csv",
            "metrics/matched_magnitude_accuracy.json",
            "metrics/mechanism_summary.csv",
            "metrics/protocol_rank_transfer.csv",
            "metrics/protocol_ratios.csv",
            "metrics/summary.json",
            "paper_data/figure_12_robustness.csv",
            "paper_data/figure_6_matched_magnitude.csv",
            "paper_data/figure_6_mechanism_benchmark.csv",
            "paper_data/figure_7_protocol_audit.csv",
            "paper_data/figure_7_protocol_rank_transfer.csv",
            "paper_data/manifest.json",
            "paper_data/table_1_access_query_structure.csv",
        }
    ),
    "attribution": frozenset(
        {
            "metrics/attribution_headlines.json",
            "metrics/attribution_reference_audit.json",
            "metrics/bootstrap_with_m.parquet",
            "metrics/endpoint_m/source_audit.json",
            "metrics/formal_table_inputs.json",
            *(
                f"metrics/formal_tables/table_{number:02d}.csv"
                for number in (2, 3, 4, 6, 7, 8, 9, 10, 11)
            ),
            "metrics/method_results.csv",
            "metrics/pairwise_differences.csv",
            "metrics/per_model_results.csv",
            "metrics/reference_primary_quality.parquet",
            "metrics/reference_quality_summary.csv",
            "metrics/timing_summary.csv",
            "paper_data/attribution_tables.json",
            "paper_data/table_02_funnybirds_idsds_attribution.csv",
            "paper_data/table_02_funnybirds_idsds_attribution.tex",
            "paper_data/table_03_dinov2_g_stress_test.csv",
            "paper_data/table_03_dinov2_g_stress_test.tex",
            "paper_data/table_04_endpoint_m_pairwise.csv",
            "paper_data/table_04_endpoint_m_pairwise.tex",
            "paper_data/table_06_complete_cross_dataset_attribution.csv",
            "paper_data/table_06_complete_cross_dataset_attribution.tex",
            "paper_data/table_07_paired_endpoint_trajectory.csv",
            "paper_data/table_07_paired_endpoint_trajectory.tex",
            "paper_data/table_08_architecture_endpoint_ablation.csv",
            "paper_data/table_08_architecture_endpoint_ablation.tex",
            "paper_data/table_09_idsds_full50k.csv",
            "paper_data/table_09_idsds_full50k.tex",
            "paper_data/table_10_imagenet_compute.csv",
            "paper_data/table_10_imagenet_compute.tex",
            "paper_data/table_11_partimagenet_boundary.csv",
            "paper_data/table_11_partimagenet_boundary.tex",
        }
    ),
    "covertype": frozenset(
        {
            "metrics/analysis_summary.json",
            "metrics/bootstrap.csv",
            "metrics/costs.csv",
            "metrics/fixed_semantic.csv",
            "metrics/matched_magnitude.csv",
            "metrics/model_family_audit.csv",
            "metrics/model_manifest.csv",
            "metrics/model_results.csv",
            "metrics/module_c_model_decaf.csv",
            "metrics/module_f_model_decaf.csv",
            "metrics/rank_statistics.csv",
            "metrics/shap_interaction_cost_summary.csv",
            "paper_data/manifest.json",
            "paper_data/tables/bootstrap.csv",
            "paper_data/tables/costs.csv",
            "paper_data/tables/fixed_semantic.csv",
            "paper_data/tables/matched_magnitude.csv",
            "paper_data/tables/model_manifest.csv",
            "paper_data/tables/rank_statistics.csv",
            "paper_data/tables/table_5_covertype_behavior_and_cost.csv",
            "paper_data/tables/table_12_covertype_design.csv",
            "paper_data/tables/table_13_covertype_behavior_alignment.csv",
            "paper_data/tables/table_14_covertype_model_family_audit.csv",
            "paper_data/tables/table_15_fixed_semantics_and_magnitude.csv",
            "paper_data/tables/table_16_covertype_cost.csv",
        }
    ),
}

ATTRIBUTION_TABLE_ROWS = {2: 28, 3: 8, 4: 12, 6: 28, 7: 12, 8: 6, 9: 16, 10: 39, 11: 75}
CSV_ROW_CONTRACTS: dict[str, dict[str, int]] = {
    "controlled": {
        "paper_data/controlled/figure_04_bootstrap.csv": 6,
        "paper_data/controlled/figure_05_bootstrap.csv": 1134,
    },
    "imagenet9": {
        "metrics/decaf_scores.csv": 432,
        "metrics/protocol_ratios.csv": 8,
        "metrics/protocol_rank_transfer.csv": 20,
        "paper_data/figure_6_matched_magnitude.csv": 6,
        "paper_data/figure_7_protocol_rank_transfer.csv": 20,
    },
    "attribution": {
        **{
            f"metrics/formal_tables/table_{number:02d}.csv": rows
            for number, rows in ATTRIBUTION_TABLE_ROWS.items()
        },
        "metrics/pairwise_differences.csv": 12,
        **{
            path: ATTRIBUTION_TABLE_ROWS[number]
            for number, path in {
                2: "paper_data/table_02_funnybirds_idsds_attribution.csv",
                3: "paper_data/table_03_dinov2_g_stress_test.csv",
                4: "paper_data/table_04_endpoint_m_pairwise.csv",
                6: "paper_data/table_06_complete_cross_dataset_attribution.csv",
                7: "paper_data/table_07_paired_endpoint_trajectory.csv",
                8: "paper_data/table_08_architecture_endpoint_ablation.csv",
                9: "paper_data/table_09_idsds_full50k.csv",
                10: "paper_data/table_10_imagenet_compute.csv",
                11: "paper_data/table_11_partimagenet_boundary.csv",
            }.items()
        },
    },
    "covertype": {
        "metrics/bootstrap.csv": 204,
        "metrics/model_results.csv": 135,
        "metrics/module_c_model_decaf.csv": 90,
        "metrics/module_f_model_decaf.csv": 45,
        "paper_data/tables/bootstrap.csv": 204,
        "paper_data/tables/model_manifest.csv": 135,
    },
}
PARQUET_ROW_CONTRACTS: dict[str, dict[str, int]] = {
    "attribution": {
        "metrics/bootstrap_with_m.parquet": 68_000,
        "metrics/reference_primary_quality.parquet": 172_347,
    }
}


@dataclass(frozen=True)
class _FamilyAdapter:
    family: str
    module: str
    analyze: Callable[[RunContext], Mapping[str, Any] | None]
    paper: Callable[[RunContext], Mapping[str, Any] | None]


def _family_adapters() -> tuple[_FamilyAdapter, ...]:
    from decaf.experiments.attribution.analyze import analyze as attribution_analyze
    from decaf.experiments.attribution.paper import paper as attribution_paper
    from decaf.experiments.controlled.cli import analyze_handler, paper_handler
    from decaf.experiments.covertype.analyze import analyze as covertype_analyze
    from decaf.experiments.covertype.paper import paper as covertype_paper
    from decaf.experiments.imagenet9.analyze import analyze as imagenet9_analyze
    from decaf.experiments.imagenet9.paper import paper as imagenet9_paper

    return (
        _FamilyAdapter(
            "controlled", "decaf.experiments.controlled.cli", analyze_handler, paper_handler
        ),
        _FamilyAdapter(
            "imagenet9", "decaf.experiments.imagenet9.cli", imagenet9_analyze, imagenet9_paper
        ),
        _FamilyAdapter(
            "attribution",
            "decaf.experiments.attribution.cli",
            attribution_analyze,
            attribution_paper,
        ),
        _FamilyAdapter(
            "covertype", "decaf.experiments.covertype.cli", covertype_analyze, covertype_paper
        ),
    )


def _row_count(path: Path) -> int | None:
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
    if path.suffix == ".parquet":
        return int(parquet.ParquetFile(path).metadata.num_rows)
    return None


def _inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory derived outputs while excluding copied sealed-reference members."""

    rows: list[dict[str, Any]] = []
    for directory, role in (("metrics", "family_analysis"), ("paper_data", "family_paper_data")):
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            relative = path.relative_to(root)
            if (
                directory == "paper_data"
                and len(relative.parts) > 1
                and relative.parts[1] in COPIED_PAPER_DIRECTORIES
            ):
                continue
            if path.is_symlink():
                raise FamilyReplayError(
                    f"derived family artifact must not be a symlink: {relative}"
                )
            size = path.stat().st_size
            if size <= 0:
                raise FamilyReplayError(f"derived family artifact is empty: {relative}")
            row = {
                "path": relative.as_posix(),
                "role": role,
                "sha256": sha256_file(path),
                "size_bytes": size,
            }
            row_count = _row_count(path)
            if row_count is not None:
                row["row_count"] = row_count
            rows.append(row)
    return rows


def _assert_artifact_paths(family: str, inventory: Sequence[Mapping[str, Any]]) -> None:
    expected = EXPECTED_ARTIFACTS[family]
    observed = [str(row.get("path", "")) for row in inventory]
    if len(observed) != len(set(observed)):
        raise FamilyReplayError(f"{family} derived inventory contains duplicate paths")
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if missing or unexpected:
        raise FamilyReplayError(
            f"{family} derived inventory differs from the sealed contract; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _expect_fields(
    family: str,
    stage: str,
    details: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    failures = {
        name: {"expected": value, "actual": details.get(name)}
        for name, value in expected.items()
        if details.get(name) != value
    }
    if failures:
        raise FamilyReplayError(f"{family} {stage} result differs from contract: {failures}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FamilyReplayError(f"cannot read family JSON artifact {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise FamilyReplayError(f"family JSON artifact must be an object: {path.name}")
    return payload


def _validate_result_contract(
    family: str, analysis: Mapping[str, Any], paper: Mapping[str, Any]
) -> None:
    if family == "controlled":
        _expect_fields(
            family,
            "analyze",
            analysis,
            {
                "status": "verified",
                "headline_assertions": 11,
                "reference_inputs": 14,
                "input_source": "sealed_reference_archives",
                "model_inference_performed_here": False,
            },
        )
        _expect_fields(family, "paper", paper, {"artifacts": 15})
    elif family == "imagenet9":
        _expect_fields(
            family,
            "analyze",
            analysis,
            {
                "source_mode": "sealed_reference_replay",
                "score_rows": 432,
                "model_count": 72,
                "protocol_ratio_rows": 8,
                "protocol_rank_transfer_rows": 20,
                "reference_input_count": 11,
            },
        )
        accuracy = analysis.get("matched_magnitude_accuracy")
        if not isinstance(accuracy, Mapping) or accuracy.get("rows") != 8_289:
            raise FamilyReplayError("imagenet9 matched-magnitude row contract differs")
        _expect_fields(family, "paper", paper, {"paper_assets": 4})
    elif family == "attribution":
        _expect_fields(
            family,
            "analyze",
            analysis,
            {
                "source_mode": "sealed_reference_replay",
                "inference_performed": False,
                "materialized_run_ids": ["A0", "A1", "A2", "A3"],
                "primary_quality_rows": 172_347,
                "bootstrap_rows": 68_000,
                "pairwise_rows": 12,
                "partimagenet_common_support": 3_586,
                "headline_assertions": 7,
                "formal_table_count": 9,
            },
        )
        _expect_fields(
            family,
            "paper",
            paper,
            {
                "table_count": 9,
                "nonempty_tables": 9,
                "registered_tables": [2, 3, 4, 6, 7, 8, 9, 10, 11],
            },
        )
    elif family == "covertype":
        _expect_fields(
            family,
            "analyze",
            analysis,
            {
                "source_mode": "sealed_reference_replay",
                "reference_run_id": "T0",
                "reference_input_count": 13,
                "model_count": 135,
                "module_c_models": 90,
                "module_f_models": 45,
                "all_decaf_identities_passed": True,
            },
        )
        canonical = analysis.get("canonical_fragility_correlation")
        spearman = canonical.get("spearman") if isinstance(canonical, Mapping) else None
        if not isinstance(spearman, (int, float)) or not math.isclose(
            float(spearman), 0.9741138295203292, rel_tol=0.0, abs_tol=1e-14
        ):
            raise FamilyReplayError("covertype canonical fragility correlation differs")
        _expect_fields(
            family,
            "paper",
            paper,
            {"table_count": 6, "machine_readable_files": 12, "formal_model_count": 135},
        )
    else:  # pragma: no cover - guarded by FAMILY_ORDER
        raise FamilyReplayError(f"unknown family replay contract: {family}")


def _validate_file_contract(family: str, root: Path) -> dict[str, Any]:
    observed_rows: dict[str, int] = {}
    for relative, expected in CSV_ROW_CONTRACTS.get(family, {}).items():
        actual = _row_count(root / relative)
        if actual != expected:
            raise FamilyReplayError(
                f"{family} row count differs for {relative}: expected {expected}, found {actual}"
            )
        observed_rows[relative] = expected
    for relative, expected in PARQUET_ROW_CONTRACTS.get(family, {}).items():
        actual = _row_count(root / relative)
        if actual != expected:
            raise FamilyReplayError(
                f"{family} row count differs for {relative}: expected {expected}, found {actual}"
            )
        observed_rows[relative] = expected

    if family == "attribution":
        audit = _read_json(root / "metrics/attribution_reference_audit.json")
        endpoint = _read_json(root / "metrics/endpoint_m/source_audit.json")
        formal = _read_json(root / "metrics/formal_table_inputs.json")
        tables = _read_json(root / "paper_data/attribution_tables.json")
        partimagenet = audit.get("partimagenet")
        if (
            audit.get("status") != "passed"
            or not isinstance(partimagenet, Mapping)
            or partimagenet.get("included_rows") != 3_586
            or endpoint.get("passed") is not True
        ):
            raise FamilyReplayError("attribution audit/common-support contract differs")
        expected_rows = ATTRIBUTION_TABLE_ROWS
        formal_rows = {
            int(row.get("table", -1)): int(row.get("rows", -1))
            for row in formal.get("tables", [])
            if isinstance(row, Mapping)
        }
        paper_rows = {
            int(row.get("table", -1)): int(row.get("rows", -1))
            for row in tables.get("tables", [])
            if isinstance(row, Mapping) and row.get("schema_only") is False
        }
        if formal_rows != expected_rows or paper_rows != expected_rows:
            raise FamilyReplayError("attribution formal table manifest differs")
    elif family == "covertype":
        manifest = _read_json(root / "paper_data/manifest.json")
        registered = manifest.get("tables")
        expected_tables = sorted(
            path.removeprefix("paper_data/tables/")
            for path in EXPECTED_ARTIFACTS[family]
            if path.startswith("paper_data/tables/")
        )
        if (
            registered != expected_tables
            or manifest.get("source_model_count") != 135
            or manifest.get("formal_model_count") != 135
        ):
            raise FamilyReplayError("covertype six-table manifest differs")
    return {"status": "passed", "row_counts": dict(sorted(observed_rows.items()))}


def _validate_family(
    family: str,
    root: Path,
    analysis: Mapping[str, Any],
    paper: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _assert_artifact_paths(family, inventory)
    _validate_result_contract(family, analysis, paper)
    result = _validate_file_contract(family, root)
    return {
        **result,
        "artifact_count": len(inventory),
        "expected_artifact_count": len(EXPECTED_ARTIFACTS[family]),
    }


def _context(repo: Path, output: Path, experiment: str) -> RunContext:
    config = load_profile(experiment, "paper", repo / "configs" / experiment / "paper.yaml")
    return RunContext.create(
        experiment=experiment,
        profile="paper",
        stage="analysis-replay",
        output=output,
        config=config,
        workers=bounded_workers(config.get("max_workers")),
        resume=False,
    )


def _stage_command(
    adapter: _FamilyAdapter,
    stage: str,
    context_path: Path,
    output_root: Path,
) -> list[str]:
    relative = context_path.relative_to(output_root).as_posix()
    command = [
        "python",
        "-m",
        adapter.module,
        "--stage",
        stage,
        "--profile",
        "paper",
        "--output",
        f"<replay-root>/{relative}",
    ]
    if stage == "paper":
        command.append("--resume")
    return command


def _run_stage(
    context: RunContext,
    stage: str,
    handler: Callable[[RunContext], Mapping[str, Any] | None],
    command: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = utc_now()
    started_clock = time.monotonic()
    try:
        details = dict(handler(context) or {})
    except Exception as error:
        context.record_stage(
            stage,
            "failed",
            started_at=started_at,
            details={"error": f"{type(error).__name__}: {error}"},
        )
        context.set_status("failed", completed_stages=[], error=str(error))
        raise
    details["elapsed_seconds"] = round(time.monotonic() - started_clock, 6)
    context.record_stage(stage, "completed", started_at=started_at, details=details)
    return details, {
        "stage": stage,
        "status": "completed",
        "execution": "in_process_registered_cli_handler",
        "command": list(command),
        "result": details,
    }


def _new_invocation_root(container: Path) -> Path:
    container.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="invocation-", dir=container)).resolve()


def _validate_recorded_inventory(root: Path, recorded: Sequence[Mapping[str, Any]]) -> None:
    current = _inventory(root)
    if current != [dict(row) for row in recorded]:
        raise FamilyReplayError(f"derived artifact inventory drifted after replay: {root.name}")


def _validate_receipt_payload(output: Path, receipt: Mapping[str, Any]) -> None:
    if receipt.get("status") != "completed" or receipt.get("family_count") != 4:
        raise FamilyReplayError("family replay receipt is not a completed four-family replay")
    invocation_value = receipt.get("invocation_path")
    if not isinstance(invocation_value, str):
        raise FamilyReplayError("family replay receipt has no portable invocation path")
    invocation = (output / invocation_value).resolve()
    if not invocation.is_relative_to(output) or not invocation.is_dir():
        raise FamilyReplayError("family replay invocation path escapes or is absent")
    rows = receipt.get("families")
    if not isinstance(rows, list) or [
        row.get("family") for row in rows if isinstance(row, Mapping)
    ] != list(FAMILY_ORDER):
        raise FamilyReplayError("family replay receipt family order differs")
    if set(path.name for path in invocation.iterdir()) != set(FAMILY_ORDER):
        raise FamilyReplayError(
            "family replay invocation contains stale or missing family directories"
        )
    for row in rows:
        if not isinstance(row, Mapping):
            raise FamilyReplayError("family replay receipt contains a malformed family row")
        family = str(row["family"])
        expected_path = f"{invocation_value}/{family}"
        if row.get("path") != expected_path or row.get("status") != "completed":
            raise FamilyReplayError(f"{family} replay receipt path/status differs")
        family_root = (output / expected_path).resolve()
        if not family_root.is_relative_to(invocation):
            raise FamilyReplayError(f"{family} replay path escapes its invocation")
        artifacts = row.get("artifacts")
        analysis = row.get("analysis")
        paper = row.get("paper")
        if (
            not isinstance(artifacts, list)
            or not isinstance(analysis, Mapping)
            or not isinstance(paper, Mapping)
        ):
            raise FamilyReplayError(f"{family} replay receipt is malformed")
        _validate_recorded_inventory(family_root, artifacts)
        contract = _validate_family(family, family_root, analysis, paper, artifacts)
        if row.get("contract") != contract:
            raise FamilyReplayError(f"{family} recorded contract summary drifted")


def validate_family_replay_receipt(
    output_root: str | Path, receipt: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Re-hash and revalidate a completed four-family replay receipt."""

    output = Path(output_root).resolve()
    if receipt is None:
        receipt = _read_json(output / "family_replays/family_replay_receipt.json")
    _validate_receipt_payload(output, receipt)
    return dict(receipt)


def replay_family_adapters(
    output_root: str | Path,
    *,
    repo_root: str | Path,
    reference_roots: Sequence[str | Path],
) -> dict[str, Any]:
    """Execute all registered CPU analysis+paper paths against sealed archives.

    Each call owns a new invocation directory and each family uses a non-resumed
    run directory.  Exact outputs and scientific cardinalities are checked before
    the completed receipt is published, so prior files can never mask missing work.
    """

    repo = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    container = output / "family_replays"
    invocation = _new_invocation_root(container)
    invocation_relative = invocation.relative_to(output).as_posix()
    receipt_path = container / "family_replay_receipt.json"
    base_receipt: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "invocation_path": invocation_relative,
        "family_count": 0,
        "families": [],
    }
    atomic_json(receipt_path, base_receipt)
    receipts: list[dict[str, Any]] = []
    active_family: str | None = None
    previous = os.environ.get(REFERENCE_ROOT_ENV)
    try:
        roots = resolve_reference_roots(reference_roots)
        os.environ[REFERENCE_ROOT_ENV] = os.pathsep.join(str(root) for root in roots)
        for adapter in _family_adapters():
            active_family = adapter.family
            context = _context(repo, invocation / adapter.family, adapter.family)
            analysis, analyze_stage = _run_stage(
                context,
                "analyze",
                adapter.analyze,
                _stage_command(adapter, "analyze", context.path, output),
            )
            paper_details, paper_stage = _run_stage(
                context,
                "paper",
                adapter.paper,
                _stage_command(adapter, "paper", context.path, output),
            )
            inventory = _inventory(context.path)
            contract = _validate_family(
                adapter.family, context.path, analysis, paper_details, inventory
            )
            context.set_status("completed", completed_stages=["analyze", "paper"])
            receipts.append(
                {
                    "family": adapter.family,
                    "path": context.path.relative_to(output).as_posix(),
                    "status": "completed",
                    "stages": [analyze_stage, paper_stage],
                    "analysis": analysis,
                    "paper": paper_details,
                    "contract": contract,
                    "artifacts": inventory,
                }
            )
        receipt = {
            "schema_version": 2,
            "status": "completed",
            "invocation_path": invocation_relative,
            "family_count": len(receipts),
            "families": receipts,
        }
        _validate_receipt_payload(output, receipt)
        atomic_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        atomic_json(
            receipt_path,
            {
                **base_receipt,
                "status": "failed",
                "family_count": len(receipts),
                "families": receipts,
                "failed_family": active_family,
                "error_type": type(error).__name__,
            },
        )
        raise
    finally:
        if previous is None:
            os.environ.pop(REFERENCE_ROOT_ENV, None)
        else:
            os.environ[REFERENCE_ROOT_ENV] = previous


__all__ = [
    "FamilyReplayError",
    "replay_family_adapters",
    "validate_family_replay_receipt",
]
