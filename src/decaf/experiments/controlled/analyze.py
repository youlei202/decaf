"""Frozen-schema adapters and controlled-family analyses."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.core.manifests import atomic_write_json
from decaf.paper.analysis_replay import select_figure_02, select_figure_03, select_figure_04
from decaf.paper.manifest import repository_root
from decaf.paper.reference import (
    discover_archive,
    load_reference_runs,
    materialize_inputs,
    receipt_dict,
    reference_roots,
    verify_archive,
)


@dataclass(frozen=True, slots=True)
class TableSchema:
    """Minimum frozen table contract used at the C0/C1/C2 boundary."""

    columns: frozenset[str]
    rows: int | None = None
    unique: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ControlledReferenceBundle:
    """Validated machine-readable tables for controlled Figures 2--5 and 8--11."""

    frames: Mapping[str, pd.DataFrame]
    sources: Mapping[str, Path]

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.frames[name]


SCHEMAS: Mapping[str, TableSchema] = {
    "c0_response": TableSchema(
        frozenset(
            {
                "model_id",
                "task",
                "architecture",
                "model_seed",
                "factor",
                "mean_auc_abs",
                "mean_auc_align",
                "mean_auc_opp",
                "mean_auc_null",
                "endpoint_class",
                "is_intended",
            }
        ),
        rows=180,
        unique=("model_id", "factor"),
    ),
    "c0_methods": TableSchema(
        frozenset({"method", "false_null_order", "coverage", "stable_edge_precision"}),
        rows=9,
        unique=("method",),
    ),
    "c0_false_null": TableSchema(
        frozenset(
            {
                "model_id",
                "task",
                "architecture",
                "model_seed",
                "method",
                "supported_factor",
                "null_factor",
                "false_null_order",
            }
        ),
        rows=1575,
    ),
    "c1_aggregate": TableSchema(
        frozenset(
            {
                "model_id",
                "module",
                "factor",
                "variant",
                "architecture",
                "seed",
                "p_train",
                "epoch",
                "trajectory_id",
                "V_rev",
                "geometry",
                "primary_geometry",
                "E",
                "C",
                "F",
                "Abs",
                "null_prediction_change_rate",
                "confidence_fragility",
            }
        ),
        rows=948,
        unique=("model_id", "factor", "geometry"),
    ),
    "c1_stages": TableSchema(
        frozenset(
            {
                "model_id",
                "module",
                "factor",
                "variant",
                "architecture",
                "seed",
                "p_train",
                "epoch",
                "trajectory_id",
                "V_rev",
                "geometry",
                "primary_geometry",
                "noise_seed",
                "counterfactual_map",
                "stage_index",
                "alpha",
                "U_abs",
                "U_align",
                "U_opp",
                "U_null",
            }
        ),
        rows=159264,
    ),
    "c1_bootstrap": TableSchema(
        frozenset(
            {
                "model_id",
                "module",
                "variant",
                "architecture",
                "factor",
                "geometry",
                "sample_id",
                "seed",
                "E",
                "C",
                "F",
                "Abs",
            }
        ),
        rows=647168,
    ),
    "c1_e_correlations": TableSchema(
        frozenset(
            {
                "metric",
                "definition",
                "n",
                "pooled_spearman",
                "within_trajectory_mean_spearman",
                "estimable_trajectories",
            }
        ),
        rows=5,
        unique=("metric",),
    ),
    "c1_f_validation": TableSchema(
        frozenset({"section", "variant", "architecture", "F_mean", "axis", "behavior", "spearman"}),
        rows=21,
    ),
    "c1_geometry": TableSchema(
        frozenset({"geometry", "metric", "n_pairs", "spearman_vs_cmmr"}),
        rows=25,
        unique=("geometry", "metric"),
    ),
    "c2_epsilon": TableSchema(
        frozenset(
            {
                "model_id",
                "task",
                "architecture",
                "seed",
                "wall_map",
                "epsilon",
                "E",
                "C",
                "F",
                "Abs",
                "Net",
                "phi_C",
            }
        ),
        rows=1260,
        unique=("model_id", "wall_map", "epsilon"),
    ),
    "c2_behavior": TableSchema(
        frozenset(
            {
                "model_id",
                "task",
                "architecture",
                "seed",
                "wall_map",
                "epsilon",
                "preserve_rate",
                "pairwise_swap_rate",
                "collapse_rate",
            }
        ),
        rows=1260,
        unique=("model_id", "wall_map", "epsilon"),
    ),
    "c2_models": TableSchema(
        frozenset(
            {
                "model_id",
                "task",
                "architecture",
                "seed",
                "test_pairs",
                "endpoint_accuracy",
                "balanced_clean_accuracy",
                "swapped_context_prediction_change",
                "map_semantics_valid",
            }
        ),
        rows=30,
        unique=("model_id",),
    ),
    "c2_bootstrap": TableSchema(
        frozenset(
            {"task", "architecture", "epsilon", "metric", "estimate", "ci_low", "ci_high", "models"}
        ),
        rows=1134,
        unique=("task", "architecture", "epsilon", "metric"),
    ),
    "c2_seeds": TableSchema(
        frozenset({"task", "architecture", "seed", "E", "C", "F", "Abs", "Net", "phi_C"}),
        rows=30,
        unique=("task", "architecture", "seed"),
    ),
}

REFERENCE_MEMBERS: Mapping[str, Mapping[str, str]] = {
    "C0": {
        "c0_response": "benchmark/response_contamination.csv",
        "c0_methods": "benchmark/method_metrics.csv",
        "c0_false_null": "benchmark/null_false_evidence.csv",
    },
    "C1": {
        "c1_aggregate": "results/endpoint_behavior_v1_measurement/aggregate_summary.parquet",
        "c1_stages": "results/endpoint_behavior_v1_measurement/stage_summary.parquet",
        "c1_bootstrap": "results/endpoint_behavior_v1_measurement/per_sample_bootstrap.parquet",
        "c1_e_correlations": (
            "results/endpoint_behavior_v1_measurement/tables/"
            "T03_module_e_correlations.csv"
        ),
        "c1_f_validation": (
            "results/endpoint_behavior_v1_measurement/tables/"
            "T05_module_f_validation.csv"
        ),
        "c1_geometry": "results/endpoint_behavior_v1_measurement/tables/T07_geometry_transfer.csv",
    },
    "C2": {
        "c2_epsilon": "evaluation/epsilon_curves.csv",
        "c2_behavior": "evaluation/behavior_rates.csv",
        "c2_models": "evaluation/model_level_results.csv",
        "c2_bootstrap": "bootstrap/bootstrap_summary.csv",
        "c2_seeds": "tables/T06_seed_results.csv",
    },
}


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"controlled reference input is missing: {path}")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def validate_frame(
    frame: pd.DataFrame,
    schema: TableSchema,
    *,
    label: str,
    strict_cardinality: bool = True,
) -> pd.DataFrame:
    """Validate required columns, row count, unique keys, and finite metrics."""

    missing = schema.columns - set(frame)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if strict_cardinality and schema.rows is not None and len(frame) != schema.rows:
        raise ValueError(f"{label} row count changed: expected {schema.rows}, found {len(frame)}")
    if schema.unique and frame.duplicated(list(schema.unique)).any():
        raise ValueError(f"{label} has duplicate keys: {schema.unique}")
    return frame


def load_reference_bundle(
    paper_data_root: str | Path,
    *,
    strict_cardinality: bool = True,
) -> ControlledReferenceBundle:
    """Load already materialized C0/C1/C2 inputs through frozen adapters."""

    root = Path(paper_data_root)
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, Path] = {}
    for run_id, members in REFERENCE_MEMBERS.items():
        run_root = root / run_id if (root / run_id).is_dir() else root
        for name, relative in members.items():
            path = run_root / relative
            frames[name] = validate_frame(
                _read_frame(path),
                SCHEMAS[name],
                label=name,
                strict_cardinality=strict_cardinality,
            )
            sources[name] = path.resolve()
    _validate_cross_table_keys(frames, strict_cardinality=strict_cardinality)
    return ControlledReferenceBundle(frames, sources)


def _validate_cross_table_keys(
    frames: Mapping[str, pd.DataFrame],
    *,
    strict_cardinality: bool,
) -> None:
    left = frames["c2_epsilon"]
    right = frames["c2_behavior"]
    keys = ["model_id", "task", "architecture", "seed", "wall_map", "epsilon"]
    joined = left[keys].merge(right[keys], how="outer", on=keys, indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("C2 epsilon and behavior grids are not one-to-one aligned")
    if strict_cardinality:
        aggregate = frames["c1_aggregate"]
        primary = aggregate.loc[
            aggregate["primary_geometry"].astype(str).str.lower().isin({"1", "true", "yes"})
        ]
        counts = primary.groupby("module")["model_id"].nunique().to_dict()
        if counts != {"C": 18, "E": 52, "F": 18}:
            raise ValueError(f"C1 primary checkpoint counts changed: {counts}")


def materialize_controlled_references(
    destination: str | Path,
    *,
    reference_root: str | Path | Iterable[str | Path] | None = None,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Verify sealed C0/C1/C2 archives and materialize only required members."""

    repo = Path(repo_root).resolve() if repo_root else repository_root()
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = load_reference_runs(repo / "manifests" / "reference_runs")
    roots = reference_roots(reference_root)
    receipts: list[dict[str, Any]] = []
    for run_id in ("C0", "C1", "C2"):
        run = runs[run_id]
        archive = discover_archive(run, roots)
        verify_archive(archive, run)
        members = set(REFERENCE_MEMBERS[run_id].values())
        materialized = materialize_inputs(run, archive, members, output)
        receipts.extend(receipt_dict(item) for item in materialized)
    return receipts


def summarize_headlines(bundle: ControlledReferenceBundle) -> dict[str, Any]:
    """Recompute the registered Controlled headline values from sealed rows."""

    figure_02 = select_figure_02(bundle["c0_response"])
    methods = bundle["c0_methods"].set_index("method")
    e_row = bundle["c1_e_correlations"].loc[bundle["c1_e_correlations"]["metric"].eq("E_margin")]
    if len(e_row) != 1:
        raise ValueError("C1 evidence table must contain one E_margin row")
    f_table = bundle["c1_f_validation"]
    f_rows = f_table.loc[
        f_table["section"].eq("variant_summary") & f_table["architecture"].eq("small_vit")
    ]
    regimes = {
        variant: float(f_rows.loc[f_rows["variant"].eq(variant), "F_mean"].iloc[0])
        for variant in ("robust", "neutral", "fragile")
    }
    c2_keys = ["model_id", "task", "architecture", "seed", "wall_map", "epsilon"]
    c2 = bundle["c2_epsilon"][c2_keys + ["C", "Abs"]].merge(
        bundle["c2_behavior"][c2_keys + ["pairwise_swap_rate"]],
        on=c2_keys,
        how="inner",
        validate="one_to_one",
    )
    seed_rows = bundle["c2_seeds"]
    return {
        "figure_02": {
            "intended_abs": figure_02["intended_factor"]["mean_auc_abs"],
            "endpoint_null_abs": figure_02["endpoint_null_factor"]["mean_auc_abs"],
            "abs_cmmr_false_null": float(methods.loc["Abs-CMMR", "false_null_order"]),
            "align_cmmr_false_null": float(methods.loc["Align-CMMR", "false_null_order"]),
            "representative": figure_02,
        },
        "figure_03": {
            "evidence_correspondence": float(e_row.iloc[0]["pooled_spearman"]),
            "representative": select_figure_03(bundle["c1_stages"]),
        },
        "figure_04": {
            "fragility_regimes": regimes,
            "representative": select_figure_04(bundle["c1_aggregate"]),
        },
        "figure_05": {
            "invert_c": float(seed_rows.loc[seed_rows["task"].eq("invert"), "C"].mean()),
            "c_swap_spearman": float(c2["C"].corr(c2["pairwise_swap_rate"], method="spearman")),
            "abs_swap_spearman": float(c2["Abs"].corr(c2["pairwise_swap_rate"], method="spearman")),
        },
    }


def assert_headline_targets(
    summary: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply config-owned tolerances to every named scalar headline."""

    actual = {
        "figure_02_intended_abs": summary["figure_02"]["intended_abs"],
        "figure_02_endpoint_null_abs": summary["figure_02"]["endpoint_null_abs"],
        "figure_02_abs_false_null": summary["figure_02"]["abs_cmmr_false_null"],
        "figure_02_align_false_null": summary["figure_02"]["align_cmmr_false_null"],
        "figure_03_evidence_correspondence": summary["figure_03"]["evidence_correspondence"],
        "figure_04_robust_f": summary["figure_04"]["fragility_regimes"]["robust"],
        "figure_04_neutral_f": summary["figure_04"]["fragility_regimes"]["neutral"],
        "figure_04_fragile_f": summary["figure_04"]["fragility_regimes"]["fragile"],
        "figure_05_invert_c": summary["figure_05"]["invert_c"],
        "figure_05_c_swap": summary["figure_05"]["c_swap_spearman"],
        "figure_05_abs_swap": summary["figure_05"]["abs_swap_spearman"],
    }
    results: dict[str, Any] = {}
    for name, target in targets.items():
        if name not in actual:
            raise ValueError(f"unknown controlled headline target: {name}")
        expected = float(target["expected"])
        tolerance = float(target["tolerance"])
        observed = float(actual[name])
        passed = bool(np.isfinite(observed) and abs(observed - expected) <= tolerance)
        results[name] = {
            "actual": observed,
            "expected": expected,
            "tolerance": tolerance,
            "status": "verified" if passed else "mismatch",
        }
        if not passed:
            raise ValueError(f"controlled headline assertion failed: {name}: {results[name]}")
    return results


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def analyze_reference_bundle(
    bundle: ControlledReferenceBundle,
    output: str | Path,
    *,
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Write normalized metrics and verified headline assertions."""

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    summary = summarize_headlines(bundle)
    assertions = assert_headline_targets(summary, targets)
    atomic_write_json(destination / "controlled_headlines.json", summary)
    atomic_write_json(destination / "controlled_assertions.json", assertions)
    rows = [
        {
            "experiment": "controlled",
            "metric": name,
            "value": result["actual"],
            "status": result["status"],
        }
        for name, result in sorted(assertions.items())
    ]
    _atomic_csv(pd.DataFrame(rows), destination / "controlled_metrics.csv")
    return {"headline_assertions": len(assertions), "status": "verified"}


def analyze_smoke(raw_root: str | Path, output: str | Path) -> dict[str, Any]:
    """Aggregate tiny score-oracle JSON outputs without paper-scale claims."""

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(raw_root).rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        for metric, value in metrics.items():
            array = np.asarray(value, dtype=np.float64)
            rows.append(
                {
                    "experiment": "controlled",
                    "model_id": payload.get("model_id", "unknown"),
                    "metric": metric,
                    "value": float(np.mean(array)),
                    "n_values": int(array.size),
                    "gpu_verification": "pending",
                }
            )
    if not rows:
        raise ValueError("controlled smoke analysis found no score-oracle outputs")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(rows), destination / "controlled_smoke_metrics.csv")
    summary = {
        "schema_version": 1,
        "status": "completed",
        "rows": len(rows),
        "scope": "cpu_score_oracle",
        "gpu_real_shard_verification": "pending",
    }
    atomic_write_json(destination / "controlled_smoke_summary.json", summary)
    return summary


__all__ = [
    "ControlledReferenceBundle",
    "REFERENCE_MEMBERS",
    "SCHEMAS",
    "TableSchema",
    "analyze_reference_bundle",
    "analyze_smoke",
    "assert_headline_targets",
    "load_reference_bundle",
    "materialize_controlled_references",
    "summarize_headlines",
    "validate_frame",
]
