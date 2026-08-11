"""Asset-semantic canonical paper data derived from family replay outputs.

The global renderer is intentionally unable to inspect arbitrary raw columns.
Every non-missing paper asset is first converted by an asset-specific contract
to this module's frozen portable schema, then validated and rendered from that
canonical file only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.experiments.common import atomic_json, atomic_text

from .manifest import VisualAsset, load_visual_manifest
from .reference import sha256_file

CANONICAL_COLUMNS = (
    "artifact_id",
    "panel_id",
    "series",
    "x",
    "y",
    "estimate",
    "ci_low",
    "ci_high",
    "n",
    "source_sha256",
    "record_json",
)
CANONICAL_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "columns": CANONICAL_COLUMNS,
            "null_policy": {
                "ci_low": "nullable",
                "ci_high": "nullable",
                "y": "nullable only for explicitly non-estimable source rows",
                "estimate": "nullable only for explicitly non-estimable source rows",
            },
            "record_json": "canonical JSON object preserving semantic identity fields",
            "schema_version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


class SemanticDataError(RuntimeError):
    """Raised when an asset cannot satisfy its registered semantic contract."""


@dataclass(frozen=True)
class RawInputs:
    """Hash-bound index of members materialized by sealed replay."""

    root: Path
    items: Mapping[tuple[str, str], Mapping[str, Any]]


@dataclass(frozen=True)
class FamilyOutputs:
    """Hash-bound index of derived outputs from one family adapter."""

    root: Path
    artifacts: Mapping[str, Mapping[str, Any]]


_FIGURE_SPECS: dict[str, dict[str, Any]] = {
    "figure_02": {
        "operation": "fixed task/factor matched grid plus method false-null ordering",
        "panels": {"matched_abs": 12, "false_null_order": 2, "false_null_evidence": 1},
        "representative_case_ids": ["figure_02"],
        "requirements": [
            (
                "task=object_shape; factors={object_shape,floor_color}; "
                "identical architecture/seed grid"
            ),
            "methods={Abs-CMMR,Align-CMMR}",
            "group null_false_evidence by method without row sampling",
        ],
    },
    "figure_03": {
        "operation": "Module-E correlations and deterministic maximum-V_rev-range trajectory",
        "panels": {
            "evidence_correspondence": 5,
            "representative_trajectory": 1,
            "representative_checkpoints": 6,
        },
        "representative_case_ids": ["figure_03"],
        "requirements": [
            "metric correlations from T03_module_e_correlations",
            (
                "trajectory_id/architecture/training_correlation/seed from "
                "representative_cases receipt"
            ),
            "primary-geometry Module-E stage and aggregate rows only",
        ],
    },
    "figure_04": {
        "operation": (
            "Module-F regime association, model bootstrap CI, and frozen representative model"
        ),
        "panels": {"fragility_regimes": 6, "model_bootstrap": 6, "representative_fragility": 1},
        "representative_case_ids": ["figure_04"],
        "requirements": [
            "section=variant_summary",
            "bootstrap over primary-geometry floor_color model means",
            "representative model_id and geometry from representative_cases receipt",
        ],
    },
    "figure_05": {
        "operation": (
            "joined contradiction calibration, behavior, seed, bootstrap CI, and model transfer"
        ),
        "panels": {
            "epsilon_curves": 1,
            "behavior": 1,
            "seed_results": 1,
            "bootstrap": 1,
            "model_transfer": 1,
        },
        "requirements": [
            "epsilon/behavior one-to-one join on model_id,task,architecture,seed,wall_map,epsilon",
            "C and Abs against pairwise_swap_rate",
            "retain registered bootstrap confidence bounds and all seed rows",
        ],
    },
    "figure_06": {
        "operation": "four registered ImageNet-9 mechanism panels",
        "panels": {
            "fixed_semantics": 11,
            "matched_magnitude": 6,
            "per_bin_auroc": 1,
            "heldout_architecture_family": 10,
        },
        "requirements": [
            "cross-check fixed_semantic against T04 fixed_semantic rows",
            "bind matched_abs summary to matched_pairs cardinality",
            "retain every per-bin row including non-estimable reason",
            "leave-one-architecture-family-out summary only",
        ],
    },
    "figure_07": {
        "operation": "recomputed patch/blend ratios and model-rank transfer with ledger coverage",
        "panels": {"patch_blend_ratios": 8, "rank_transfer": 20, "stage_coverage": 1},
        "requirements": [
            "epsilon=0.02 filtered means by pair_type/path",
            "patch_A and patch_B divided by blend for Abs and F",
            "Spearman model ranking transfer for M,E,C,F,Abs",
            "model_decaf/profile/sample identities validated by family analyzer",
        ],
    },
    "figure_08": {
        "operation": "complete controlled task/architecture/seed/factor atlas",
        "panels": {"controlled_atlas": 180},
        "requirements": ["retain all C0 response_contamination cells without sampling"],
    },
    "figure_09": {
        "operation": "all primary-geometry Module-E stage and checkpoint trajectories",
        "panels": {"all_evidence_trajectories": 1, "checkpoint_summary": 1},
        "requirements": [
            "module=E",
            "primary_geometry=true",
            "retain every trajectory_id/factor/epoch",
        ],
    },
    "figure_10": {
        "operation": "cross-geometry transfer relative to CMMR",
        "panels": {"geometry_transfer": 25},
        "requirements": ["complete 5 geometries x 5 metrics grid", "n_pairs and Spearman retained"],
    },
    "figure_11": {
        "operation": "contradiction calibration and registered wall-map transfer",
        "panels": {"calibration_transfer": 1, "model_transfer": 30, "seed_results": 30},
        "requirements": [
            "mean epsilon curves by task,architecture,wall_map,epsilon",
            "retain all model-level transfer and seed-level rows",
        ],
    },
    "figure_12": {
        "operation": "ImageNet-9 model/sample robustness with terminal ledger coverage",
        "panels": {"model_robustness": 1, "sample_robustness": 1, "stage_coverage": 1},
        "requirements": [
            "aggregate profiles by architecture_family/training_regime",
            "aggregate sample F and endpoint-active rate by registered protocol",
            "retain every stage-ledger terminal record",
        ],
    },
}

_TABLE_SPECS: dict[str, dict[str, Any]] = {
    "table_01": {
        "family": "imagenet9",
        "path": "paper_data/table_1_access_query_structure.csv",
        "operation": "generate method access/query structure from registered config",
        "required_columns": ["method_id", "access", "nominal_queries", "requires_gradients"],
        "display_columns": ["method_id", "access", "nominal_queries", "requires_gradients"],
        "estimate_column": "nominal_queries",
        "minimum_rows": 6,
    },
    "table_02": {
        "family": "attribution",
        "path": "metrics/formal_tables/table_02.csv",
        "operation": "join quality, timing, memory, and query summaries by registered method",
        "required_columns": [
            "dataset",
            "method",
            "mean",
            "ci90_low",
            "ci90_high",
            "ci95_low",
            "ci95_high",
        ],
        "display_columns": [
            "dataset",
            "method",
            "mean",
            "ci90_low",
            "ci90_high",
            "ci95_low",
            "ci95_high",
        ],
        "estimate_column": "mean",
        "minimum_rows": 28,
    },
    "table_03": {
        "family": "attribution",
        "path": "metrics/formal_tables/table_03.csv",
        "operation": "validated one-to-one DINOv2-g quality/timing join",
        "required_columns": [
            "dataset",
            "model",
            "method",
            "common_support_spearman",
            "wall_seconds_per_image",
            "peak_allocated_bytes",
        ],
        "display_columns": [
            "method",
            "common_support_spearman",
            "wall_seconds_per_image",
            "peak_allocated_gib",
            "forward_rows_per_image",
        ],
        "estimate_column": "common_support_spearman",
        "minimum_rows": 8,
    },
    "table_04": {
        "family": "attribution",
        "path": "metrics/formal_tables/table_04.csv",
        "operation": "recompute paired trajectory-minus-endpoint contrasts",
        "required_columns": [
            "dataset",
            "left_method",
            "right_method",
            "mean_paired_difference",
            "ci90_low",
            "ci90_high",
            "ci95_low",
            "ci95_high",
        ],
        "display_columns": [
            "dataset",
            "left_method",
            "right_method",
            "mean_paired_difference",
            "ci95_low",
            "ci95_high",
        ],
        "estimate_column": "mean_paired_difference",
        "minimum_rows": 12,
    },
    "table_05": {
        "family": "covertype",
        "path": "paper_data/tables/table_5_covertype_behavior_and_cost.csv",
        "operation": "join behavior ranks, bootstrap uncertainty, and normalized method cost",
        "required_columns": [
            "section",
            "method",
            "estimate",
            "spearman",
            "spearman_ci90_low",
            "spearman_ci90_high",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "bootstrap_repetitions",
            "wall_seconds",
            "shap_interaction_relative_cost",
        ],
        "display_columns": [
            "section",
            "method",
            "module",
            "outcome",
            "estimate",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "wall_seconds",
            "shap_interaction_relative_cost",
        ],
        "estimate_column": "estimate",
        "minimum_rows": 66,
    },
    "table_06": {
        "family": "attribution",
        "path": "metrics/formal_tables/table_06.csv",
        "operation": "complete A2/A3 quality join across datasets and methods",
        "required_columns": [
            "dataset",
            "method",
            "mean",
            "ci90_low",
            "ci90_high",
            "ci95_low",
            "ci95_high",
        ],
        "display_columns": [
            "dataset",
            "method",
            "mean",
            "ci90_low",
            "ci90_high",
            "ci95_low",
            "ci95_high",
        ],
        "estimate_column": "mean",
        "minimum_rows": 28,
    },
    "table_07": {
        "family": "attribution",
        "path": "metrics/formal_tables/table_07.csv",
        "operation": "paired endpoint-versus-trajectory contrasts",
        "required_columns": [
            "dataset",
            "left_method",
            "right_method",
            "mean_paired_difference",
            "ci95_low",
            "ci95_high",
        ],
        "display_columns": [
            "dataset",
            "left_method",
            "right_method",
            "mean_paired_difference",
            "ci95_low",
            "ci95_high",
        ],
        "estimate_column": "mean_paired_difference",
        "minimum_rows": 12,
    },
    "table_08": {
        "family": "attribution",
        "path": "metrics/formal_tables/table_08.csv",
        "operation": "aggregate endpoint ablation by registered architecture",
        "required_columns": ["dataset", "model", "Endpoint M"],
        "display_columns": [
            "dataset",
            "model",
            "Endpoint M",
            "DECAF-3",
            "DECAF-5",
            "DECAF-9",
            "n_images",
        ],
        "estimate_column": "Endpoint M",
        "minimum_rows": 6,
    },
    "table_09": {
        "family": "attribution",
        "path": "metrics/formal_tables/table_09.csv",
        "operation": "join full-50k scale check with Endpoint-M results",
        "required_columns": [
            "dataset",
            "level",
            "method",
            "mean",
            "ci95_low",
            "ci95_high",
            "n_images",
        ],
        "display_columns": [
            "level",
            "model",
            "method",
            "mean",
            "ci95_low",
            "ci95_high",
            "n_images",
        ],
        "estimate_column": "mean",
        "minimum_rows": 16,
    },
    "table_10": {
        "family": "attribution",
        "path": "metrics/formal_tables/table_10.csv",
        "operation": "one-to-one timing/memory/query join on dataset,model,method",
        "required_columns": [
            "dataset",
            "model",
            "method",
            "timing_wall_sec_per_image",
            "memory_peak_allocated_gib",
            "queries_query_count_per_image",
        ],
        "display_columns": [
            "dataset",
            "model",
            "method",
            "timing_wall_sec_per_image",
            "memory_peak_allocated_gib",
            "queries_query_count_per_image",
        ],
        "estimate_column": "timing_wall_sec_per_image",
        "minimum_rows": 39,
    },
    "table_11": {
        "family": "attribution",
        "path": "metrics/formal_tables/table_11.csv",
        "operation": (
            "recompute exact PartImageNet common support and bootstrap summaries from A0/A1"
        ),
        "required_columns": [
            "dataset",
            "method",
            "metric",
            "mean",
            "ci90_low",
            "ci90_high",
            "ci95_low",
            "ci95_high",
            "number_of_common_images_total",
            "number_of_part_observations",
        ],
        "display_columns": [
            "method",
            "metric",
            "mean",
            "ci95_low",
            "ci95_high",
            "number_of_common_images_total",
            "number_of_part_observations",
        ],
        "estimate_column": "mean",
        "exact_rows": 75,
    },
    "table_12": {
        "family": "covertype",
        "path": "paper_data/tables/table_12_covertype_design.csv",
        "operation": "derive design from effective config, model inventory, and scheduler plan",
        "required_columns": ["module", "models", "design", "strengths_or_regimes"],
        "display_columns": ["module", "models", "design", "strengths_or_regimes"],
        "estimate_column": "models",
        "exact_rows": 2,
    },
    "table_13": {
        "family": "covertype",
        "path": "paper_data/tables/table_13_covertype_behavior_alignment.csv",
        "operation": "complete E/C/F behavior alignment including canonical endpoint-null F",
        "required_columns": ["module", "outcome", "score_column", "spearman", "kendall_tau"],
        "display_columns": [
            "module",
            "outcome",
            "score_column",
            "spearman",
            "kendall_tau",
            "canonical_expression",
        ],
        "estimate_column": "spearman",
        "minimum_rows": 3,
    },
    "table_14": {
        "family": "covertype",
        "path": "paper_data/tables/table_14_covertype_model_family_audit.csv",
        "operation": "audit Module-C and Module-F coverage by model family/regime",
        "required_columns": [
            "module",
            "model_family",
            "regime",
            "M",
            "E",
            "C",
            "F",
            "Abs",
            "Net",
            "behavior_rows",
        ],
        "display_columns": [
            "module",
            "model_family",
            "regime",
            "M",
            "E",
            "C",
            "F",
            "behavior_rows",
        ],
        "estimate_column": "F",
        "minimum_rows": 15,
    },
    "table_15": {
        "family": "covertype",
        "path": "paper_data/tables/table_15_fixed_semantics_and_magnitude.csv",
        "operation": "recompute fixed-semantic discrimination and magnitude conditioning",
        "required_columns": [
            "section",
            "method",
            "Macro_AUROC",
            "within_bin_macro_AUROC",
            "matched_pair_accuracy",
        ],
        "display_columns": [
            "section",
            "method",
            "Macro_AUROC",
            "Macro_AUPRC",
            "within_bin_macro_AUROC",
            "within_bin_macro_AUPRC",
            "matched_pair_accuracy",
            "pairwise_mechanism_ranking_accuracy",
        ],
        "estimate_column": "Macro_AUROC",
        "exact_rows": 34,
    },
    "table_16": {
        "family": "covertype",
        "path": "paper_data/tables/table_16_covertype_cost.csv",
        "operation": "join complete method and SHAP-interaction cost summaries",
        "required_columns": [
            "method",
            "cost_source",
            "wall_seconds",
            "cpu_seconds",
            "predicted_rows",
            "peak_rss_bytes",
            "cumulative_shard_wall_seconds",
            "completed_tree_models",
        ],
        "display_columns": [
            "method",
            "cost_source",
            "wall_seconds",
            "cpu_seconds",
            "predicted_rows",
            "peak_rss_bytes",
            "cumulative_shard_wall_seconds",
            "completed_tree_models",
        ],
        "estimate_column": "wall_seconds",
        "minimum_rows": 2,
    },
}


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _contract(asset: VisualAsset) -> dict[str, Any]:
    spec = _FIGURE_SPECS.get(asset.asset_id) or _TABLE_SPECS.get(asset.asset_id)
    if spec is None:
        raise SemanticDataError(f"no semantic contract registered for {asset.asset_id}")
    return {
        "schema_version": 1,
        "asset_id": asset.asset_id,
        "kind": asset.kind,
        "manifest_operation": asset.generation_contract["operation"],
        "raw_inputs": [{"run_id": item.run_id, "member": item.member} for item in asset.raw_inputs],
        **spec,
    }


def semantic_contract_sha256(asset: VisualAsset) -> str:
    """Return the frozen semantic-contract digest for one asset."""

    return _json_hash(_contract(asset))


def semantic_contract(asset: VisualAsset) -> Mapping[str, Any]:
    """Expose a copy of the machine-readable asset contract for audits/tests."""

    return json.loads(json.dumps(_contract(asset)))


def _frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SemanticDataError(f"required derived or raw input is missing: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise SemanticDataError(f"unsupported semantic tabular input: {path}")


def _require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SemanticDataError(f"{label} is missing required columns: {missing}")


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"1", "true", "yes"})


def _safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value) if not isinstance(value, (str, int)) else value
    if isinstance(text, str):
        windows_absolute = re.match(r"^[A-Za-z]:[\\/]", text) is not None
        if Path(text).is_absolute() or windows_absolute:
            normalized = text.replace("\\", "/").rstrip("/")
            return normalized.rsplit("/", maxsplit=1)[-1] or "root"
        private_fragments = (
            "/" + "work" + "/" + "Users" + "/",
            "/" + "home" + "/",
            "/" + "tmp" + "/",
        )
        if any(fragment in text for fragment in private_fragments):
            raise SemanticDataError("canonical record contains an embedded private path")
    return text


def _combined_hash(hashes: Sequence[str]) -> str:
    clean = sorted(set(map(str, hashes)))
    if len(clean) == 1:
        return clean[0]
    return _json_hash(clean)


def _source_lineage_closure(
    raw: RawInputs, families: Mapping[str, FamilyOutputs]
) -> dict[str, tuple[str, ...]]:
    allowed = {
        str(item.get("sha256"))
        for item in raw.items.values()
        if isinstance(item.get("sha256"), str)
    }
    for family in families.values():
        allowed.update(
            str(item.get("sha256"))
            for item in family.artifacts.values()
            if isinstance(item.get("sha256"), str)
        )
    if not allowed or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in allowed):
        raise SemanticDataError("sealed source inventory contains invalid hashes")
    closure = {value: (value,) for value in sorted(allowed)}
    for left, right in combinations(sorted(allowed), 2):
        closure.setdefault(_combined_hash((left, right)), (left, right))
    return closure


def _resolve_source_lineage(
    asset_id: str,
    frame: pd.DataFrame,
    closure: Mapping[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    observed = sorted(set(frame["source_sha256"].astype(str)))
    unresolved = [value for value in observed if value not in closure]
    if unresolved:
        raise SemanticDataError(
            f"{asset_id} contains source hashes outside sealed replay closure: {unresolved}"
        )
    return {value: list(closure[value]) for value in observed}


def _canonical_panel(
    frame: pd.DataFrame,
    *,
    asset_id: str,
    panel_id: str,
    series: str | Any,
    x: str | Any,
    estimate: str | Any,
    source_hashes: Sequence[str] | None = None,
    ci_low: str | Any | None = None,
    ci_high: str | Any | None = None,
    n: str | Any = 1,
) -> pd.DataFrame:
    rows = frame.reset_index(drop=True).copy()
    if rows.empty:
        raise SemanticDataError(f"{asset_id}/{panel_id} produced no rows")

    def values(specification: str | Any | None, default: Any) -> pd.Series:
        if specification is None:
            return pd.Series([default] * len(rows))
        if isinstance(specification, str) and specification in rows.columns:
            return rows[specification].reset_index(drop=True)
        if isinstance(specification, pd.Series):
            if len(specification) != len(rows):
                raise SemanticDataError(f"{asset_id}/{panel_id} vector length drifted")
            return specification.reset_index(drop=True)
        if isinstance(specification, (list, tuple, np.ndarray)):
            if len(specification) != len(rows):
                raise SemanticDataError(f"{asset_id}/{panel_id} vector length drifted")
            return pd.Series(specification)
        return pd.Series([specification] * len(rows))

    estimates = pd.to_numeric(values(estimate, np.nan), errors="coerce")
    if source_hashes is None and "source_sha256" in rows:
        provenance = rows["source_sha256"].astype(str)
    else:
        provenance = pd.Series([_combined_hash(source_hashes or ())] * len(rows))
    records = []
    for record in rows.to_dict("records"):
        records.append(
            json.dumps(
                {str(key): _safe(value) for key, value in record.items()},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return pd.DataFrame(
        {
            "artifact_id": asset_id,
            "panel_id": panel_id,
            "series": values(series, "all").map(_safe),
            "x": values(x, np.arange(len(rows))).map(_safe),
            "y": estimates,
            "estimate": estimates,
            "ci_low": pd.to_numeric(values(ci_low, np.nan), errors="coerce"),
            "ci_high": pd.to_numeric(values(ci_high, np.nan), errors="coerce"),
            "n": pd.to_numeric(values(n, 1), errors="coerce").fillna(1),
            "source_sha256": provenance,
            "record_json": records,
        },
        columns=CANONICAL_COLUMNS,
    )


def _canonical_existing(frame: pd.DataFrame, *, asset_id: str) -> pd.DataFrame:
    _require(frame, CANONICAL_COLUMNS[:-1], asset_id)
    panels: list[pd.DataFrame] = []
    for panel_id, rows in frame.groupby("panel_id", sort=False, dropna=False):
        semantic = rows.drop(columns=[column for column in CANONICAL_COLUMNS if column in rows])
        records = [
            json.dumps(
                {str(key): _safe(value) for key, value in record.items()},
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in semantic.to_dict("records")
        ]
        normalized = rows[list(CANONICAL_COLUMNS[:-1])].copy()
        normalized["artifact_id"] = asset_id
        normalized["panel_id"] = str(panel_id)
        normalized["record_json"] = records
        panels.append(normalized[list(CANONICAL_COLUMNS)])
    return pd.concat(panels, ignore_index=True)


def _raw_inputs(raw_root: Path, replay_receipt: Mapping[str, Any]) -> RawInputs:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in replay_receipt.get("inputs", ()):
        key = (str(item.get("run_id")), str(item.get("requested_suffix")))
        if key in indexed:
            raise SemanticDataError(f"duplicate replay input receipt: {key}")
        indexed[key] = item
    if not indexed:
        raise SemanticDataError("sealed replay receipt has no materialized input inventory")
    return RawInputs(raw_root.resolve(), indexed)


def _raw_path(raw: RawInputs, run_id: str, member: str) -> Path:
    item = raw.items.get((run_id, member))
    if item is None:
        raise SemanticDataError(f"missing sealed receipt for {run_id}:{member}")
    path = (raw.root / run_id / member).resolve()
    if not path.is_relative_to(raw.root) or not path.is_file():
        raise SemanticDataError(f"missing registered raw input {run_id}:{member}")
    if path.stat().st_size != int(item.get("size_bytes", -1)):
        raise SemanticDataError(f"materialized raw input size drifted: {run_id}:{member}")
    if sha256_file(path) != item.get("sha256"):
        raise SemanticDataError(f"materialized raw input hash drifted: {run_id}:{member}")
    return path


def _input_hashes(asset: VisualAsset, raw: RawInputs) -> list[str]:
    hashes: list[str] = []
    for item in asset.raw_inputs:
        _raw_path(raw, item.run_id, item.member)
        hashes.append(str(raw.items[(item.run_id, item.member)]["sha256"]))
    return hashes


def _family_paths(output_root: Path, family_receipt: Mapping[str, Any]) -> dict[str, FamilyOutputs]:
    from .family_replay import validate_family_replay_receipt

    validate_family_replay_receipt(output_root, family_receipt)
    families = family_receipt.get("families")
    if not isinstance(families, list):
        raise SemanticDataError("family replay receipt has no family inventory")
    result: dict[str, FamilyOutputs] = {}
    for item in families:
        if item.get("status") != "completed":
            raise SemanticDataError(f"family replay did not complete: {item}")
        path = (output_root / str(item["path"])).resolve()
        if not path.is_relative_to(output_root.resolve()) or not path.is_dir():
            raise SemanticDataError(f"family replay path is invalid: {path}")
        indexed: dict[str, Mapping[str, Any]] = {}
        for artifact in item.get("artifacts", ()):
            relative = str(artifact.get("path", ""))
            if not relative or relative in indexed:
                raise SemanticDataError(
                    f"{item.get('family')} family artifact inventory is not unique"
                )
            indexed[relative] = artifact
        result[str(item["family"])] = FamilyOutputs(path, indexed)
    expected = {"controlled", "imagenet9", "attribution", "covertype"}
    if set(result) != expected:
        raise SemanticDataError(f"family replay set differs: {sorted(result)}")
    return result


def _family_file(family: FamilyOutputs, relative: str) -> Path:
    item = family.artifacts.get(relative)
    if item is None:
        raise SemanticDataError(f"derived family output is unreceipted: {relative}")
    path = (family.root / relative).resolve()
    if not path.is_relative_to(family.root) or not path.is_file():
        raise SemanticDataError(f"derived family output is missing: {relative}")
    if path.stat().st_size != int(item.get("size_bytes", -1)):
        raise SemanticDataError(f"derived family output size drifted: {relative}")
    if sha256_file(path) != item.get("sha256"):
        raise SemanticDataError(f"derived family output hash drifted: {relative}")
    return path


def _controlled_asset(
    asset: VisualAsset,
    raw: RawInputs,
    family: FamilyOutputs,
    representatives: Mapping[str, Any],
) -> pd.DataFrame:
    base = "paper_data/controlled"
    files: dict[str, tuple[str, ...]] = {
        "figure_02": ("figure_02_matched_abs.csv", "figure_02_false_null.csv"),
        "figure_03": ("figure_03_correlations.csv", "figure_03_trajectory.csv"),
        "figure_04": ("figure_04_regimes.csv", "figure_04_bootstrap.csv"),
        "figure_05": (
            "figure_05_epsilon.csv",
            "figure_05_behavior.csv",
            "figure_05_seed.csv",
            "figure_05_bootstrap.csv",
        ),
        "figure_08": ("figure_08_atlas.csv",),
        "figure_09": ("figure_09_all_evidence_trajectories.csv",),
        "figure_10": ("figure_10_geometry_transfer.csv",),
        "figure_11": ("figure_11_calibration_transfer.csv",),
    }
    paths = [_family_file(family, f"{base}/{name}") for name in files[asset.asset_id]]
    existing = _canonical_existing(
        pd.concat([_frame(path) for path in paths]), asset_id=asset.asset_id
    )
    extra: list[pd.DataFrame] = []
    inputs = {
        (item.run_id, item.member): _raw_path(raw, item.run_id, item.member)
        for item in asset.raw_inputs
    }

    if asset.asset_id == "figure_02":
        path = inputs[("C0", "benchmark/null_false_evidence.csv")]
        frame = _frame(path)
        _require(frame, ["method", "false_null_order"], "Figure 2 false-null evidence")
        grouped = frame.groupby("method", as_index=False, sort=True).agg(
            false_null_order=("false_null_order", "mean"), n_pairs=("false_null_order", "size")
        )
        extra.append(
            _canonical_panel(
                grouped,
                asset_id=asset.asset_id,
                panel_id="false_null_evidence",
                series="method",
                x="method",
                estimate="false_null_order",
                n="n_pairs",
                source_hashes=[sha256_file(path)],
            )
        )
    elif asset.asset_id == "figure_03":
        case = representatives.get("figure_03", {})
        if case.get("status") != "verified":
            raise SemanticDataError("Figure 3 representative case is not verified")
        selected = case["resolved"]
        path = inputs[("C1", "results/endpoint_behavior_v1_measurement/aggregate_summary.parquet")]
        frame = _frame(path)
        _require(
            frame,
            [
                "module",
                "trajectory_id",
                "architecture",
                "seed",
                "epoch",
                "V_rev",
                "primary_geometry",
            ],
            "Figure 3 aggregate",
        )
        rows = frame.loc[
            frame["module"].eq("E")
            & _truthy(frame["primary_geometry"])
            & frame["trajectory_id"].astype(str).eq(str(selected["trajectory_id"]))
            & frame["architecture"].astype(str).eq(str(selected["architecture"]))
            & pd.to_numeric(frame["seed"], errors="coerce").eq(int(selected["seed"]))
        ]
        grouped = rows.groupby("epoch", as_index=False, sort=True).agg(
            V_rev=("V_rev", "mean"), n=("model_id", "nunique")
        )
        if sorted(grouped["epoch"].astype(int)) != list(map(int, selected["checkpoint_epochs"])):
            raise SemanticDataError("Figure 3 representative checkpoint inventory drifted")
        extra.append(
            _canonical_panel(
                grouped,
                asset_id=asset.asset_id,
                panel_id="representative_checkpoints",
                series="V_rev",
                x="epoch",
                estimate="V_rev",
                n="n",
                source_hashes=[sha256_file(path)],
            )
        )
    elif asset.asset_id == "figure_04":
        case = representatives.get("figure_04", {})
        if case.get("status") != "verified":
            raise SemanticDataError("Figure 4 representative case is not verified")
        selected = case["resolved"]
        path = inputs[("C1", "results/endpoint_behavior_v1_measurement/aggregate_summary.parquet")]
        frame = _frame(path)
        _require(
            frame, ["model_id", "geometry", "factor", "F", "primary_geometry"], "Figure 4 aggregate"
        )
        rows = frame.loc[
            frame["model_id"].astype(str).eq(str(selected["model_id"]))
            & frame["geometry"].astype(str).eq(str(selected["geometry"]))
            & _truthy(frame["primary_geometry"])
        ]
        grouped = rows.groupby("factor", as_index=False, sort=True).agg(
            F=("F", "mean"), n=("F", "size")
        )
        extra.append(
            _canonical_panel(
                grouped,
                asset_id=asset.asset_id,
                panel_id="representative_fragility",
                series="factor",
                x="factor",
                estimate="F",
                n="n",
                source_hashes=[sha256_file(path)],
            )
        )
    elif asset.asset_id == "figure_05":
        path = inputs[("C2", "evaluation/model_level_results.csv")]
        frame = _frame(path)
        _require(
            frame,
            ["model_id", "task", "endpoint_pair_accuracy", "swapped_pair_accuracy"],
            "Figure 5 model transfer",
        )
        frame = frame.copy()
        frame["accuracy_transfer"] = (
            frame["swapped_pair_accuracy"] - frame["endpoint_pair_accuracy"]
        )
        extra.append(
            _canonical_panel(
                frame,
                asset_id=asset.asset_id,
                panel_id="model_transfer",
                series="task",
                x="endpoint_pair_accuracy",
                estimate="accuracy_transfer",
                source_hashes=[sha256_file(path)],
            )
        )
    elif asset.asset_id == "figure_09":
        path = inputs[("C1", "results/endpoint_behavior_v1_measurement/aggregate_summary.parquet")]
        frame = _frame(path)
        _require(
            frame,
            ["module", "primary_geometry", "trajectory_id", "epoch", "E"],
            "Figure 9 aggregate",
        )
        rows = frame.loc[frame["module"].eq("E") & _truthy(frame["primary_geometry"])]
        grouped = rows.groupby(["trajectory_id", "epoch"], as_index=False, sort=True).agg(
            E=("E", "mean"), n=("model_id", "nunique")
        )
        extra.append(
            _canonical_panel(
                grouped,
                asset_id=asset.asset_id,
                panel_id="checkpoint_summary",
                series="trajectory_id",
                x="epoch",
                estimate="E",
                n="n",
                source_hashes=[sha256_file(path)],
            )
        )
    elif asset.asset_id == "figure_11":
        model_path = inputs[("C2", "evaluation/model_level_results.csv")]
        seed_path = inputs[("C2", "tables/T06_seed_results.csv")]
        model = _frame(model_path)
        seed = _frame(seed_path)
        _require(
            model,
            ["model_id", "task", "endpoint_pair_accuracy", "swapped_pair_accuracy"],
            "Figure 11 model transfer",
        )
        _require(seed, ["task", "seed", "C"], "Figure 11 seed results")
        model = model.copy()
        model["transfer"] = model["swapped_pair_accuracy"] - model["endpoint_pair_accuracy"]
        extra.extend(
            [
                _canonical_panel(
                    model,
                    asset_id=asset.asset_id,
                    panel_id="model_transfer",
                    series="task",
                    x="endpoint_pair_accuracy",
                    estimate="transfer",
                    source_hashes=[sha256_file(model_path)],
                ),
                _canonical_panel(
                    seed,
                    asset_id=asset.asset_id,
                    panel_id="seed_results",
                    series="task",
                    x="seed",
                    estimate="C",
                    source_hashes=[sha256_file(seed_path)],
                ),
            ]
        )
    return pd.concat([existing, *extra], ignore_index=True)


def _i9_figure(asset: VisualAsset, raw: RawInputs, family: FamilyOutputs) -> pd.DataFrame:
    inputs = {
        (item.run_id, item.member): _raw_path(raw, item.run_id, item.member)
        for item in asset.raw_inputs
    }
    panels: list[pd.DataFrame] = []
    if asset.asset_id == "figure_06":
        fixed_path = inputs[("I9", "results/mechanism_benchmark/fixed_semantic.csv")]
        table_path = inputs[("I9", "results/tables/T04_mechanism_benchmark.csv")]
        fixed, table = _frame(fixed_path), _frame(table_path)
        required = ["method", "Macro_AUROC", "Macro_AUPRC", "AUROC_E", "AUROC_C", "AUROC_F"]
        _require(fixed, required, "Figure 6 fixed semantic")
        _require(table, [*required, "evaluation"], "Figure 6 mechanism table")
        table_fixed = table.loc[table["evaluation"].eq("fixed_semantic")].copy()
        table_fixed["method_key"] = (
            table_fixed["method"].astype(str).str.replace(" (fixed semantic)", "", regex=False)
        )
        fixed = fixed.copy()
        fixed["method_key"] = (
            fixed["method"].astype(str).str.replace(" (fixed semantic)", "", regex=False)
        )
        joined = fixed.merge(
            table_fixed[["method_key", "Macro_AUROC"]],
            on="method_key",
            suffixes=("", "_table"),
            validate="one_to_one",
        )
        if len(joined) != 11 or not np.allclose(
            joined["Macro_AUROC"], joined["Macro_AUROC_table"], equal_nan=True
        ):
            raise SemanticDataError("Figure 6 fixed-semantic sources do not agree")
        panels.append(
            _canonical_panel(
                fixed,
                asset_id=asset.asset_id,
                panel_id="fixed_semantics",
                series="method",
                x="method",
                estimate="Macro_AUROC",
                source_hashes=[sha256_file(fixed_path), sha256_file(table_path)],
            )
        )

        matched_path = inputs[("I9", "results/matched_abs/matched_abs_benchmark.csv")]
        pairs_path = inputs[("I9", "results/matched_abs/matched_pairs.parquet")]
        matched, pairs = _frame(matched_path), _frame(pairs_path)
        _require(
            matched,
            ["method", "matched_pair_accuracy", "matched_pairs", "within_bin_macro_AUROC"],
            "Figure 6 matched magnitude",
        )
        if set(pd.to_numeric(matched["matched_pairs"], errors="raise").astype(int)) != {len(pairs)}:
            raise SemanticDataError("Figure 6 matched-pair cardinality disagrees with summary")
        panels.append(
            _canonical_panel(
                matched,
                asset_id=asset.asset_id,
                panel_id="matched_magnitude",
                series="method",
                x="within_bin_macro_AUROC",
                estimate="matched_pair_accuracy",
                n="matched_pairs",
                source_hashes=[sha256_file(matched_path), sha256_file(pairs_path)],
            )
        )

        bins_path = inputs[("I9", "results/tables/T05A_per_bin_auroc.csv")]
        bins = _frame(bins_path)
        _require(
            bins,
            ["method", "mechanism_label", "abs_bin", "AUROC", "sample_count", "reason"],
            "Figure 6 per-bin AUROC",
        )
        bins = bins.copy()
        bins["series_id"] = bins["method"].astype(str) + "/" + bins["mechanism_label"].astype(str)
        panels.append(
            _canonical_panel(
                bins,
                asset_id=asset.asset_id,
                panel_id="per_bin_auroc",
                series="series_id",
                x="abs_bin",
                estimate="AUROC",
                n="sample_count",
                source_hashes=[sha256_file(bins_path)],
            )
        )

        heldout_path = inputs[
            (
                "I9",
                "results/mechanism_benchmark/probe_leave_one_architecture_family_out_summary.csv",
            )
        ]
        heldout = _frame(heldout_path)
        _require(
            heldout,
            ["method", "estimator", "Macro_AUROC", "fold_count", "validation"],
            "Figure 6 held-out family",
        )
        if set(heldout["validation"].astype(str)) != {"leave_one_architecture_family_out"}:
            raise SemanticDataError("Figure 6 held-out-family validation scope drifted")
        panels.append(
            _canonical_panel(
                heldout,
                asset_id=asset.asset_id,
                panel_id="heldout_architecture_family",
                series="estimator",
                x="method",
                estimate="Macro_AUROC",
                n="fold_count",
                source_hashes=[sha256_file(heldout_path)],
            )
        )
    elif asset.asset_id == "figure_07":
        ratios_path = _family_file(family, "metrics/protocol_ratios.csv")
        rank_path = _family_file(family, "metrics/protocol_rank_transfer.csv")
        ratios, rank = _frame(ratios_path), _frame(rank_path)
        _require(
            ratios,
            [
                "pair_type",
                "patch_path",
                "metric",
                "ratio_mean",
                "patch_rows",
                "blend_rows",
                "operation",
            ],
            "Figure 7 ratios",
        )
        _require(
            rank,
            ["pair_type", "patch_path", "metric", "model_count", "spearman_rank_transfer"],
            "Figure 7 rank transfer",
        )
        if set(ratios["operation"].astype(str)) != {"filtered_mean_ratio"}:
            raise SemanticDataError("Figure 7 ratio operation drifted")
        panels.extend(
            [
                _canonical_panel(
                    ratios,
                    asset_id=asset.asset_id,
                    panel_id="patch_blend_ratios",
                    series="metric",
                    x="pair_type",
                    estimate="ratio_mean",
                    n="patch_rows",
                    source_hashes=[sha256_file(ratios_path)],
                ),
                _canonical_panel(
                    rank,
                    asset_id=asset.asset_id,
                    panel_id="rank_transfer",
                    series="metric",
                    x="pair_type",
                    estimate="spearman_rank_transfer",
                    n="model_count",
                    source_hashes=[sha256_file(rank_path)],
                ),
            ]
        )
        ledger_path = inputs[("I9", "results/stage_ledger.jsonl")]
        ledger = _frame(ledger_path)
        _require(ledger, ["stage", "status"], "Figure 7 stage ledger")
        coverage = (
            ledger.groupby(["stage", "status"], as_index=False, sort=True)
            .size()
            .rename(columns={"size": "records"})
        )
        panels.append(
            _canonical_panel(
                coverage,
                asset_id=asset.asset_id,
                panel_id="stage_coverage",
                series="status",
                x="stage",
                estimate="records",
                n="records",
                source_hashes=[sha256_file(ledger_path)],
            )
        )
    else:
        profile_path = inputs[("I9", "results/tables/T03_decaf_profiles.csv")]
        sample_path = inputs[("I9", "results/sample_decaf.parquet")]
        ledger_path = inputs[("I9", "results/stage_ledger.jsonl")]
        profile, sample, ledger = _frame(profile_path), _frame(sample_path), _frame(ledger_path)
        profile_required = [
            "architecture_family",
            "training_regime",
            "challenge_robust_accuracy",
            "challenge_drop",
            "null_corruption_fragility",
        ]
        _require(profile, profile_required, "Figure 12 model robustness")
        model = profile.groupby(
            ["architecture_family", "training_regime"], as_index=False, sort=True
        )[["challenge_robust_accuracy", "challenge_drop", "null_corruption_fragility"]].mean()
        _require(
            sample,
            [
                "architecture_family",
                "training_regime",
                "pair_type",
                "path",
                "epsilon",
                "F",
                "endpoint_active",
            ],
            "Figure 12 sample robustness",
        )
        sample_summary = sample.groupby(
            ["architecture_family", "training_regime", "pair_type", "path", "epsilon"],
            as_index=False,
            sort=True,
        ).agg(F=("F", "mean"), endpoint_active_rate=("endpoint_active", "mean"), n=("F", "size"))
        _require(ledger, ["stage", "status"], "Figure 12 stage ledger")
        coverage = (
            ledger.groupby(["stage", "status"], as_index=False, sort=True)
            .size()
            .rename(columns={"size": "records"})
        )
        panels.extend(
            [
                _canonical_panel(
                    model,
                    asset_id=asset.asset_id,
                    panel_id="model_robustness",
                    series="training_regime",
                    x="architecture_family",
                    estimate="challenge_robust_accuracy",
                    source_hashes=[sha256_file(profile_path)],
                ),
                _canonical_panel(
                    sample_summary,
                    asset_id=asset.asset_id,
                    panel_id="sample_robustness",
                    series="pair_type",
                    x="epsilon",
                    estimate="F",
                    n="n",
                    source_hashes=[sha256_file(sample_path)],
                ),
                _canonical_panel(
                    coverage,
                    asset_id=asset.asset_id,
                    panel_id="stage_coverage",
                    series="status",
                    x="stage",
                    estimate="records",
                    n="records",
                    source_hashes=[sha256_file(ledger_path)],
                ),
            ]
        )
    return pd.concat(panels, ignore_index=True)


def _table_asset(asset: VisualAsset, family_paths: Mapping[str, FamilyOutputs]) -> pd.DataFrame:
    spec = _TABLE_SPECS[asset.asset_id]
    source = _family_file(family_paths[str(spec["family"])], str(spec["path"]))
    frame = _frame(source)
    _require(frame, spec["required_columns"], asset.asset_id)
    _require(frame, spec["display_columns"], f"{asset.asset_id} display schema")
    if "exact_rows" in spec and len(frame) != int(spec["exact_rows"]):
        raise SemanticDataError(
            f"{asset.asset_id} expected {spec['exact_rows']} rows, received {len(frame)}"
        )
    if len(frame) < int(spec.get("minimum_rows", 1)):
        raise SemanticDataError(f"{asset.asset_id} has too few rows: {len(frame)}")
    display = list(spec["display_columns"])
    semantic = frame.copy()
    semantic["display_json"] = semantic[display].apply(
        lambda row: json.dumps(
            {key: _safe(value) for key, value in row.items()}, sort_keys=True, separators=(",", ":")
        ),
        axis=1,
    )
    series = next(
        (column for column in ("method", "dataset", "module", "model") if column in semantic),
        "display_json",
    )
    estimate = str(spec["estimate_column"])
    _require(semantic, [estimate], f"{asset.asset_id} scientific value schema")
    if pd.to_numeric(semantic[estimate], errors="coerce").notna().sum() < 1:
        raise SemanticDataError(f"{asset.asset_id} has no finite scientific values")

    interval_tables = {
        "table_02",
        "table_04",
        "table_06",
        "table_07",
        "table_09",
        "table_11",
    }
    if asset.asset_id in interval_tables:
        intervals = ["ci90_low", "ci90_high", "ci95_low", "ci95_high"]
        _require(semantic, intervals, f"{asset.asset_id} confidence intervals")
        if semantic[intervals].apply(pd.to_numeric, errors="coerce").isna().any().any():
            raise SemanticDataError(f"{asset.asset_id} has missing confidence intervals")
    if asset.asset_id == "table_03":
        interval_columns = [
            "spearman_ci90_low",
            "spearman_ci90_high",
            "spearman_ci95_low",
            "spearman_ci95_high",
        ]
        _require(semantic, interval_columns, "table_03 confidence intervals")
        if semantic[interval_columns].apply(pd.to_numeric, errors="coerce").isna().any().any():
            raise SemanticDataError("table_03 has missing confidence intervals")
    if asset.asset_id == "table_05":
        expected_sections = {
            "behavior_rank": 51,
            "canonical_endpoint_null": 1,
            "method_cost": 13,
            "shap_interaction_completion": 1,
        }
        observed_sections = semantic.groupby("section", sort=True).size().to_dict()
        if observed_sections != expected_sections:
            raise SemanticDataError(f"table_05 section cardinality drifted: {observed_sections}")
        scientific = semantic["section"].isin({"behavior_rank", "canonical_endpoint_null"})
        interval_columns = [
            "spearman_ci90_low",
            "spearman_ci90_high",
            "spearman_ci95_low",
            "spearman_ci95_high",
        ]
        if (
            semantic.loc[scientific, interval_columns]
            .apply(pd.to_numeric, errors="coerce")
            .isna()
            .any()
            .any()
        ):
            raise SemanticDataError("table_05 behavior rows have missing confidence intervals")
        canonical = semantic.loc[semantic["section"].eq("canonical_endpoint_null"), "spearman"]
        if len(canonical) != 1 or not np.isclose(
            float(canonical.iloc[0]), 0.9741, rtol=0.0, atol=0.00005
        ):
            raise SemanticDataError("table_05 canonical endpoint-null F result drifted")
        relative = pd.to_numeric(
            semantic["shap_interaction_relative_cost"], errors="coerce"
        ).dropna()
        if relative.empty or not np.allclose(relative, 10662.38729531742, rtol=0.0, atol=0.000001):
            raise SemanticDataError("table_05 normalized SHAP-interaction cost drifted")
    if asset.asset_id == "table_11":
        support = pd.to_numeric(semantic["number_of_common_images_total"], errors="coerce")
        if support.isna().any() or set(support.astype(int)) != {3586}:
            raise SemanticDataError("table_11 exact common-support cardinality drifted")
    if asset.asset_id == "table_15":
        section_counts = semantic.groupby("section", sort=True).size().to_dict()
        if section_counts != {"fixed_semantic": 17, "matched_magnitude": 17}:
            raise SemanticDataError(f"table_15 section cardinality drifted: {section_counts}")
        fixed = semantic.loc[semantic["section"].eq("fixed_semantic"), "Macro_AUROC"]
        matched = semantic.loc[
            semantic["section"].eq("matched_magnitude"), "within_bin_macro_AUROC"
        ]
        if fixed.notna().sum() < 1 or matched.notna().sum() < 1:
            raise SemanticDataError("table_15 scientific result columns are empty")
    if asset.asset_id == "table_16":
        sections = semantic.groupby("cost_source", sort=True).size().to_dict()
        expected_sections = {
            "pipeline_cost_table": 13,
            "shap_interaction_completion_summary": 1,
        }
        if sections != expected_sections:
            raise SemanticDataError(f"table_16 cost-source cardinality drifted: {sections}")
    ci_low = next(
        (
            column
            for column in ("ci95_low", "spearman_ci95_low", "ci_low", "ci90_low")
            if column in semantic
        ),
        None,
    )
    ci_high = next(
        (
            column
            for column in ("ci95_high", "spearman_ci95_high", "ci_high", "ci90_high")
            if column in semantic
        ),
        None,
    )
    count = next(
        (
            column
            for column in ("n", "n_models", "number_of_models", "n_images")
            if column in semantic
        ),
        1,
    )
    return _canonical_panel(
        semantic,
        asset_id=asset.asset_id,
        panel_id="table_body",
        series=series,
        x=np.arange(len(semantic), dtype=np.int64),
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n=count,
        source_hashes=[sha256_file(source)],
    )


def _validate_frame(
    asset: VisualAsset, frame: pd.DataFrame, receipt: Mapping[str, Any] | None = None
) -> dict[str, int]:
    if list(frame.columns) != list(CANONICAL_COLUMNS):
        raise SemanticDataError(f"{asset.asset_id} canonical schema drifted: {list(frame.columns)}")
    if frame.empty or set(frame["artifact_id"].astype(str)) != {asset.asset_id}:
        raise SemanticDataError(f"{asset.asset_id} canonical identity/cardinality is invalid")
    contract = _contract(asset)
    expected_panels = set(contract.get("panels", {"table_body": 1}))
    cardinality = {
        str(key): int(value) for key, value in frame.groupby("panel_id", sort=True).size().items()
    }
    if set(cardinality) != expected_panels:
        raise SemanticDataError(f"{asset.asset_id} panel structure drifted: {cardinality}")
    minima = contract.get("panels", {"table_body": int(contract.get("minimum_rows", 1))})
    for panel, minimum in minima.items():
        if cardinality[str(panel)] < int(minimum):
            raise SemanticDataError(f"{asset.asset_id}/{panel} expected at least {minimum} rows")
    if "exact_rows" in contract and len(frame) != int(contract["exact_rows"]):
        raise SemanticDataError(
            f"{asset.asset_id} expected exactly {contract['exact_rows']} canonical rows"
        )
    if (
        asset.kind == "figure"
        and pd.to_numeric(frame["estimate"], errors="coerce").notna().sum() < 2
    ):
        raise SemanticDataError(f"{asset.asset_id} has insufficient finite estimates")
    if not frame["source_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise SemanticDataError(f"{asset.asset_id} has invalid source hashes")
    for value in frame["record_json"]:
        decoded = json.loads(str(value))
        if not isinstance(decoded, dict):
            raise SemanticDataError(f"{asset.asset_id} contains non-object record_json")
    if receipt is not None:
        observed_sources = sorted(set(frame["source_sha256"].astype(str)))
        lineage = receipt.get("source_lineage")
        if not isinstance(lineage, Mapping) or set(lineage) != set(observed_sources):
            raise SemanticDataError(f"{asset.asset_id} receipt source lineage drifted")
        for source_hash, components in lineage.items():
            if (
                not isinstance(components, list)
                or not components
                or _combined_hash([str(value) for value in components]) != source_hash
            ):
                raise SemanticDataError(
                    f"{asset.asset_id} receipt source lineage is not hash-consistent"
                )
        expected = {
            "semantic_contract_sha256": semantic_contract_sha256(asset),
            "schema_sha256": CANONICAL_SCHEMA_SHA256,
            "row_count": len(frame),
            "panel_count": len(cardinality),
            "panel_cardinality": cardinality,
            "resolved_source_sha256s": observed_sources,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise SemanticDataError(f"{asset.asset_id} receipt {key} drifted")
    return cardinality


def canonical_asset_path(paper_data_root: str | Path, asset: VisualAsset) -> Path:
    subdirectory = "figures" if asset.kind == "figure" else "tables"
    return Path(paper_data_root) / "canonical" / subdirectory / f"{asset.asset_id}.csv"


def materialize_canonical_assets(
    output_root: str | Path,
    *,
    repo_root: str | Path,
    replay_receipt: Mapping[str, Any],
    family_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and receipt one canonical CSV for every non-missing visual asset."""

    output = Path(output_root).resolve()
    repo = Path(repo_root).resolve()
    raw_root = output / str(replay_receipt["paper_data_directory"])
    raw = _raw_inputs(raw_root, replay_receipt)
    family_paths = _family_paths(output, family_receipt)
    source_closure = _source_lineage_closure(raw, family_paths)
    manifest = load_visual_manifest(repo / "paper" / "visual_manifest.yaml")
    representatives = replay_receipt.get("representative_cases", {})
    rows: list[dict[str, Any]] = []
    for asset in manifest.assets.values():
        if asset.status == "source_missing":
            continue
        if asset.asset_id in _FIGURE_SPECS:
            if asset.asset_id in {"figure_06", "figure_07", "figure_12"}:
                frame = _i9_figure(asset, raw, family_paths["imagenet9"])
            else:
                frame = _controlled_asset(asset, raw, family_paths["controlled"], representatives)
        else:
            frame = _table_asset(asset, family_paths)
        frame = frame[list(CANONICAL_COLUMNS)]
        cardinality = _validate_frame(asset, frame)
        source_lineage = _resolve_source_lineage(asset.asset_id, frame, source_closure)
        destination = canonical_asset_path(raw_root, asset)
        atomic_text(destination, frame.to_csv(index=False, lineterminator="\n"))
        source_hashes = _input_hashes(asset, raw)
        representative_ids = list(_contract(asset).get("representative_case_ids", ()))
        rows.append(
            {
                "asset_id": asset.asset_id,
                "kind": asset.kind,
                "path": destination.relative_to(output).as_posix(),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "semantic_contract_sha256": semantic_contract_sha256(asset),
                "schema_sha256": CANONICAL_SCHEMA_SHA256,
                "row_count": len(frame),
                "panel_count": len(cardinality),
                "panel_cardinality": cardinality,
                "source_sha256s": sorted(source_hashes),
                "resolved_source_sha256s": sorted(source_lineage),
                "source_lineage": source_lineage,
                "representative_case_ids": representative_ids,
            }
        )
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "required_columns": list(CANONICAL_COLUMNS),
        "schema_sha256": CANONICAL_SCHEMA_SHA256,
        "artifact_count": len(rows),
        "contract_set_sha256": _json_hash(
            {
                asset.asset_id: semantic_contract_sha256(asset)
                for asset in manifest.assets.values()
                if asset.status != "source_missing"
            }
        ),
        "artifacts": rows,
    }
    if len(rows) != 27:
        raise SemanticDataError(f"canonical artifact count must be 27, received {len(rows)}")
    receipt_path = raw_root / "canonical" / "canonical_receipt.json"
    atomic_json(receipt_path, receipt)
    receipt["path"] = receipt_path.relative_to(output).as_posix()
    receipt["sha256"] = sha256_file(receipt_path)
    receipt["size_bytes"] = receipt_path.stat().st_size
    return receipt


def load_canonical_asset(
    paper_data_root: str | Path,
    asset: VisualAsset,
    canonical_receipt: Mapping[str, Any],
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Load one canonical asset and verify bytes, schema, contract, and cardinality."""

    matches = [
        item
        for item in canonical_receipt.get("artifacts", ())
        if item.get("asset_id") == asset.asset_id
    ]
    if len(matches) != 1:
        raise SemanticDataError(f"canonical receipt identity missing/duplicated: {asset.asset_id}")
    item = matches[0]
    path = canonical_asset_path(paper_data_root, asset)
    expected_relative = (
        Path("paper_data")
        / "canonical"
        / ("figures" if asset.kind == "figure" else "tables")
        / f"{asset.asset_id}.csv"
    )
    if Path(str(item.get("path", ""))) != expected_relative:
        raise SemanticDataError(f"canonical receipt path drifted for {asset.asset_id}")
    if sha256_file(path) != item.get("sha256") or path.stat().st_size != int(
        item.get("size_bytes", -1)
    ):
        raise SemanticDataError(f"canonical bytes drifted for {asset.asset_id}")
    frame = pd.read_csv(path)
    _validate_frame(asset, frame, item)
    return frame, item


def canonical_cardinality_text(cardinality: Mapping[str, Any]) -> str:
    return ",".join(f"{key}:{int(cardinality[key])}" for key in sorted(cardinality))


def canonical_cardinality_sha256(cardinality: Mapping[str, Any]) -> str:
    return _json_hash({str(key): int(value) for key, value in cardinality.items()})


__all__ = [
    "CANONICAL_COLUMNS",
    "CANONICAL_SCHEMA_SHA256",
    "SemanticDataError",
    "canonical_asset_path",
    "canonical_cardinality_sha256",
    "canonical_cardinality_text",
    "load_canonical_asset",
    "materialize_canonical_assets",
    "semantic_contract",
    "semantic_contract_sha256",
]
