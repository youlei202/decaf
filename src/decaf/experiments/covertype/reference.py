"""Fail-closed replay of the sealed T0 Covertype analysis inputs."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr

from decaf.experiments.common import RunContext, atomic_json, atomic_text, repository_root
from decaf.experiments.covertype.evaluate import formal_specs
from decaf.paper.reference import (
    ReferenceRun,
    discover_archive,
    load_reference_runs,
    materialize_inputs,
    receipt_dict,
    reference_roots,
    resolve_member,
    sha256_file,
    verify_archive,
)

REFERENCE_RUN_ID = "T0"
REFERENCE_SOURCE_MODE = "sealed_reference_replay"
EXPECTED_NATURAL_DATA_FINGERPRINT = (
    "07aa4349a338b765e0c143407ecc0acd4ccdf35ed0e13c8014519fdc013ade9c"
)
EXPECTED_CANONICAL_FRAGILITY_SPEARMAN = 0.9741138295203292

RANK_COLUMNS = {
    "module",
    "outcome",
    "mechanism",
    "method",
    "score_column",
    "n_units",
    "spearman",
    "kendall_tau",
}
BOOTSTRAP_COLUMNS = {
    "analysis",
    "method",
    "module",
    "outcome",
    "metric",
    "point_estimate",
    "ci95_low",
    "ci95_high",
    "bootstrap_repetitions",
    "valid_repetitions",
}
FIXED_COLUMNS = {
    "method",
    "evaluation",
    "n_units",
    "Macro_AUROC",
    "Macro_AUPRC",
    "primary_unit_definition",
}
MATCHED_COLUMNS = {
    "method",
    "n_units",
    "within_bin_macro_AUROC",
    "within_bin_macro_AUPRC",
    "matched_pair_accuracy",
    "matched_pairs",
}
COST_COLUMNS = {
    "method",
    "wall_seconds",
    "cpu_seconds",
    "predicted_rows",
    "peak_rss_bytes",
}
MODEL_COLUMNS = {
    "model_id",
    "module",
    "regime",
    "strength",
    "model_family",
    "seed",
    "natural_data_fingerprint",
    "M_Z",
    "E_Z",
    "C_Z",
    "F_Z",
    "Abs_Z",
    "M_U",
    "E_U",
    "C_U",
    "F_U",
    "Abs_U",
    "null_context_prediction_change_rate",
}


def _reference_run(context: RunContext) -> ReferenceRun:
    runs = load_reference_runs(repository_root() / "manifests" / "reference_runs")
    run_id = str(context.config.get("reference_run", REFERENCE_RUN_ID))
    if run_id != REFERENCE_RUN_ID:
        raise ValueError(f"Covertype paper replay requires reference run {REFERENCE_RUN_ID}")
    run = runs[run_id]
    if run.family != "covertype" or run.scientific_status != "sealed_reference_run":
        raise ValueError("T0 is not registered as the sealed Covertype reference run")
    if not run.analysis_inputs:
        raise ValueError("T0 does not declare any analysis inputs")
    return run


def _member_sha256(bundle: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with bundle.open(member) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("sealed Covertype reference receipt is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("sealed Covertype reference receipt is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("sealed Covertype reference receipt must be a JSON object")
    return payload


def validate_materialized_covertype_reference(context: RunContext) -> dict[str, Any]:
    """Rebind every materialized input to the current T0 manifest and archive."""

    run = _reference_run(context)
    receipt = _load_receipt(context.path / "receipts" / "covertype_reference_inputs.json")
    required_identity = {
        "schema_version": 1,
        "status": "completed",
        "source_mode": REFERENCE_SOURCE_MODE,
        "run_id": run.run_id,
        "family": run.family,
        "scientific_status": run.scientific_status,
        "archive_filename": run.archive_filename,
        "archive_sha256": run.archive_sha256,
        "archive_size_bytes": run.archive_size_bytes,
        "archive_member_count": run.archive_member_count,
    }
    for field, expected in required_identity.items():
        if receipt.get(field) != expected:
            raise ValueError(f"sealed Covertype reference receipt {field} differs")

    inputs = receipt.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("sealed Covertype reference receipt has no input inventory")
    expected_suffixes = set(run.analysis_inputs)
    observed_suffixes = {
        str(item.get("requested_suffix")) for item in inputs if isinstance(item, Mapping)
    }
    if observed_suffixes != expected_suffixes or len(inputs) != len(expected_suffixes):
        raise ValueError("sealed Covertype reference receipt inventory differs")

    materialized_root = context.path / "reference_data"
    expected_files = {f"{run.run_id}/{suffix}" for suffix in expected_suffixes}
    observed_files = {
        path.relative_to(materialized_root).as_posix()
        for path in materialized_root.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise ValueError("sealed Covertype materialized file inventory differs")

    archive = discover_archive(run, reference_roots())
    verify_archive(archive, run)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for raw_item in inputs:
            if not isinstance(raw_item, Mapping):
                raise ValueError("sealed Covertype reference input receipt is malformed")
            suffix = str(raw_item["requested_suffix"])
            relative = f"{run.run_id}/{suffix}"
            if raw_item.get("run_id") != run.run_id or raw_item.get("relative_path") != relative:
                raise ValueError(f"sealed Covertype reference identity differs: {suffix}")
            resolved = resolve_member(names, suffix)
            if raw_item.get("resolved_member") != resolved:
                raise ValueError(f"sealed Covertype archive member binding differs: {suffix}")
            archive_size = bundle.getinfo(resolved).file_size
            archive_sha256 = _member_sha256(bundle, resolved)
            if int(raw_item.get("size_bytes", -1)) != archive_size:
                raise ValueError(f"sealed Covertype archive member size differs: {suffix}")
            if raw_item.get("sha256") != archive_sha256:
                raise ValueError(f"sealed Covertype archive member hash differs: {suffix}")
            path = materialized_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"sealed Covertype reference input is missing: {suffix}")
            if path.stat().st_size != archive_size:
                raise ValueError(f"sealed Covertype reference size differs: {suffix}")
            if sha256_file(path) != archive_sha256:
                raise ValueError(f"sealed Covertype reference hash differs: {suffix}")
    return {
        "source_mode": REFERENCE_SOURCE_MODE,
        "reference_run_id": run.run_id,
        "reference_input_count": len(inputs),
        "archive_sha256": run.archive_sha256,
    }


def _read_csv(root: Path, relative: str, columns: set[str], rows: int) -> pd.DataFrame:
    frame = pd.read_csv(root / relative)
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"T0 input {relative} is missing columns: {missing}")
    if len(frame) != rows:
        raise ValueError(
            f"T0 input {relative} row count differs: expected {rows}, got {len(frame)}"
        )
    return frame


def _semantic_grid(frame: pd.DataFrame) -> set[tuple[str, str, float | None, str, int]]:
    rows: set[tuple[str, str, float | None, str, int]] = set()
    for record in frame[["module", "regime", "strength", "model_family", "seed"]].itertuples(
        index=False
    ):
        strength = None if pd.isna(record.strength) else float(record.strength)
        rows.add(
            (
                str(record.module),
                str(record.regime),
                strength,
                str(record.model_family),
                int(record.seed),
            )
        )
    return rows


def _normalize_models(
    module_c: pd.DataFrame, module_f: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized: list[pd.DataFrame] = []
    for frame, suffix in ((module_c, "Z"), (module_f, "U")):
        current = frame.copy()
        for component in ("M", "E", "C", "F", "Abs", "Net"):
            current[component] = current[f"{component}_{suffix}"].astype(float)
        residual = np.abs(current["Abs"] - current[["E", "C", "F"]].sum(axis=1))
        if not np.isfinite(residual).all() or float(residual.max()) > 1e-12:
            raise AssertionError(f"T0 Module {current['module'].iloc[0]} DECAF identity drifted")
        current["decaf_identity_passed"] = True
        normalized.append(current)
    c_frame, f_frame = normalized
    combined = pd.concat(normalized, ignore_index=True, sort=False)
    return c_frame, f_frame, combined.sort_values("model_id", kind="stable").reset_index(drop=True)


def _validate_source_contract(context: RunContext) -> dict[str, pd.DataFrame]:
    root = context.path / "reference_data" / REFERENCE_RUN_ID
    rank = _read_csv(root, "results/benchmark/rank_statistics.csv", RANK_COLUMNS, 51)
    bootstrap = _read_csv(root, "results/benchmark/bootstrap.csv", BOOTSTRAP_COLUMNS, 204)
    fixed = _read_csv(root, "results/benchmark/fixed_semantic.csv", FIXED_COLUMNS, 17)
    matched = _read_csv(root, "results/benchmark/matched_magnitude.csv", MATCHED_COLUMNS, 17)
    costs = _read_csv(root, "results/tables/costs.csv", COST_COLUMNS, 13)
    shap_costs = _read_csv(
        root,
        "results/baselines/shap_interaction_cost_summary.csv",
        {
            "method",
            "expected_tree_models",
            "completed_tree_models",
            "expected_shards",
            "completed_shards",
            "status",
        },
        1,
    )
    module_c = _read_csv(root, "results/module_c/model_decaf.csv", MODEL_COLUMNS, 90)
    module_f = _read_csv(root, "results/module_f/model_decaf.csv", MODEL_COLUMNS, 45)
    model_manifest = _read_csv(
        root,
        "results/inventory/model_manifest.csv",
        {
            "module",
            "regime",
            "strength",
            "model_family",
            "seed",
            "model_id",
            "checkpoint_path",
            "status",
        },
        135,
    )

    if module_c["model_id"].duplicated().any() or module_f["model_id"].duplicated().any():
        raise ValueError("T0 model tables contain duplicate model IDs")
    if set(module_c["model_id"]) & set(module_f["model_id"]):
        raise ValueError("T0 Module C and F model IDs overlap")
    model_ids = set(module_c["model_id"]) | set(module_f["model_id"])
    if (
        set(model_manifest["model_id"]) != model_ids
        or model_manifest["model_id"].duplicated().any()
    ):
        raise ValueError("T0 model manifest differs from the model tables")
    if set(model_manifest["status"].astype(str)) != {"completed"}:
        raise ValueError("T0 model manifest contains non-completed models")

    expected_grid = {
        (spec.module, spec.regime, spec.strength, spec.model_family, spec.seed)
        for spec in formal_specs()
    }
    observed_grid = _semantic_grid(pd.concat((module_c, module_f), ignore_index=True))
    if observed_grid != expected_grid:
        raise ValueError("T0 model tables differ from the formal 135-model semantic grid")
    fingerprints = set(pd.concat((module_c, module_f))["natural_data_fingerprint"].astype(str))
    if fingerprints != {EXPECTED_NATURAL_DATA_FINGERPRINT}:
        raise ValueError("T0 model tables have an unexpected natural-data fingerprint")

    split_manifest = json.loads(
        (root / "results/data/split_manifest.json").read_text(encoding="utf-8")
    )
    if split_manifest.get("fingerprint") != EXPECTED_NATURAL_DATA_FINGERPRINT:
        raise ValueError("T0 split manifest fingerprint differs")
    if split_manifest.get("split_sizes") != {"train": 144000, "validation": 48000, "test": 48000}:
        raise ValueError("T0 split manifest sizes differ")
    effective_config = yaml.safe_load(
        (root / "results/effective_config.yaml").read_text(encoding="utf-8")
    )
    if (
        not isinstance(effective_config, Mapping)
        or effective_config.get("experiment_name") != "DECAF Covertype Contextual Mechanisms v1"
    ):
        raise ValueError("T0 effective configuration identity differs")
    scheduler = json.loads(
        (root / "results/inventory/shap_interaction_scheduler.json").read_text(encoding="utf-8")
    )
    if scheduler.get("status") != "completed" or scheduler.get("formal_total_shards") != 216:
        raise ValueError("T0 SHAP interaction scheduler contract differs")
    shap_manifest = pd.read_parquet(
        root / "results/shap_interaction_stratified_128_manifest.parquet"
    )
    required_shap_columns = {
        "protocol_version",
        "schedule_id",
        "module",
        "seed",
        "shard_id",
        "source_index",
    }
    if len(shap_manifest) != 768 or required_shap_columns - set(shap_manifest.columns):
        raise ValueError("T0 SHAP interaction sample manifest contract differs")
    if (
        int(shap_costs.at[0, "expected_tree_models"]) != 54
        or int(shap_costs.at[0, "completed_tree_models"]) != 54
        or int(shap_costs.at[0, "expected_shards"]) != 216
        or int(shap_costs.at[0, "completed_shards"]) != 216
        or str(shap_costs.at[0, "status"]) != "complete"
    ):
        raise ValueError("T0 SHAP interaction completion summary differs")
    return {
        "rank": rank,
        "bootstrap": bootstrap,
        "fixed": fixed,
        "matched": matched,
        "costs": costs,
        "shap_costs": shap_costs,
        "module_c": module_c,
        "module_f": module_f,
        "model_manifest": model_manifest,
    }


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def analyze_sealed_reference(context: RunContext) -> dict[str, Any]:
    """Materialize T0 and produce run-local analysis tables from its sealed rows."""

    run = _reference_run(context)
    archive = discover_archive(run, reference_roots())
    verify_archive(archive, run)
    materialized = materialize_inputs(
        run, archive, run.analysis_inputs, context.path / "reference_data"
    )
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "source_mode": REFERENCE_SOURCE_MODE,
        "run_id": run.run_id,
        "family": run.family,
        "scientific_status": run.scientific_status,
        "archive_filename": run.archive_filename,
        "archive_sha256": run.archive_sha256,
        "archive_size_bytes": run.archive_size_bytes,
        "archive_member_count": run.archive_member_count,
        "inputs": [receipt_dict(item) for item in materialized],
    }
    atomic_json(context.path / "receipts" / "covertype_reference_inputs.json", receipt)
    validate_materialized_covertype_reference(context)
    tables = _validate_source_contract(context)
    module_c, module_f, model = _normalize_models(tables["module_c"], tables["module_f"])

    correlation = float(
        spearmanr(module_f["F"], module_f["null_context_prediction_change_rate"]).statistic
    )
    tau = float(
        kendalltau(module_f["F"], module_f["null_context_prediction_change_rate"]).statistic
    )
    if not np.isclose(correlation, EXPECTED_CANONICAL_FRAGILITY_SPEARMAN, rtol=0.0, atol=1e-14):
        raise AssertionError("T0 canonical endpoint-null fragility correlation drifted")
    canonical = {
        "component": "F",
        "score_column": "F_U",
        "outcome": "null_context_prediction_change_rate",
        "expression": "correlation(F, null_context_prediction_change_rate)",
        "n": len(module_f),
        "spearman": correlation,
        "kendall_tau": tau,
    }

    family_numeric = ["M", "E", "C", "F", "Abs", "Net", "behavior_rows"]
    family_audit = (
        model.groupby(["module", "model_family", "regime"], dropna=False)[family_numeric]
        .mean(numeric_only=True)
        .reset_index()
    )
    sanitized_manifest = tables["model_manifest"].drop(columns=["checkpoint_path"])
    metrics = context.path / "metrics"
    for name, frame in (
        ("rank_statistics.csv", tables["rank"]),
        ("bootstrap.csv", tables["bootstrap"]),
        ("fixed_semantic.csv", tables["fixed"]),
        ("matched_magnitude.csv", tables["matched"]),
        ("costs.csv", tables["costs"]),
        ("shap_interaction_cost_summary.csv", tables["shap_costs"]),
        ("module_c_model_decaf.csv", module_c),
        ("module_f_model_decaf.csv", module_f),
        ("model_results.csv", model),
        ("model_family_audit.csv", family_audit),
        ("model_manifest.csv", sanitized_manifest),
    ):
        _write_frame(metrics / name, frame)

    summary = {
        "schema_version": 1,
        "source_mode": REFERENCE_SOURCE_MODE,
        "reference_run_id": run.run_id,
        "reference_archive_sha256": run.archive_sha256,
        "reference_input_count": len(materialized),
        "model_count": len(model),
        "module_c_models": len(module_c),
        "module_f_models": len(module_f),
        "natural_data_fingerprint": EXPECTED_NATURAL_DATA_FINGERPRINT,
        "all_decaf_identities_passed": bool(model["decaf_identity_passed"].all()),
        "canonical_fragility_correlation": canonical,
    }
    atomic_json(metrics / "analysis_summary.json", summary)
    return summary


__all__ = [
    "EXPECTED_CANONICAL_FRAGILITY_SPEARMAN",
    "REFERENCE_RUN_ID",
    "REFERENCE_SOURCE_MODE",
    "analyze_sealed_reference",
    "validate_materialized_covertype_reference",
]
