"""Uniform command-line interface for the controlled experiment family."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from decaf.core.manifests import atomic_write_json, sha256_file
from decaf.experiments.common import (
    RunContext,
    load_profile,
    make_parser,
    repository_root,
    run_cli,
)
from decaf.experiments.controlled.analyze import (
    analyze_reference_bundle,
    analyze_smoke,
    controlled_reference_complete,
    load_reference_bundle,
    materialize_controlled_analysis_outputs,
    materialize_controlled_references,
    reference_bundle_receipts,
)
from decaf.experiments.controlled.data import (
    resolve_shapes3d_root,
    validate_shapes3d_asset,
)
from decaf.experiments.controlled.evaluate import (
    build_members,
    checkpoint_bindings_from_manifest,
    configuration_sha256,
    execute_members,
    materialized_member_executor,
    member_contract_sha256,
    plan_counts,
    prepared_run_bindings,
    resolve_materialized_output_root,
    smoke_executor,
    validate_materialized_member_bundle,
    write_jobs_manifest,
)
from decaf.experiments.controlled.models import (
    expected_contradiction_models,
    validate_c0_no_retraining_bundle,
    validate_c1_checkpoint_bundle,
    validate_c2_checkpoint_bundle,
)
from decaf.experiments.controlled.paper import (
    write_reference_paper_data,
    write_smoke_paper_data,
)
from decaf.experiments.controlled.train import (
    c1_factory_training_jobs,
    selected_c1_checkpoints,
)

EXPERIMENT = "controlled"


def build_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build and audit the complete static controlled schedule."""

    members = build_members(config)
    counts = plan_counts(config, members)
    expected = {str(key): int(value) for key, value in config.get("expected_plan", {}).items()}
    assertions: dict[str, Any] = {}
    for name, value in expected.items():
        actual = counts.get(name)
        passed = actual == value
        assertions[name] = {"expected": value, "actual": actual, "passed": passed}
        if not passed:
            raise ValueError(
                f"controlled static-plan assertion failed: {name}: {actual} != {value}"
            )
    if any(member.phase.startswith("c0_train") for member in members):
        raise AssertionError("C0 no-retraining contract was violated")
    c1_measurements = [member for member in members if member.phase == "c1_measure"]
    if not c1_measurements or any(
        len(member.dependencies) != 1 or not member.dependencies[0].startswith("c1_train__")
        for member in c1_measurements
    ):
        raise AssertionError("C1 measurement jobs are not closed over factory jobs")
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "profile": str(config.get("profile", "unknown")),
        "configuration_sha256": configuration_sha256(config),
        "member_contract_sha256": member_contract_sha256(members),
        "scientific_counts": counts,
        "assertions": assertions,
        "contracts": {
            "c0_no_retraining": True,
            "shared_noise_within_pairs": True,
            "accumulation_dtype": "float64",
            "paper_compute_contract": "hash_registered_materialized_accelerator_outputs_v2",
            "prepared_input_lineage_bound": True,
            "checkpoint_cache_universe_exact": True,
            "gpu_execution_performed_by_this_cli": False,
            "c1_checkpoint_producer_coverage": True,
            "c2_checkpoint_producer_coverage": True,
            "analysis_inputs_from_compute": True,
            "unique_output_paths": True,
            "unique_receipt_paths": True,
        },
        "members": [member.as_dict() for member in members],
    }


def _checkpoint_cache_root() -> Path:
    root = os.environ.get("DECAF_CACHE_ROOT")
    if not root:
        raise RuntimeError("DECAF_CACHE_ROOT is required for paper-profile controlled compute")
    return Path(root).expanduser().resolve() / "checkpoints" / "controlled"


def _checkpoint_inventory() -> tuple[Path, dict[str, Mapping[str, Any]]]:
    path = repository_root() / "manifests" / "checkpoints" / "controlled.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("kind") != "checkpoint_inventory":
        raise ValueError("Controlled checkpoint inventory has an unsupported schema")
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("Controlled checkpoint inventory groups must be a list")
    groups = {
        str(group["id"]): group
        for group in raw_groups
        if isinstance(group, Mapping) and "id" in group
    }
    expected = {"base_models", "selected_evidence_states", "context_swap"}
    if set(groups) != expected:
        raise ValueError("Controlled checkpoint inventory group coverage changed")
    return path, groups


def _manifest_asset_path(manifest: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else manifest.parent / path


def _validate_paper_checkpoint_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate C0/C1/C2 manifest identity, producer coverage, and local bytes."""

    cache = _checkpoint_cache_root()
    assets = config["assets"]
    c0_manifest = cache / str(assets["c0_model_manifest"])
    c1_manifest = cache / str(assets["c1_model_manifest"])
    c2_manifest = cache / str(assets["c2_model_manifest"])
    inventory_path, inventory = _checkpoint_inventory()
    c0_rows = validate_c0_no_retraining_bundle(
        c0_manifest,
        expected_registry_sha256=str(inventory["base_models"]["portable_registry_sha256"]),
    )
    selected = selected_c1_checkpoints(config["endpoint_behavior"])
    c1_rows = validate_c1_checkpoint_bundle(
        c1_manifest,
        selected,
        expected_registry_sha256=str(
            inventory["selected_evidence_states"]["portable_registry_sha256"]
        ),
    )
    contradiction = config["contradiction"]
    c2_registry = expected_contradiction_models(
        tuple(map(str, contradiction["tasks"])),
        tuple(map(str, contradiction["architectures"])),
        tuple(map(int, contradiction["seeds"])),
    )
    c2_rows = validate_c2_checkpoint_bundle(
        c2_manifest,
        c2_registry,
        expected_registry_sha256=str(inventory["context_swap"]["portable_registry_sha256"]),
    )
    c1_jobs = c1_factory_training_jobs(config["endpoint_behavior"])

    return {
        "schema_version": 1,
        "root_environment": "DECAF_CACHE_ROOT",
        "verification": "local_byte_identity",
        "gpu_execution_verified_here": False,
        "canonical_inventory": {
            "path": "manifests/checkpoints/controlled.yaml",
            "sha256": sha256_file(inventory_path),
        },
        "items": [
            {
                "id": "c0_base_models",
                "count": len(c0_rows),
                "no_retraining": True,
                "source_manifest": str(assets["c0_model_manifest"]),
                "source_manifest_sha256": sha256_file(c0_manifest),
                "portable_registry_sha256": c0_rows.attrs["logical_registry_sha256"],
                "checkpoints": [
                    {
                        "model_id": str(row.model_id),
                        "logical_path": f"checkpoints/c0/{row.model_id}.pt",
                        "bytes": _manifest_asset_path(
                            c0_manifest, row.checkpoint_path
                        ).stat().st_size,
                        "sha256": str(row.checkpoint_sha256),
                    }
                    for row in c0_rows.itertuples(index=False)
                ],
                "probability_caches": [
                    {
                        "model_id": str(row.model_id),
                        "logical_path": f"probability_caches/c0/{row.model_id}.npy",
                        "bytes": _manifest_asset_path(
                            c0_manifest, row.probability_cache_path
                        ).stat().st_size,
                        "sha256": str(row.probability_cache_sha256),
                    }
                    for row in c0_rows.itertuples(index=False)
                ],
            },
            {
                "id": "c1_selected_checkpoints",
                "count": len(c1_rows),
                "training_jobs": len(c1_jobs),
                "producer_member_coverage": c1_rows["producer_member_id"].nunique(),
                "source_manifest": str(assets["c1_model_manifest"]),
                "source_manifest_sha256": sha256_file(c1_manifest),
                "portable_registry_sha256": c1_rows.attrs["logical_registry_sha256"],
                "checkpoints": [
                    {
                        "model_id": str(row.model_id),
                        "logical_path": f"checkpoints/c1/{row.model_id}.pt",
                        "bytes": _manifest_asset_path(
                            c1_manifest, row.checkpoint_path
                        ).stat().st_size,
                        "sha256": str(row.checkpoint_sha256),
                        "producer_member_id": str(row.producer_member_id),
                    }
                    for row in c1_rows.itertuples(index=False)
                ],
            },
            {
                "id": "c2_context_swap",
                "count": len(c2_rows),
                "training_jobs": len(c2_registry),
                "producer_member_coverage": c2_rows["producer_member_id"].nunique(),
                "source_manifest": str(assets["c2_model_manifest"]),
                "source_manifest_sha256": sha256_file(c2_manifest),
                "portable_registry_sha256": c2_rows.attrs["logical_registry_sha256"],
                "checkpoints": [
                    {
                        "model_id": str(row.model_id),
                        "logical_path": f"checkpoints/c2/{row.model_id}.pt",
                        "bytes": _manifest_asset_path(
                            c2_manifest, row.checkpoint_path
                        ).stat().st_size,
                        "sha256": str(row.checkpoint_sha256),
                        "producer_member_id": str(row.producer_member_id),
                    }
                    for row in c2_rows.itertuples(index=False)
                ],
            },
        ],
    }


def prepare_handler(context: RunContext) -> Mapping[str, Any]:
    """Validate public assets and persist the deterministic schedule."""

    plan = build_plan(context.config)
    write_jobs_manifest(
        context.path / "manifests" / "jobs.jsonl",
        build_members(context.config),
    )
    profile = str(context.config.get("profile", context.profile))
    if profile == "smoke":
        data_manifest = {
            "schema_version": 1,
            "items": [
                {
                    "id": "controlled_cpu_oracle",
                    "kind": "synthetic_scores",
                    "paper_equivalent": False,
                }
            ],
        }
        checkpoint_manifest = {
            "schema_version": 1,
            "items": [],
            "scope": "cpu_score_oracle",
            "gpu_real_shard_verification": "pending",
        }
    else:
        dataset_root = resolve_shapes3d_root(context.config["assets"].get("dataset_root"))
        asset = validate_shapes3d_asset(dataset_root)
        data_manifest = {"schema_version": 1, "items": [asset.public_record()]}
        checkpoint_manifest = _validate_paper_checkpoint_inputs(context.config)
    atomic_write_json(context.path / "manifests" / "data.json", data_manifest)
    atomic_write_json(context.path / "manifests" / "checkpoints.json", checkpoint_manifest)
    atomic_write_json(context.path / "manifests" / "plan.json", plan)
    return {"members": len(plan["members"]), "scientific_counts": plan["scientific_counts"]}


def compute_handler(context: RunContext) -> Mapping[str, Any]:
    """Run the CPU oracle or ingest a complete hash-registered GPU bundle."""

    members = build_members(context.config)
    profile = str(context.config.get("profile", context.profile))
    if profile == "smoke":
        run_bindings = prepared_run_bindings(context, members)
        return execute_members(
            context,
            members,
            smoke_executor,
            run_bindings=run_bindings,
        )

    if not context.stage_completed("prepare"):
        raise RuntimeError(
            "paper Controlled compute requires a completed prepare stage in the same run; "
            "run --stage prepare first, then resume with --stage compute"
        )
    prepared_plan_path = context.path / "manifests" / "plan.json"
    if not prepared_plan_path.is_file():
        raise FileNotFoundError("paper Controlled compute is missing manifests/plan.json")
    prepared_plan = json.loads(prepared_plan_path.read_text(encoding="utf-8"))
    config_digest = configuration_sha256(context.config)
    if prepared_plan.get("configuration_sha256") != config_digest:
        raise ValueError("prepared Controlled configuration fingerprint mismatch")
    run_bindings = prepared_run_bindings(context, members)

    execution = context.config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise ValueError("controlled execution config must be a mapping")
    source_root = resolve_materialized_output_root(context.config)
    member_bundle = validate_materialized_member_bundle(
        source_root,
        members,
        run_bindings=run_bindings,
        checkpoint_bindings=checkpoint_bindings_from_manifest(
            context.path / "manifests" / "checkpoints.json"
        ),
        manifest_relative=str(execution.get("member_manifest", "manifests/members.json")),
    )
    result = execute_members(
        context,
        members,
        materialized_member_executor(member_bundle),
        run_bindings=run_bindings,
    )
    member_manifest_digest = sha256_file(member_bundle.manifest)
    analysis_receipts = materialize_controlled_analysis_outputs(
        source_root,
        context.path / "paper_data" / "reference",
        run_bindings=run_bindings,
        member_manifest_sha256=member_manifest_digest,
        manifest_relative=str(execution.get("analysis_manifest", "manifests/analysis.json")),
        analysis_prefix=str(execution.get("analysis_root", "analysis")),
    )
    atomic_write_json(
        context.path / "receipts" / "controlled_materialized_analysis_inputs.json",
        {
            "schema_version": 2,
            "source_kind": "materialized_accelerator_analysis",
            "byte_identity_verified": True,
            "gpu_execution_performed_here": False,
            "run_bindings": run_bindings,
            "member_manifest_sha256": member_manifest_digest,
            "analysis_manifest_sha256": sha256_file(
                source_root
                / str(execution.get("analysis_manifest", "manifests/analysis.json"))
            ),
            "items": analysis_receipts,
        },
    )
    return {
        **result,
        "source": "materialized_accelerator_outputs",
        "producer_declared_execution_class": member_bundle.producer_execution_class,
        "member_manifest_sha256": member_manifest_digest,
        "analysis_inputs": len(analysis_receipts),
        "byte_identity_verified": True,
        "gpu_execution_performed_here": False,
    }


def analyze_handler(context: RunContext) -> Mapping[str, Any]:
    """Analyze CPU-oracle outputs or verified C0/C1/C2 reference tables."""

    profile = str(context.config.get("profile", context.profile))
    if profile == "smoke":
        return analyze_smoke(context.path / "raw", context.path / "metrics")
    paper_data = context.path / "paper_data" / "reference"
    if controlled_reference_complete(paper_data):
        receipts = reference_bundle_receipts(
            paper_data,
            source_kind="materialized_accelerator_analysis",
        )
        source_kind = "materialized_accelerator_analysis"
    else:
        receipts = materialize_controlled_references(
            paper_data,
            # Let the shared resolver split DECAF_REFERENCE_RUNS_ROOT on
            # os.pathsep. Passing the raw string here collapses multiple roots
            # into one invalid path.
            reference_root=None,
            repo_root=repository_root(),
        )
        source_kind = "sealed_reference_archives"
    atomic_write_json(context.path / "receipts" / "controlled_reference_inputs.json", receipts)
    bundle = load_reference_bundle(paper_data)
    result = analyze_reference_bundle(
        bundle,
        context.path / "metrics",
        targets=context.config["headline_targets"],
    )
    return {
        **result,
        "reference_inputs": len(receipts),
        "input_source": source_kind,
        "model_inference_performed_here": False,
    }


def paper_handler(context: RunContext) -> Mapping[str, Any]:
    """Generate family-local, machine-readable controlled panel data."""

    profile = str(context.config.get("profile", context.profile))
    destination = context.path / "paper_data" / "controlled"
    if profile == "smoke":
        return write_smoke_paper_data(
            context.path / "metrics" / "controlled_smoke_metrics.csv",
            destination,
        )
    summary_path = context.path / "metrics" / "controlled_headlines.json"
    if not summary_path.is_file():
        raise FileNotFoundError("controlled analysis must complete before the paper stage")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bundle = load_reference_bundle(context.path / "paper_data" / "reference")
    return write_reference_paper_data(bundle, destination, headline_summary=summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser(EXPERIMENT)
    args = parser.parse_args(argv)
    config = load_profile(EXPERIMENT, args.profile, args.config)
    plan = build_plan(config)
    return run_cli(
        experiment=EXPERIMENT,
        args=args,
        plan=plan,
        handlers={
            "prepare": prepare_handler,
            "compute": compute_handler,
            "analyze": analyze_handler,
            "paper": paper_handler,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "analyze_handler",
    "build_plan",
    "compute_handler",
    "main",
    "paper_handler",
    "prepare_handler",
]
