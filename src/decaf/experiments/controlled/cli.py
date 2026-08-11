"""Uniform command-line interface for the controlled experiment family."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from decaf.core.manifests import atomic_write_json
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
    load_reference_bundle,
    materialize_controlled_references,
)
from decaf.experiments.controlled.data import (
    resolve_shapes3d_root,
    validate_shapes3d_asset,
)
from decaf.experiments.controlled.evaluate import (
    build_members,
    execute_members,
    plan_counts,
    smoke_executor,
    write_jobs_manifest,
)
from decaf.experiments.controlled.models import validate_c0_no_retraining_bundle
from decaf.experiments.controlled.paper import (
    write_reference_paper_data,
    write_smoke_paper_data,
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
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "profile": str(config.get("profile", "unknown")),
        "scientific_counts": counts,
        "assertions": assertions,
        "contracts": {
            "c0_no_retraining": True,
            "shared_noise_within_pairs": True,
            "accumulation_dtype": "float64",
            "paper_compute_verification": "pending_gpu_real_shard",
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
        cache = _checkpoint_cache_root()
        c0_manifest = cache / str(context.config["assets"]["c0_model_manifest"])
        c0_rows = validate_c0_no_retraining_bundle(c0_manifest)
        data_manifest = {"schema_version": 1, "items": [asset.public_record()]}
        checkpoint_manifest = {
            "schema_version": 1,
            "items": [
                {"id": "c0_base_models", "count": len(c0_rows), "no_retraining": True},
                {
                    "id": "c1_selected_checkpoints",
                    "count": plan["scientific_counts"]["endpoint_behavior_checkpoints"],
                },
                {
                    "id": "c2_context_swap",
                    "count": plan["scientific_counts"]["contradiction_models"],
                },
            ],
        }
    atomic_write_json(context.path / "manifests" / "data.json", data_manifest)
    atomic_write_json(context.path / "manifests" / "checkpoints.json", checkpoint_manifest)
    atomic_write_json(context.path / "manifests" / "plan.json", plan)
    return {"members": len(plan["members"]), "scientific_counts": plan["scientific_counts"]}


def compute_handler(context: RunContext) -> Mapping[str, Any]:
    """Run the CPU oracle or stop before unsupported paper-scale GPU work."""

    members = build_members(context.config)
    profile = str(context.config.get("profile", context.profile))
    if profile != "smoke":
        raise RuntimeError(
            "paper-profile Controlled compute requires the registered accelerator backend "
            "and authorized checkpoint bytes; inspect --plan-only or run --stage analyze "
            "against sealed reference archives"
        )
    return execute_members(context, members, smoke_executor)


def analyze_handler(context: RunContext) -> Mapping[str, Any]:
    """Analyze CPU-oracle outputs or verified C0/C1/C2 reference tables."""

    profile = str(context.config.get("profile", context.profile))
    if profile == "smoke":
        return analyze_smoke(context.path / "raw", context.path / "metrics")
    paper_data = context.path / "paper_data" / "reference"
    receipts = materialize_controlled_references(
        paper_data,
        reference_root=os.environ.get("DECAF_REFERENCE_RUNS_ROOT"),
        repo_root=repository_root(),
    )
    atomic_write_json(context.path / "receipts" / "controlled_reference_inputs.json", receipts)
    bundle = load_reference_bundle(paper_data)
    result = analyze_reference_bundle(
        bundle,
        context.path / "metrics",
        targets=context.config["headline_targets"],
    )
    return {**result, "reference_inputs": len(receipts), "model_inference_performed": False}


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
