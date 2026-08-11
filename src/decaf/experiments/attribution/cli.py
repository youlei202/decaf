"""Uniform command-line entry point for the attribution family."""

from __future__ import annotations

import json
from collections.abc import Sequence

from decaf.experiments.attribution.analyze import analyze
from decaf.experiments.attribution.evaluate import (
    compute,
    prepare,
    validate_compute_members,
)
from decaf.experiments.attribution.paper import paper
from decaf.experiments.attribution.plan import (
    build_plan,
    canonical_sha256,
    validate_plan,
)
from decaf.experiments.common import (
    RunContext,
    load_profile,
    make_parser,
    repository_root,
    run_cli,
)

PROFILES = ("smoke", "main", "paper", "large-model", "boundary")


def _validate_prepared_resume(context: RunContext) -> dict[str, object]:
    path = context.path / "manifests/plan.json"
    if not path.is_file():
        raise FileNotFoundError("resumed prepare stage has no stored plan")
    stored = json.loads(path.read_text(encoding="utf-8"))
    validate_plan(stored, raise_on_error=True)
    current = build_plan(context.config)
    if canonical_sha256(stored) != canonical_sha256(current):
        raise RuntimeError("resumed prepare plan does not match the current config")
    return {"plan_contract_sha256": stored["plan_contract_sha256"]}


def _validate_analysis_source_resume(context: RunContext) -> dict[str, object]:
    if (context.path / "manifests/plan.json").is_file():
        return validate_compute_members(context)
    from decaf.experiments.attribution.reference import (
        validate_materialized_attribution_references,
    )

    validate_materialized_attribution_references(context.path)
    return {"source_mode": "sealed_reference_replay", "validated": True}


def main(argv: Sequence[str] | None = None) -> int:
    """Plan or run one attribution profile."""

    parser = make_parser("attribution", profiles=PROFILES)
    args = parser.parse_args(argv)
    if args.profile == "large-model" and args.config is None:
        args.config = repository_root() / "configs/attribution/large_model.yaml"
    config = load_profile("attribution", args.profile, args.config)
    plan = build_plan(config)
    return run_cli(
        experiment="attribution",
        args=args,
        plan=plan,
        handlers={
            "prepare": prepare,
            "compute": compute,
            "analyze": analyze,
            "paper": paper,
        },
        resume_validators={
            "prepare": _validate_prepared_resume,
            "compute": validate_compute_members,
            "analyze": _validate_analysis_source_resume,
            "paper": _validate_analysis_source_resume,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROFILES", "main"]
