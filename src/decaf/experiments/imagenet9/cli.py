"""Uniform command-line entrypoint for the ImageNet-9 reproduction family."""

from __future__ import annotations

from collections.abc import Sequence

from decaf.experiments.common import RunContext, load_profile, make_parser, run_cli
from decaf.experiments.imagenet9.analyze import analyze
from decaf.experiments.imagenet9.evaluate import (
    build_execution_plan,
    compute,
    prepare,
    validate_compute_resume,
    validate_prepare_resume,
)
from decaf.experiments.imagenet9.gpu_runtime import b200_enabled, validate_downstream_resume
from decaf.experiments.imagenet9.paper import paper, validate_reference_replay

build_plan = build_execution_plan


def _validate_analyze_resume(context: RunContext) -> dict[str, object]:
    if b200_enabled(context.config):
        return validate_downstream_resume(context, "analyze")
    return validate_reference_replay(context)


def _validate_paper_resume(context: RunContext) -> dict[str, object]:
    if b200_enabled(context.config):
        return validate_downstream_resume(context, "paper")
    return validate_reference_replay(context)


def main(argv: Sequence[str] | None = None) -> int:
    """Run prepare, compute, analyze, paper, all, or a static plan audit."""

    parser = make_parser("imagenet9")
    arguments = parser.parse_args(argv)
    config = load_profile("imagenet9", arguments.profile, arguments.config)
    return run_cli(
        experiment="imagenet9",
        args=arguments,
        plan=build_execution_plan(config),
        handlers={
            "prepare": prepare,
            "compute": compute,
            "analyze": analyze,
            "paper": paper,
        },
        resume_validators={
            "prepare": validate_prepare_resume,
            "compute": validate_compute_resume,
            "analyze": _validate_analyze_resume,
            "paper": _validate_paper_resume,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
