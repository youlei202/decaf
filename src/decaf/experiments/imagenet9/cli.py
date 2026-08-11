"""Uniform command-line entrypoint for the ImageNet-9 reproduction family."""

from __future__ import annotations

from collections.abc import Sequence

from decaf.experiments.common import load_profile, make_parser, run_cli
from decaf.experiments.imagenet9.analyze import analyze
from decaf.experiments.imagenet9.evaluate import build_formal_plan, compute, prepare
from decaf.experiments.imagenet9.paper import paper, validate_reference_replay

build_plan = build_formal_plan


def main(argv: Sequence[str] | None = None) -> int:
    """Run prepare, compute, analyze, paper, all, or a static plan audit."""

    parser = make_parser("imagenet9")
    arguments = parser.parse_args(argv)
    config = load_profile("imagenet9", arguments.profile, arguments.config)
    return run_cli(
        experiment="imagenet9",
        args=arguments,
        plan=build_formal_plan(config),
        handlers={
            "prepare": prepare,
            "compute": compute,
            "analyze": analyze,
            "paper": paper,
        },
        resume_validators={
            "analyze": validate_reference_replay,
            "paper": validate_reference_replay,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
