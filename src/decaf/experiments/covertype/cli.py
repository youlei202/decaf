"""Uniform command-line entrypoint for Covertype reproduction."""

from __future__ import annotations

from collections.abc import Sequence

from decaf.experiments.common import make_parser, run_cli
from decaf.experiments.covertype.analyze import analyze
from decaf.experiments.covertype.evaluate import build_formal_plan, compute, prepare
from decaf.experiments.covertype.paper import paper


def main(argv: Sequence[str] | None = None) -> int:
    """Run prepare, compute, analyze, paper, all, or a static plan audit."""

    parser = make_parser("covertype", profiles=("smoke", "integration", "paper"))
    arguments = parser.parse_args(argv)
    return run_cli(
        experiment="covertype",
        args=arguments,
        plan=build_formal_plan(),
        handlers={
            "prepare": prepare,
            "compute": compute,
            "analyze": analyze,
            "paper": paper,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
