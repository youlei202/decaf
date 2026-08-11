"""Uniform command-line entrypoint for Covertype reproduction."""

from __future__ import annotations

import json
from collections.abc import Sequence

from decaf.experiments.common import RunContext, make_parser, run_cli
from decaf.experiments.covertype.analyze import analyze
from decaf.experiments.covertype.evaluate import (
    build_formal_plan,
    compute,
    prepare,
    validate_compute_resume,
    validate_prepare_resume,
)
from decaf.experiments.covertype.paper import paper
from decaf.experiments.covertype.reference import (
    REFERENCE_SOURCE_MODE,
    validate_materialized_covertype_reference,
)


def _validate_analysis_source_resume(context: RunContext) -> dict[str, object]:
    summary_path = context.path / "metrics" / "analysis_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("resumed Covertype analysis summary is missing")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("resumed Covertype analysis summary is not valid JSON") from error
    if not isinstance(summary, dict):
        raise ValueError("resumed Covertype analysis summary must be a JSON object")
    if summary.get("source_mode") == REFERENCE_SOURCE_MODE:
        return validate_materialized_covertype_reference(context)
    if summary.get("source_mode") != "computed_run":
        raise ValueError("resumed Covertype analysis source mode is invalid")
    return validate_compute_resume(context)


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
        resume_validators={
            "prepare": validate_prepare_resume,
            "compute": validate_compute_resume,
            "analyze": _validate_analysis_source_resume,
            "paper": _validate_analysis_source_resume,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
