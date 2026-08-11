from __future__ import annotations

import json
from pathlib import Path

from decaf.experiments.common import (
    RunContext,
    atomic_json,
    bounded_workers,
    execute_run,
    requested_stages,
)


def _context(tmp_path: Path) -> RunContext:
    return RunContext.create(
        experiment="example",
        profile="smoke",
        stage="all",
        output=tmp_path / "run",
        config={"experiment": "example"},
        workers=1,
        resume=False,
    )


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    atomic_json(target, {"status": "running"})
    atomic_json(target, {"status": "completed", "items": [1, 2]})
    assert json.loads(target.read_text()) == {
        "items": [1, 2],
        "status": "completed",
    }
    assert not list(tmp_path.glob(".*.tmp"))


def test_standard_run_schema_and_terminal_status(tmp_path: Path) -> None:
    context = _context(tmp_path)
    handlers = {
        stage: (lambda _context, name=stage: {"name": name}) for stage in requested_stages("all")
    }
    assert execute_run(context, handlers) == 0
    expected = {
        "run.json",
        "config.yaml",
        "environment.json",
        "manifests",
        "raw",
        "metrics",
        "paper_data",
        "receipts",
        "logs",
    }
    assert expected.issubset({path.name for path in context.path.iterdir()})
    assert json.loads(context.run_receipt_path.read_text())["status"] == "completed"
    assert all(context.stage_completed(stage) for stage in requested_stages("all"))


def test_failed_handler_never_leaves_running_receipt(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def fail(_context: RunContext) -> None:
        raise RuntimeError("expected failure")

    try:
        execute_run(context, {"prepare": fail})
    except RuntimeError:
        pass
    assert json.loads(context.run_receipt_path.read_text())["status"] == "failed"
    assert json.loads(context.stage_receipt("prepare").read_text())["status"] == "failed"


def test_cpu_worker_count_is_bounded() -> None:
    assert 1 <= bounded_workers(10_000) <= 32
