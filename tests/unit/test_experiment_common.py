from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from decaf.experiments.common import (
    RunContext,
    TerminationRequested,
    atomic_json,
    bounded_workers,
    execute_run,
    parse_devices,
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


def test_device_parser_rejects_duplicates_and_negative_ids() -> None:
    assert parse_devices("0") == (0,)
    assert parse_devices("0, 2") == (0, 2)
    with pytest.raises(Exception, match="unique non-negative"):
        parse_devices("0,0")
    with pytest.raises(Exception, match="unique non-negative"):
        parse_devices("-1")


def test_sigterm_terminalizes_global_and_stage_receipts(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def terminate(_context: RunContext) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(TerminationRequested, match="signal"):
        execute_run(context, {"prepare": terminate})
    assert json.loads(context.run_receipt_path.read_text())["status"] == "failed"
    assert json.loads(context.stage_receipt("prepare").read_text())["status"] == "failed"
