from __future__ import annotations

import runpy
from pathlib import Path

import pytest


def _module() -> dict[str, object]:
    return runpy.run_path("scripts/reproduce/run_full_pytest_audit.py")


def test_passed_test_count_reads_terminal_pytest_summary() -> None:
    parser = _module()["passed_test_count"]
    assert callable(parser)
    assert parser("progress\n184 passed, 2 skipped in 6.20s\n") == 184
    assert parser("7 passed in 0.12s\n") == 7
    with pytest.raises(ValueError, match="no terminal"):
        parser("collection failed\n")


def test_receipt_is_portable_and_hash_bound(tmp_path: Path) -> None:
    builder = _module()["build_receipt"]
    assert callable(builder)
    log = tmp_path / "full_pytest.log"
    log.write_text("3 passed in 0.10s\n", encoding="utf-8")
    receipt = builder(
        commit="a" * 40,
        tree="b" * 40,
        started_at="2026-08-12T18:00:00.000000Z",
        finished_at="2026-08-12T18:00:01.000000Z",
        elapsed_seconds=1.0,
        exit_code=0,
        passed_tests=3,
        log_path=log,
    )
    assert receipt["command"] == ["python", "-m", "pytest"]
    assert receipt["status"] == "passed"
    assert receipt["tracked_worktree_clean"] is True
    assert receipt["output_log"]["path"] == "verification/final_audit/full_pytest.log"
    assert len(receipt["output_log"]["sha256"]) == 64
