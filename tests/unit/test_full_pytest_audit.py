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


def test_pinned_environment_removes_b200_gate_and_records_only_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _module()["_pinned_environment"]
    assert callable(builder)
    covertype = tmp_path / "covertype"
    covertype.mkdir()
    (covertype / "covertype_balanced_240000_split7601.npz").write_bytes(b"archive")
    idsds = tmp_path / "idsds.parquet"
    idsds.write_bytes(b"manifest")
    reference = tmp_path / "reference.zip"
    reference.write_bytes(b"zip")
    monkeypatch.setenv("DECAF_B200_VERIFY", "1")
    monkeypatch.setenv("DECAF_DATA_ROOT", "wrong")

    environment, contract = builder(
        covertype_data_root=covertype,
        idsds_manifest=idsds,
        reference_runs_root=str(reference),
    )

    assert "DECAF_B200_VERIFY" not in environment
    assert environment["DECAF_DATA_ROOT"] == str(covertype.resolve())
    assert environment["DECAF_IDSDS_MANIFEST"] == str(idsds.resolve())
    assert contract["b200_gate_removed"] is True
    assert str(tmp_path) not in str(contract)
    assert set(contract["assets"]) == {
        "covertype_archive",
        "idsds_manifest",
        "reference_run_archives",
    }


def test_pinned_environment_requires_real_covertype_archive(tmp_path: Path) -> None:
    builder = _module()["_pinned_environment"]
    assert callable(builder)
    with pytest.raises(FileNotFoundError, match="Covertype archive"):
        builder(
            covertype_data_root=tmp_path,
            idsds_manifest=None,
            reference_runs_root=None,
        )
