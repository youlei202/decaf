from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from decaf.paper.reference import (
    ReferenceError,
    ReferenceRun,
    discover_archive,
    materialize_inputs,
    resolve_member,
    verify_archive,
)


def _fixture_archive(path: Path) -> ReferenceRun:
    content = "model,value\na,1\nb,2\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("historical-prefix/benchmark/example.csv", content)
        bundle.writestr("historical-prefix/audit/receipt.json", "{}\n")
    return ReferenceRun(
        run_id="X0",
        family="fixture",
        scientific_status="sealed_reference",
        archive_filename=path.name,
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        archive_size_bytes=path.stat().st_size,
        archive_member_count=2,
        analysis_inputs=("benchmark/example.csv",),
    )


def test_recursive_discovery_verification_and_materialization(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "archives"
    nested.mkdir(parents=True)
    archive = nested / "fixture.zip"
    run = _fixture_archive(archive)

    discovered = discover_archive(run, [tmp_path])
    verify_archive(discovered, run)
    receipts = materialize_inputs(run, discovered, run.analysis_inputs, tmp_path / "paper_data")

    assert discovered == archive
    assert len(receipts) == 1
    assert receipts[0].resolved_member == "historical-prefix/benchmark/example.csv"
    assert receipts[0].relative_path == "X0/benchmark/example.csv"
    assert receipts[0].row_count == 2
    assert (tmp_path / "paper_data" / receipts[0].relative_path).read_text() == (
        "model,value\na,1\nb,2\n"
    )


def test_member_resolution_rejects_ambiguity_and_traversal() -> None:
    with pytest.raises(ReferenceError, match="ambiguous"):
        resolve_member(["left/data.csv", "right/data.csv"], "data.csv")
    with pytest.raises(ReferenceError, match="unsafe"):
        resolve_member(["prefix/data.csv"], "../data.csv")
