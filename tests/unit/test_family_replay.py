"""Focused fail-closed tests for the four-family sealed replay orchestrator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from decaf.experiments.controlled import cli as controlled_cli
from decaf.paper.family_replay import (
    EXPECTED_ARTIFACTS,
    FamilyReplayError,
    _assert_artifact_paths,
    _FamilyAdapter,
    _inventory,
    _new_invocation_root,
    _stage_command,
    _validate_recorded_inventory,
    replay_family_adapters,
)
from decaf.paper.reference import ReferenceError, reference_roots


def test_controlled_analysis_resolves_pathsep_reference_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("DECAF_REFERENCE_RUNS_ROOT", os.pathsep.join((str(first), str(second))))
    monkeypatch.setattr(controlled_cli, "controlled_reference_complete", lambda _: False)

    class ResolvedRoots(RuntimeError):
        pass

    def materialize(
        destination: Path, *, reference_root: object, repo_root: Path
    ) -> list[dict[str, object]]:
        assert destination == tmp_path / "paper_data/reference"
        assert repo_root.is_dir()
        assert reference_root is None
        assert reference_roots(reference_root) == (first.resolve(), second.resolve())
        raise ResolvedRoots

    monkeypatch.setattr(controlled_cli, "materialize_controlled_references", materialize)
    context = SimpleNamespace(path=tmp_path, config={"profile": "paper"}, profile="paper")
    with pytest.raises(ResolvedRoots):
        controlled_cli.analyze_handler(context)


def test_inventory_excludes_every_copied_reference_directory(tmp_path: Path) -> None:
    derived = tmp_path / "metrics/derived.csv"
    derived.parent.mkdir(parents=True)
    derived.write_text("value\n1\n", encoding="utf-8")
    for directory in ("reference", "reference_inputs", "source_assets"):
        copied = tmp_path / "paper_data" / directory / "copied.csv"
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_text("value\n2\n", encoding="utf-8")

    assert _inventory(tmp_path) == [
        {
            "path": "metrics/derived.csv",
            "role": "family_analysis",
            "row_count": 1,
            "sha256": "1a80986111952a11d02e84dbed98ae00f279469aad0615d17fa81911f8a6b428",
            "size_bytes": 8,
        }
    ]


def test_exact_family_inventory_rejects_missing_and_stale_outputs() -> None:
    expected = sorted(EXPECTED_ARTIFACTS["controlled"])
    complete = [{"path": path} for path in expected]
    _assert_artifact_paths("controlled", complete)

    with pytest.raises(FamilyReplayError, match="missing="):
        _assert_artifact_paths("controlled", complete[:-1])
    with pytest.raises(FamilyReplayError, match="unexpected="):
        _assert_artifact_paths("controlled", [*complete, {"path": "metrics/stale.csv"}])


def test_recorded_inventory_rejects_post_replay_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics/result.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("value\n1\n", encoding="utf-8")
    recorded = _inventory(tmp_path)
    artifact.write_text("value\n9\n", encoding="utf-8")

    with pytest.raises(FamilyReplayError, match="inventory drifted"):
        _validate_recorded_inventory(tmp_path, recorded)


def test_each_invocation_owns_a_fresh_directory(tmp_path: Path) -> None:
    container = tmp_path / "family_replays"
    first = _new_invocation_root(container)
    (first / "stale.txt").write_text("old", encoding="utf-8")
    second = _new_invocation_root(container)

    assert first != second
    assert not (second / "stale.txt").exists()
    assert first.parent == second.parent == container.resolve()


def test_stage_receipt_command_is_cli_equivalent_and_portable(tmp_path: Path) -> None:
    adapter = _FamilyAdapter(
        "controlled", "decaf.experiments.controlled.cli", lambda _: {}, lambda _: {}
    )
    context = tmp_path / "family_replays/invocation-1/controlled"
    context.mkdir(parents=True)

    analyze = _stage_command(adapter, "analyze", context, tmp_path)
    paper = _stage_command(adapter, "paper", context, tmp_path)

    assert analyze[:3] == ["python", "-m", "decaf.experiments.controlled.cli"]
    assert analyze[-1] == "<replay-root>/family_replays/invocation-1/controlled"
    assert "--resume" not in analyze
    assert paper[-1] == "--resume"
    assert str(tmp_path) not in " ".join(analyze + paper)


def test_invalid_reference_root_replaces_stale_global_receipt_with_failure(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    receipt = tmp_path / "family_replays/family_replay_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status":"completed","family_count":4}\n', encoding="utf-8")

    with pytest.raises(ReferenceError, match="do not exist"):
        replay_family_adapters(
            tmp_path,
            repo_root=repo,
            reference_roots=[tmp_path / "missing-archive-root"],
        )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["family_count"] == 0
    assert payload["families"] == []
    assert payload["error_type"] == "ReferenceError"
    assert payload["invocation_path"].startswith("family_replays/invocation-")


def test_registered_exact_artifact_counts_cover_all_four_families() -> None:
    assert {family: len(paths) for family, paths in EXPECTED_ARTIFACTS.items()} == {
        "controlled": 19,
        "imagenet9": 13,
        "attribution": 39,
        "covertype": 25,
    }
    assert all("reference_inputs" not in path for path in EXPECTED_ARTIFACTS["attribution"])
