from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml

from decaf.paper.manifest import load_visual_manifest
from decaf.paper.reference import load_reference_runs

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "reproduce" / "package_release.py"
SPEC = spec_from_file_location("decaf_package_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PACKAGE_RELEASE = module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE_RELEASE)
_archive_repository = PACKAGE_RELEASE._archive_repository
_require_passed_reports = PACKAGE_RELEASE._require_passed_reports
_validate_historical_drift = PACKAGE_RELEASE._validate_historical_drift
_validate_package_manifest = PACKAGE_RELEASE._validate_package_manifest
_validate_provenance_manifests = PACKAGE_RELEASE._validate_provenance_manifests
_validate_source_snapshot_recovery = PACKAGE_RELEASE._validate_source_snapshot_recovery
_validate_source_snapshots = PACKAGE_RELEASE._validate_source_snapshots
_write_package_manifest = PACKAGE_RELEASE._write_package_manifest

REPOSITORY_IDENTITY = {
    "commit": "a" * 40,
    "tree": "b" * 40,
    "tracked_worktree_clean": True,
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
    ).stdout


def _passed_steps() -> dict[str, object]:
    families = {
        family: {"status": "passed"}
        for family in ("controlled", "imagenet9", "attribution", "covertype")
    }
    return {
        "quality": {"status": "passed"},
        "analysis_replay": {"status": "passed"},
        "unit": {"status": "passed"},
        "integration_cpu": {"status": "passed"},
        "full_plan": {"status": "passed", "families": families},
        "repository_audit": {"passed": True},
    }


def _bound_report(payload: dict[str, object]) -> dict[str, object]:
    return {
        **payload,
        "repository_commit": REPOSITORY_IDENTITY["commit"],
        "repository_tree": REPOSITORY_IDENTITY["tree"],
        "tracked_worktree_clean": True,
    }


def test_release_reports_require_every_structured_gate(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "analysis_replay.json",
        _bound_report(
            {
                "status": "passed",
                "reference_runs_verified": 9,
                "figures_regenerated": 12,
                "tables_regenerated": 16,
            }
        ),
    )
    _write_json(
        tmp_path / "cpu_verification.json",
        _bound_report({"status": "passed", "mode": "all-cpu", "steps": _passed_steps()}),
    )
    _write_json(tmp_path / "repository_audit.json", {"passed": True})

    analysis, cpu = _require_passed_reports(tmp_path, REPOSITORY_IDENTITY)

    assert analysis["reference_runs_verified"] == 9
    assert cpu["steps"]["quality"]["status"] == "passed"


def test_release_reports_fail_when_a_family_plan_is_absent(tmp_path: Path) -> None:
    steps = _passed_steps()
    full_plan = steps["full_plan"]
    assert isinstance(full_plan, dict)
    families = full_plan["families"]
    assert isinstance(families, dict)
    del families["attribution"]
    _write_json(
        tmp_path / "analysis_replay.json",
        _bound_report(
            {
                "status": "passed",
                "reference_runs_verified": 9,
                "figures_regenerated": 12,
                "tables_regenerated": 16,
            }
        ),
    )
    _write_json(
        tmp_path / "cpu_verification.json",
        _bound_report({"status": "passed", "mode": "all-cpu", "steps": steps}),
    )
    _write_json(tmp_path / "repository_audit.json", {"passed": True})

    with pytest.raises(RuntimeError, match="omits attribution"):
        _require_passed_reports(tmp_path, REPOSITORY_IDENTITY)


def test_release_reports_reject_a_stale_repository_identity(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "analysis_replay.json",
        _bound_report(
            {
                "status": "passed",
                "reference_runs_verified": 9,
                "figures_regenerated": 12,
                "tables_regenerated": 16,
            }
        ),
    )
    stale_cpu = _bound_report({"status": "passed", "mode": "all-cpu", "steps": _passed_steps()})
    stale_cpu["repository_commit"] = "c" * 40
    _write_json(tmp_path / "cpu_verification.json", stale_cpu)
    _write_json(tmp_path / "repository_audit.json", {"passed": True})

    with pytest.raises(RuntimeError, match="CPU verification.*packaged commit"):
        _require_passed_reports(tmp_path, REPOSITORY_IDENTITY)


def test_drift_receipt_recomputes_live_tree_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical"
    historical.mkdir()
    _git(historical, "init", "-q")
    (historical / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    _git(historical, "add", "tracked.txt")
    _git(
        historical,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    frozen_status = _git(historical, "status", "--porcelain=v1", "--untracked-files=all")
    frozen_tracked = _git(historical, "diff", "--no-ext-diff", "--binary")
    frozen_staged = _git(historical, "diff", "--cached", "--no-ext-diff", "--binary")
    frozen_head = _git(historical, "rev-parse", "HEAD").decode().strip()
    frozen_fingerprint = hashlib.sha256(
        frozen_status + b"\0" + frozen_tracked + b"\0" + frozen_staged
    ).hexdigest()
    _write_json(
        tmp_path / "historical_git_state.json",
        {
            "absolute_path": str(historical),
            "captured_at": "2026-01-01T00:00:00+00:00",
            "working_tree_fingerprint": frozen_fingerprint,
            "records": {
                "head": {"stdout": frozen_head + "\n"},
                "status_porcelain": {"stdout": frozen_status.decode()},
                "diff": {"stdout": frozen_tracked.decode()},
                "diff_staged": {"stdout": frozen_staged.decode()},
            },
        },
    )

    external = historical / "external.txt"
    external.write_text("external\n", encoding="utf-8")
    current_status = _git(historical, "status", "--porcelain=v1", "--untracked-files=all")
    current_tracked = _git(historical, "diff", "--no-ext-diff", "--binary")
    current_staged = _git(historical, "diff", "--cached", "--no-ext-diff", "--binary")
    current_fingerprint = hashlib.sha256(
        current_status + b"\0" + current_tracked + b"\0" + current_staged
    ).hexdigest()
    empty_sha = hashlib.sha256(b"").hexdigest()
    drift_path = tmp_path / "drift.json"
    _write_json(
        drift_path,
        {
            "status": "documented_external_drift",
            "historical_repository": str(historical),
            "historical_repository_modified_by_restructure": False,
            "head_unchanged": True,
            "frozen_head": frozen_head,
            "current_head": frozen_head,
            "frozen_working_tree_fingerprint": frozen_fingerprint,
            "current_working_tree_fingerprint": current_fingerprint,
            "tracked_diff": {
                "unchanged": True,
                "frozen_sha256": empty_sha,
                "current_sha256": empty_sha,
                "current_size_bytes": 0,
            },
            "staged_diff": {
                "unchanged": True,
                "frozen_sha256": empty_sha,
                "current_sha256": empty_sha,
                "current_size_bytes": 0,
            },
            "external_drift_detected": True,
            "only_additional_untracked_paths": True,
            "initial_status_line_count": 0,
            "current_status_line_count": 1,
            "added_status_lines": ["?? external.txt"],
            "removed_status_lines": [],
            "added_untracked_files": [
                {
                    "path": "external.txt",
                    "size_bytes": external.stat().st_size,
                    "sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
                }
            ],
        },
    )

    def forge_current_observation() -> None:
        report = json.loads(drift_path.read_text(encoding="utf-8"))
        live_status = _git(historical, "status", "--porcelain=v1", "--untracked-files=all")
        live_tracked = _git(historical, "diff", "--no-ext-diff", "--binary")
        live_staged = _git(historical, "diff", "--cached", "--no-ext-diff", "--binary")
        live_lines = live_status.decode().splitlines()
        report["current_head"] = _git(historical, "rev-parse", "HEAD").decode().strip()
        report["current_working_tree_fingerprint"] = hashlib.sha256(
            live_status + b"\0" + live_tracked + b"\0" + live_staged
        ).hexdigest()
        report["current_status_line_count"] = len(live_lines)
        report["added_status_lines"] = sorted(set(live_lines))
        report["removed_status_lines"] = []
        report["tracked_diff"]["current_sha256"] = hashlib.sha256(live_tracked).hexdigest()
        report["tracked_diff"]["current_size_bytes"] = len(live_tracked)
        report["staged_diff"]["current_sha256"] = hashlib.sha256(live_staged).hexdigest()
        report["staged_diff"]["current_size_bytes"] = len(live_staged)
        _write_json(drift_path, report)

    assert _validate_historical_drift(drift_path, tmp_path)["external_drift_detected"] is True
    external.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="added-file bytes"):
        _validate_historical_drift(drift_path, tmp_path)
    external.write_text("external\n", encoding="utf-8")

    (historical / "tracked.txt").write_text("changed\n", encoding="utf-8")
    forge_current_observation()
    with pytest.raises(RuntimeError, match="live tracked diff differs"):
        _validate_historical_drift(drift_path, tmp_path)

    (historical / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    _git(
        historical,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "unexpected head",
    )
    forge_current_observation()
    with pytest.raises(RuntimeError, match="live HEAD differs"):
        _validate_historical_drift(drift_path, tmp_path)


def test_snapshot_recovery_receipt_verifies_live_archives(tmp_path: Path) -> None:
    recovery_records = {
        name: {
            "repaired_sha256_match": True,
            "current_source_byte_compare": {"status": "passed"},
        }
        for name in (
            "controlled",
            "endpoint_behavior_v1",
            "imagenet9",
            "attribution",
            "covertype",
        )
    }
    snapshot_manifest = {"schema_version": 1, "snapshots": {}}
    for name, recovery in recovery_records.items():
        path = tmp_path / f"{name}.tar.gz"
        path.write_bytes(f"snapshot:{name}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        recovery["repaired_sha256"] = digest
        recovery["size_bytes"] = path.stat().st_size
        snapshot_manifest["snapshots"][name] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    _write_json(
        tmp_path / "source_snapshot_recovery.json",
        {"status": "repaired_and_verified", "snapshots": recovery_records},
    )
    (tmp_path / "source_snapshots.yaml").write_text(
        yaml.safe_dump(snapshot_manifest),
        encoding="utf-8",
    )

    recovery = _validate_source_snapshot_recovery(tmp_path)
    assert recovery["status"] == "repaired_and_verified"
    assert _validate_source_snapshots(tmp_path, recovery)["count"] == 5


def test_repository_archive_uses_captured_commit_not_symbolic_head(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    source = repository / "value.txt"
    source.write_text("captured\n", encoding="utf-8")
    _git(repository, "add", "value.txt")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "captured",
    )
    captured = _git(repository, "rev-parse", "HEAD").decode().strip()
    source.write_text("later\n", encoding="utf-8")
    _git(repository, "add", "value.txt")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "later",
    )

    destination = tmp_path / "archive"
    _archive_repository(repository, destination, captured)

    assert (destination / "value.txt").read_text(encoding="utf-8") == "captured\n"


def test_provenance_manifests_cross_check_tracked_contracts(tmp_path: Path) -> None:
    tracked_runs = load_reference_runs(REPOSITORY / "manifests/reference_runs")
    run_records = []
    server_archives = []
    for run in tracked_runs.values():
        archive_path = f"/sealed/{run.archive_filename}"
        run_records.append(
            {
                "id": run.run_id,
                "family": run.family,
                "scientific_status": run.scientific_status,
                "archive_path": archive_path,
                "archive_exists": True,
                "archive_sha256": run.archive_sha256,
                "archive_size_bytes": run.archive_size_bytes,
                "archive_member_count": run.archive_member_count,
            }
        )
        server_archives.append(
            {
                "path": archive_path,
                "exists": True,
                "kind": "file",
                "size_bytes": run.archive_size_bytes,
            }
        )
    (tmp_path / "reference_runs.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reference_run_count": 9,
                "all_archives_present": True,
                "runs": run_records,
            }
        ),
        encoding="utf-8",
    )
    _write_json(tmp_path / "historical_git_state.json", {"absolute_path": "/historical"})
    _write_json(
        tmp_path / "server_inventory.json",
        {
            "schema_version": 1,
            "historical_repository": {"path": "/historical"},
            "reference_archives": server_archives,
        },
    )
    visual = load_visual_manifest(REPOSITORY / "paper/visual_manifest.yaml")
    with (tmp_path / "paper_artifact_provenance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "artifact_type",
            "artifact_number",
            "reference_runs",
            "declared_input",
            "generator_target",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for asset in visual.assets.values():
            writer.writerow(
                {
                    "artifact_type": asset.kind,
                    "artifact_number": asset.number,
                    "reference_runs": "+".join(asset.run_ids) or "none",
                    "declared_input": (
                        asset.raw_inputs[0].member if asset.raw_inputs else "paper-only source"
                    ),
                    "generator_target": asset.tex_target,
                }
            )

    assert _validate_provenance_manifests(tmp_path, REPOSITORY) == {
        "status": "passed",
        "reference_run_count": 9,
        "paper_asset_count": 28,
        "paper_provenance_row_count": 28,
    }

    inventory = yaml.safe_load((tmp_path / "reference_runs.yaml").read_text(encoding="utf-8"))
    inventory["runs"][0]["archive_sha256"] = "0" * 64
    (tmp_path / "reference_runs.yaml").write_text(
        yaml.safe_dump(inventory),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tracked manifest"):
        _validate_provenance_manifests(tmp_path, REPOSITORY)


def test_package_manifest_rejects_payload_tamper(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "repository").mkdir(parents=True)
    payload = package / "repository" / "README.md"
    payload.write_text("release\n", encoding="utf-8")

    receipt = _write_package_manifest(package)

    assert receipt["file_count"] == 1
    assert _validate_package_manifest(package)["status"] == "passed"
    payload.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bytes differ"):
        _validate_package_manifest(package)
