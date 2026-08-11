from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reproduce" / "package_release.py"
SPEC = spec_from_file_location("decaf_package_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PACKAGE_RELEASE = module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE_RELEASE)
_require_passed_reports = PACKAGE_RELEASE._require_passed_reports
_validate_historical_drift = PACKAGE_RELEASE._validate_historical_drift
_validate_source_snapshot_recovery = PACKAGE_RELEASE._validate_source_snapshot_recovery


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_release_reports_require_every_structured_gate(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "analysis_replay.json",
        {
            "status": "passed",
            "reference_runs_verified": 9,
            "figures_regenerated": 12,
            "tables_regenerated": 16,
        },
    )
    _write_json(
        tmp_path / "cpu_verification.json",
        {"status": "passed", "mode": "all-cpu", "steps": _passed_steps()},
    )
    _write_json(tmp_path / "repository_audit.json", {"passed": True})

    analysis, cpu = _require_passed_reports(tmp_path)

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
        {
            "status": "passed",
            "reference_runs_verified": 9,
            "figures_regenerated": 12,
            "tables_regenerated": 16,
        },
    )
    _write_json(
        tmp_path / "cpu_verification.json",
        {"status": "passed", "mode": "all-cpu", "steps": steps},
    )
    _write_json(tmp_path / "repository_audit.json", {"passed": True})

    with pytest.raises(RuntimeError, match="omits attribution"):
        _require_passed_reports(tmp_path)


def test_drift_and_snapshot_recovery_receipts_fail_closed(tmp_path: Path) -> None:
    drift_path = tmp_path / "drift.json"
    _write_json(
        drift_path,
        {
            "status": "documented_external_drift",
            "historical_repository_modified_by_restructure": False,
            "head_unchanged": True,
            "tracked_diff": {"unchanged": True},
            "staged_diff": {"unchanged": True},
            "external_drift_detected": True,
            "only_additional_untracked_paths": True,
            "added_untracked_files": [{"path": "external.txt"}],
        },
    )
    recovery_records = {
        name: {
            "repaired_sha256_match": True,
            "current_source_byte_compare": {"status": "passed"},
        }
        for name in ("controlled", "endpoint", "imagenet9", "attribution", "covertype")
    }
    _write_json(
        tmp_path / "source_snapshot_recovery.json",
        {"status": "repaired_and_verified", "snapshots": recovery_records},
    )

    assert _validate_historical_drift(drift_path)["external_drift_detected"] is True
    assert _validate_source_snapshot_recovery(tmp_path)["status"] == "repaired_and_verified"

    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    drift["tracked_diff"]["unchanged"] = False
    _write_json(drift_path, drift)
    with pytest.raises(RuntimeError, match="tracked diff"):
        _validate_historical_drift(drift_path)
