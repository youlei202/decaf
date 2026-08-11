"""Pinned-real-data CPU integration for the Covertype experiment family."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from decaf.experiments.covertype.cli import main

EXPECTED_ARCHIVE_SHA256 = "681f893d49757e4d588115430b072980df2f4c281acedb1183b53ef5b4e443de"


def test_covertype_real_shard_trains_evaluates_analyzes_and_resumes(tmp_path: Path) -> None:
    data_root = os.environ.get("DECAF_DATA_ROOT")
    assert data_root, (
        "DECAF_DATA_ROOT must point to the pinned real Covertype cache directory; "
        "integration-cpu intentionally fails instead of using synthetic data"
    )
    archive = Path(data_root) / "covertype_balanced_240000_split7601.npz"
    assert archive.is_file(), f"pinned real Covertype cache is missing: {archive}"

    output = tmp_path / "covertype-real-integration"
    assert main(["--stage", "all", "--profile", "integration", "--output", str(output)]) == 0

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    data = json.loads((output / "manifests" / "data.json").read_text(encoding="utf-8"))
    compute_receipt = json.loads(
        (output / "receipts" / "compute_members.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (output / "metrics" / "analysis_summary.json").read_text(encoding="utf-8")
    )
    paper_manifest = json.loads(
        (output / "paper_data" / "manifest.json").read_text(encoding="utf-8")
    )
    member_paths = sorted((output / "raw" / "members").glob("*.json"))

    assert run["status"] == "completed"
    assert data["source_kind"] == "sklearn.datasets.fetch_covtype"
    assert data["fallback_reason"] is None
    assert not data["fixture_is_smoke_only"]
    assert data["rows"] == {"train": 1200, "validation": 400, "test": 400}
    assert data["source_archive"]["archive_sha256"] == EXPECTED_ARCHIVE_SHA256
    assert data["source_archive"]["transport"] == "pinned_npz_cache"
    assert data["source_archive"]["root_environment_variable"] == "DECAF_DATA_ROOT"
    assert data["source_archive"]["fixed_shard"]["selection"] == (
        "lowest_source_indices_per_class_within_frozen_split"
    )
    assert len(member_paths) == 2
    member_receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "receipts" / "members").glob("*.json"))
    ]
    assert len(member_receipts) == 2
    for receipt in member_receipts:
        details = receipt["details"]
        assert len(details["artifact_sha256"]) == 64
        assert details["artifact_size_bytes"] > 0
        assert details["dataset_fingerprint"] == data["fingerprint"]
        assert details["record_identity"]["model_id"] == receipt["member_id"]
    assert compute_receipt["status"] == "completed"
    assert compute_receipt["all_processes_exited"]
    assert analysis["all_decaf_identities_passed"]
    assert analysis["module_c_models"] == 1
    assert analysis["module_f_models"] == 1
    assert analysis["canonical_fragility_correlation"]["expression"] == (
        "correlation(F, null_context_prediction_change_rate)"
    )
    assert paper_manifest["formal_model_count"] == 135
    for table_number in (5, 12, 13, 14, 15, 16):
        assert list((output / "paper_data" / "tables").glob(f"table_{table_number}_*.csv"))

    frame = pd.read_csv(output / "metrics" / "model_results.csv")
    assert len(frame) == 2
    assert set(frame["model_family"]) == {"hist_gradient_boosting"}
    assert set(frame["seed"]) == {7701}
    assert set(frame.loc[frame["module"] == "C", "regime"]) == {"direct"}
    assert set(frame.loc[frame["module"] == "F", "regime"]) == {"fragile"}
    assert frame["decaf_identity_passed"].all()
    assert frame["baseline_permutation_factor_importance"].notna().all()
    assert set(frame["model_implementation"]) == {
        "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier"
    }

    before = {path.name: path.stat().st_mtime_ns for path in member_paths}
    (output / "receipts" / "compute.json").unlink()
    assert (
        main(
            [
                "--stage",
                "compute",
                "--profile",
                "integration",
                "--output",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    resumed_compute = json.loads(
        (output / "receipts" / "compute_members.json").read_text(encoding="utf-8")
    )
    assert resumed_compute["status"] == "completed"
    assert resumed_compute["details"]["resumed_members"] == 2
    assert before == {path.name: path.stat().st_mtime_ns for path in member_paths}

    assert (
        main(
            [
                "--stage",
                "all",
                "--profile",
                "integration",
                "--output",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    after = {path.name: path.stat().st_mtime_ns for path in member_paths}
    assert before == after

    victim = member_paths[0]
    victim.write_text(victim.read_text(encoding="utf-8") + " ", encoding="utf-8")
    (output / "receipts" / "compute.json").unlink()
    with pytest.raises(RuntimeError, match="artifact (size|hash) mismatch"):
        main(
            [
                "--stage",
                "compute",
                "--profile",
                "integration",
                "--output",
                str(output),
                "--resume",
            ]
        )
