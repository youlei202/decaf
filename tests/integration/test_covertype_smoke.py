"""Real CPU end-to-end integration for the Covertype experiment family."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from decaf.experiments.covertype.cli import main


def test_covertype_smoke_trains_evaluates_analyzes_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "covertype-smoke"
    assert main(["--stage", "all", "--profile", "smoke", "--output", str(output)]) == 0

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
    assert data["source_kind"] == "deterministic_synthetic_covtype_fixture"
    assert data["fixture_is_smoke_only"]
    assert len(member_paths) == 4
    assert compute_receipt["status"] == "completed"
    assert compute_receipt["all_processes_exited"]
    assert analysis["all_decaf_identities_passed"]
    assert analysis["module_c_models"] == 2
    assert analysis["module_f_models"] == 2
    assert analysis["canonical_fragility_correlation"]["expression"] == (
        "correlation(F, null_context_prediction_change_rate)"
    )
    assert paper_manifest["formal_model_count"] == 135
    for table_number in (5, 12, 13, 14, 15, 16):
        assert list((output / "paper_data" / "tables").glob(f"table_{table_number}_*.csv"))

    frame = pd.read_csv(output / "metrics" / "model_results.csv")
    assert len(frame) == 4
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
                "smoke",
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
    assert resumed_compute["details"]["resumed_members"] == 4
    assert before == {path.name: path.stat().st_mtime_ns for path in member_paths}

    assert (
        main(
            [
                "--stage",
                "all",
                "--profile",
                "smoke",
                "--output",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    after = {path.name: path.stat().st_mtime_ns for path in member_paths}
    assert before == after
