"""End-to-end replay of the sealed T0 Covertype analysis inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from decaf.experiments.covertype.cli import main
from decaf.experiments.covertype.reference import (
    EXPECTED_CANONICAL_FRAGILITY_SPEARMAN,
)


def test_covertype_sealed_reference_analysis_paper_and_resume(tmp_path: Path) -> None:
    if not os.environ.get("DECAF_REFERENCE_RUNS_ROOT"):
        pytest.skip("DECAF_REFERENCE_RUNS_ROOT is not configured")
    output = tmp_path / "covertype-reference"

    assert main(["--profile", "paper", "--stage", "analyze", "--output", str(output)]) == 0
    assert not list((output / "raw" / "members").glob("*.json"))
    assert (
        main(
            [
                "--profile",
                "paper",
                "--stage",
                "paper",
                "--output",
                str(output),
                "--resume",
            ]
        )
        == 0
    )

    summary = json.loads((output / "metrics" / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["source_mode"] == "sealed_reference_replay"
    assert summary["reference_run_id"] == "T0"
    assert summary["reference_input_count"] == 13
    assert summary["model_count"] == 135
    assert summary["module_c_models"] == 90
    assert summary["module_f_models"] == 45
    assert summary["all_decaf_identities_passed"]
    assert summary["canonical_fragility_correlation"]["spearman"] == pytest.approx(
        EXPECTED_CANONICAL_FRAGILITY_SPEARMAN,
        abs=1e-14,
    )

    receipt = json.loads(
        (output / "receipts" / "covertype_reference_inputs.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "completed"
    assert receipt["run_id"] == "T0"
    assert len(receipt["inputs"]) == 13
    for item in receipt["inputs"]:
        materialized = output / "reference_data" / item["relative_path"]
        assert materialized.is_file()
        assert materialized.stat().st_size == item["size_bytes"]

    manifest = json.loads((output / "paper_data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "sealed_reference_replay"
    assert manifest["reference_input_count"] == 13
    assert len(manifest["tables"]) == 12
    for table_number in (5, 12, 13, 14, 15, 16):
        assert list((output / "paper_data" / "tables").glob(f"table_{table_number}_*.csv"))
    public_model_manifest = pd.read_csv(output / "paper_data" / "tables" / "model_manifest.csv")
    assert len(public_model_manifest) == 135
    assert "checkpoint_path" not in public_model_manifest

    assert (
        main(
            [
                "--profile",
                "paper",
                "--stage",
                "analyze",
                "--output",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    tampered = output / "reference_data" / "T0" / "results" / "benchmark" / "rank_statistics.csv"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference (size|hash) differs"):
        main(
            [
                "--profile",
                "paper",
                "--stage",
                "analyze",
                "--output",
                str(output),
                "--resume",
            ]
        )
