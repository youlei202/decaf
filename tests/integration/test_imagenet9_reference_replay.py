"""End-to-end CPU replay of the sealed ImageNet-9 analysis inputs."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from decaf.experiments.imagenet9.cli import main


def test_imagenet9_sealed_reference_analysis_and_paper(tmp_path: object) -> None:
    if not os.environ.get("DECAF_REFERENCE_RUNS_ROOT"):
        pytest.skip("DECAF_REFERENCE_RUNS_ROOT is not configured")
    output = tmp_path / "imagenet9-reference"  # type: ignore[operator]

    assert main(["--profile", "paper", "--stage", "analyze", "--output", str(output)]) == 0
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

    summary = json.loads((output / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_mode"] == "sealed_reference_replay"
    assert summary["reference_input_count"] == 11
    assert summary["reference_headlines"]["abs_accuracy"] == pytest.approx(
        0.3501829734185867, abs=1e-14
    )
    assert summary["reference_headlines"]["decaf_accuracy"] == pytest.approx(
        0.9639884183858124, abs=1e-14
    )
    ratios = pd.read_csv(output / "metrics" / "protocol_ratios.csv").set_index(
        ["pair_type", "patch_path", "metric"]
    )
    assert ratios.at[("same_rand", "patch_A", "Abs"), "ratio_mean"] == pytest.approx(
        1.7372284770086692, abs=1e-12
    )
    assert ratios.at[("same_next", "patch_A", "F"), "ratio_mean"] == pytest.approx(
        4.70751149983746, abs=1e-12
    )
    manifest = json.loads((output / "paper_data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["historical_headlines_asserted_here"] is True
    assert {asset: len(items) for asset, items in manifest["source_inputs"].items()} == {
        "figure_6": 6,
        "figure_7": 4,
        "figure_12": 3,
        "table_1": 1,
    }
    for items in manifest["source_inputs"].values():
        for item in items:
            materialized = output / item["materialized_path"]
            assert materialized.is_file()
            assert materialized.stat().st_size == item["size_bytes"]
    assert (
        manifest["supporting_outputs"]["figure_7"]["protocol_rank_transfer"]
        == "paper_data/figure_7_protocol_rank_transfer.csv"
    )

    tampered = output / "reference_data" / "I9" / "results" / "stage_ledger.jsonl"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference (size|hash) differs"):
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
