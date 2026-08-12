"""Regression coverage for single-B200 attribution paper-table routing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from decaf.experiments.attribution.paper import paper
from decaf.experiments.common import load_profile


def _context(tmp_path, profile: str):
    return SimpleNamespace(
        path=tmp_path,
        profile=profile,
        config=load_profile("attribution", profile),
    )


def _manifest(path):
    return json.loads((path / "paper_data/attribution_tables.json").read_text())


def test_main_b200_scopes_produce_nonempty_registered_tables(tmp_path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    primary = pd.DataFrame(
        [
            {
                "scope": "smoke_idsds_primary",
                "dataset": "imagenet1k_idsds",
                "model": "resnet50",
                "method": "decaf_5",
                "score": 0.5,
            },
            {
                "scope": "smoke_funnybirds_primary",
                "dataset": "funnybirds",
                "model": "funnybirds_resnet50",
                "method": "decaf_5",
                "score": 0.6,
            },
        ]
    )
    primary.to_csv(metrics / "method_results.csv", index=False)
    primary.to_csv(metrics / "pairwise_differences.csv", index=False)
    primary.to_csv(metrics / "per_model_results.csv", index=False)
    primary.iloc[:1].to_csv(metrics / "timing_summary.csv", index=False)

    result = paper(_context(tmp_path, "smoke"))
    rows = {row["table"]: row for row in _manifest(tmp_path)["tables"]}

    assert result["nonempty_tables"] >= 5
    assert all(not rows[number]["schema_only"] for number in (2, 4, 6, 7, 8))


def test_large_model_and_boundary_aliases_are_not_schema_only(tmp_path) -> None:
    dino = tmp_path / "dino"
    dino_metrics = dino / "metrics"
    dino_metrics.mkdir(parents=True)
    quality = pd.DataFrame(
        [
            {
                "scope": "smoke_dinov2_g_quality",
                "dataset": "imagenet1k_idsds",
                "model": "dinov2_vit_g_14",
                "method": "decaf_5",
                "score": 0.5,
            }
        ]
    )
    timing = quality.drop(columns="scope").assign(elapsed_seconds=1.0)
    quality.to_csv(dino_metrics / "per_model_results.csv", index=False)
    timing.to_csv(dino_metrics / "timing_summary.csv", index=False)
    paper(_context(dino, "large-model-smoke"))
    dino_rows = {row["table"]: row for row in _manifest(dino)["tables"]}
    assert not dino_rows[3]["schema_only"]
    assert not dino_rows[10]["schema_only"]

    boundary = tmp_path / "boundary"
    boundary_metrics = boundary / "metrics"
    boundary_metrics.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "scope": "smoke_partimagenet_boundary",
                "dataset": "partimagenet",
                "model": "resnet50",
                "method": "decaf_5",
                "score": 0.5,
            }
        ]
    ).to_csv(boundary_metrics / "method_results.csv", index=False)
    paper(_context(boundary, "boundary-smoke"))
    boundary_rows = {row["table"]: row for row in _manifest(boundary)["tables"]}
    assert not boundary_rows[11]["schema_only"]
