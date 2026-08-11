from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from decaf.experiments.controlled.analyze import (
    SCHEMAS,
    ControlledReferenceBundle,
    TableSchema,
    assert_headline_targets,
    validate_frame,
)
from decaf.experiments.controlled.paper import (
    PANEL_COLUMNS,
    _module_f_bootstrap,
    panel_frame,
    write_smoke_paper_data,
)


def test_frozen_schema_cardinalities_are_registered() -> None:
    assert SCHEMAS["c0_response"].rows == 180
    assert SCHEMAS["c1_aggregate"].rows == 948
    assert SCHEMAS["c1_stages"].rows == 159264
    assert SCHEMAS["c2_epsilon"].rows == 1260
    assert SCHEMAS["c2_models"].rows == 30


def test_schema_validator_rejects_missing_columns_and_duplicate_keys() -> None:
    schema = TableSchema(
        frozenset({"model_id", "factor", "value"}), rows=2, unique=("model_id", "factor")
    )
    with pytest.raises(ValueError, match="missing columns"):
        validate_frame(pd.DataFrame({"model_id": ["m"]}), schema, label="fixture")
    duplicate = pd.DataFrame({"model_id": ["m", "m"], "factor": ["f", "f"], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate keys"):
        validate_frame(duplicate, schema, label="fixture")


def test_headline_targets_are_config_owned_and_fail_closed() -> None:
    summary = {
        "figure_02": {
            "intended_abs": 0.05,
            "endpoint_null_abs": 0.05,
            "abs_cmmr_false_null": 0.07,
            "align_cmmr_false_null": 0.0,
        },
        "figure_03": {"evidence_correspondence": 0.936},
        "figure_04": {"fragility_regimes": {"robust": 0.01, "neutral": 0.098, "fragile": 0.59}},
        "figure_05": {"invert_c": 0.5, "c_swap_spearman": 0.961, "abs_swap_spearman": -0.036},
    }
    targets = {"figure_03_evidence_correspondence": {"expected": 0.936, "tolerance": 0.0}}
    assert (
        assert_headline_targets(summary, targets)["figure_03_evidence_correspondence"]["status"]
        == "verified"
    )
    with pytest.raises(ValueError, match="headline assertion failed"):
        assert_headline_targets(
            summary,
            {"figure_03_evidence_correspondence": {"expected": 0.5, "tolerance": 0.01}},
        )


def test_panel_data_has_portable_identity_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "metrics.csv"
    frame = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "metric": ["E", "C"],
            "value": [0.4, 0.2],
            "n_values": [3, 3],
        }
    )
    frame.to_csv(source, index=False)
    panel = panel_frame(
        frame,
        artifact_id="fixture",
        panel_id="a",
        source=source,
        series="metric",
        x="model_id",
        estimate="value",
        n="n_values",
    )
    assert tuple(panel.columns[: len(PANEL_COLUMNS)]) == PANEL_COLUMNS
    assert panel["source_sha256"].str.len().eq(64).all()
    result = write_smoke_paper_data(source, tmp_path / "paper")
    assert result["artifacts"] == 1
    receipt = json.loads((tmp_path / "paper" / "controlled_receipt.json").read_text())
    assert receipt["gpu_real_shard_verification"] == "pending"


def test_module_f_bootstrap_uses_only_the_endpoint_null_factor(tmp_path: Path) -> None:
    source = tmp_path / "per_sample_bootstrap.parquet"
    floor_values = (
        0.6078392242306401,
        0.48760556160305396,
        0.6754459869087547,
    )
    rows = []
    for seed, floor_value in zip((5301, 5302, 5303), floor_values, strict=True):
        model_id = f"f__fragile__small_vit__seed_{seed}"
        rows.extend(
            (
                {
                    "model_id": model_id,
                    "module": "F",
                    "variant": "fragile",
                    "architecture": "small_vit",
                    "factor": "floor_color",
                    "primary_geometry": 1.0,
                    "F": floor_value,
                },
                {
                    "model_id": model_id,
                    "module": "F",
                    "variant": "fragile",
                    "architecture": "small_vit",
                    "factor": "object_shape",
                    "primary_geometry": 1.0,
                    "F": 0.0,
                },
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(source, index=False)
    bundle = ControlledReferenceBundle({"c1_bootstrap": frame}, {"c1_bootstrap": source})

    result = _module_f_bootstrap(bundle)

    assert len(result) == 1
    assert result.iloc[0]["estimate"] == pytest.approx(0.5902969242474829, abs=1e-15)
