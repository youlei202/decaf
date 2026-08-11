from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from decaf.paper import analysis_replay
from decaf.paper.analysis_replay import (
    select_figure_02,
    select_figure_03,
    select_figure_04,
)
from decaf.paper.manifest import VisualAsset, VisualManifest


def _figure_02_frame() -> pd.DataFrame:
    rows = []
    for architecture, seed, supported, endpoint_null in (
        ("resnet18", 1, 0.04, 0.05),
        ("small_vit", 2, 0.06, 0.07),
    ):
        model_id = f"object_shape__{architecture}__seed_{seed}"
        rows.extend(
            [
                {
                    "model_id": model_id,
                    "task": "object_shape",
                    "architecture": architecture,
                    "model_seed": seed,
                    "factor": "object_shape",
                    "mean_auc_abs": supported,
                    "endpoint_class": "endpoint_supported",
                    "is_intended": True,
                },
                {
                    "model_id": model_id,
                    "task": "object_shape",
                    "architecture": architecture,
                    "model_seed": seed,
                    "factor": "floor_color",
                    "mean_auc_abs": endpoint_null,
                    "endpoint_class": "endpoint_null",
                    "is_intended": False,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_figure_02_selector_aggregates_the_fixed_matched_grid() -> None:
    selected = select_figure_02(_figure_02_frame())

    assert selected["architecture_scope"] == ["resnet18", "small_vit"]
    assert selected["model_seeds"] == [1, 2]
    assert selected["rows_per_factor"] == 2
    assert selected["intended_factor"]["mean_auc_abs"] == pytest.approx(0.05)
    assert selected["endpoint_null_factor"]["mean_auc_abs"] == pytest.approx(0.06)


def test_figure_02_selector_rejects_duplicate_grid_keys() -> None:
    frame = _figure_02_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        select_figure_02(duplicated)


def test_figure_03_selector_uses_range_then_stable_tie_breaks() -> None:
    frame = pd.DataFrame(
        {
            "module": ["E"] * 6,
            "primary_geometry": [True] * 6,
            "architecture": ["z_arch"] * 3 + ["a_arch"] * 3,
            "p_train": [0.9] * 6,
            "seed": [2] * 3 + [1] * 3,
            "trajectory_id": ["z"] * 3 + ["a"] * 3,
            "epoch": [1, 2, 3, 1, 2, 3],
            "V_rev": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        }
    )

    selected = select_figure_03(frame)

    assert selected["architecture"] == "a_arch"
    assert selected["trajectory_id"] == "a"
    assert selected["checkpoint_epochs"] == [1, 2, 3]


def test_figure_04_selector_uses_maximum_f_with_stable_tie_breaks() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["later", "first"],
            "module": ["F", "F"],
            "primary_geometry": [True, True],
            "architecture": ["z_arch", "a_arch"],
            "seed": [2, 1],
            "variant": ["fragile", "fragile"],
            "geometry": ["cmmr", "cmmr"],
            "epoch": [20, 20],
            "F": [0.8, 0.8],
            "null_prediction_change_rate": [0.7, 0.6],
            "confidence_fragility": [0.9, 0.85],
        }
    )

    assert select_figure_04(frame)["model_id"] == "first"


def test_unknown_numerical_assertion_operation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assertions = tuple(
        {
            "id": f"unknown_operation_{index:02d}",
            "operation": "unsupported_operation",
            "expected": 0.0,
        }
        for index in range(27)
    )
    asset = VisualAsset(
        asset_id="figure_01",
        kind="figure",
        number=1,
        title="Fail-closed test",
        run_ids=(),
        raw_inputs=(),
        generator="decaf.paper.render.render_source_missing_asset",
        tex_target="paper/generated/figures/figure_01.tex",
        status="source_missing",
        generation_contract={
            "status": "source_missing",
            "operation": "Historical source unavailable.",
        },
        headline_assertions=assertions,
    )
    manifest = VisualManifest(schema_version=1, assets={"figure_01": asset})
    monkeypatch.setattr(analysis_replay, "load_visual_manifest", lambda _: manifest)

    with pytest.raises(ValueError, match="unsupported numerical operation"):
        analysis_replay.replay_headline_assertions(tmp_path, tmp_path)
