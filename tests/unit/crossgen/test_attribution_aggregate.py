from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decaf.core.trajectories import trajectory_scores
from tools.crossgen import attribution_aggregate as aggregate
from tools.crossgen.legacy_attribution_export import _trajectory_rows
from tools.crossgen.schema import NEUTRAL_COLUMNS, write_trajectory_record


def _neutral_frame() -> pd.DataFrame:
    grid = np.asarray([0.0, 0.5, 1.0])
    q_plus = np.asarray([0.5, 0.6, 0.8])
    q_minus = np.asarray([[0.5, 0.5, 0.4], [0.5, 0.61, 0.795]])
    endpoint = q_plus[-1] - q_minus[:, -1]
    summaries = [
        trajectory_scores(grid, q_plus - branch, effect, 0.02)
        for branch, effect in zip(q_minus, endpoint, strict=True)
    ]
    historical = {
        name: np.asarray([float(summary[name]) for summary in summaries])
        for name in aggregate.SUMMARY_NAMES
    }
    rows = _trajectory_rows(
        dataset="imagenet1k_idsds",
        reference_run="sealed",
        model_id="resnet50",
        checkpoint_sha256="a" * 64,
        image_id="image",
        target=7,
        method="decaf_3",
        part_names=("patch_00", "patch_01"),
        stage_t=grid,
        q_plus=q_plus,
        q_minus=q_minus,
        historical=historical,
        historical_endpoint_d=endpoint,
        counterfactual_map="normalized_zero_4x4_patch_deletion",
        metadata={"sealed_spearman": 0.75},
    )
    return pd.DataFrame(rows, columns=NEUTRAL_COLUMNS)


def _empty_funny_spearman() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[*aggregate.IMAGE_KEYS, "historical_spearman"]
    )


def test_historical_units_parse_exact_six_segment_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregate, "EXPECTED_UNITS", 2)
    neutral = _neutral_frame()
    result = aggregate.historical_units(neutral, _empty_funny_spearman())
    assert result["unit_id"].nunique() == 2
    assert result["factor_or_part_id"].tolist() == ["patch_00", "patch_01"]
    assert result["historical_reference"].eq("normalized_zero").all()
    assert result["historical_intervention_operator"].eq(
        "endpoint_part_deletion"
    ).all()

    changed = neutral.copy()
    changed["sample_or_pair_id"] = "different-image"
    with pytest.raises(ValueError, match="identity differs from columns"):
        aggregate.historical_units(changed, _empty_funny_spearman())


def test_combine_rejects_unit_overlap_across_source_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "trajectories"
        / "attribution__imagenet1k_idsds__resnet50.parquet"
    )
    write_trajectory_record(_neutral_frame(), path)
    monkeypatch.setattr(
        aggregate,
        "MODELS",
        {"imagenet1k_idsds": ("resnet50", "resnet50")},
    )
    with pytest.raises(ValueError, match="unit IDs overlap"):
        aggregate.combine_legacy_records(tmp_path)


def test_expand_current_is_vector_element_one_to_one_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregate, "EXPECTED_UNITS", 2)
    monkeypatch.setattr(aggregate, "EXPECTED_IMAGES", 1)
    frame = pd.DataFrame(
        [
            {
                "dataset": "imagenet1k_idsds",
                "model": "resnet50",
                "method": "decaf_3",
                "image_id": "image",
                "part_names": np.asarray(["patch_00", "patch_01"]),
                "patch_scores": np.asarray([0.2, 0.1]),
                "decaf_M": np.asarray([0.4, 0.1]),
                "decaf_E": np.asarray([0.2, 0.1]),
                "decaf_C": np.asarray([0.03, 0.02]),
                "decaf_F": np.asarray([0.01, 0.0]),
                "decaf_Abs": np.asarray([0.24, 0.12]),
                "endpoint_effects": np.asarray([0.4, -0.1]),
                "spearman": 0.75,
                "target_class": 7,
                "reference": "normalized_zero",
                "intervention_operator": "endpoint_part_deletion",
                "checkpoint_assets_json": json.dumps(
                    [{"sha256": "a" * 64}]
                ),
                "finite_complete": True,
                "numeric_audit_passed": True,
                "input_domain": "normalized_model_input",
                "model_preprocess_inside_forward": False,
            }
        ]
    )
    result = aggregate.expand_current([frame])
    assert result["factor_or_part_id"].tolist() == ["patch_00", "patch_01"]
    assert result["vector_index"].tolist() == [0, 1]
    assert result["current_counterfactual_map"].eq(
        "normalized_zero_4x4_patch_deletion"
    ).all()

    changed = frame.copy()
    changed["reference"] = "pixel_zero"
    with pytest.raises(ValueError, match="endpoint provenance differs"):
        aggregate.expand_current([changed])


def _comparison_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    historical_rows: list[dict[str, object]] = []
    current_rows: list[dict[str, object]] = []
    # Deliberately unequal vector lengths prove Spearman is not part-weighted.
    for image_id, part_count, spearman_error in (
        ("image-many-parts", 10, 0.0),
        ("image-one-part", 1, 0.008),
    ):
        for index in range(part_count):
            factor = f"patch_{index:02d}"
            unit_id = (
                "attribution::imagenet1k_idsds::resnet50::decaf_3::"
                f"{image_id}::{factor}"
            )
            keys = {
                "unit_id": unit_id,
                "dataset": "imagenet1k_idsds",
                "model": "resnet50",
                "method": "decaf_3",
                "image_id": image_id,
                "factor_or_part_id": factor,
            }
            historical_rows.append(
                {
                    **keys,
                    "historical_checkpoint_sha256": "a" * 64,
                    "historical_target": 7,
                    "historical_counterfactual_map": (
                        "normalized_zero_4x4_patch_deletion"
                    ),
                    "historical_reference": "normalized_zero",
                    "historical_intervention_operator": (
                        "endpoint_part_deletion"
                    ),
                    "historical_endpoint_d": 0.4,
                    "historical_score": 0.2,
                    "historical_M": 0.4,
                    "historical_E": 0.2,
                    "historical_C": 0.03,
                    "historical_F": 0.01,
                    "historical_Abs": 0.24,
                    "historical_spearman": 0.5,
                }
            )
            current_rows.append(
                {
                    **keys,
                    "current_checkpoint_sha256": "a" * 64,
                    "current_target": 7,
                    "current_counterfactual_map": (
                        "normalized_zero_4x4_patch_deletion"
                    ),
                    "current_reference": "normalized_zero",
                    "current_intervention_operator": (
                        "endpoint_part_deletion"
                    ),
                    "current_endpoint_d": 0.4,
                    "current_score": 0.2,
                    "current_M": 0.4,
                    "current_E": 0.2,
                    "current_C": 0.03,
                    "current_F": 0.01,
                    "current_Abs": 0.24,
                    "current_spearman": 0.5 + spearman_error,
                }
            )
    return pd.DataFrame(historical_rows), pd.DataFrame(current_rows)


def test_compare_uses_unique_image_method_spearman_not_part_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical, current = _comparison_frames()
    monkeypatch.setattr(aggregate, "EXPECTED_UNITS", 11)
    monkeypatch.setattr(aggregate, "EXPECTED_IMAGES", 2)
    comparison, spearman, summary = aggregate.compare_e2e_frames(
        historical,
        current,
    )
    assert len(comparison) == 11
    assert len(spearman) == 2
    assert summary["image_method_count"] == 2
    assert summary["spearman"]["mean_signed_error"] == pytest.approx(0.004)
    assert summary["spearman"]["median_absolute_error"] == pytest.approx(0.004)
    assert summary["identity_agreement"] == 1.0
    assert summary["non_boundary_tier_a_or_b_fraction"] == 1.0

    changed = current.copy()
    changed["current_reference"] = "pixel_zero"
    comparison, _, summary = aggregate.compare_e2e_frames(
        historical,
        changed,
    )
    assert comparison["hard_mismatch"].all()
    assert summary["reference_agreement"] == 0.0
    assert summary["identity_agreement"] == 0.0
