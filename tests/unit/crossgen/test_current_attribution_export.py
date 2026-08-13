from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.crossgen import current_attribution_export as current_export
from tools.crossgen.current_attribution_export import (
    _bind_quality,
    _quality_jobs,
    _read_selection,
    _target_jobs,
)


def _selection() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "imagenet1k_idsds",
        "model_id": "resnet50",
        "selection": "first_eight_in_frozen_candidate_order",
        "image_ids": [f"image-{index}" for index in range(8)],
        "targets": list(range(8)),
    }


def _quality_frame() -> pd.DataFrame:
    rows = []
    for index in range(8):
        e = np.asarray([0.2, 0.05])
        c = np.asarray([0.1, 0.02])
        f = np.asarray([0.01, 0.0])
        rows.append(
            {
                "image_id": f"image-{index}",
                "target_class": index,
                "patch_scores": e,
                "endpoint_effects": np.zeros(2),
                "quality_target_effects": np.zeros(2),
                "decaf_M": np.zeros(2),
                "decaf_E": e,
                "decaf_C": c,
                "decaf_F": f,
                "decaf_Abs": e + c + f,
                "spearman": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _target_frame(
    effects: np.ndarray,
    *,
    reference: str,
    intervention_operator: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [f"image-{index}" for index in range(8)],
            "target_effects": [effects.copy() for _ in range(8)],
            "reference": [reference] * 8,
            "intervention_operator": [intervention_operator] * 8,
        }
    )


def test_selection_and_job_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps(_selection()), encoding="utf-8")
    assert _read_selection(
        manifest,
        dataset="imagenet1k_idsds",
        model_id="resnet50",
    )["targets"] == list(range(8))

    endpoint = {
        "member_id": "endpoint",
        "method_id": "__deletion_targets__",
        "dataset": "imagenet1k_idsds",
        "model_id": "resnet50",
    }
    quality = []
    for method in ("decaf_3", "decaf_5", "decaf_9"):
        quality.append(
            {
                "member_id": method,
                "method_id": method,
                "dataset": "imagenet1k_idsds",
                "model_id": "resnet50",
                "scope": "smoke_idsds_primary",
                "kind": "quality",
                "image_start": 0,
                "image_stop": 8,
                "image_count": 8,
                "depends_on": [
                    {
                        "member_id": "endpoint",
                        "method_id": "__deletion_targets__",
                    }
                ],
            }
        )
    jobs = [endpoint, *quality]
    selected = _quality_jobs(
        jobs,
        dataset="imagenet1k_idsds",
        model_id="resnet50",
        methods=("decaf_3", "decaf_5", "decaf_9"),
    )
    assert [job["method_id"] for job in selected] == ["decaf_3", "decaf_5", "decaf_9"]
    targets = _target_jobs(jobs, selected, dataset="imagenet1k_idsds")
    assert targets["__deletion_targets__"]["member_id"] == "endpoint"


def test_bind_quality_restores_endpoint_and_unsigned_primary() -> None:
    endpoint = _target_frame(
        np.asarray([0.4, -0.1]),
        reference="normalized_zero",
        intervention_operator="endpoint_part_deletion",
    )
    result = _bind_quality(
        _quality_frame(),
        {"__deletion_targets__": endpoint},
        _selection(),
        dataset="imagenet1k_idsds",
    )
    assert len(result) == 8
    assert all(np.array_equal(value, [0.4, 0.1]) for value in result["decaf_M"])
    assert all(np.array_equal(value, [0.2, 0.05]) for value in result["patch_scores"])
    assert np.isfinite(result["spearman"]).all()
    assert result["reference"].eq("normalized_zero").all()
    assert result["intervention_operator"].eq("endpoint_part_deletion").all()


def test_bind_quality_rejects_signed_primary() -> None:
    frame = _quality_frame()
    frame.at[0, "patch_scores"] = np.asarray([-0.2, 0.05])
    endpoint = _target_frame(
        np.asarray([0.4, -0.1]),
        reference="normalized_zero",
        intervention_operator="endpoint_part_deletion",
    )
    with pytest.raises(ValueError, match="unsigned E"):
        _bind_quality(
            frame,
            {"__deletion_targets__": endpoint},
            _selection(),
            dataset="imagenet1k_idsds",
        )


def test_bind_funnybirds_quality_uses_two_operator_mean() -> None:
    selection = _selection()
    selection["dataset"] = "funnybirds"
    selection["model_id"] = "funnybirds_resnet50"
    endpoint = _target_frame(
        np.asarray([0.4, -0.1]),
        reference="locked_gaussian_blur_k31_sigma12_raw_rgb",
        intervention_operator="endpoint_part_deletion",
    )
    background = endpoint.copy()
    background["target_effects"] = [
        np.asarray([0.3, -0.2]) for _ in range(8)
    ]
    background["intervention_operator"] = "background_texture"
    telea = endpoint.copy()
    telea["target_effects"] = [np.asarray([0.1, -0.4]) for _ in range(8)]
    telea["intervention_operator"] = "telea_dilate3"
    result = _bind_quality(
        _quality_frame(),
        {
            "__part_deletion_targets__": endpoint,
            "__heldout_background_texture__": background,
            "__heldout_telea_dilate3__": telea,
        },
        selection,
        dataset="funnybirds",
    )
    assert result["quality_aggregation"].eq("equal_mean_within_image").all()
    expected = (
        result["spearman_background_texture"]
        + result["spearman_telea_dilate3"]
    ) / 2.0
    assert np.array_equal(result["spearman"], expected)
    assert all(
        np.allclose(value, [0.2, -0.3])
        for value in result["quality_target_effects"]
    )


def test_bind_quality_rejects_missing_or_changed_target_provenance() -> None:
    endpoint = _target_frame(
        np.asarray([0.4, -0.1]),
        reference="normalized_zero",
        intervention_operator="endpoint_part_deletion",
    )
    with pytest.raises(ValueError, match="provenance columns absent"):
        _bind_quality(
            _quality_frame(),
            {"__deletion_targets__": endpoint.drop(columns="reference")},
            _selection(),
            dataset="imagenet1k_idsds",
        )

    endpoint.at[0, "reference"] = "pixel_zero"
    with pytest.raises(ValueError, match="target reference differs"):
        _bind_quality(
            _quality_frame(),
            {"__deletion_targets__": endpoint},
            _selection(),
            dataset="imagenet1k_idsds",
        )


@pytest.mark.parametrize(
    ("dataset", "model_id"),
    [
        ("imagenet1k_idsds", "resnet50"),
        ("imagenet1k_idsds", "vgg16"),
        ("imagenet1k_idsds", "vit_base_patch16_224"),
        ("funnybirds", "funnybirds_resnet50"),
        ("funnybirds", "funnybirds_vgg16"),
        ("funnybirds", "funnybirds_vit_b_16"),
    ],
)
def test_main_parser_accepts_all_six_dataset_model_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset: str,
    model_id: str,
) -> None:
    observed: dict[str, object] = {}

    def fake_export(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"dataset": dataset, "model_id": model_id}

    monkeypatch.setattr(current_export, "export_current_attribution", fake_export)
    assert (
        current_export.main(
            [
                "--dataset",
                dataset,
                "--model",
                model_id,
                "--sample-manifest",
                str(tmp_path / "selection.json"),
                "--output",
                str(tmp_path / "output.parquet"),
                "--receipt",
                str(tmp_path / "receipt.json"),
            ]
        )
        == 0
    )
    assert observed["dataset"] == dataset
    assert observed["model_id"] == model_id
