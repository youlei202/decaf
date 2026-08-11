"""Scientific aggregation definitions for the ImageNet-9 replay."""

from __future__ import annotations

import pandas as pd
import pytest

from decaf.experiments.imagenet9.analyze import (
    matched_magnitude_accuracy,
    protocol_ratios,
)


def _score_rows() -> pd.DataFrame:
    rows = []
    for pair_id, blend_abs, patch_abs, blend_f, patch_f in (
        ("p0", 1.0, 4.0, 2.0, 8.0),
        ("p1", 9.0, 16.0, 6.0, 12.0),
    ):
        for reveal_path, abs_score, f_score in (
            ("blend", blend_abs, blend_f),
            ("patch_A", patch_abs, patch_f),
        ):
            rows.append(
                {
                    "pair_id": pair_id,
                    "pair_type": "same_rand",
                    "model_id": "model",
                    "reveal_path": reveal_path,
                    "M": 1.0,
                    "E": 0.0,
                    "C": 0.0,
                    "F": f_score,
                    "Abs": abs_score,
                    "Net": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_protocol_ratio_is_ratio_of_filtered_means_not_mean_of_pair_ratios() -> None:
    ratios = protocol_ratios(_score_rows()).set_index(["pair_type", "patch_path", "metric"])

    assert ratios.at[("same_rand", "patch_A", "Abs"), "ratio_mean"] == pytest.approx(2.0)
    assert ratios.at[("same_rand", "patch_A", "F"), "ratio_mean"] == pytest.approx(2.5)
    assert set(ratios["operation"]) == {"filtered_mean_ratio"}


def test_matched_magnitude_accuracy_uses_the_sealed_benchmark_rows() -> None:
    benchmark = pd.DataFrame(
        {
            "method": ["Abs", "DECAF"],
            "matched_pair_accuracy": [0.35, 0.96],
            "matched_pairs": [8289, 8289],
        }
    )

    result = matched_magnitude_accuracy(_score_rows(), benchmark)

    assert result == {
        "rows": 8289,
        "decaf_accuracy": 0.96,
        "abs_accuracy": 0.35,
        "source": "sealed_matched_abs_benchmark",
    }


def test_matched_magnitude_accuracy_recomputes_raw_pair_means() -> None:
    benchmark = pd.DataFrame(
        {
            "method": ["Abs", "DECAF"],
            "matched_pair_accuracy": [0.25, 0.75],
            "matched_pairs": [4, 4],
        }
    )
    pairs = pd.DataFrame(
        {
            "abs_matched_pair_accuracy": [0.0, 0.0, 0.0, 1.0],
            "decaf_matched_pair_accuracy": [0.0, 1.0, 1.0, 1.0],
        }
    )

    result = matched_magnitude_accuracy(_score_rows(), benchmark, pairs)

    assert result["rows"] == 4
    assert result["abs_accuracy"] == 0.25
    assert result["decaf_accuracy"] == 0.75
    assert result["source"] == "recomputed_from_sealed_matched_pairs"
