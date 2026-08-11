"""Tiny CPU score-oracle coverage of paths and the authoritative DECAF core."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decaf.experiments.imagenet9.evaluate import (
    evaluate_response_frame,
    validate_response_frame,
)
from decaf.experiments.imagenet9.reveal import blend_path, patch_path


def test_blend_and_patch_paths_preserve_endpoints() -> None:
    original = np.zeros((2, 2, 1), dtype=np.float32)
    counterfactual = np.ones((2, 2, 1), dtype=np.float32)
    alpha = [0.0, 0.5, 1.0]

    blended = blend_path(original, counterfactual, alpha)
    patched = patch_path(original, counterfactual, [[0, 1], [2, 3]], alpha)

    np.testing.assert_array_equal(blended[0], original)
    np.testing.assert_array_equal(blended[-1], counterfactual)
    np.testing.assert_array_equal(patched[0], original)
    np.testing.assert_array_equal(patched[-1], counterfactual)
    assert patched[1].sum() == pytest.approx(2.0)


def test_response_evaluation_routes_all_three_mechanisms() -> None:
    rows: list[dict[str, object]] = []
    for pair_id, expected, response in (
        ("aligned", "E", [0.0, 0.4, 0.8]),
        ("opposed", "C", [0.0, 0.8, -0.1]),
        ("null", "F", [0.0, 0.4, 0.01]),
    ):
        for stage_index, (alpha, value) in enumerate(zip([0.0, 0.5, 1.0], response, strict=True)):
            rows.append(
                {
                    "pair_id": pair_id,
                    "model_id": "oracle",
                    "reveal_path": "blend",
                    "expected_component": expected,
                    "stage_index": stage_index,
                    "alpha": alpha,
                    "response": value,
                }
            )
    scores = evaluate_response_frame(pd.DataFrame(rows), epsilon=0.02)

    assert scores.set_index("pair_id")["predicted_component"].to_dict() == {
        "aligned": "E",
        "null": "F",
        "opposed": "C",
    }
    np.testing.assert_allclose(scores["Abs"], scores["E"] + scores["C"] + scores["F"])


def test_response_schema_rejects_incomplete_or_non_monotonic_paths() -> None:
    incomplete = pd.DataFrame([{"pair_id": "x"}])
    with pytest.raises(ValueError, match="missing columns"):
        validate_response_frame(incomplete)

    bad = pd.DataFrame(
        {
            "pair_id": ["x", "x", "x"],
            "model_id": ["m", "m", "m"],
            "reveal_path": ["blend", "blend", "blend"],
            "stage_index": [0, 1, 2],
            "alpha": [0.0, 0.75, 0.5],
            "response": [0.0, 0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="strictly span"):
        validate_response_frame(bad)
