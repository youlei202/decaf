from __future__ import annotations

import numpy as np
import pandas as pd

from decaf.core.metrics import endpoint_magnitude
from decaf.experiments.attribution.endpoint import (
    append_endpoint_m,
    audit_endpoint_identity,
    endpoint_m_quality,
)


def test_endpoint_m_is_derived_from_persisted_endpoint_in_analysis() -> None:
    effects = np.array([-2.0, 0.2, 1.1, -0.4], dtype=np.float64)
    quality_target = np.array([0.4, -1.0, 2.2, 0.1], dtype=np.float64)
    magnitude = endpoint_magnitude(effects)
    frame = pd.DataFrame(
        [
            {
                "scope": "unit",
                "dataset": "oracle",
                "model": "oracle_linear",
                "method": "decaf_5",
                "image_id": "image-0",
                "spearman": 0.5,
                "endpoint_effects": effects,
                "quality_target_effects": quality_target,
                "decaf_M": magnitude,
                "patch_scores": effects,
            }
        ]
    )
    combined, audit = append_endpoint_m(frame)
    endpoint = combined.loc[combined["method"] == "endpoint_m"].iloc[0]
    assert audit["passed"] is True
    assert audit["rows"] == 1
    np.testing.assert_array_equal(endpoint["patch_scores"], magnitude)
    assert endpoint["source_method"] == "decaf_5"
    assert endpoint["spearman"] == endpoint_m_quality(magnitude, quality_target)[0]
    assert audit["roles_separated"] is True


def test_endpoint_identity_audit_rejects_drift() -> None:
    effects = np.array([[-1.0, 0.5, 2.0]], dtype=np.float64)
    magnitude = endpoint_magnitude(effects)
    assert audit_endpoint_identity(magnitude, effects)["passed"] is True
    drifted = magnitude.copy()
    drifted[0, 0] += 1.0e-4
    assert audit_endpoint_identity(drifted, effects)["passed"] is False


def test_endpoint_m_analysis_accepts_variable_length_semantic_parts() -> None:
    rows = []
    for image_id, effects in (
        ("short", np.array([-1.0, 0.2, 2.0])),
        ("long", np.array([-0.7, 0.1, 0.9, 1.4, -2.0])),
    ):
        rows.append(
            {
                "scope": "partimagenet_boundary",
                "dataset": "partimagenet",
                "model": "resnet50",
                "method": "decaf_5",
                "image_id": image_id,
                "spearman": 0.5,
                "endpoint_effects": effects,
                "quality_target_effects": effects.copy(),
                "decaf_M": endpoint_magnitude(effects),
            }
        )
    combined, audit = append_endpoint_m(pd.DataFrame(rows))
    assert audit["passed"] is True
    assert audit["variable_length_rows"] is True
    assert len(combined.loc[combined["method"] == "endpoint_m"]) == 2
