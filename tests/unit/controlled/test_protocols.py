from __future__ import annotations

import numpy as np
import pytest

from decaf.experiments.controlled.protocols import (
    analytic_context_mixture,
    decompose_score_trajectory,
    geometry_specs,
    shared_gaussian_increments,
    transform_increments,
)


def test_score_oracle_routes_evidence_contradiction_and_fragility() -> None:
    evidence = decompose_score_trajectory([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], endpoint=1.0)
    contradiction = decompose_score_trajectory([0.0, 0.5, 1.0], [0.0, -0.5, -1.0], endpoint=1.0)
    fragility = decompose_score_trajectory([0.0, 0.5, 1.0], [0.0, 0.5, 0.25], endpoint=0.0)
    assert float(evidence["E"]) == pytest.approx(float(evidence["Abs"]))
    assert float(contradiction["C"]) == pytest.approx(float(contradiction["Abs"]))
    assert float(fragility["F"]) == pytest.approx(float(fragility["Abs"]))
    assert all(result["numeric_audit"]["passed"] for result in (evidence, contradiction, fragility))


def test_context_mixture_recovers_half_contradiction_at_half_mismatch() -> None:
    endpoint = np.asarray([1.0, 1.0, -1.0, -1.0])
    curves = analytic_context_mixture(endpoint, -endpoint, [0.0, 0.5])
    assert curves["C"][-1] == pytest.approx(0.5)
    assert curves["E"][-1] == pytest.approx(0.5)
    assert np.allclose(curves["Abs"], curves["E"] + curves["C"] + curves["F"])


def test_registered_geometries_are_deterministic_and_trace_matched() -> None:
    draw = shared_gaussian_increments((8, 4), seed=9)
    assert np.array_equal(draw, shared_gaussian_increments((8, 4), seed=9))
    for geometry in ("cmmr", "pixel_trace_matched", "diagonal", "power_beta_0.25"):
        transformed = transform_increments(draw, geometry)
        assert np.allclose(np.sum(draw * draw, axis=-1), np.sum(transformed * transformed, axis=-1))
    specs = geometry_specs(
        {
            "alpha_grid": [0.0, 0.5, 1.0],
            "geometries": ["cmmr", "pixel", "diagonal", "power"],
            "power_betas": [0.25, 0.5, 0.75],
        }
    )
    assert len(specs) == 6
    assert sum(spec.primary for spec in specs) == 1
