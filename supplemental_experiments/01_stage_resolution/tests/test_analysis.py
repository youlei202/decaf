"""Analytic trajectories test numerical claims; these are not experiment results."""
# ruff: noqa: E402 -- tests import this standalone experiment after setting its path.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from stage_analysis import nested_indices, sample_diagnostics, summarize


def test_linear_active_and_endpoint_null_analytic_integrals():
    grid = np.linspace(0, 1, 65)
    # Linear active path has E=integral(0.4t)=0.2 and C=F=0 on every grid.
    # Endpoint-null triangle has integral 0.25, all routed to F on every grid.
    response = np.stack([0.4 * grid, np.minimum(grid, 1 - grid)])
    frame = sample_diagnostics(response, ["active", "null"], [3, 5, 9, 17, 33, 65])
    active = frame[frame.pair_id == "active"]
    null = frame[frame.pair_id == "null"]
    np.testing.assert_allclose(active.E, 0.2)
    np.testing.assert_allclose(active[["C", "F"]], 0)
    np.testing.assert_allclose(null.F, 0.25)
    np.testing.assert_allclose(null[["E", "C"]], 0)
    assert active.C_relative_error.isna().all()


def test_exact_conservation_does_not_imply_resolution_stability():
    grid = np.linspace(0, 1, 65)
    response = 0.4 * grid
    # A narrow negative pulse at 1/16 disappears on the nine-stage grid.
    response[4] = -0.8
    frame = sample_diagnostics(response[None], ["narrow_reversal"], [9, 65])
    coarse, fine = frame.iloc[0], frame.iloc[1]
    assert coarse.C == 0 and fine.C > 0
    assert coarse.missed_contradiction
    assert coarse.C_absolute_error > 0
    np.testing.assert_allclose(frame.conservation_residual, 0, atol=1e-15)


def test_threshold_equality_is_active_and_fixed_for_all_grids():
    grid = np.linspace(0, 1, 65)
    frame = sample_diagnostics(np.stack([grid * 0.02, grid * 0.019]), ["edge", "below"], [3, 65])
    assert frame.loc[frame.pair_id == "edge", "endpoint_active"].all()
    assert not frame.loc[frame.pair_id == "below", "endpoint_active"].any()
    np.testing.assert_allclose(frame.loc[frame.pair_id == "below", "E"], 0)


def test_reject_non_nested_grids():
    with pytest.raises(ValueError, match="not a nested"):
        nested_indices(65, [7, 65])
    with pytest.raises(ValueError, match="unique"):
        nested_indices(65, [9, 9, 65])


def test_model_ranking_and_identical_support_are_enforced():
    grid = np.linspace(0, 1, 65)
    frames = []
    for index, scale in enumerate([0.1, 0.2, 0.3]):
        frame = sample_diagnostics(np.stack([scale * grid, 0.01 * grid]), ["a", "b"], [9, 65])
        frame["model_id"], frame["path"], frame["pair_type"] = str(index), "blend", "same_rand"
        frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    means, ranks = summarize(samples)
    np.testing.assert_allclose(ranks.loc[ranks.component == "E", "model_rank_spearman"], 1)
    assert ranks.loc[ranks.component == "C", "model_rank_spearman"].isna().all()
    assert len(means) == 3 * 2 * 4
    changed = samples.copy()
    changed.loc[changed.model_id == "2", "pair_id"] += "changed"
    with pytest.raises(ValueError, match="identical complete pair support"):
        summarize(changed)


def test_common_neutral_is_required():
    with pytest.raises(ValueError, match="common neutral"):
        sample_diagnostics(np.ones((1, 65)) * 0.1, ["bad"], [9, 65])
