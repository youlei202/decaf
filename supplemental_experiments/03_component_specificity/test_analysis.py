"""Statistical contract tests; fixture numbers are never scientific run outputs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    'specificity_analysis', Path(__file__).with_name('analyze.py')
)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def test_constant_correlations_are_missing_not_zero():
    values = np.column_stack([np.ones(5), np.arange(5), -np.arange(5)])
    result = analysis.rank_correlations(values, np.arange(5))
    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1:], [1, -1])
    assert np.isnan(analysis.rank_correlations(values, np.ones(5))).all()


def test_cluster_resampling_keeps_whole_trajectories():
    frame = pd.DataFrame({'cluster_id': ['a'] * 2 + ['b'] * 3 + ['c'] * 2 + ['d'] * 3,
                          'architecture': ['R'] * 5 + ['V'] * 5})
    idx = analysis.resample_blocks(frame, ['architecture'], np.random.default_rng(17))
    counts = np.bincount(idx, minlength=len(frame))
    for _, sub in frame.groupby('cluster_id'):
        assert len(set(counts[sub.index])) == 1
    assert counts[:5].sum() in (4, 5, 6)
    assert counts[5:].sum() in (4, 5, 6)


def test_paired_bootstrap_preserves_zero_difference_and_common_support():
    frame = pd.DataFrame({'model_id': list('abcdef'), 'cluster_id': list('abcdef'),
                          'architecture': ['R'] * 6, 'E': [0, 1, 4, 3, 5, 6],
                          'C': [0, 1, 4, 3, 5, 6], 'target': [0, 1, 2, 3, 4, 5]})
    design = analysis.Design('unit_test', frame, ['E', 'C'], ['target'], ['architecture'],
                             [], [('E', 'C')])
    corr, diffs, _ = analysis.analyze_design(design, 40, 0.95, 123)
    assert corr.rho.nunique() == 1
    np.testing.assert_array_equal(diffs[['difference', 'ci_low', 'ci_high']].to_numpy(), 0)
    frame.loc[0, 'C'] = np.nan
    corr, _, support = analysis.analyze_design(design, 40, 0.95, 123)
    assert corr.n_rows.eq(5).all()
    assert support.excluded_rows.eq(1).all()


def test_one_cluster_reports_point_only_without_pseudoreplicate_ci():
    frame = pd.DataFrame({'model_id': list('abcd'), 'cluster_id': ['same'] * 4,
                          'architecture': ['R'] * 4, 'E': [1, 2, 3, 4], 'target': [1, 2, 3, 4]})
    design = analysis.Design('unit_test', frame, ['E'], ['target'], ['architecture'], [], [])
    corr, _, _ = analysis.analyze_design(design, 40, 0.95, 123)
    assert corr.rho.iloc[0] == 1
    assert pd.isna(corr.ci_low.iloc[0])
    assert corr.status.iloc[0] == 'point_only_insufficient_independent_clusters'


def test_evidence_rejects_missing_factor_support():
    frame = pd.DataFrame({'model_id': ['one'], 'module': ['E'], 'factor': ['wall_color'],
                          'architecture': ['R'], 'seed': [1], 'p_train': [0.95], 'epoch': [1],
                          'trajectory_id': ['tr1'], 'V_rev': [0.4], 'E': [0.2], 'C': [0.1],
                          'F': [0.0], 'Abs': [0.3], 'M': [0.4]})
    with pytest.raises(ValueError, match='exactly wall_color and object_shape'):
        analysis.evidence_design(frame, None)


def test_context_behavior_join_rejects_unmatched_rows():
    keys = {'model_id': ['m1'], 'task': ['invert'], 'architecture': ['R'], 'seed': [1],
            'wall_map': [1], 'epsilon': [0.2]}
    scores = pd.DataFrame(dict(keys, E=[0.4], C=[0.1], F=[0.0], Abs=[0.5]))
    behavior = pd.DataFrame(dict(keys, preserve_rate=[0.8], pairwise_swap_rate=[0.2],
                                 collapse_rate=[0.0]))
    behavior.loc[0, 'epsilon'] = 0.3
    with pytest.raises(ValueError, match='different row support'):
        analysis.contradiction_design(scores, behavior, None)
