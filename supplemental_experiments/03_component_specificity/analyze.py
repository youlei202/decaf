#!/usr/bin/env python3
"""Cross-component comparisons on existing measured controlled experiments.

No inference, training, or synthetic scientific data are performed here. Each
analysis retains its original task support; C1-E, C1-F, and C2 are never pooled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

C1_MEMBER = 'C1/results/endpoint_behavior_v1_measurement/aggregate_summary.parquet'
C2_SCORES = 'C2/evaluation/epsilon_curves.csv'
C2_BEHAVIOR = 'C2/evaluation/behavior_rates.csv'
COMPONENTS = ('E', 'C', 'F', 'Abs')
EVIDENCE_SCORES = (*COMPONENTS, 'M')


@dataclass
class Design:
    name: str
    frame: pd.DataFrame
    predictors: list[str]
    outcomes: list[str]
    strata: list[str]
    group_columns: list[str]
    comparisons: list[tuple[str, str]]


def require(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f'{label}: missing columns {sorted(missing)}')


def unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f'{label}: missing or duplicate keys {keys}')


def invariant(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> None:
    for column in columns:
        if frame.groupby(keys, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f'{column} changes within {keys}')


def select_primary(frame: pd.DataFrame, geometry: str | None) -> pd.DataFrame:
    if geometry:
        selected = frame.loc[frame.geometry.eq(geometry)].copy()
    else:
        require(frame, ['primary_geometry'], 'C1')
        flag = frame.primary_geometry.astype(str).str.lower().isin({'true', '1', 'yes'})
        selected = frame.loc[flag].copy()
    if selected.empty or selected.geometry.nunique() != 1:
        raise ValueError('Choose exactly one nonempty C1 geometry; do not pool reveal paths')
    unique(selected, ['model_id', 'factor'], 'C1 selected geometry')
    values = selected[list(COMPONENTS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1e-12).any():
        raise ValueError('C1 contains invalid/nonfinite component measurements')
    if not np.allclose(values[:, :3].sum(1), values[:, 3], rtol=1e-7, atol=1e-10):
        raise ValueError('C1 components fail E+C+F=Abs')
    return selected


def evidence_design(primary: pd.DataFrame, expected: int | None) -> Design:
    data = primary.loc[primary.module.eq('E')].copy()
    meta = ['architecture', 'seed', 'p_train', 'epoch', 'trajectory_id', 'V_rev']
    require(data, ['model_id', 'factor', *meta, *EVIDENCE_SCORES], 'C1 evidence')
    if expected is not None and data.model_id.nunique() != expected:
        raise ValueError(f'Expected {expected} evidence checkpoints, got {data.model_id.nunique()}')
    if data.empty or set(data.factor) != {'wall_color', 'object_shape'}:
        raise ValueError('Evidence requires exactly wall_color and object_shape per checkpoint')
    if not data.groupby('model_id').size().eq(2).all():
        raise ValueError('Evidence factors do not share identical checkpoint support')
    invariant(data, ['model_id'], meta)
    output = data.groupby('model_id', sort=True)[meta].first()
    if output.trajectory_id.isna().any() or output.trajectory_id.astype(str).eq('').any():
        raise ValueError('Evidence needs trajectory IDs for checkpoint-cluster resampling')
    invariant(output.reset_index(), ['trajectory_id'], ['architecture', 'p_train', 'seed'])
    output['cluster_id'] = output.trajectory_id
    output['task'] = 'shortcut_reversal'
    wide = data.pivot(index='model_id', columns='factor', values=list(EVIDENCE_SCORES))
    predictors = []
    for name in EVIDENCE_SCORES:
        for factor, suffix in [('wall_color', 'wall'), ('object_shape', 'shape')]:
            key = f'{name}_{suffix}'
            output[key] = wide[name][factor]
            predictors.append(key)
        key = f'{name}_margin'
        output[key] = output[f'{name}_wall'] - output[f'{name}_shape']
        predictors.append(key)
    comparisons = [(f'E_{kind}', f'{other}_{kind}')
                   for kind in ('margin', 'wall', 'shape') for other in ('C', 'F', 'Abs', 'M')]
    return Design('C1_evidence', output.reset_index(), predictors, ['V_rev'],
                  ['architecture', 'p_train'], ['architecture', 'p_train', 'trajectory_id'],
                  comparisons)


def fragility_design(primary: pd.DataFrame, expected: int | None) -> Design:
    data = primary.loc[primary.module.eq('F') & primary.factor.eq('floor_color')].copy()
    columns = ['model_id', 'architecture', 'seed', 'variant', 'null_prediction_change_rate',
               'endpoint_null_fraction', 'clean_accuracy', *COMPONENTS]
    require(data, columns, 'C1 fragility')
    if expected is not None and len(data) != expected:
        raise ValueError(f'Expected {expected} fragility models, got {len(data)}')
    data = data[columns].copy()
    unique(data, ['model_id'], 'C1 fragility')
    data['cluster_id'] = data.architecture.astype(str) + ':' + data.seed.astype(str)
    data['task'] = 'floor_null_sensitivity'
    return Design('C1_fragility', data, list(COMPONENTS), ['null_prediction_change_rate'],
                  ['architecture'], ['architecture', 'variant'],
                  [('F', other) for other in ('E', 'C', 'Abs')])


def contradiction_design(scores: pd.DataFrame, behavior: pd.DataFrame,
                         expected: int | None) -> Design:
    keys = ['model_id', 'task', 'architecture', 'seed', 'wall_map', 'epsilon']
    targets = ['preserve_rate', 'pairwise_swap_rate', 'collapse_rate']
    require(scores, [*keys, *COMPONENTS], 'C2 scores')
    require(behavior, [*keys, *targets], 'C2 independent behavior')
    unique(scores, keys, 'C2 scores')
    unique(behavior, keys, 'C2 behavior')
    data = scores[[*keys, *COMPONENTS]].merge(behavior[[*keys, *targets]], on=keys,
                how='outer', validate='one_to_one', indicator=True)
    if not data['_merge'].eq('both').all():
        raise ValueError('C2 scores and independent behavior have different row support')
    data = data.drop(columns='_merge')
    if expected is not None and data.model_id.nunique() != expected:
        raise ValueError(f'Expected {expected} C2 models, got {data.model_id.nunique()}')
    if data.empty:
        raise ValueError('C2 is empty')
    values = data[list(COMPONENTS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1e-12).any():
        raise ValueError('C2 has invalid/nonfinite component measurements')
    if not np.allclose(values[:, :3].sum(1), values[:, 3], rtol=1e-7, atol=1e-10):
        raise ValueError('C2 components fail E+C+F=Abs')
    # epsilon in these historical C2 files is CONTEXT MISMATCH probability, not
    # the endpoint-null gate. Keep all values/maps in their training seed block.
    data = data.rename(columns={'epsilon': 'context_mismatch_probability'})
    data['cluster_id'] = data.architecture.astype(str) + ':' + data.seed.astype(str)
    return Design('C2_context_swap', data, list(COMPONENTS), targets,
                  ['architecture'], ['architecture', 'task'], list(combinations(COMPONENTS, 2)))


def rank_correlations(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorized Spearman; mathematically undefined constant ranks stay NaN."""
    if len(target) < 3:
        return np.full(values.shape[1], np.nan)
    x = rankdata(values, axis=0)
    y = rankdata(target)
    x -= x.mean(axis=0)
    y -= y.mean()
    denominator = np.sqrt((x * x).sum(0) * (y * y).sum())
    return np.divide((x * y[:, None]).sum(0), denominator,
                     out=np.full(values.shape[1], np.nan), where=denominator > 0)


def resample_blocks(frame: pd.DataFrame, strata: list[str], rng: np.random.Generator):
    """Sample complete clusters within design strata, preserving duplicates."""
    invariant(frame, ['cluster_id'], strata)
    blocks = []
    for _, part in frame.groupby(strata, sort=True, dropna=False):
        ids = part.cluster_id.unique()
        selections = rng.choice(ids, len(ids), replace=True)
        blocks.extend(part.index[part.cluster_id.eq(value)].to_numpy() for value in selections)
    return np.concatenate(blocks)


def subgroup_frames(design: Design):
    yield 'all', 'all', design.frame
    for column in design.group_columns:
        for value, sub in design.frame.groupby(column, sort=True, dropna=False):
            yield column, str(value), sub
    if 'task' in design.group_columns and 'architecture' in design.group_columns:
        for key, sub in design.frame.groupby(['task', 'architecture'], sort=True):
            yield 'task+architecture', '|'.join(map(str, key)), sub


def interval(values: np.ndarray, confidence: float, eligible: bool):
    finite = values[np.isfinite(values)]
    if not eligible or len(finite) < max(20, len(values) * 0.5):
        return np.nan, np.nan, len(finite)
    tail = (1 - confidence) / 2
    low, high = np.quantile(finite, [tail, 1 - tail])
    return float(low), float(high), len(finite)


def analyze_design(design: Design, repetitions: int, confidence: float, seed: int):
    correlations, differences, support = [], [], []
    for group_by, group, raw in subgroup_frames(design):
        for outcome in design.outcomes:
            # A single common complete-case support for all predictors in this
            # comparison; no pairwise deletion which could favor one component.
            numeric = raw[[*design.predictors, outcome]].apply(pd.to_numeric, errors='raise')
            valid = np.isfinite(numeric.to_numpy()).all(axis=1)
            data = raw.loc[valid].reset_index(drop=True)
            x = data[design.predictors].to_numpy(float)
            y = data[outcome].to_numpy(float)
            common = dict(experiment=design.name, group_by=group_by, group=group,
                          outcome=outcome, n_rows=len(data), n_clusters=data.cluster_id.nunique(),
                          excluded_rows=int((~valid).sum()), n_models=data.model_id.nunique())
            support.append(common)
            observed = rank_correlations(x, y)
            bootstrap = np.full((repetitions, len(design.predictors)), np.nan)
            n_by_stratum = data.groupby(design.strata, dropna=False).cluster_id.nunique()
            eligible = bool(len(n_by_stratum) and n_by_stratum.min() >= 2)
            if eligible:
                # Stable subgroup seeds and common draws for all scores/outcomes.
                token = f'{design.name}|{group_by}|{group}|{seed}'
                sub_seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little')
                rng = np.random.default_rng(sub_seed)
                invariant(data, ['cluster_id'], design.strata)
                blocks_by_stratum = []
                for _, part in data.groupby(design.strata, sort=True, dropna=False):
                    blocks_by_stratum.append([group.index.to_numpy()
                        for _, group in part.groupby('cluster_id', sort=True)])
                for repetition in range(repetitions):
                    selected = []
                    for blocks in blocks_by_stratum:
                        draw = rng.integers(0, len(blocks), size=len(blocks))
                        selected.extend(blocks[i] for i in draw)
                    idx = np.concatenate(selected)
                    bootstrap[repetition] = rank_correlations(x[idx], y[idx])
            for i, predictor in enumerate(design.predictors):
                low, high, valid_draws = interval(bootstrap[:, i], confidence, eligible)
                status = 'estimated'
                if not np.isfinite(observed[i]):
                    status = ('insufficient_observations' if len(data) < 3 else
                              'constant_outcome' if len(np.unique(y)) < 2 else 'constant_predictor')
                elif not eligible:
                    status = 'point_only_insufficient_independent_clusters'
                elif not np.isfinite(low):
                    status = 'point_only_insufficient_valid_bootstraps'
                correlations.append(dict(common, predictor=predictor, rho=observed[i],
                    abs_rho=abs(observed[i]), ci_low=low, ci_high=high, status=status,
                    bootstrap_valid=valid_draws, bootstrap_repetitions=repetitions,
                    confidence_level=confidence))
            for left, right in design.comparisons:
                li, ri = design.predictors.index(left), design.predictors.index(right)
                for scale in ['signed_rho', 'absolute_rho']:
                    point, draws = observed, bootstrap
                    if scale == 'absolute_rho':
                        point, draws = np.abs(point), np.abs(draws)
                    delta = point[li] - point[ri]
                    low, high, valid_draws = interval(draws[:, li] - draws[:, ri],
                                                     confidence, eligible)
                    differences.append(dict(common, left=left, right=right, scale=scale,
                        difference=delta, ci_low=low, ci_high=high, bootstrap_valid=valid_draws,
                        bootstrap_repetitions=repetitions, confidence_level=confidence,
                        status='estimated' if np.isfinite(low) else 'undefined_or_point_only'))
    return pd.DataFrame(correlations), pd.DataFrame(differences), pd.DataFrame(support)


def fingerprint(path: Path) -> dict:
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--c1-aggregate', type=Path)
    parser.add_argument('--c2-scores', type=Path)
    parser.add_argument('--c2-behavior', type=Path)
    parser.add_argument('--geometry', help='One C1 geometry; default uses primary_geometry')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--allow-partial', action='store_true',
                        help='Permit other model counts; output is explicitly marked partial')
    args = parser.parse_args(argv)
    if args.bootstrap < 20 or not 0 < args.confidence < 1:
        parser.error('bootstrap must be >=20 and confidence strictly between 0 and 1')
    if args.reference_root:
        args.c1_aggregate = args.c1_aggregate or args.reference_root / C1_MEMBER
        args.c2_scores = args.c2_scores or args.reference_root / C2_SCORES
        args.c2_behavior = args.c2_behavior or args.reference_root / C2_BEHAVIOR
    if not args.c1_aggregate and not args.c2_scores:
        parser.error('Provide --reference-root or explicit measured input paths')
    if bool(args.c2_scores) != bool(args.c2_behavior):
        parser.error('--c2-scores and --c2-behavior must be provided together')
    designs, inputs = [], []
    if args.c1_aggregate:
        frame = (pd.read_parquet(args.c1_aggregate) if args.c1_aggregate.suffix == '.parquet'
                 else pd.read_csv(args.c1_aggregate))
        require(frame, ['module', 'factor', 'geometry', *COMPONENTS], 'C1')
        primary = select_primary(frame, args.geometry)
        designs.extend([evidence_design(primary, None if args.allow_partial else 52),
                        fragility_design(primary, None if args.allow_partial else 18)])
        inputs.append(fingerprint(args.c1_aggregate))
    if args.c2_scores:
        designs.append(contradiction_design(pd.read_csv(args.c2_scores),
                       pd.read_csv(args.c2_behavior), None if args.allow_partial else 30))
        inputs.extend([fingerprint(args.c2_scores), fingerprint(args.c2_behavior)])
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output is nonempty; choose a new run directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    all_correlations, all_differences, all_support = [], [], []
    for design in designs:
        print(f'Analyzing {design.name}: {len(design.frame)} measured rows', flush=True)
        design.frame.to_csv(args.output / f'{design.name}_observations.csv', index=False)
        correlations, differences, support = analyze_design(design, args.bootstrap,
                                                             args.confidence, args.seed)
        all_correlations.append(correlations)
        all_differences.append(differences)
        all_support.append(support)
    pd.concat(all_correlations).to_csv(args.output / 'correlations.csv', index=False)
    pd.concat(all_differences).to_csv(
        args.output / 'paired_correlation_differences.csv', index=False
    )
    pd.concat(all_support).to_csv(args.output / 'support.csv', index=False)
    receipt = dict(status='completed_measured_cache_reanalysis',
                   scope='partial' if args.allow_partial else 'registered_model_counts',
                   inputs=inputs, script=fingerprint(Path(__file__)),
                   bootstrap_repetitions=args.bootstrap, confidence_level=args.confidence,
                   seed=args.seed, geometry=args.geometry or 'registered_primary',
                   experiments=[d.name for d in designs],
                   inference='paired stratified cluster percentile bootstrap; marginal intervals',
                   note='No new model inference; modules are analyzed separately, never pooled.')
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Completed measured-data analysis: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
