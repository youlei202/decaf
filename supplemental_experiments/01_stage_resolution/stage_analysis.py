"""Nested-grid diagnostics; all grids reuse one measured signed trajectory."""

from __future__ import annotations

import numpy as np
import pandas as pd

from decaf.core.trajectories import trajectory_scores

COMPONENTS = ("E", "C", "F", "Abs")


def nested_indices(finest: int, counts: list[int]) -> dict[int, np.ndarray]:
    if finest < 3 or not counts or max(counts) != finest:
        raise ValueError("finest must be the largest requested count (at least three)")
    if len(counts) != len(set(counts)):
        raise ValueError("stage counts must be unique")
    result = {}
    for count in counts:
        if count < 3 or (finest - 1) % (count - 1):
            raise ValueError(f"{count} is not a nested uniform grid of {finest}")
        result[count] = np.arange(0, finest, (finest - 1) // (count - 1))
    return result


def safe_ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(
        numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0
    )


def spearman(left, right) -> float:
    left, right = np.asarray(left), np.asarray(right)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(pd.Series(left).rank().corr(pd.Series(right).rank()))


def sample_diagnostics(response, pair_ids, counts, *, epsilon=0.02) -> pd.DataFrame:
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 2 or response.shape[0] != len(pair_ids):
        raise ValueError("response must be [pairs, stages] with matching pair_ids")
    if len(set(map(str, pair_ids))) != len(pair_ids):
        raise ValueError("pair_ids must be unique within a cache shard")
    if not np.allclose(response[:, 0], 0, atol=1e-7, rtol=0):
        raise ValueError("paired reveal must start at a common neutral score")
    finest = response.shape[1]
    indices = nested_indices(finest, list(counts))
    grid = np.linspace(0, 1, finest)
    reference = trajectory_scores(grid, response, epsilon=epsilon)
    frames = []
    for count, selected in indices.items():
        values = trajectory_scores(grid[selected], response[:, selected], epsilon=epsilon)
        frame = pd.DataFrame(
            {
                "pair_id": list(map(str, pair_ids)),
                "stages": count,
                "reference_stages": finest,
                "endpoint_delta": response[:, -1],
                "endpoint_active": np.abs(response[:, -1]) >= epsilon,
            }
        )
        for component in COMPONENTS:
            value, ref = values[component], reference[component]
            frame[component] = value
            frame[f"{component}_reference"] = ref
            frame[f"{component}_absolute_error"] = np.abs(value - ref)
            frame[f"{component}_relative_error"] = safe_ratio(np.abs(value - ref), ref)
            if component != "Abs":
                share = safe_ratio(value, values["Abs"])
                ref_share = safe_ratio(ref, reference["Abs"])
                frame[f"{component}_share"] = share
                frame[f"{component}_share_reference"] = ref_share
                frame[f"{component}_share_absolute_error"] = np.abs(share - ref_share)
        frame["conservation_residual"] = (frame["Abs"] - frame["E"] - frame["C"] - frame["F"]).abs()
        # Positive reference contradiction missed completely by a coarser grid.
        frame["missed_contradiction"] = (frame.C == 0) & (frame.C_reference > 0)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-model errors and across-model ranking retention.

    Zero-reference relative errors and constant-vector ranks stay undefined.
    Model ranks use the SAME complete pair support for every compared model.
    """
    keys = ["model_id", "path", "pair_type", "stages"]
    rows = []
    for key, group in samples.groupby(keys, sort=True):
        common = dict(zip(keys, key, strict=True))
        for component in COMPONENTS:
            value = group[component].to_numpy()
            ref = group[f"{component}_reference"].to_numpy()
            error = np.abs(value - ref)
            record = {
                **common,
                "component": component,
                "n_pairs": len(group),
                "mean": value.mean(),
                "reference_mean": ref.mean(),
                "mean_absolute_error": error.mean(),
                "p95_absolute_error": np.quantile(error, 0.95),
                "absolute_mean_bias": abs(value.mean() - ref.mean()),
                "normalized_mae": safe_ratio(error.mean(), ref.mean()).item(),
                "relative_mean_bias": safe_ratio(abs(value.mean() - ref.mean()), ref.mean()).item(),
                "zero_reference_pairs": int(np.count_nonzero(ref == 0)),
                "pair_rank_spearman": spearman(value, ref),
                "conservation_max_residual": group.conservation_residual.max(),
                "missed_contradiction_pairs": int(group.missed_contradiction.sum()),
            }
            if component != "Abs":
                record["aggregate_share"] = safe_ratio(value.sum(), group.Abs.sum()).item()
                record["reference_aggregate_share"] = safe_ratio(
                    ref.sum(), group.Abs_reference.sum()
                ).item()
                record["share_mean_absolute_error"] = group[
                    f"{component}_share_absolute_error"
                ].mean()
            rows.append(record)
    means = pd.DataFrame(rows)
    ranks = []
    for key, group in means.groupby(["path", "pair_type", "stages", "component"], sort=True):
        path, pair_type, stages, component = key
        support = (
            samples[
                (samples.path == path)
                & (samples.pair_type == pair_type)
                & (samples.stages == stages)
            ]
            .groupby("model_id")
            .pair_id.agg(lambda x: tuple(sorted(x)))
        )
        if support.nunique() != 1:
            raise ValueError("model ranking requires identical complete pair support")
        ranks.append(
            {
                "path": path,
                "pair_type": pair_type,
                "stages": stages,
                "component": component,
                "n_models": len(group),
                "model_rank_spearman": spearman(group["mean"], group.reference_mean),
                "model_share_rank_spearman": (
                    spearman(group.aggregate_share, group.reference_aggregate_share)
                    if component != "Abs"
                    else float("nan")
                ),
            }
        )
    return means, pd.DataFrame(ranks)
