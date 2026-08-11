"""Ordinary attribution analysis, including the Endpoint-M baseline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.experiments.attribution.endpoint import append_endpoint_m
from decaf.experiments.attribution.evaluate import atomic_parquet, load_quality_members
from decaf.experiments.attribution.timing import TIMING_COLUMNS, summarize_timing
from decaf.experiments.common import RunContext, atomic_json

GROUP_COLUMNS = ("scope", "dataset", "method")
PRIMARY_PAIR_DIRECTIONS = (
    ("decaf_3", "endpoint_m"),
    ("decaf_5", "endpoint_m"),
    ("decaf_9", "endpoint_m"),
    ("endpoint_m", "ig_32"),
    ("endpoint_m", "ig_u_32"),
    ("endpoint_m", "kernel_shap_512"),
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically persist a CSV analysis artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _statistics(config: Mapping[str, Any]) -> tuple[int, int]:
    value = config.get("statistics", {})
    if not isinstance(value, Mapping):
        raise TypeError("statistics configuration must be a mapping")
    replicates = int(value.get("bootstrap_replicates", 1_000))
    seed = int(value.get("bootstrap_seed", 8_218))
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    return replicates, seed


def _stable_seed(seed: int, replicate: int, dataset: str, model: str) -> int:
    payload = f"{int(seed)}\0{int(replicate)}\0{dataset}\0{model}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _shared_bootstrap(
    group: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, pd.DataFrame],
    dict[str, np.ndarray],
]:
    """Sample images within fixed models, sharing every draw across methods."""

    methods = tuple(sorted(group["method"].astype(str).unique()))
    models = tuple(sorted(group["model"].astype(str).unique()))
    matrices: dict[str, pd.DataFrame] = {}
    for model in models:
        selected = group.loc[group["model"].astype(str).eq(model)]
        wide = selected.pivot(index="image_id", columns="method", values="spearman")
        if set(wide.columns) != set(methods):
            raise ValueError(f"method inventory differs for model {model}")
        wide = wide.loc[:, list(methods)].sort_index()
        if wide.empty or wide.isna().any().any() or len(wide) * len(methods) != len(selected):
            raise ValueError(f"quality is not exactly paired for model {model}")
        matrices[model] = wide
    draws = {method: np.empty(replicates, dtype=np.float64) for method in methods}
    dataset = "/".join(
        (
            str(group["scope"].iloc[0]),
            str(group["dataset"].iloc[0]),
        )
    )
    for replicate in range(replicates):
        sampled_models: list[np.ndarray] = []
        for model in models:
            values = matrices[model].to_numpy(dtype=np.float64)
            generator = np.random.default_rng(_stable_seed(seed, replicate, dataset, model))
            indices = generator.integers(0, len(values), size=len(values))
            sampled_models.append(values[indices].mean(axis=0))
        macro = np.stack(sampled_models).mean(axis=0)
        for method, value in zip(methods, macro, strict=True):
            draws[method][replicate] = float(value)
    return methods, models, matrices, draws


def summarize_quality(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return scope-level and per-model quality summaries."""

    required = {*GROUP_COLUMNS, "model", "image_id", "spearman"}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"quality results are invalid; missing={missing}")
    per_model_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for (scope, dataset), group in frame.groupby(["scope", "dataset"], sort=True, observed=True):
        methods, models, matrices, draws = _shared_bootstrap(
            group, replicates=replicates, seed=seed
        )
        image_counts = {model: len(matrices[model]) for model in models}
        for method in methods:
            model_means: list[float] = []
            model_medians: list[float] = []
            for model in models:
                values = matrices[model][method].to_numpy(dtype=np.float64)
                model_means.append(float(values.mean()))
                model_medians.append(float(np.median(values)))
                per_model_rows.append(
                    {
                        "scope": scope,
                        "dataset": dataset,
                        "method": method,
                        "model": model,
                        "mean": model_means[-1],
                        "median": model_medians[-1],
                        "sem": (
                            float(values.std(ddof=1) / np.sqrt(len(values)))
                            if len(values) > 1
                            else 0.0
                        ),
                        "n_images": len(values),
                    }
                )
            quantiles = np.quantile(draws[method], [0.025, 0.05, 0.95, 0.975])
            rows.append(
                {
                    "scope": scope,
                    "dataset": dataset,
                    "method": method,
                    "mean": float(np.mean(model_means)),
                    "median": float(np.mean(model_medians)),
                    "sem": (float(draws[method].std(ddof=1)) if replicates > 1 else 0.0),
                    "ci90_low": float(quantiles[1]),
                    "ci90_high": float(quantiles[2]),
                    "ci95_low": float(quantiles[0]),
                    "ci95_high": float(quantiles[3]),
                    "n_models": len(models),
                    "n_images_total": int(sum(image_counts.values())),
                    "n_images_per_model_json": json.dumps(image_counts, sort_keys=True),
                    "bootstrap_replicates": replicates,
                    "model_aggregation": "equal_weight_macro_average",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(per_model_rows)


def summarize_pairwise(
    frame: pd.DataFrame,
    *,
    anchor_method: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize every available registered pair as exact left-minus-right."""

    rows: list[dict[str, Any]] = []
    for (scope, dataset), group in frame.groupby(["scope", "dataset"], sort=True, observed=True):
        methods, models, matrices, method_draws = _shared_bootstrap(
            group, replicates=replicates, seed=seed
        )
        available = set(methods)
        pairs = [
            pair
            for pair in PRIMARY_PAIR_DIRECTIONS
            if pair[0] in available and pair[1] in available
        ]
        if (
            set(
                (
                    "decaf_3",
                    "decaf_5",
                    "decaf_9",
                    "endpoint_m",
                    "ig_32",
                    "ig_u_32",
                    "kernel_shap_512",
                )
            ).issubset(available)
            and len(pairs) != 6
        ):
            raise AssertionError("primary Attribution pair inventory is incomplete")
        if not pairs and {anchor_method, "endpoint_m"}.issubset(available):
            pairs = [(anchor_method, "endpoint_m")]
        for left_method, right_method in pairs:
            model_differences: dict[str, np.ndarray] = {}
            for model in models:
                wide = matrices[model]
                model_differences[model] = wide[left_method].to_numpy(dtype=np.float64) - wide[
                    right_method
                ].to_numpy(dtype=np.float64)
            bootstrap = method_draws[left_method] - method_draws[right_method]
            quantiles = np.quantile(bootstrap, [0.025, 0.05, 0.95, 0.975])
            model_means = {
                model: float(values.mean()) for model, values in model_differences.items()
            }
            model_medians = {
                model: float(np.median(values)) for model, values in model_differences.items()
            }
            rows.append(
                {
                    "scope": scope,
                    "dataset": dataset,
                    "left_method": left_method,
                    "right_method": right_method,
                    "mean_paired_difference": float(np.mean(list(model_means.values()))),
                    "median_paired_difference": float(np.mean(list(model_medians.values()))),
                    "ci90_low": float(quantiles[1]),
                    "ci90_high": float(quantiles[2]),
                    "ci95_low": float(quantiles[0]),
                    "ci95_high": float(quantiles[3]),
                    "bootstrap_probability_difference_gt_zero": float(np.mean(bootstrap > 0.0)),
                    "model_win_count": int(sum(value > 0.0 for value in model_means.values())),
                    "model_tie_count": int(sum(value == 0.0 for value in model_means.values())),
                    "n_models": len(models),
                    "n_paired_image_clusters": int(
                        sum(len(values) for values in model_differences.values())
                    ),
                    "n_per_model_json": json.dumps(
                        {model: len(values) for model, values in model_differences.items()},
                        sort_keys=True,
                    ),
                    "bootstrap_replicates": replicates,
                    "pairing": "same_resampled_images_within_each_model",
                    "sign_convention": "left_minus_right",
                    "median_aggregation": ("equal_weight_mean_of_within_model_medians"),
                }
            )
    return pd.DataFrame(rows)


def _timing_members(root: Path) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    required = set(TIMING_COLUMNS)
    plan_path = root / "manifests/plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        paths = [
            root / str(job["output_path"])
            for job in plan.get("members", [])
            if job.get("kind") in {"timing", "large_model_timing"}
        ]
    else:
        paths = sorted((root / "raw/members").rglob("*.parquet"))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"attribution timing member is absent: {path}")
        frame = pd.read_parquet(path)
        if required.issubset(frame.columns):
            frames.append(frame.loc[:, list(TIMING_COLUMNS)])
    return pd.concat(frames, ignore_index=True) if frames else None


def analyze(context: RunContext) -> dict[str, Any]:
    """Run the registered attribution analysis and generate Endpoint M."""

    if context.profile == "paper" and not (context.path / "manifests/plan.json").is_file():
        from decaf.experiments.attribution.reference import analyze_reference

        return analyze_reference(context)
    frame = load_quality_members(context)
    replicates, seed = _statistics(context.config)
    statistics = context.config.get("statistics", {})
    anchor = (
        str(statistics.get("endpoint_anchor", "decaf_5"))
        if isinstance(statistics, Mapping)
        else "decaf_5"
    )
    combined, endpoint_audit = append_endpoint_m(frame, anchor_method=anchor)
    method_results, per_model_results = summarize_quality(
        combined,
        replicates=replicates,
        seed=seed,
    )
    pairwise = summarize_pairwise(
        combined,
        anchor_method=anchor,
        replicates=replicates,
        seed=seed,
    )
    metrics = context.path / "metrics"
    atomic_parquet(combined, metrics / "per_image_with_endpoint_m.parquet")
    atomic_csv(method_results, metrics / "method_results.csv")
    atomic_csv(per_model_results, metrics / "per_model_results.csv")
    atomic_csv(pairwise, metrics / "pairwise_differences.csv")
    atomic_json(
        metrics / "endpoint_m/source_audit.json",
        {
            "schema_version": 1,
            "generated_in_stage": "analyze",
            "inference_performed": False,
            "source_column": "decaf_M",
            **endpoint_audit,
        },
    )
    timing = _timing_members(context.path)
    timing_rows = 0
    if timing is not None:
        timing_summary = summarize_timing(timing)
        atomic_csv(timing_summary, metrics / "timing_summary.csv")
        timing_rows = len(timing_summary)
    return {
        "input_rows": len(frame),
        "endpoint_m_rows": endpoint_audit["rows"],
        "endpoint_m_generated": True,
        "endpoint_m_stage": "analyze",
        "endpoint_identity_passed": endpoint_audit["passed"],
        "method_summary_rows": len(method_results),
        "pairwise_rows": len(pairwise),
        "timing_summary_rows": timing_rows,
        "bootstrap_replicates": replicates,
    }


__all__ = [
    "analyze",
    "atomic_csv",
    "summarize_pairwise",
    "summarize_quality",
]
