"""Canonical realized-behavior analysis for the Covertype family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from decaf.experiments.common import RunContext, atomic_json, atomic_text


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def load_member_frame(run_path: Path) -> pd.DataFrame:
    """Load exactly one row from every completed member artifact."""

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_path / "raw" / "members").glob("*.json"))
    ]
    if not rows:
        raise FileNotFoundError("no Covertype member artifacts are available")
    frame = pd.DataFrame(rows)
    if frame["model_id"].duplicated().any():
        raise ValueError("Covertype member artifacts contain duplicate model IDs")
    return frame.sort_values("model_id", kind="stable").reset_index(drop=True)


def _safe_rank(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 2 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return {"n": int(len(pair)), "spearman": None, "kendall_tau": None}
    return {
        "n": int(len(pair)),
        "spearman": float(spearmanr(pair["x"], pair["y"]).statistic),
        "kendall_tau": float(kendalltau(pair["x"], pair["y"]).statistic),
    }


def canonical_fragility_correlation(frame: pd.DataFrame) -> dict[str, Any]:
    """Correlate F with endpoint-null realized prediction change.

    This is the canonical implementation of the paper's endpoint-null
    fragility validation; it deliberately names the realized outcome instead
    of accepting an anonymous external correlation value.
    """

    required = {"F", "null_context_prediction_change_rate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fragility analysis is missing columns: {sorted(missing)}")
    result = _safe_rank(
        frame["F"],
        frame["null_context_prediction_change_rate"],
    )
    return {
        "component": "F",
        "outcome": "null_context_prediction_change_rate",
        "expression": "correlation(F, null_context_prediction_change_rate)",
        **result,
    }


def _bootstrap_rank(
    x: pd.Series,
    y: pd.Series,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna().reset_index(drop=True)
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return {"valid_repetitions": 0, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        sampled = pair.iloc[rng.integers(0, len(pair), size=len(pair))]
        if sampled["x"].nunique() < 2 or sampled["y"].nunique() < 2:
            continue
        value = float(spearmanr(sampled["x"], sampled["y"]).statistic)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"valid_repetitions": 0, "ci_low": None, "ci_high": None}
    return {
        "valid_repetitions": len(values),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def analyze(context: RunContext) -> dict[str, Any]:
    """Generate normalized model tables, rank statistics, and confidence intervals."""

    frame = load_member_frame(context.path)
    module_c = frame.loc[frame["module"].eq("C")].copy()
    module_f = frame.loc[frame["module"].eq("F")].copy()
    _atomic_frame(context.path / "metrics" / "model_results.csv", frame)
    _atomic_frame(context.path / "metrics" / "module_c_model_decaf.csv", module_c)
    _atomic_frame(context.path / "metrics" / "module_f_model_decaf.csv", module_f)

    definitions = (
        ("C", "E", "preserve_rate", "Evidence"),
        ("C", "C", "invert_rate", "Contradiction"),
        ("F", "F", "null_context_prediction_change_rate", "Fragility"),
    )
    rank_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    repetitions = int(context.config["analysis"]["bootstrap_repetitions"])
    bootstrap_seed = int(context.config["analysis"]["bootstrap_seed"])
    for offset, (module, component, outcome, label) in enumerate(definitions):
        subset = frame.loc[frame["module"].eq(module)]
        rank = _safe_rank(subset[component], subset[outcome])
        rank_rows.append(
            {
                "module": module,
                "semantic_label": label,
                "component": component,
                "outcome": outcome,
                **rank,
            }
        )
        bootstrap_rows.append(
            {
                "module": module,
                "semantic_label": label,
                "component": component,
                "outcome": outcome,
                "repetitions": repetitions,
                **_bootstrap_rank(
                    subset[component],
                    subset[outcome],
                    repetitions=repetitions,
                    seed=bootstrap_seed + offset,
                ),
            }
        )
    rank_statistics = pd.DataFrame(rank_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    _atomic_frame(context.path / "metrics" / "rank_statistics.csv", rank_statistics)
    _atomic_frame(context.path / "metrics" / "bootstrap.csv", bootstrap)

    numeric = [
        "M",
        "E",
        "C",
        "F",
        "Abs",
        "Net",
        "endpoint_active_rate",
        "baseline_permutation_factor_importance",
        "wall_seconds",
        "prediction_rows",
    ]
    family_audit = (
        frame.groupby(["module", "model_family", "regime"], dropna=False)[numeric]
        .mean(numeric_only=True)
        .reset_index()
    )
    _atomic_frame(context.path / "metrics" / "model_family_audit.csv", family_audit)
    costs = frame[["model_id", "module", "model_family", "wall_seconds", "prediction_rows"]].copy()
    _atomic_frame(context.path / "metrics" / "costs.csv", costs)

    fragility = canonical_fragility_correlation(module_f)
    summary = {
        "schema_version": 1,
        "model_count": len(frame),
        "module_c_models": len(module_c),
        "module_f_models": len(module_f),
        "all_decaf_identities_passed": bool(frame["decaf_identity_passed"].all()),
        "canonical_fragility_correlation": fragility,
    }
    atomic_json(context.path / "metrics" / "analysis_summary.json", summary)
    return summary


__all__ = [
    "analyze",
    "canonical_fragility_correlation",
    "load_member_frame",
]
