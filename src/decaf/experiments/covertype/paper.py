"""Machine-readable Table 5 and Tables 12--16 for Covertype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from decaf.experiments.common import RunContext, atomic_json, atomic_text
from decaf.experiments.covertype.evaluate import build_formal_plan
from decaf.experiments.covertype.reference import (
    REFERENCE_SOURCE_MODE,
    validate_materialized_covertype_reference,
)


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required Covertype analysis table is missing: {path}")
    return pd.read_csv(path)


def _write(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _fixed_semantic(frame: pd.DataFrame) -> pd.DataFrame:
    units = frame.loc[
        ((frame["module"] == "C") & frame["regime"].isin(["direct", "invert"]))
        | ((frame["module"] == "F") & frame["regime"].eq("fragile"))
    ].copy()
    units["semantic_label"] = np.select(
        [
            (units["module"] == "C") & units["regime"].eq("direct"),
            (units["module"] == "C") & units["regime"].eq("invert"),
        ],
        ["Evidence", "Contradiction"],
        default="Fragility",
    )
    rows: list[dict[str, Any]] = []
    for label, score in (
        ("Evidence", "E"),
        ("Contradiction", "C"),
        ("Fragility", "F"),
    ):
        target = units["semantic_label"].eq(label).astype(np.int8)
        if target.nunique() < 2:
            auroc = None
            auprc = None
        else:
            auroc = float(roc_auc_score(target, units[score]))
            auprc = float(average_precision_score(target, units[score]))
        rows.append(
            {
                "method": "DECAF (fixed semantics)",
                "semantic_label": label,
                "score_column": score,
                "n_units": len(units),
                "auroc": auroc,
                "auprc": auprc,
                "positive_definition": {
                    "Evidence": "Module C direct",
                    "Contradiction": "Module C invert",
                    "Fragility": "Module F fragile",
                }[label],
            }
        )
    return pd.DataFrame(rows)


def _magnitude_conditioning(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component in ("E", "C", "F"):
        pair = frame[["Abs", component]].dropna()
        correlation = (
            float(pair.corr(method="spearman").iloc[0, 1])
            if len(pair) > 1 and pair["Abs"].nunique() > 1 and pair[component].nunique() > 1
            else None
        )
        rows.append(
            {
                "method": "DECAF",
                "component": component,
                "conditioning_variable": "Abs",
                "n_units": len(pair),
                "spearman_with_abs": correlation,
            }
        )
    return pd.DataFrame(rows)


def _design_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "module": "C",
                "models": 90,
                "design": "2 strengths x 3 mechanisms x 5 families x 3 seeds",
                "strengths_or_regimes": "0.75;0.95 / direct;gate;invert",
            },
            {
                "module": "F",
                "models": 45,
                "design": "3 regimes x 5 families x 3 seeds",
                "strengths_or_regimes": "robust;mild;fragile",
            },
        ]
    )


def _bootstrap_endpoint_null(
    model: pd.DataFrame, *, repetitions: int = 500, seed: int = 7601
) -> dict[str, Any]:
    """Recompute uncertainty for the canonical endpoint-null F association."""

    required = {"F", "null_context_prediction_change_rate"}
    missing = sorted(required - set(model.columns))
    if missing:
        raise ValueError(f"canonical endpoint-null bootstrap is missing columns: {missing}")
    pair = model[list(required)].dropna().reset_index(drop=True)
    if len(pair) < 3:
        raise ValueError("canonical endpoint-null bootstrap has too few models")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        sampled = pair.iloc[rng.integers(0, len(pair), size=len(pair))]
        value = float(sampled.corr(method="spearman").iloc[0, 1])
        if np.isfinite(value):
            values.append(value)
    if len(values) != repetitions:
        raise ValueError("canonical endpoint-null bootstrap produced invalid replicates")
    return {
        "ci90_low": float(np.quantile(values, 0.05)),
        "ci90_high": float(np.quantile(values, 0.95)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "bootstrap_repetitions": repetitions,
        "valid_repetitions": len(values),
        "bootstrap_seed": seed,
    }


def _sealed_tables(
    context: RunContext,
    summary: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    validation = validate_materialized_covertype_reference(context)
    metrics = context.path / "metrics"
    rank = _read(metrics / "rank_statistics.csv")
    bootstrap = _read(metrics / "bootstrap.csv")
    fixed = _read(metrics / "fixed_semantic.csv")
    matched = _read(metrics / "matched_magnitude.csv")
    family = _read(metrics / "model_family_audit.csv")
    model = _read(metrics / "model_results.csv")
    costs = _read(metrics / "costs.csv")
    shap_costs = _read(metrics / "shap_interaction_cost_summary.csv")
    model_manifest = _read(metrics / "model_manifest.csv")

    rank_bootstrap = bootstrap.loc[
        bootstrap["analysis"].eq("rank") & bootstrap["metric"].eq("spearman")
    ].copy()
    join_keys = ["module", "outcome", "method"]
    if rank_bootstrap.duplicated(join_keys).any() or len(rank_bootstrap) != len(rank):
        raise ValueError("sealed Covertype rank bootstrap inventory drifted")
    rank_bootstrap = rank_bootstrap.rename(
        columns={
            "point_estimate": "bootstrap_point_estimate",
            "ci90_low": "spearman_ci90_low",
            "ci90_high": "spearman_ci90_high",
            "ci95_low": "spearman_ci95_low",
            "ci95_high": "spearman_ci95_high",
        }
    )
    uncertainty_columns = [
        *join_keys,
        "bootstrap_point_estimate",
        "spearman_ci90_low",
        "spearman_ci90_high",
        "spearman_ci95_low",
        "spearman_ci95_high",
        "bootstrap_repetitions",
        "valid_repetitions",
        "bootstrap_seed",
    ]
    behavior = rank.merge(
        rank_bootstrap[uncertainty_columns],
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    if not np.allclose(
        behavior["spearman"], behavior["bootstrap_point_estimate"], rtol=0.0, atol=1e-12
    ):
        raise ValueError("sealed Covertype rank point estimates disagree with bootstrap")
    behavior.insert(0, "section", "behavior_rank")
    behavior["estimate"] = behavior["spearman"]

    canonical = summary["canonical_fragility_correlation"]
    module_f = model.loc[model["module"].eq("F")].copy()
    canonical_ci = _bootstrap_endpoint_null(module_f)
    canonical_row = pd.DataFrame(
        [
            {
                "section": "canonical_endpoint_null",
                "module": "F",
                "outcome": canonical["outcome"],
                "mechanism": "Fragility",
                "method": "DECAF (canonical endpoint-null)",
                "score_column": canonical["score_column"],
                "n_units": canonical["n"],
                "spearman": canonical["spearman"],
                "kendall_tau": canonical["kendall_tau"],
                "estimate": canonical["spearman"],
                "bootstrap_point_estimate": canonical["spearman"],
                "spearman_ci90_low": canonical_ci["ci90_low"],
                "spearman_ci90_high": canonical_ci["ci90_high"],
                "spearman_ci95_low": canonical_ci["ci95_low"],
                "spearman_ci95_high": canonical_ci["ci95_high"],
                "bootstrap_repetitions": canonical_ci["bootstrap_repetitions"],
                "valid_repetitions": canonical_ci["valid_repetitions"],
                "bootstrap_seed": canonical_ci["bootstrap_seed"],
            }
        ]
    )
    cost_rows = costs.copy()
    cost_rows.insert(0, "section", "method_cost")
    cost_rows["estimate"] = cost_rows["wall_seconds"]
    shap_rows = shap_costs.copy()
    shap_rows.insert(0, "section", "shap_interaction_completion")
    shap_rows["estimate"] = shap_rows["cumulative_shard_wall_seconds"]
    completed_models = int(model_manifest["status"].eq("completed").sum())
    completed_tree_models = int(shap_costs["completed_tree_models"].iloc[0])
    decaf_wall = float(costs.loc[costs["method"].eq("DECAF"), "wall_seconds"].iloc[0])
    shap_wall = float(costs.loc[costs["method"].eq("SHAP interaction"), "wall_seconds"].iloc[0])
    relative_cost = (shap_wall / completed_tree_models) / (decaf_wall / completed_models)
    table_5 = pd.concat(
        (behavior, canonical_row, cost_rows, shap_rows), ignore_index=True, sort=False
    )
    table_5["reference_cost_rows"] = len(costs)
    table_5["total_pipeline_wall_seconds"] = float(costs["wall_seconds"].sum())
    table_5["total_prediction_rows"] = int(costs["predicted_rows"].sum())
    table_5["shap_interaction_relative_cost"] = relative_cost
    table_13 = pd.concat(
        (rank, canonical_row.drop(columns=["section"])), ignore_index=True, sort=False
    )
    table_13["canonical_expression"] = ""
    table_13.loc[
        table_13["method"].eq("DECAF (canonical endpoint-null)"),
        "canonical_expression",
    ] = canonical["expression"]
    table_15 = pd.concat(
        (
            fixed.assign(section="fixed_semantic"),
            matched.assign(section="matched_magnitude"),
        ),
        ignore_index=True,
        sort=False,
    )
    table_16 = pd.concat(
        (
            costs.assign(cost_source="pipeline_cost_table"),
            shap_costs.assign(cost_source="shap_interaction_completion_summary"),
        ),
        ignore_index=True,
        sort=False,
    )
    receipt = json.loads(
        (context.path / "receipts" / "covertype_reference_inputs.json").read_text(encoding="utf-8")
    )
    outputs = {
        "table_5_covertype_behavior_and_cost.csv": table_5,
        "table_12_covertype_design.csv": _design_table(),
        "table_13_covertype_behavior_alignment.csv": table_13,
        "table_14_covertype_model_family_audit.csv": family,
        "table_15_fixed_semantics_and_magnitude.csv": table_15,
        "table_16_covertype_cost.csv": table_16,
        "rank_statistics.csv": rank,
        "bootstrap.csv": bootstrap,
        "fixed_semantic.csv": fixed,
        "matched_magnitude.csv": matched,
        "costs.csv": costs,
        "model_manifest.csv": model_manifest,
    }
    if len(model) != 135 or validation["reference_input_count"] != len(receipt["inputs"]):
        raise ValueError("sealed Covertype paper inputs differ from the T0 analysis contract")
    return outputs, receipt["inputs"]


def paper(context: RunContext) -> dict[str, Any]:
    """Emit the six registered Covertype paper tables and their source aliases."""

    metrics = context.path / "metrics"
    destination = context.path / "paper_data" / "tables"
    summary = json.loads((metrics / "analysis_summary.json").read_text(encoding="utf-8"))
    if summary.get("source_mode") == REFERENCE_SOURCE_MODE:
        outputs, source_inputs = _sealed_tables(context, summary)
        model_count = int(summary["model_count"])
    else:
        rank = _read(metrics / "rank_statistics.csv")
        bootstrap = _read(metrics / "bootstrap.csv")
        family = _read(metrics / "model_family_audit.csv")
        model = _read(metrics / "model_results.csv")
        costs = _read(metrics / "costs.csv")
        source_inputs = []
        model_count = len(model)
        table_5 = rank.merge(
            bootstrap[
                [
                    "module",
                    "component",
                    "outcome",
                    "valid_repetitions",
                    "ci_low",
                    "ci_high",
                ]
            ],
            on=["module", "component", "outcome"],
            how="left",
        )
        table_5["mean_pipeline_wall_seconds"] = float(costs["wall_seconds"].mean())
        table_5["total_prediction_rows"] = int(costs["prediction_rows"].sum())
        table_13 = rank.copy()
        table_13["canonical_expression"] = ""
        mask = table_13["outcome"].eq("null_context_prediction_change_rate")
        table_13.loc[mask, "canonical_expression"] = (
            "correlation(F, null_context_prediction_change_rate)"
        )
        fixed = _fixed_semantic(model)
        magnitude = _magnitude_conditioning(model)
        table_15 = pd.concat(
            (
                fixed.assign(section="fixed_semantic"),
                magnitude.assign(section="magnitude_conditioning"),
            ),
            ignore_index=True,
            sort=False,
        )
        table_16 = (
            costs.groupby(["module", "model_family"], dropna=False)[
                ["wall_seconds", "prediction_rows"]
            ]
            .sum()
            .reset_index()
        )
        table_16.insert(0, "method", "DECAF pipeline with permutation baseline")
        plan = build_formal_plan()
        outputs = {
            "table_5_covertype_behavior_and_cost.csv": table_5,
            "table_12_covertype_design.csv": _design_table(),
            "table_13_covertype_behavior_alignment.csv": table_13,
            "table_14_covertype_model_family_audit.csv": family,
            "table_15_fixed_semantics_and_magnitude.csv": table_15,
            "table_16_covertype_cost.csv": table_16,
            "rank_statistics.csv": rank,
            "bootstrap.csv": bootstrap,
            "fixed_semantic.csv": fixed,
            "matched_magnitude.csv": magnitude,
            "costs.csv": table_16,
            "model_manifest.csv": pd.DataFrame(plan["jobs"]),
        }
    plan = build_formal_plan()
    for name, frame in outputs.items():
        _write(destination / name, frame)
    manifest = {
        "schema_version": 1,
        "experiment": "covertype",
        "tables": sorted(outputs),
        "source_mode": summary.get("source_mode", "computed_run"),
        "source_model_count": model_count,
        "formal_model_count": plan["counts"]["total_models"],
        "canonical_fragility_correlation": summary["canonical_fragility_correlation"],
        "reference_input_count": len(source_inputs),
        "source_inputs": source_inputs,
    }
    atomic_json(context.path / "paper_data" / "manifest.json", manifest)
    return {
        "table_count": 6,
        "machine_readable_files": len(outputs),
        "formal_model_count": 135,
    }


__all__ = ["paper"]
