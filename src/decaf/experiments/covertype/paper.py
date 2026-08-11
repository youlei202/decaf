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


def paper(context: RunContext) -> dict[str, Any]:
    """Emit the six registered Covertype paper tables and their source aliases."""

    metrics = context.path / "metrics"
    destination = context.path / "paper_data" / "tables"
    rank = _read(metrics / "rank_statistics.csv")
    bootstrap = _read(metrics / "bootstrap.csv")
    family = _read(metrics / "model_family_audit.csv")
    model = _read(metrics / "model_results.csv")
    costs = _read(metrics / "costs.csv")
    summary = json.loads((metrics / "analysis_summary.json").read_text(encoding="utf-8"))
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
    plan = build_formal_plan()
    table_12 = pd.DataFrame(
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
    table_13 = rank.copy()
    table_13["canonical_expression"] = ""
    mask = table_13["outcome"].eq("null_context_prediction_change_rate")
    table_13.loc[mask, "canonical_expression"] = (
        "correlation(F, null_context_prediction_change_rate)"
    )
    table_14 = family
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
        costs.groupby(["module", "model_family"], dropna=False)[["wall_seconds", "prediction_rows"]]
        .sum()
        .reset_index()
    )
    table_16.insert(0, "method", "DECAF pipeline with permutation baseline")
    outputs = {
        "table_5_covertype_behavior_and_cost.csv": table_5,
        "table_12_covertype_design.csv": table_12,
        "table_13_covertype_behavior_alignment.csv": table_13,
        "table_14_covertype_model_family_audit.csv": table_14,
        "table_15_fixed_semantics_and_magnitude.csv": table_15,
        "table_16_covertype_cost.csv": table_16,
        "rank_statistics.csv": rank,
        "bootstrap.csv": bootstrap,
        "fixed_semantic.csv": fixed,
        "matched_magnitude.csv": magnitude,
        "costs.csv": table_16,
        "model_manifest.csv": pd.DataFrame(plan["jobs"]),
    }
    for name, frame in outputs.items():
        _write(destination / name, frame)
    manifest = {
        "schema_version": 1,
        "experiment": "covertype",
        "tables": sorted(outputs),
        "source_model_count": len(model),
        "formal_model_count": plan["counts"]["total_models"],
        "canonical_fragility_correlation": summary["canonical_fragility_correlation"],
    }
    atomic_json(context.path / "paper_data" / "manifest.json", manifest)
    return {
        "table_count": 6,
        "machine_readable_files": len(outputs),
        "formal_model_count": 135,
    }


__all__ = ["paper"]
