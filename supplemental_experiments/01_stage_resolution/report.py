#!/usr/bin/env python3
"""Summarize completed measured results and verify fixed endpoints across paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_review_summary(output, destination, status, manifest, metrics, ranks, checks):
    samples = pd.read_parquet(output / "analysis/sample_metrics.parquet")
    rows = []
    for key, group in metrics[metrics.stages.isin([9, 17, 33])].groupby(
        ["path", "pair_type", "stages", "component"], sort=True
    ):
        path, pair_type, stages, component = key
        rank = ranks[
            (ranks.path == path)
            & (ranks.pair_type == pair_type)
            & (ranks.stages == stages)
            & (ranks.component == component)
        ].iloc[0]
        reference_mean = group.reference_mean.mean()
        rows.append(
            {
                "path": path,
                "pair_type": pair_type,
                "stages": stages,
                "component": component,
                "n_models": len(group),
                "mean": group["mean"].mean(),
                "reference_mean": reference_mean,
                "signed_mean_bias": (group["mean"] - group.reference_mean).mean(),
                "normalized_mae_pooled": (
                    group.mean_absolute_error.mean() / reference_mean
                    if reference_mean > 0
                    else None
                ),
                "model_normalized_mae_median": group.normalized_mae.median(),
                "model_normalized_mae_max": group.normalized_mae.max(),
                "undefined_model_relative_error_count": group.normalized_mae.isna().sum(),
                "max_model_absolute_mae": group.mean_absolute_error.max(),
                "model_rank_spearman": rank.model_rank_spearman,
            }
        )
    missed = []
    for key, group in samples[samples.stages.isin([9, 17, 33])].groupby(
        ["path", "pair_type", "stages"], sort=True
    ):
        positive = int((group.C_reference > 0).sum())
        count = int(group.missed_contradiction.sum())
        missed_reference = group.loc[group.missed_contradiction, "C_reference"]
        total_reference_mass = group.C_reference.sum()
        missed.append(
            {
                "path": key[0],
                "pair_type": key[1],
                "stages": key[2],
                "reference_C_positive_samples": positive,
                "missed_C_samples": count,
                "missed_C_fraction_of_reference_positive": count / positive if positive else None,
                "missed_C_reference_mass_fraction": (
                    missed_reference.sum() / total_reference_mass
                    if total_reference_mass > 0
                    else None
                ),
                "missed_C_reference_median": missed_reference.median(),
                "missed_C_reference_max": missed_reference.max(),
                "definition": (
                    "coarse C equals zero and finest C is strictly positive; no magnitude floor"
                ),
            }
        )
    receipts = [json.loads(p.read_text()) for p in (output / "cache").glob("*.json")]
    latest_receipt_time = max(p.stat().st_mtime for p in (output / "cache").glob("*.json"))
    times_path = output / "process_times.json"
    starts = json.loads(times_path.read_text()) if times_path.exists() else []
    runtime = {
        "sum_shard_wall_seconds": sum(r["elapsed_seconds"] for r in receipts),
        "workers": len(starts) if starts else None,
        "wall_seconds_from_process_start_to_last_receipt": (
            latest_receipt_time - min(r["process_start_unix"] for r in starts) if starts else None
        ),
        "timing_scope": (
            "Inference workers including preparation/model loads, through final receipt; "
            "excludes subsequent CPU analysis."
        ),
        "devices": sorted({r["device"] for r in receipts}),
        "torch_versions": sorted({r["torch"] for r in receipts}),
        "torchvision_versions": sorted({r["torchvision"] for r in receipts}),
    }
    summary = {
        "experiment": "stage_resolution",
        "complete": True,
        "support_description": manifest["config"].get("support_description"),
        "models": [
            {"model_id": m["model_id"], "checkpoint_sha256": m["checkpoint_sha256"]}
            for m in manifest["models"]
        ],
        "n_models": status["models"],
        "n_source_pairs": status["source_pairs"],
        "n_typed_pairs_per_model_path": status["typed_pairs"],
        "stage_counts": status["stage_counts"],
        "epsilon": status["epsilon"],
        "model_input_evaluations": status["model_input_evaluations"],
        "plan_sha256": manifest["plan_sha256"],
        "manifest_file_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        "code_sha256_at_inference": manifest["code_sha256"],
        "analysis_code_sha256": status["analysis_code_sha256"],
        "report_source_sha256": checks["report_source_sha256"],
        "config_sha256": manifest["config_sha256"],
        "data_sha256": manifest["data_sha256"],
        "runtime": runtime,
        "endpoint_checks": checks,
        "conservation_max_residual": status["conservation_max_residual"],
        "component_results": rows,
        "missed_contradiction": missed,
        "limitations": [
            "The 65-point grid is a sampled numerical reference, not an exact continuous integral.",
            "New public pretrained-model support; not a rerun of the original 72-model zoo.",
            "Small/zero contradiction reference means yield large/undefined relative errors.",
            "No prespecified numerical tolerance was used to declare nine stages sufficient.",
        ],
        "run_directory": str(output.resolve()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False) + "\n")


def summarize_completed(output: Path, summary_destination: Path | None = None):
    status = json.loads((output / "analysis_status.json").read_text())
    if not status["complete"]:
        raise ValueError("report requires a complete analyzed run")
    manifest = json.loads((output / "manifest.json").read_text())
    metrics = pd.read_csv(output / "analysis/model_metrics.csv")
    ranks = pd.read_csv(output / "analysis/model_rank_stability.csv")
    rows = []
    for keys, group in metrics.groupby(["path", "stages", "component"], sort=True):
        row = dict(zip(["path", "stages", "component"], keys, strict=True))
        row.update(
            {
                "n_model_pairtype_groups": len(group),
                "defined_normalized_mae_groups": int(group.normalized_mae.notna().sum()),
                "median_normalized_mae": group.normalized_mae.median(),
                "max_normalized_mae": group.normalized_mae.max(),
                "median_mean_absolute_error": group.mean_absolute_error.median(),
                "max_mean_absolute_error": group.mean_absolute_error.max(),
                "max_p95_absolute_error": group.p95_absolute_error.max(),
                "max_relative_mean_bias": group.relative_mean_bias.max(),
                "min_pair_rank_spearman": group.pair_rank_spearman.min(),
            }
        )
        if keys[-1] != "Abs":
            share_change = abs(group.aggregate_share - group.reference_aggregate_share)
            row["median_aggregate_share_change_pp"] = 100 * share_change.median()
            row["max_aggregate_share_change_pp"] = 100 * share_change.max()
        row["min_model_rank_spearman"] = ranks.loc[
            (ranks.path == keys[0]) & (ranks.stages == keys[1]) & (ranks.component == keys[2]),
            "model_rank_spearman",
        ].min()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "analysis/grid_summary.csv", index=False)

    endpoints, comparisons, gate_changes, maximum = {}, 0, 0, 0.0
    epsilon = manifest["config"]["epsilon"]
    for task in manifest["tasks"]:
        cache = output / "cache" / f"{task['task_id']}.npz"
        receipt = json.loads(cache.with_suffix(".json").read_text())
        if hashlib.sha256(cache.read_bytes()).hexdigest() != receipt["cache_sha256"]:
            raise ValueError(f"cache integrity changed after analysis: {cache}")
        with np.load(cache, allow_pickle=False) as data:
            for index, pair_id in enumerate(data["pair_ids"]):
                key = (task["model_id"], str(pair_id))
                current = np.array([data["q_plus"][index, -1], data["q_minus"][index, -1]])
                if key in endpoints:
                    previous = endpoints[key]
                    maximum = max(maximum, float(abs(previous - current).max()))
                    gate_changes += int(
                        (abs(previous[0] - previous[1]) >= epsilon)
                        != (abs(current[0] - current[1]) >= epsilon)
                    )
                    comparisons += 1
                else:
                    endpoints[key] = current
    checks = {
        "plan_sha256": manifest["plan_sha256"],
        "endpoint_comparisons_across_paths": comparisons,
        "max_endpoint_score_difference": maximum,
        "endpoint_gate_changes_across_paths": gate_changes,
        "endpoint_check_passed": maximum <= 1e-7 and gate_changes == 0,
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "analysis/endpoint_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    if not checks["endpoint_check_passed"]:
        raise ValueError(f"paired endpoint identity check failed: {checks}")
    if summary_destination is not None:
        write_review_summary(output, summary_destination, status, manifest, metrics, ranks, checks)
    print(json.dumps(checks, indent=2))
    print(summary.loc[summary.stages.isin([9, 17, 33])].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path, help="write a compact reviewable JSON summary")
    args = parser.parse_args()
    summarize_completed(args.output, args.summary)
