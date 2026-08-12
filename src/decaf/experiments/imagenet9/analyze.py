"""Run-derived ImageNet-9 mechanism and protocol summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.experiments.common import RunContext, atomic_json, atomic_text
from decaf.paper.reference import (
    discover_archive,
    load_reference_runs,
    materialize_inputs,
    receipt_dict,
    reference_roots,
    verify_archive,
)

SCORE_COLUMNS = {
    "pair_id",
    "model_id",
    "reveal_path",
    "M",
    "E",
    "C",
    "F",
    "Abs",
    "Net",
}


def validate_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the computed score table used by all downstream summaries."""

    missing = sorted(SCORE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"ImageNet-9 score table is missing columns: {missing}")
    normalized = frame.copy()
    for column in ("M", "E", "C", "F", "Abs", "Net"):
        normalized[column] = normalized[column].astype(float)
        if not np.isfinite(normalized[column]).all():
            raise ValueError(f"ImageNet-9 score column is non-finite: {column}")
    if (normalized[["M", "E", "C", "F", "Abs"]] < 0).any().any():
        raise ValueError("ImageNet-9 magnitude scores must be non-negative")
    return normalized


def matched_magnitude_accuracy(
    frame: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    matched_pairs: pd.DataFrame | None = None,
) -> dict[str, float | int | str | None]:
    """Compare three-way DECAF routing to an Abs-only magnitude baseline."""

    scored = validate_score_frame(frame)
    if matched_pairs is not None:
        columns = {"abs_matched_pair_accuracy", "decaf_matched_pair_accuracy"}
        missing = sorted(columns - set(matched_pairs.columns))
        if missing:
            raise ValueError(f"raw matched pairs are missing columns: {missing}")
        result: dict[str, float | int | str | None] = {
            "rows": len(matched_pairs),
            "decaf_accuracy": float(matched_pairs["decaf_matched_pair_accuracy"].mean()),
            "abs_accuracy": float(matched_pairs["abs_matched_pair_accuracy"].mean()),
            "source": "recomputed_from_sealed_matched_pairs",
        }
        if benchmark is not None:
            published = matched_magnitude_accuracy(frame, benchmark)
            for key in ("decaf_accuracy", "abs_accuracy"):
                if not np.isclose(float(result[key]), float(published[key]), rtol=0.0, atol=1e-14):
                    raise AssertionError(f"raw matched-pair recomputation drifted: {key}")
            if int(result["rows"]) != int(published["rows"]):
                raise AssertionError("raw matched-pair count differs from the sealed summary")
        return result
    if benchmark is not None:
        required = {"method", "matched_pair_accuracy", "matched_pairs"}
        missing = sorted(required - set(benchmark.columns))
        if missing:
            raise ValueError(f"matched-magnitude benchmark is missing columns: {missing}")
        selected = benchmark[benchmark["method"].isin(("Abs", "DECAF"))]
        if set(selected["method"]) != {"Abs", "DECAF"} or len(selected) != 2:
            raise ValueError("matched-magnitude benchmark must have one Abs and one DECAF row")
        values = selected.set_index("method")
        matched_pairs = {int(value) for value in values["matched_pairs"]}
        if len(matched_pairs) != 1:
            raise ValueError("Abs and DECAF benchmark rows use different matched pairs")
        return {
            "rows": matched_pairs.pop(),
            "decaf_accuracy": float(values.at["DECAF", "matched_pair_accuracy"]),
            "abs_accuracy": float(values.at["Abs", "matched_pair_accuracy"]),
            "source": "sealed_matched_abs_benchmark",
        }
    if "expected_component" not in scored:
        return {
            "rows": len(scored),
            "decaf_accuracy": None,
            "abs_accuracy": None,
            "source": "unavailable_requires_matched_abs_pairing",
        }
    expected = scored["expected_component"].astype(str)
    decaf_prediction = scored[["E", "C", "F"]].idxmax(axis=1)
    # Abs has no direction or endpoint-null semantics.  Its deterministic
    # matched-magnitude baseline predicts the active aligned class only.
    abs_prediction = pd.Series("E", index=scored.index)
    return {
        "rows": len(scored),
        "decaf_accuracy": float((decaf_prediction == expected).mean()),
        "abs_accuracy": float((abs_prediction == expected).mean()),
        "source": "smoke_diagnostic",
    }


def protocol_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the registered filtered-mean patch/blend ratios."""

    scored = validate_score_frame(frame)
    if "pair_type" not in scored:
        scored["pair_type"] = "all"
    scored = (
        scored.groupby(["pair_type", "model_id", "reveal_path"], as_index=False, sort=True)[
            ["M", "E", "C", "F", "Abs"]
        ]
        .mean()
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for pair_type, group in scored.groupby("pair_type", sort=True):
        blend = group[group["reveal_path"] == "blend"]
        if blend.empty:
            raise ValueError(f"protocol ratio has no blend rows for {pair_type}")
        for path in sorted(set(group["reveal_path"].astype(str)) - {"blend"}):
            patch = group[group["reveal_path"] == path]
            for metric in ("Abs", "F"):
                denominator = float(blend[metric].mean())
                numerator = float(patch[metric].mean())
                if abs(denominator) <= np.finfo(np.float64).eps:
                    raise ValueError(
                        f"protocol ratio denominator is zero: {pair_type}/{path}/{metric}"
                    )
                rows.append(
                    {
                        "pair_type": str(pair_type),
                        "patch_path": path,
                        "metric": metric,
                        "patch_rows": len(patch),
                        "blend_rows": len(blend),
                        "patch_mean": numerator,
                        "blend_mean": denominator,
                        "ratio_mean": numerator / denominator,
                        "operation": "filtered_mean_ratio",
                    }
                )
    return pd.DataFrame(rows)


def mechanism_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mechanism scores by model and reveal protocol."""

    scored = validate_score_frame(frame)
    groups = ["model_id", "reveal_path"]
    if "pair_type" in scored:
        groups.insert(1, "pair_type")
    return (
        scored.groupby(groups, as_index=False, sort=True)[["M", "E", "C", "F", "Abs", "Net"]]
        .mean()
        .sort_values(groups, kind="stable")
        .reset_index(drop=True)
    )


def protocol_rank_transfer(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure model-ranking transfer from blend to each patch protocol."""

    scored = validate_score_frame(frame)
    if "pair_type" not in scored:
        scored["pair_type"] = "all"
    scored = (
        scored.groupby(["pair_type", "model_id", "reveal_path"], as_index=False, sort=True)[
            ["M", "E", "C", "F", "Abs"]
        ]
        .mean()
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for pair_type, group in scored.groupby("pair_type", sort=True):
        blend = group[group["reveal_path"] == "blend"].set_index("model_id")
        for path in sorted(set(group["reveal_path"].astype(str)) - {"blend"}):
            patch = group[group["reveal_path"] == path].set_index("model_id")
            for metric in ("M", "E", "C", "F", "Abs"):
                joined = patch[[metric]].join(
                    blend[[metric]], how="inner", lsuffix="_patch", rsuffix="_blend"
                )
                correlation = (
                    float(
                        joined[f"{metric}_patch"].corr(joined[f"{metric}_blend"], method="spearman")
                    )
                    if len(joined) >= 2
                    else float("nan")
                )
                rows.append(
                    {
                        "pair_type": str(pair_type),
                        "patch_path": path,
                        "metric": metric,
                        "model_count": len(joined),
                        "spearman_rank_transfer": correlation,
                    }
                )
    return pd.DataFrame(rows)


def _materialize_reference_scores(
    context: RunContext,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    repository = Path(__file__).resolve().parents[4]
    runs = load_reference_runs(repository / "manifests" / "reference_runs")
    run = runs[str(context.config.get("reference_run", "I9"))]
    archive = discover_archive(run, reference_roots())
    verify_archive(archive, run)
    output = context.path / "reference_data"
    receipts = materialize_inputs(run, archive, run.analysis_inputs, output)
    receipt_payload = [receipt_dict(receipt) for receipt in receipts]
    atomic_json(
        context.path / "receipts" / "reference_replay.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": run.run_id,
            "archive_filename": run.archive_filename,
            "archive_sha256": run.archive_sha256,
            "inputs": receipt_payload,
        },
    )
    profiles_path = output / "I9" / "results" / "tables" / "T03_decaf_profiles.csv"
    profiles = pd.read_csv(profiles_path)
    epsilon = float(context.config["experiment_grid"]["epsilon"])
    profiles = profiles[np.isclose(profiles["epsilon"].astype(float), epsilon)].copy()
    profiles = profiles.rename(columns={"path": "reveal_path"})
    profiles["pair_id"] = "aggregate__" + profiles["pair_type"].astype(str)
    scores = validate_score_frame(profiles)
    atomic_text(context.path / "metrics" / "decaf_scores.csv", scores.to_csv(index=False))
    benchmark_path = output / "I9" / "results" / "matched_abs" / "matched_abs_benchmark.csv"
    matched_pairs_path = output / "I9" / "results" / "matched_abs" / "matched_pairs.parquet"
    return (
        scores,
        pd.read_csv(benchmark_path),
        pd.read_parquet(matched_pairs_path),
        receipt_payload,
    )


def _assert_reference_headlines(
    accuracy: dict[str, float | int | str | None],
    ratios: pd.DataFrame,
) -> dict[str, float]:
    expected_accuracy = {
        "abs_accuracy": 0.3501829734185867,
        "decaf_accuracy": 0.9639884183858124,
    }
    observed: dict[str, float] = {}
    for key, expected in expected_accuracy.items():
        value = float(accuracy[key])
        if not np.isclose(value, expected, rtol=0.0, atol=1e-14):
            raise AssertionError(f"ImageNet-9 reference headline drifted: {key}")
        observed[key] = value
    ratio_expected = {
        ("same_rand", "patch_A", "Abs"): 1.7372284770086692,
        ("same_rand", "patch_A", "F"): 4.3762687997406715,
        ("same_next", "patch_A", "Abs"): 1.7929210632916266,
        ("same_next", "patch_A", "F"): 4.70751149983746,
    }
    indexed = ratios.set_index(["pair_type", "patch_path", "metric"])
    for key, expected in ratio_expected.items():
        value = float(indexed.at[key, "ratio_mean"])
        if not np.isclose(value, expected, rtol=0.0, atol=1e-12):
            raise AssertionError(f"ImageNet-9 reference headline drifted: {key}")
        observed["__".join(key)] = value
    return observed


def analyze(context: RunContext) -> dict[str, Any]:
    """Generate all ImageNet-9 analysis outputs from run-local score rows."""

    source = context.path / "metrics" / "decaf_scores.csv"
    reference_receipts: list[dict[str, Any]] = []
    benchmark: pd.DataFrame | None = None
    matched_pairs: pd.DataFrame | None = None
    source_mode = "computed_run"
    if source.is_file():
        scores = validate_score_frame(pd.read_csv(source))
    elif context.profile == "paper":
        scores, benchmark, matched_pairs, reference_receipts = _materialize_reference_scores(
            context
        )
        source_mode = "sealed_reference_replay"
    else:
        raise FileNotFoundError("compute must produce metrics/decaf_scores.csv before analyze")
    accuracy = matched_magnitude_accuracy(scores, benchmark, matched_pairs)
    ratios = protocol_ratios(scores)
    rank_transfer = protocol_rank_transfer(scores)
    summary = mechanism_summary(scores)
    baseline_path = context.path / "metrics" / "baseline_scores.csv"
    baseline_rows = 0
    baseline_methods: list[str] = []
    if baseline_path.is_file():
        baselines = pd.read_csv(baseline_path)
        required = {"pair_id", "pair_type", "model_id", "method_id", "score"}
        missing = sorted(required - set(baselines.columns))
        if missing:
            raise ValueError(f"baseline score table is missing columns: {missing}")
        baselines["score"] = baselines["score"].astype(float)
        if not np.isfinite(baselines["score"]).all():
            raise ValueError("baseline score table contains non-finite scores")
        baseline_summary = (
            baselines.groupby(["model_id", "method_id", "pair_type"], as_index=False, sort=True)[
                "score"
            ]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        atomic_text(
            context.path / "metrics" / "baseline_summary.csv",
            baseline_summary.to_csv(index=False),
        )
        baseline_rows = len(baselines)
        baseline_methods = sorted(baselines["method_id"].astype(str).unique())
    atomic_json(context.path / "metrics" / "matched_magnitude_accuracy.json", accuracy)
    atomic_text(context.path / "metrics" / "protocol_ratios.csv", ratios.to_csv(index=False))
    atomic_text(
        context.path / "metrics" / "protocol_rank_transfer.csv",
        rank_transfer.to_csv(index=False),
    )
    atomic_text(context.path / "metrics" / "mechanism_summary.csv", summary.to_csv(index=False))
    headlines = (
        _assert_reference_headlines(accuracy, ratios)
        if source_mode == "sealed_reference_replay"
        else {}
    )
    from decaf.experiments.imagenet9.gpu_runtime import (
        b200_enabled,
        write_downstream_receipt,
    )

    gpu_verified = False
    if b200_enabled(context.config):
        compute_receipt = context.path / "receipts" / "imagenet9_b200_compute.json"
        if not compute_receipt.is_file():
            raise FileNotFoundError("real ImageNet-9 analysis requires the B200 compute receipt")
        import json

        gpu_verified = bool(
            json.loads(compute_receipt.read_text(encoding="utf-8")).get("gpu_inference_verified")
        )
        if not gpu_verified:
            raise ValueError("real ImageNet-9 compute receipt does not verify GPU inference")
    report = {
        "schema_version": 1,
        "source": "metrics/decaf_scores.csv",
        "score_rows": len(scores),
        "model_count": int(scores["model_id"].nunique()),
        "reveal_paths": sorted(scores["reveal_path"].astype(str).unique()),
        "matched_magnitude_accuracy": accuracy,
        "protocol_ratio_rows": len(ratios),
        "protocol_rank_transfer_rows": len(rank_transfer),
        "baseline_rows": baseline_rows,
        "baseline_methods": baseline_methods,
        "source_mode": source_mode,
        "reference_input_count": len(reference_receipts),
        "reference_headlines": headlines,
        "gpu_inference_verified": gpu_verified,
    }
    atomic_json(context.path / "metrics" / "summary.json", report)
    if gpu_verified:
        artifacts = [
            "metrics/summary.json",
            "metrics/matched_magnitude_accuracy.json",
            "metrics/protocol_ratios.csv",
            "metrics/protocol_rank_transfer.csv",
            "metrics/mechanism_summary.csv",
        ]
        if (context.path / "metrics" / "baseline_summary.csv").is_file():
            artifacts.append("metrics/baseline_summary.csv")
        write_downstream_receipt(context, "analyze", artifacts)
    return report


__all__ = [
    "analyze",
    "matched_magnitude_accuracy",
    "mechanism_summary",
    "protocol_ratios",
    "protocol_rank_transfer",
    "validate_score_frame",
]
