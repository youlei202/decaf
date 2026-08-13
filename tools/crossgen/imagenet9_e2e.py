"""Run and compare the exact ImageNet-9 cross-generation B200 bridge."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.experiments.common import atomic_json, atomic_text, load_profile
from decaf.experiments.imagenet9.evaluate import evaluate_response_frame
from decaf.experiments.imagenet9.gpu_models import (
    load_model,
    load_official_mapping,
    probability_model,
    resolve_b200_assets,
)
from decaf.experiments.imagenet9.gpu_runtime import _require_single_b200, _scan_frame
from tools.crossgen.compare_core import (
    BOUNDARY_ABS,
    HARD_MISMATCH_ABS,
    TIER_A_ATOL,
    TIER_A_RTOL,
    TIER_B_ABS,
    compare_record,
)
from tools.crossgen.legacy_imagenet9_export import (
    ALPHA,
    DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    EPSILON,
    HISTORICAL_RESULTS,
    MODEL_BINDINGS,
    PAIR_TYPES,
    REVEAL_PATHS,
    WEIGHT_CACHE_ROOT,
    _atomic_parquet,
    export_legacy_trajectory,
    prepare_bridge,
    sealed_summaries,
)
from tools.crossgen.schema import sha256_file

SUMMARY_NAMES = ("M", "E", "C", "F", "Abs")


def _asset_environment() -> dict[str, str]:
    return {
        "DECAF_DATA_ROOT": str(DATASET_ROOT),
        "DECAF_IMAGENET9_WEIGHT_CACHE_ROOT": str(WEIGHT_CACHE_ROOT),
        "DECAF_IMAGENET9_CHECKPOINT_ROOT": str(HISTORICAL_RESULTS / "training/checkpoints"),
    }


def _load_orders(output_root: Path) -> dict[str, Any]:
    path = output_root / "manifests/imagenet9_historical_patch_orders.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    orders = payload.get("orders")
    if not isinstance(orders, dict) or set(orders) != {"patch_A", "patch_B"}:
        raise TypeError("explicit historical patch-order manifest is malformed")
    for reveal_path, mapping in orders.items():
        if not isinstance(mapping, dict) or len(mapping) != 16:
            raise ValueError(f"{reveal_path} must contain exactly 16 typed orders")
        for pair_id, order in mapping.items():
            if (
                not isinstance(pair_id, str)
                or not isinstance(order, list)
                or len(order) != 64
                or set(map(int, order)) != set(range(64))
            ):
                raise ValueError(f"invalid explicit patch order: {reveal_path}/{pair_id}")
    return orders


def run_current_executor(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Run exact selected units through the current repository's scan executor."""

    root = Path(output_root)
    prepare_bridge(root)
    config = load_profile("imagenet9", "smoke")
    settings = dict(config["b200_smoke"])
    settings["explicit_patch_orders"] = _load_orders(root)
    assets = resolve_b200_assets(config, _asset_environment())
    pairs = pd.read_csv(root / "manifests/imagenet9_selection.csv")
    if len(pairs) != 16 or pairs["pair_id"].astype(str).duplicated().any():
        raise ValueError("current executor selection must contain 16 unique typed pairs")
    torch, gpu_name = _require_single_b200()
    mapping = load_official_mapping(assets.mapping, torch)
    frames: list[pd.DataFrame] = []
    load_records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    for asset in assets.models:
        binding = next(item for item in MODEL_BINDINGS if item.current_model_id == asset.model_id)
        if asset.checkpoint_sha256 != binding.checkpoint_sha256:
            raise ValueError(f"current checkpoint identity differs: {asset.model_id}")
        base_model, load_record = load_model(asset, device="cuda:0")
        model = probability_model(
            base_model,
            mapping_matrix=mapping,
            output_classes=asset.output_classes,
        ).to(device=torch.device("cuda:0"))
        try:
            for reveal_path in REVEAL_PATHS:
                member = {
                    "model_id": asset.model_id,
                    "reveal_path": reveal_path,
                }
                frame = _scan_frame(
                    member,
                    pairs,
                    assets=assets,
                    settings=settings,
                    model=model,
                    torch=torch,
                )
                frame["checkpoint_sha256"] = asset.checkpoint_sha256
                frames.append(frame)
        finally:
            load_records.append({"model_id": asset.model_id, **load_record})
            del model, base_model
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize(torch.device("cuda:0"))
    result = pd.concat(frames, ignore_index=True)
    expected = len(MODEL_BINDINGS) * 16 * len(REVEAL_PATHS) * len(ALPHA)
    keys = ["model_id", "pair_id", "reveal_path", "stage_index"]
    if len(result) != expected or result.duplicated(keys).any():
        raise AssertionError(
            f"current ImageNet-9 executor produced {len(result)} rows, expected {expected}"
        )
    output = root / "trajectories/imagenet9_current_e2e_scans.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(result.sort_values(keys, kind="stable"), output)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "experiment_family": "imagenet9",
        "executor": "current decaf.experiments.imagenet9.gpu_runtime._scan_frame",
        "repository_commit": commit,
        "gpu_name": gpu_name,
        "gpu_count": int(torch.cuda.device_count()),
        "models": load_records,
        "source_pairs_per_model": 8,
        "typed_pairs_per_model": 16,
        "paths": list(REVEAL_PATHS),
        "stages": len(ALPHA),
        "units": len(MODEL_BINDINGS) * 16 * len(REVEAL_PATHS),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "patch_order_manifest": str(root / "manifests/imagenet9_historical_patch_orders.json"),
        "patch_order_manifest_sha256": sha256_file(
            root / "manifests/imagenet9_historical_patch_orders.json"
        ),
        "patch_order_injection": True,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(torch.device("cuda:0"))),
    }
    receipt_path = root / "provenance/imagenet9_current_e2e.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_path, receipt)
    return output


def _current_scores(current_scans: pd.DataFrame) -> pd.DataFrame:
    scores = evaluate_response_frame(current_scans, epsilon=EPSILON)
    return scores.rename(columns={name: f"current_{name}" for name in SUMMARY_NAMES})


def _binding_by_current() -> dict[str, Any]:
    return {binding.current_model_id: binding for binding in MODEL_BINDINGS}


def _dominant(row: pd.Series, prefix: str) -> str:
    values = {name: float(row[f"{prefix}_{name}"]) for name in ("E", "C", "F")}
    maximum = max(values.values())
    return "|".join(name for name in ("E", "C", "F") if values[name] == maximum)


def _historical_summary_table() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for binding in MODEL_BINDINGS:
        frame = sealed_summaries(binding).copy()
        frame["model_id"] = binding.current_model_id
        frame["historical_model_id"] = binding.historical_model_id
        frame["pair_id"] = frame["pair_id"].astype(str) + "__" + frame["pair_type"].astype(str)
        frame = frame.rename(
            columns={
                "path": "reveal_path",
                "endpoint_delta": "historical_endpoint_d",
                "endpoint_active": "historical_gate",
                **{name: f"historical_{name}" for name in SUMMARY_NAMES},
            }
        )
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    expected = len(MODEL_BINDINGS) * 16 * len(REVEAL_PATHS)
    keys = ["model_id", "pair_id", "reveal_path"]
    if len(result) != expected or result.duplicated(keys).any():
        raise AssertionError("sealed historical summary table is not exactly 144 units")
    return result


def _fraction(values: pd.Series) -> float:
    return float(values.astype(bool).mean()) if len(values) else float("nan")


def _stats(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("comparison statistics require non-empty finite values")
    return {
        "median_absolute_error": float(np.median(array)),
        "p95_absolute_error": float(np.percentile(array, 95)),
        "maximum_absolute_error": float(np.max(array)),
    }


def _selected_ratios(
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for (model_id, pair_type), group in comparison.groupby(["model_id", "pair_type"], sort=True):
        by_path = group.set_index("reveal_path")
        for patch_path in ("patch_A", "patch_B"):
            for name in ("E", "C", "F", "Abs"):
                current_blend = float(by_path.loc["blend", f"current_{name}"].mean())
                current_patch = float(by_path.loc[patch_path, f"current_{name}"].mean())
                historical_blend = float(by_path.loc["blend", f"historical_{name}"].mean())
                historical_patch = float(by_path.loc[patch_path, f"historical_{name}"].mean())
                current_ratio = (
                    current_patch / current_blend
                    if abs(current_blend) > np.finfo(np.float64).eps
                    else float("nan")
                )
                historical_ratio = (
                    historical_patch / historical_blend
                    if abs(historical_blend) > np.finfo(np.float64).eps
                    else float("nan")
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "pair_type": pair_type,
                        "patch_path": patch_path,
                        "metric": name,
                        "current_patch_mean": current_patch,
                        "current_blend_mean": current_blend,
                        "current_ratio": current_ratio,
                        "historical_patch_mean": historical_patch,
                        "historical_blend_mean": historical_blend,
                        "historical_ratio": historical_ratio,
                        "ratio_absolute_error": (
                            abs(current_ratio - historical_ratio)
                            if np.isfinite(current_ratio) and np.isfinite(historical_ratio)
                            else float("nan")
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    json_rows = []
    for row in frame.to_dict("records"):
        json_rows.append(
            {
                key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                for key, value in row.items()
            }
        )
    return frame, json_rows


def compare_e2e(
    current_scan_path: str | Path,
    legacy_scan_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Compare current executor stages and summaries to the exact historical units."""

    root = Path(output_root)
    current = pd.read_parquet(current_scan_path)
    legacy = pd.read_parquet(legacy_scan_path)
    stage_keys = ["model_id", "pair_id", "reveal_path", "stage_index"]
    stage = current.merge(
        legacy.loc[
            :,
            [
                *stage_keys,
                "alpha",
                "score_plus",
                "score_minus",
                "response",
                "source_pair_id",
                "historical_model_id",
            ],
        ],
        on=stage_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_current", "_legacy"),
    )
    expected_stages = len(MODEL_BINDINGS) * 16 * len(REVEAL_PATHS) * len(ALPHA)
    if len(stage) != expected_stages:
        raise AssertionError(
            f"current/legacy stage join has {len(stage)} rows, expected {expected_stages}"
        )
    if not np.array_equal(
        stage["alpha_current"].to_numpy(dtype=np.float64),
        stage["alpha_legacy"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("current and historical alpha identities differ")
    stage["response_abs_error"] = np.abs(
        stage["response_current"].to_numpy(dtype=np.float64)
        - stage["response_legacy"].to_numpy(dtype=np.float64)
    )
    stage_aggregates = (
        stage.groupby(stage_keys[:-1], sort=True)
        .agg(
            stage_response_median_abs_error=("response_abs_error", "median"),
            stage_response_p95_abs_error=(
                "response_abs_error",
                lambda values: float(np.percentile(values, 95)),
            ),
            stage_response_max_abs_error=("response_abs_error", "max"),
        )
        .reset_index()
    )
    endpoints = stage[stage["stage_index"].astype(int) == len(ALPHA) - 1].copy()
    endpoints["current_legacy_endpoint_abs_error"] = np.abs(
        endpoints["response_current"] - endpoints["response_legacy"]
    )
    stage_aggregates = stage_aggregates.merge(
        endpoints.loc[
            :,
            [
                *stage_keys[:-1],
                "response_legacy",
                "current_legacy_endpoint_abs_error",
            ],
        ].rename(columns={"response_legacy": "legacy_fresh_endpoint_d"}),
        on=stage_keys[:-1],
        validate="one_to_one",
    )

    current_scores = _current_scores(current).rename(
        columns={
            "endpoint_delta": "current_endpoint_d",
            "endpoint_active": "current_gate",
        }
    )
    historical = _historical_summary_table()
    unit_keys = ["model_id", "pair_id", "reveal_path"]
    comparison = current_scores.merge(
        historical,
        on=unit_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_current_label", "_historical_label"),
    ).merge(stage_aggregates, on=unit_keys, validate="one_to_one")
    expected_units = len(MODEL_BINDINGS) * 16 * len(REVEAL_PATHS)
    if len(comparison) != expected_units:
        raise AssertionError(
            f"current/historical unit join has {len(comparison)} rows, expected {expected_units}"
        )
    if "pair_type_current_label" in comparison:
        if not (
            comparison["pair_type_current_label"].astype(str)
            == comparison["pair_type_historical_label"].astype(str)
        ).all():
            raise ValueError("current and historical pair-type identities differ")
        comparison["pair_type"] = comparison["pair_type_current_label"].astype(str)
    elif "pair_type" not in comparison:
        comparison["pair_type"] = comparison["pair_id"].astype(str).str.rsplit("__", n=1).str[-1]

    for name in SUMMARY_NAMES:
        comparison[f"abs_error_{name}"] = np.abs(
            comparison[f"current_{name}"] - comparison[f"historical_{name}"]
        )
        comparison[f"signed_error_{name}"] = (
            comparison[f"current_{name}"] - comparison[f"historical_{name}"]
        )
    comparison["historical_gate"] = comparison["historical_gate"].astype(bool)
    comparison["current_gate"] = comparison["current_gate"].astype(bool)
    comparison["current_orientation"] = np.where(
        comparison["current_gate"],
        np.sign(comparison["current_endpoint_d"]).astype(int),
        0,
    )
    comparison["historical_orientation"] = np.where(
        comparison["historical_gate"],
        np.sign(comparison["historical_endpoint_d"]).astype(int),
        0,
    )
    comparison["boundary"] = (
        np.abs(np.abs(comparison["historical_endpoint_d"]) - EPSILON) <= BOUNDARY_ABS
    )
    comparison["gate_match"] = comparison["current_gate"] == comparison["historical_gate"]
    comparison["orientation_match"] = (
        comparison["current_orientation"] == comparison["historical_orientation"]
    )
    comparison["current_dominant"] = comparison.apply(lambda row: _dominant(row, "current"), axis=1)
    comparison["historical_dominant"] = comparison.apply(
        lambda row: _dominant(row, "historical"), axis=1
    )
    comparison["dominant_match"] = (
        comparison["current_dominant"] == comparison["historical_dominant"]
    )
    bindings = _binding_by_current()
    current_checkpoint = {
        str(model_id): set(group["checkpoint_sha256"].astype(str))
        for model_id, group in current.groupby("model_id", sort=False)
    }
    legacy_checkpoint = {
        str(model_id): set(group["checkpoint_sha256"].astype(str))
        for model_id, group in legacy.groupby("model_id", sort=False)
    }
    comparison["checkpoint_identity_match"] = comparison["model_id"].map(
        lambda model_id: (
            model_id in bindings
            and current_checkpoint.get(str(model_id)) == {bindings[str(model_id)].checkpoint_sha256}
            and legacy_checkpoint.get(str(model_id)) == {bindings[str(model_id)].checkpoint_sha256}
        )
    )
    comparison["sample_identity_match"] = (
        comparison["pair_id"].astype(str).isin(set(current["pair_id"].astype(str)))
    )
    comparison["identity_match"] = (
        comparison["checkpoint_identity_match"]
        & comparison["sample_identity_match"]
        & comparison["pair_type"].isin(PAIR_TYPES)
        & comparison["reveal_path"].isin(REVEAL_PATHS)
    )
    close = pd.DataFrame(
        {
            name: np.isclose(
                comparison[f"current_{name}"],
                comparison[f"historical_{name}"],
                atol=TIER_A_ATOL,
                rtol=TIER_A_RTOL,
            )
            for name in SUMMARY_NAMES
        }
    )
    comparison["tier_a_pass"] = close.all(axis=1)
    maximum_summary_error = comparison[[f"abs_error_{name}" for name in SUMMARY_NAMES]].max(axis=1)
    comparison["tier_b_pass"] = (
        (maximum_summary_error <= TIER_B_ABS)
        & (comparison["boundary"] | (comparison["gate_match"] & comparison["orientation_match"]))
        & comparison["dominant_match"]
    )
    comparison["tier"] = np.where(
        comparison["tier_a_pass"],
        "A",
        np.where(comparison["tier_b_pass"], "B", "FAIL"),
    )
    comparison["hard_mismatch"] = (
        (maximum_summary_error > HARD_MISMATCH_ABS)
        | (~comparison["boundary"] & (~comparison["gate_match"] | ~comparison["orientation_match"]))
        | ~comparison["dominant_match"]
        | ~comparison["identity_match"]
    )
    comparison = comparison.sort_values(unit_keys, kind="stable").reset_index(drop=True)

    comparisons = root / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)
    comparison_path = comparisons / "imagenet9.csv"
    stage_path = comparisons / "imagenet9_stage_responses.csv"
    ratio_path = comparisons / "imagenet9_selected_ratios.csv"
    atomic_text(comparison_path, comparison.to_csv(index=False))
    atomic_text(stage_path, stage.to_csv(index=False))
    ratio_frame, ratio_rows = _selected_ratios(comparison)
    atomic_text(ratio_path, ratio_frame.to_csv(index=False))

    non_boundary = comparison.loc[~comparison["boundary"]]
    tier_a = comparison["tier"].eq("A")
    tier_b = comparison["tier"].eq("B")
    summary = {
        "schema_version": 1,
        "experiment_family": "imagenet9",
        "unit_count": int(len(comparison)),
        "stage_row_count": int(len(stage)),
        "models": len(MODEL_BINDINGS),
        "source_pairs_per_model": 8,
        "typed_pairs_per_model": 16,
        "tier_a_fraction": _fraction(tier_a),
        "tier_b_fraction": _fraction(tier_b),
        "tier_a_or_b_fraction": _fraction(tier_a | tier_b),
        "hard_mismatch_fraction": _fraction(comparison["hard_mismatch"]),
        "gate_agreement": _fraction(non_boundary["gate_match"]),
        "orientation_agreement": _fraction(non_boundary["orientation_match"]),
        "dominant_mechanism_agreement": _fraction(comparison["dominant_match"]),
        "identity_agreement": _fraction(comparison["identity_match"]),
        "stage_response": _stats(stage["response_abs_error"]),
        "fresh_endpoint_response": _stats(comparison["current_legacy_endpoint_abs_error"]),
        "metrics": {name: _stats(comparison[f"abs_error_{name}"]) for name in SUMMARY_NAMES},
        "mean_signed_error": {
            name: float(comparison[f"signed_error_{name}"].mean()) for name in SUMMARY_NAMES
        },
        "selected_subset_patch_to_blend_ratios": ratio_rows,
        "comparison": str(comparison_path),
        "comparison_sha256": sha256_file(comparison_path),
        "stage_comparison": str(stage_path),
        "stage_comparison_sha256": sha256_file(stage_path),
        "ratio_comparison": str(ratio_path),
        "ratio_comparison_sha256": sha256_file(ratio_path),
        "metadata_limit": (
            "historical archives sealed summaries but not raw stages/orders; raw "
            "scores and deterministic path identity were regenerated with frozen code"
        ),
    }
    summary["acceptance"] = {
        "tier_a_or_b_at_least_95pct": summary["tier_a_or_b_fraction"] >= 0.95,
        "hard_mismatch_at_most_5pct": summary["hard_mismatch_fraction"] <= 0.05,
        "gate_at_least_99pct": summary["gate_agreement"] >= 0.99,
        "orientation_at_least_99pct": summary["orientation_agreement"] >= 0.99,
        "dominant_at_least_95pct": summary["dominant_mechanism_agreement"] >= 0.95,
        "identity_exact": summary["identity_agreement"] == 1.0,
        "no_systematic_summary_bias": max(
            abs(value) for value in summary["mean_signed_error"].values()
        )
        <= TIER_B_ABS,
    }
    summary["status"] = (
        "PASS_CORE_AND_E2E" if all(summary["acceptance"].values()) else "FAIL_NUMERICAL"
    )
    summary_path = comparisons / "imagenet9_summary.json"
    atomic_json(summary_path, summary)
    return {
        "comparison": comparison_path,
        "stage_comparison": stage_path,
        "ratios": ratio_path,
        "summary": summary,
        "summary_path": summary_path,
    }


def _compare_core_atomic(neutral_path: Path, root: Path) -> dict[str, Any]:
    comparison = root / "comparisons/imagenet9_core.csv"
    summary = root / "comparisons/imagenet9_core_summary.json"
    comparison.parent.mkdir(parents=True, exist_ok=True)
    temporary_comparison = comparison.with_name(".imagenet9_core.part.csv")
    temporary_summary = summary.with_name(".imagenet9_core_summary.part.json")
    try:
        result = compare_record(
            neutral_path,
            temporary_comparison,
            summary_output=temporary_summary,
        )
        temporary_comparison.replace(comparison)
        temporary_summary.replace(summary)
    finally:
        temporary_comparison.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
    result["summary_path"] = summary
    result["comparison_path"] = comparison
    return result


def _combine_core_and_e2e_status(core_result: dict[str, Any], e2e_result: dict[str, Any]) -> str:
    """Make the public family status fail closed across both comparison layers."""

    core = core_result["summary"]
    core_acceptance = {
        "tier_a_or_b_at_least_95pct": core["tier_a_or_b_fraction"] >= 0.95,
        "hard_mismatch_at_most_5pct": core["hard_mismatch_fraction"] <= 0.05,
        "gate_at_least_99pct": core["gate_agreement"] >= 0.99,
        "orientation_at_least_99pct": core["orientation_agreement"] >= 0.99,
        "dominant_at_least_95pct": core["dominant_mechanism_agreement"] >= 0.95,
        "identity_exact": core["identity_agreement"] == 1.0,
        "numeric_identity_exact": core["numeric_identity_fraction"] == 1.0,
    }
    summary = e2e_result["summary"]
    summary["current_core"] = {
        "unit_count": core["unit_count"],
        "tier_a_fraction": core["tier_a_fraction"],
        "tier_b_fraction": core["tier_b_fraction"],
        "tier_a_or_b_fraction": core["tier_a_or_b_fraction"],
        "hard_mismatch_fraction": core["hard_mismatch_fraction"],
        "gate_agreement": core["gate_agreement"],
        "orientation_agreement": core["orientation_agreement"],
        "dominant_mechanism_agreement": core["dominant_mechanism_agreement"],
        "identity_agreement": core["identity_agreement"],
        "numeric_identity_fraction": core["numeric_identity_fraction"],
        "acceptance": core_acceptance,
    }
    summary["status"] = (
        "PASS_CORE_AND_E2E"
        if all(summary["acceptance"].values()) and all(core_acceptance.values())
        else "FAIL_NUMERICAL"
    )
    atomic_json(e2e_result["summary_path"], summary)
    return str(summary["status"])


def run_bridge(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run legacy score export, current executor, current-core, and E2E comparisons."""

    root = Path(output_root)
    prepare_bridge(root)
    legacy_path = root / "trajectories/imagenet9_legacy_stage_scores.parquet"
    neutral_path = root / "trajectories/imagenet9.parquet"
    if not resume or not legacy_path.is_file() or not neutral_path.is_file():
        legacy_outputs = export_legacy_trajectory(root)
        legacy_path = legacy_outputs["stage_scores"]
        neutral_path = legacy_outputs["trajectory"]
    current_path = root / "trajectories/imagenet9_current_e2e_scans.parquet"
    if not resume or not current_path.is_file():
        current_path = run_current_executor(root)
    core_result = _compare_core_atomic(neutral_path, root)
    e2e_result = compare_e2e(current_path, legacy_path, root)
    status = _combine_core_and_e2e_status(core_result, e2e_result)
    return {
        "legacy_stage_scores": legacy_path,
        "trajectory": neutral_path,
        "current_stage_scores": current_path,
        "core_comparison": root / "comparisons/imagenet9_core.csv",
        "core_summary": core_result["summary_path"],
        "e2e_comparison": e2e_result["comparison"],
        "e2e_summary": e2e_result["summary_path"],
        "status": status,
    }


def compare_existing(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Recompute comparisons from already-materialized GPU score files."""

    root = Path(output_root)
    neutral_path = root / "trajectories/imagenet9.parquet"
    legacy_path = root / "trajectories/imagenet9_legacy_stage_scores.parquet"
    current_path = root / "trajectories/imagenet9_current_e2e_scans.parquet"
    missing = [
        str(path) for path in (neutral_path, legacy_path, current_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"ImageNet-9 comparison inputs are missing: {missing}")
    core_result = _compare_core_atomic(neutral_path, root)
    e2e_result = compare_e2e(current_path, legacy_path, root)
    status = _combine_core_and_e2e_status(core_result, e2e_result)
    return {
        "core_summary": core_result["summary_path"],
        "e2e_summary": e2e_result["summary_path"],
        "status": status,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("run", "compare"),
        help="run uses the B200; compare consumes existing score files on CPU",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete legacy/current score files before rebuilding comparisons",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        run_bridge(args.output_root, resume=args.resume)
        if args.action == "run"
        else compare_existing(args.output_root)
    )
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
