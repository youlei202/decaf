"""CPU-only replay of the sealed Attribution reference runs A0--A3.

The replay deliberately consumes persisted tables only.  It never imports the
model-facing stack and cannot claim that new GPU inference was performed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from decaf.experiments.attribution.endpoint import row_spearman
from decaf.experiments.common import RunContext, atomic_json, repository_root
from decaf.paper.reference import (
    discover_archive,
    load_reference_runs,
    materialize_inputs,
    receipt_dict,
    reference_roots,
    sha256_file,
    verify_archive,
)

REFERENCE_RUN_IDS = ("A0", "A1", "A2", "A3")
BOOTSTRAP_SEED = 8218
BOOTSTRAP_REPLICATES = 1_000
PRIMARY_METHODS = (
    "decaf_3",
    "decaf_5",
    "decaf_9",
    "ig_32",
    "ig_u_32",
    "kernel_shap_512",
    "endpoint_m",
)
PRIMARY_PAIRS = (
    ("decaf_3", "endpoint_m"),
    ("decaf_5", "endpoint_m"),
    ("decaf_9", "endpoint_m"),
    ("endpoint_m", "ig_32"),
    ("endpoint_m", "ig_u_32"),
    ("endpoint_m", "kernel_shap_512"),
)
TABLE8_LABELS = {
    "endpoint_m": "Endpoint M",
    "decaf_3": "DECAF-3",
    "decaf_5": "DECAF-5",
    "decaf_9": "DECAF-9",
    "ig_32": "IG-32",
    "ig_u_32": "IG-U-32",
    "kernel_shap_512": "KernelSHAP-512",
}
PRIMARY_SUPPORT_COUNTS = {
    ("funnybirds", "funnybirds_resnet50"): 499,
    ("funnybirds", "funnybirds_vgg16"): 497,
    ("funnybirds", "funnybirds_vit_b_16"): 488,
    ("imagenet", "resnet50"): 7_663,
    ("imagenet", "vgg16"): 7_189,
    ("imagenet", "vit_base_patch16_224"): 8_285,
}
PARTIMAGENET_SUPPORT_COUNTS = {
    "resnet50": 879,
    "convnext_large": 853,
    "swin_b": 892,
    "dinov2_vit_l_14": 962,
}
PARTIMAGENET_METHODS = (
    "decaf_3",
    "decaf_5",
    "decaf_9",
    "deep_lift",
    "endpoint_m",
    "exact_part_shapley",
    "gradient_shap",
    "ig_16",
    "ig_32",
    "input_x_gradient",
    "kernel_shap_512",
    "part_lime_1000",
    "part_occlusion",
    "rise_512",
    "smoothgrad_16",
)
PARTIMAGENET_METRICS = (
    "normalized_l1",
    "pearson",
    "sign_agreement",
    "spearman",
    "top1_agreement",
)
FULL50K_SUPPORT_COUNTS = {
    "resnet50": 38_460,
    "vgg16": 36_042,
    "vit_base_patch16_224": 41_374,
}
HEADLINE_TARGETS = {
    "funnybirds_decaf_5": (0.4032586441787493, 1.0e-14),
    "imagenet_decaf_5": (0.3668448266626176, 1.0e-14),
    "funnybirds_endpoint_m": (0.3235199814368542, 1.0e-14),
    "imagenet_endpoint_m": (0.3713393218889992, 1.0e-14),
    "ig32_over_decaf5_wall_time": (4.753018872418976, 1.0e-12),
    "ig32_over_decaf5_peak_memory": (2.3640978423701218, 1.0e-12),
    "partimagenet_common_support": (3_586.0, 0.0),
}


class AttributionReferenceError(RuntimeError):
    """A sealed Attribution input violated its public replay contract."""


@dataclass(frozen=True)
class AttributionReferenceBundle:
    """Materialized A0--A3 inputs rooted inside one run directory."""

    root: Path
    receipt_path: Path

    def path(self, run_id: str, member: str) -> Path:
        path = self.root / run_id / member
        if not path.is_file():
            raise AttributionReferenceError(
                f"materialized reference input is absent: {run_id}:{member}"
            )
        return path


def _reference_manifest_directory() -> Path:
    return repository_root() / "manifests" / "reference_runs"


def materialize_attribution_references(
    context: RunContext,
) -> AttributionReferenceBundle:
    """Verify and extract only A0--A3 analysis inputs, with hash receipts."""

    runs = load_reference_runs(_reference_manifest_directory())
    roots = reference_roots()
    destination = context.path / "paper_data" / "reference_inputs"
    inputs: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    for run_id in REFERENCE_RUN_IDS:
        run = runs[run_id]
        archive = discover_archive(run, roots)
        verify_archive(archive, run)
        archives.append(
            {
                "run_id": run_id,
                "filename": run.archive_filename,
                "resolved_path": str(archive),
                "sha256": sha256_file(archive),
                "size_bytes": archive.stat().st_size,
                "member_count": run.archive_member_count,
            }
        )
        inputs.extend(
            receipt_dict(item)
            for item in materialize_inputs(
                run,
                archive,
                run.analysis_inputs,
                destination,
            )
        )
    receipt_path = context.path / "receipts" / "attribution_reference_inputs.json"
    atomic_json(
        receipt_path,
        {
            "schema_version": 1,
            "run_ids": list(REFERENCE_RUN_IDS),
            "source_mode": "sealed_reference_replay",
            "inference_performed": False,
            "archives": archives,
            "inputs": inputs,
        },
    )
    return AttributionReferenceBundle(destination, receipt_path)


def validate_materialized_attribution_references(run_root: str | Path) -> None:
    """Revalidate archives and every extracted member before a resumed skip."""

    root = Path(run_root)
    receipt_path = root / "receipts" / "attribution_reference_inputs.json"
    if not receipt_path.is_file():
        raise AttributionReferenceError("formal resume requires attribution_reference_inputs.json")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttributionReferenceError("reference receipt is unreadable") from error
    if (
        payload.get("schema_version") != 1
        or tuple(payload.get("run_ids", ())) != REFERENCE_RUN_IDS
        or payload.get("source_mode") != "sealed_reference_replay"
        or payload.get("inference_performed") is not False
    ):
        raise AttributionReferenceError("reference receipt run inventory drifted")
    runs = load_reference_runs(_reference_manifest_directory())
    roots = reference_roots()
    archive_values = payload.get("archives")
    if not isinstance(archive_values, list):
        raise AttributionReferenceError("reference archive receipt inventory is invalid")
    archive_ids = [str(row.get("run_id")) for row in archive_values if isinstance(row, Mapping)]
    if len(archive_ids) != len(REFERENCE_RUN_IDS) or set(archive_ids) != set(REFERENCE_RUN_IDS):
        raise AttributionReferenceError("reference archive receipt inventory drifted")
    archive_rows = {str(row["run_id"]): row for row in archive_values}
    for run_id in REFERENCE_RUN_IDS:
        run = runs[run_id]
        archive = discover_archive(run, roots)
        verify_archive(archive, run)
        recorded = archive_rows.get(run_id)
        if recorded is None or (
            recorded.get("filename") != run.archive_filename
            or recorded.get("sha256") != sha256_file(archive)
            or recorded.get("size_bytes") != archive.stat().st_size
            or recorded.get("member_count") != run.archive_member_count
            or Path(str(recorded.get("resolved_path", ""))).resolve() != archive.resolve()
        ):
            raise AttributionReferenceError(f"reference archive receipt drifted for {run_id}")
    materialized = root / "paper_data" / "reference_inputs"
    input_values = payload.get("inputs")
    if not isinstance(input_values, list) or not all(
        isinstance(item, Mapping) for item in input_values
    ):
        raise AttributionReferenceError("reference input receipt inventory is invalid")
    expected = {
        (run_id, requested)
        for run_id in REFERENCE_RUN_IDS
        for requested in runs[run_id].analysis_inputs
    }
    observed = [
        (str(item.get("run_id")), str(item.get("requested_suffix"))) for item in input_values
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise AttributionReferenceError("reference input receipt inventory drifted")
    expected_paths: set[str] = set()
    for item in input_values:
        run_id = str(item["run_id"])
        requested = str(item["requested_suffix"])
        expected_relative = f"{run_id}/{requested}"
        if item.get("relative_path") != expected_relative or not str(
            item.get("resolved_member", "")
        ).endswith(requested):
            raise AttributionReferenceError(
                f"reference input path binding drifted: {run_id}:{requested}"
            )
        expected_paths.add(expected_relative)
        path = materialized / expected_relative
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise AttributionReferenceError(
                f"materialized reference hash drifted: {item.get('relative_path')}"
            )
    actual_paths = {
        path.relative_to(materialized).as_posix()
        for path in materialized.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise AttributionReferenceError("materialized reference file inventory drifted")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing or frame.empty:
        raise AttributionReferenceError(f"{label} is invalid; missing={missing}")


def _read_filtered_parquet(
    path: Path,
    *,
    columns: Sequence[str],
    expression: Any | None = None,
) -> pd.DataFrame:
    source = pads.dataset(str(path), format="parquet")
    try:
        table = source.to_table(columns=list(columns), filter=expression, use_threads=True)
    except Exception as error:
        raise AttributionReferenceError(f"cannot read projected Parquet {path}: {error}") from error
    return table.to_pandas()


def _array_stack(series: pd.Series, *, width: int, label: str) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for index, value in enumerate(series):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (width,) or not np.isfinite(array).all():
            raise AttributionReferenceError(
                f"{label} is invalid at row {index}: shape={array.shape}"
            )
        arrays.append(array)
    if not arrays:
        raise AttributionReferenceError(f"{label} is empty")
    return np.stack(arrays)


def _difference_statistics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or not first.size:
        raise AttributionReferenceError("alignment arrays have different shapes")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise AttributionReferenceError("alignment arrays contain non-finite values")
    difference = np.abs(first - second)
    return {
        "units": int(first.size),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "fraction_bitwise_equal": float(np.mean(first == second)),
    }


def _strict_support(bundle: AttributionReferenceBundle) -> pd.DataFrame:
    a2 = pd.read_parquet(bundle.path("A2", "results/manifests/strict_common_support.parquet"))
    a3 = pd.read_parquet(bundle.path("A3", "manifests/strict_common_support.parquet"))
    sort_keys = ["dataset", "model", "image_id"]
    pd.testing.assert_frame_equal(
        a2.sort_values(sort_keys).reset_index(drop=True),
        a3.sort_values(sort_keys).reset_index(drop=True),
        check_dtype=False,
        check_like=True,
    )
    _require_columns(a2, (*sort_keys, "included"), "strict common support")
    included = a2.loc[a2["included"].astype(bool), sort_keys].copy()
    for column in sort_keys:
        included[column] = included[column].astype(str)
    if included.duplicated(sort_keys).any():
        raise AttributionReferenceError("strict support contains duplicate keys")
    observed = included.groupby(["dataset", "model"], sort=True).size().to_dict()
    if observed != PRIMARY_SUPPORT_COUNTS:
        raise AttributionReferenceError(f"strict support counts drifted: {observed}")
    return included


def _restrict_to_support(
    frame: pd.DataFrame,
    support: pd.DataFrame,
    *,
    dataset: str,
    label: str,
) -> pd.DataFrame:
    source = frame.copy()
    for column in ("model", "image_id"):
        if source[column].isna().any():
            raise AttributionReferenceError(f"{label} has null {column}")
        source[column] = source[column].astype(str)
    keys = support.loc[support["dataset"].astype(str).eq(dataset), ["model", "image_id"]]
    selected = source.merge(keys, on=["model", "image_id"], how="inner", validate="many_to_one")
    if selected.empty:
        raise AttributionReferenceError(f"{label} has no rows on strict support")
    return selected


def validate_exact_inventory(
    frame: pd.DataFrame,
    *,
    expected_counts: Mapping[tuple[str, str], int],
    methods: Sequence[str],
) -> None:
    """Fail unless every selected method has the exact model image support."""

    _require_columns(
        frame,
        ("dataset", "model", "method", "image_id", "spearman"),
        "quality inventory",
    )
    if frame.duplicated(["dataset", "model", "method", "image_id"]).any():
        raise AttributionReferenceError("quality inventory has duplicate keys")
    for (dataset, model), count in expected_counts.items():
        selected = frame.loc[
            frame["dataset"].astype(str).eq(dataset) & frame["model"].astype(str).eq(model)
        ]
        method_ids: set[str] | None = None
        for method in methods:
            rows = selected.loc[selected["method"].astype(str).eq(method)]
            ids = set(rows["image_id"].astype(str))
            if len(ids) != count:
                raise AttributionReferenceError(
                    f"{dataset}/{model}/{method} support is {len(ids)} != {count}"
                )
            if method_ids is None:
                method_ids = ids
            elif ids != method_ids:
                raise AttributionReferenceError(
                    f"{dataset}/{model}/{method} image IDs are not exactly paired"
                )
            values = pd.to_numeric(rows["spearman"], errors="coerce").to_numpy(dtype=np.float64)
            if values.size != count or not np.isfinite(values).all():
                raise AttributionReferenceError(
                    f"{dataset}/{model}/{method} quality is incomplete/non-finite"
                )


def stable_seed(seed: int, replicate: int, dataset: str, model: str) -> int:
    """Return the historical group-order-independent bootstrap seed."""

    payload = f"{int(seed)}\0{int(replicate)}\0{dataset}\0{model}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def bootstrap_replicates(
    quality: pd.DataFrame,
    *,
    pairs: Sequence[tuple[str, str]] = PRIMARY_PAIRS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Run the fixed-model, shared-image-index paired bootstrap."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    prepared: dict[tuple[str, str], tuple[tuple[str, ...], np.ndarray]] = {}
    models_by_dataset: dict[str, tuple[str, ...]] = {}
    for dataset in sorted(quality["dataset"].astype(str).unique()):
        source = quality.loc[quality["dataset"].astype(str).eq(dataset)]
        methods = tuple(sorted(source["method"].astype(str).unique()))
        if set(methods) != set(PRIMARY_METHODS):
            raise AttributionReferenceError(
                f"bootstrap method inventory drifted for {dataset}: {methods}"
            )
        models = tuple(sorted(source["model"].astype(str).unique()))
        models_by_dataset[dataset] = models
        for model in models:
            rows = source.loc[source["model"].astype(str).eq(model)]
            wide = rows.pivot(index="image_id", columns="method", values="spearman")
            if set(wide.columns) != set(methods):
                raise AttributionReferenceError(
                    f"bootstrap inventory mismatch for {dataset}/{model}"
                )
            wide = wide.loc[:, list(methods)].sort_index()
            values = wide.to_numpy(dtype=np.float64)
            if wide.empty or not np.isfinite(values).all():
                raise AttributionReferenceError(
                    f"bootstrap matrix is incomplete for {dataset}/{model}"
                )
            prepared[(dataset, model)] = (methods, values)

    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        for dataset, models in models_by_dataset.items():
            model_means: dict[str, np.ndarray] = {}
            for model in models:
                methods, values = prepared[(dataset, model)]
                generator = np.random.default_rng(stable_seed(seed, replicate, dataset, model))
                indices = generator.integers(0, values.shape[0], size=values.shape[0])
                sampled = values[indices].mean(axis=0)
                model_means[model] = sampled
                rows.extend(
                    {
                        "dataset": dataset,
                        "statistic": "quality_mean",
                        "level": "model",
                        "model": model,
                        "method": method,
                        "left_method": None,
                        "right_method": None,
                        "replicate": replicate,
                        "value": float(value),
                    }
                    for method, value in zip(methods, sampled, strict=True)
                )
            macro = np.stack([model_means[model] for model in models]).mean(axis=0)
            rows.extend(
                {
                    "dataset": dataset,
                    "statistic": "quality_mean",
                    "level": "dataset_macro",
                    "model": "__macro__",
                    "method": method,
                    "left_method": None,
                    "right_method": None,
                    "replicate": replicate,
                    "value": float(value),
                }
                for method, value in zip(methods, macro, strict=True)
            )
            method_index = {method: index for index, method in enumerate(methods)}
            for left, right in pairs:
                differences = [
                    model_means[model][method_index[left]] - model_means[model][method_index[right]]
                    for model in models
                ]
                rows.append(
                    {
                        "dataset": dataset,
                        "statistic": "paired_difference",
                        "level": "dataset_macro",
                        "model": "__macro__",
                        "method": f"{left}-minus-{right}",
                        "left_method": left,
                        "right_method": right,
                        "replicate": replicate,
                        "value": float(np.mean(differences)),
                    }
                )
    result = pd.DataFrame(rows)
    keys = ["dataset", "statistic", "level", "model", "method", "replicate"]
    if result.duplicated(keys).any():
        raise AttributionReferenceError("bootstrap created duplicate rows")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _quantiles(values: np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise AttributionReferenceError("bootstrap statistic is invalid")
    return tuple(float(value) for value in np.quantile(array, (0.05, 0.95, 0.025, 0.975)))


def summarize_reference_quality(quality: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-model and equal-weight macro quality exactly as A3."""

    rows: list[dict[str, Any]] = []
    for dataset in sorted(quality["dataset"].astype(str).unique()):
        source = quality.loc[quality["dataset"].astype(str).eq(dataset)]
        models = tuple(sorted(source["model"].astype(str).unique()))
        methods = tuple(sorted(source["method"].astype(str).unique()))
        model_rows: list[dict[str, Any]] = []
        for model in models:
            for method in methods:
                values = source.loc[
                    source["model"].astype(str).eq(model) & source["method"].astype(str).eq(method),
                    "spearman",
                ].to_numpy(dtype=np.float64)
                draws = bootstrap.loc[
                    bootstrap["dataset"].eq(dataset)
                    & bootstrap["statistic"].eq("quality_mean")
                    & bootstrap["level"].eq("model")
                    & bootstrap["model"].eq(model)
                    & bootstrap["method"].eq(method),
                    "value",
                ].to_numpy(dtype=np.float64)
                ci90_low, ci90_high, ci95_low, ci95_high = _quantiles(draws)
                row = {
                    "dataset": dataset,
                    "level": "model",
                    "model": model,
                    "method": method,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "sem": (
                        float(np.std(values, ddof=1) / math.sqrt(values.size))
                        if values.size > 1
                        else 0.0
                    ),
                    "ci90_low": ci90_low,
                    "ci90_high": ci90_high,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "n_images": int(values.size),
                    "bootstrap_replicates": int(draws.size),
                }
                rows.append(row)
                model_rows.append(row)
        model_frame = pd.DataFrame(model_rows)
        for method in methods:
            selected = model_frame.loc[model_frame["method"].eq(method)].set_index("model")
            draws = bootstrap.loc[
                bootstrap["dataset"].eq(dataset)
                & bootstrap["statistic"].eq("quality_mean")
                & bootstrap["level"].eq("dataset_macro")
                & bootstrap["method"].eq(method),
                "value",
            ].to_numpy(dtype=np.float64)
            ci90_low, ci90_high, ci95_low, ci95_high = _quantiles(draws)
            rows.append(
                {
                    "dataset": dataset,
                    "level": "dataset_macro",
                    "model": "__macro__",
                    "method": method,
                    "mean": float(selected["mean"].mean()),
                    "median": float(selected["median"].mean()),
                    "sem": (float(np.std(draws, ddof=1)) if draws.size > 1 else 0.0),
                    "ci90_low": ci90_low,
                    "ci90_high": ci90_high,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "n_images": int(selected["n_images"].sum()),
                    "bootstrap_replicates": int(draws.size),
                    "n_models": len(models),
                    "n_images_per_model_json": json.dumps(
                        {model: int(selected.loc[model, "n_images"]) for model in models},
                        sort_keys=True,
                    ),
                    "model_aggregation": "equal_weight_macro_average",
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "level", "model", "method"], kind="stable")
        .reset_index(drop=True)
    )


def summarize_reference_pairwise(
    quality: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    pairs: Sequence[tuple[str, str]] = PRIMARY_PAIRS,
) -> pd.DataFrame:
    """Return all six exact-pivot, raw left-minus-right comparisons per dataset."""

    rows: list[dict[str, Any]] = []
    for dataset in sorted(quality["dataset"].astype(str).unique()):
        source = quality.loc[quality["dataset"].astype(str).eq(dataset)]
        models = tuple(sorted(source["model"].astype(str).unique()))
        for left, right in pairs:
            per_model: dict[str, np.ndarray] = {}
            for model in models:
                selected = source.loc[
                    source["model"].astype(str).eq(model)
                    & source["method"].astype(str).isin((left, right)),
                    ["image_id", "method", "spearman"],
                ]
                wide = selected.pivot(index="image_id", columns="method", values="spearman")
                if (
                    set(wide.columns) != {left, right}
                    or wide[[left, right]].isna().any().any()
                    or len(wide) * 2 != len(selected)
                ):
                    raise AttributionReferenceError(
                        f"unpaired comparison for {dataset}/{model}/{left}/{right}"
                    )
                per_model[model] = wide[left].to_numpy(dtype=np.float64) - wide[right].to_numpy(
                    dtype=np.float64
                )
            draws = bootstrap.loc[
                bootstrap["dataset"].eq(dataset)
                & bootstrap["statistic"].eq("paired_difference")
                & bootstrap["left_method"].eq(left)
                & bootstrap["right_method"].eq(right),
                "value",
            ].to_numpy(dtype=np.float64)
            ci90_low, ci90_high, ci95_low, ci95_high = _quantiles(draws)
            model_means = {model: float(values.mean()) for model, values in per_model.items()}
            model_medians = {model: float(np.median(values)) for model, values in per_model.items()}
            rows.append(
                {
                    "dataset": dataset,
                    "left_method": left,
                    "right_method": right,
                    "mean_paired_difference": float(np.mean(list(model_means.values()))),
                    "median_paired_difference": float(np.mean(list(model_medians.values()))),
                    "ci90_low": ci90_low,
                    "ci90_high": ci90_high,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "bootstrap_probability_difference_gt_zero": float(np.mean(draws > 0.0)),
                    "model_win_count": int(sum(value > 0.0 for value in model_means.values())),
                    "model_tie_count": int(sum(value == 0.0 for value in model_means.values())),
                    "n_models": len(models),
                    "n_paired_image_clusters": int(
                        sum(len(values) for values in per_model.values())
                    ),
                    "n_per_model_json": json.dumps(
                        {model: int(len(values)) for model, values in per_model.items()},
                        sort_keys=True,
                    ),
                    "bootstrap_replicates": int(draws.size),
                    "pairing": "same_resampled_images_within_each_model",
                    "sign_convention": "left_minus_right",
                    "median_aggregation": ("equal_weight_mean_of_within_model_medians"),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["dataset", "left_method", "right_method"], kind="stable"
    )
    if len(result) != 12:
        raise AttributionReferenceError(
            f"expected six pairwise rows per dataset, got {len(result)}"
        )
    return result.reset_index(drop=True)


def _load_funnybirds(
    bundle: AttributionReferenceBundle, support: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    models = (
        "funnybirds_resnet50",
        "funnybirds_vgg16",
        "funnybirds_vit_b_16",
    )
    part_path = bundle.path("A0", "formal/part_attribution.parquet")
    part_expression = (
        (pads.field("dataset") == "funnybirds")
        & pads.field("model").isin(models)
        & pads.field("method").isin(("endpoint_m", "part_occlusion"))
        & (pads.field("track") == "main")
        & (pads.field("reference") == "gaussian_blur_k31_sigma12")
    )
    parts = _read_filtered_parquet(
        part_path,
        columns=(
            "dataset",
            "model",
            "method",
            "image_id",
            "target_class",
            "part_group",
            "attribution_score",
            "track",
            "reference",
        ),
        expression=part_expression,
    )
    parts = _restrict_to_support(
        parts, support, dataset="funnybirds", label="FunnyBirds part attribution"
    )
    feature_keys = [
        "model",
        "image_id",
        "target_class",
        "part_group",
        "track",
        "reference",
    ]
    if parts.duplicated([*feature_keys, "method"]).any():
        raise AttributionReferenceError("FunnyBirds endpoint features contain duplicate keys")
    endpoint_parts = parts.loc[parts["method"].eq("endpoint_m")]
    signed_parts = parts.loc[parts["method"].eq("part_occlusion")]
    features = endpoint_parts.merge(
        signed_parts,
        on=feature_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_m", "_d"),
    )
    if len(features) != len(endpoint_parts) or len(features) != len(signed_parts):
        raise AttributionReferenceError(
            "FunnyBirds endpoint_m and part_occlusion features are not identical"
        )
    magnitude = pd.to_numeric(features["attribution_score_m"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    signed = pd.to_numeric(features["attribution_score_d"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    identity = _difference_statistics(magnitude, np.abs(signed))
    if identity["max_abs_difference"] != 0.0:
        raise AttributionReferenceError("FunnyBirds Endpoint M is not abs(d)")

    heldout_path = bundle.path("A0", "formal/heldout_quality.parquet")
    heldout_expression = (
        (pads.field("dataset") == "funnybirds")
        & pads.field("model").isin(models)
        & (pads.field("method") == "endpoint_m")
        & (pads.field("track") == "main")
        & (pads.field("reference") == "gaussian_blur_k31_sigma12")
    )
    heldout = _read_filtered_parquet(
        heldout_path,
        columns=(
            "dataset",
            "model",
            "method",
            "image_id",
            "track",
            "reference",
            "operator",
            "metric",
            "value",
        ),
        expression=heldout_expression,
    )
    heldout = _restrict_to_support(
        heldout, support, dataset="funnybirds", label="FunnyBirds heldout quality"
    )
    expected_operators = {"background_texture", "telea_dilate3"}
    operator_sets = heldout.groupby(["model", "image_id", "metric"], sort=False)["operator"].agg(
        lambda values: set(values.astype(str))
    )
    if not all(value == expected_operators for value in operator_sets):
        raise AttributionReferenceError(
            "FunnyBirds Endpoint M requires exactly two historical operators"
        )
    averaged = (
        heldout.groupby(["dataset", "model", "method", "image_id", "metric"], sort=True)["value"]
        .mean()
        .reset_index()
    )
    endpoint = averaged.loc[
        averaged["metric"].eq("spearman"),
        ["dataset", "model", "method", "image_id", "value"],
    ].rename(columns={"value": "spearman"})

    reused = pd.read_parquet(
        bundle.path("A2", "results/funnybirds/reused_quality.parquet"),
        columns=["dataset", "model", "method", "image_id", "spearman"],
    )
    supplement = pd.read_parquet(
        bundle.path("A2", "results/funnybirds/supplement_results.parquet"),
        columns=["dataset", "model", "method", "image_id", "spearman"],
    )
    existing = pd.concat([reused, supplement], ignore_index=True)
    existing = existing.loc[
        existing["method"]
        .astype(str)
        .isin(tuple(method for method in PRIMARY_METHODS if method != "endpoint_m"))
    ]
    existing = _restrict_to_support(
        existing, support, dataset="funnybirds", label="FunnyBirds primary quality"
    )
    quality = pd.concat([existing, endpoint], ignore_index=True)
    quality["dataset"] = "funnybirds"

    components_path = bundle.path("A0", "formal/decaf_components.parquet")
    component_expression = (
        (pads.field("dataset") == "funnybirds")
        & pads.field("model").isin(models)
        & pads.field("method").isin(("decaf_3", "decaf_5", "decaf_9"))
        & (pads.field("track") == "main")
        & (pads.field("reference") == "gaussian_blur_k31_sigma12")
    )
    components = _read_filtered_parquet(
        components_path,
        columns=("model", "method", "image_id", "target_class", "part_group", "M"),
        expression=component_expression,
    )
    components = _restrict_to_support(
        components, support, dataset="funnybirds", label="FunnyBirds DECAF components"
    )
    base = features.loc[:, ["model", "image_id", "target_class", "part_group"]].copy()
    base["endpoint_M"] = magnitude
    component_audit: list[dict[str, Any]] = []
    for (model, method), group in components.groupby(["model", "method"], sort=True):
        merged = group.merge(
            base,
            on=["model", "image_id", "target_class", "part_group"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(group):
            raise AttributionReferenceError(
                f"FunnyBirds component support mismatch for {model}/{method}"
            )
        statistics = _difference_statistics(
            merged["M"].to_numpy(dtype=np.float64),
            merged["endpoint_M"].to_numpy(dtype=np.float64),
        )
        component_audit.append({"model": str(model), "method": str(method), **statistics})
    return quality, {
        "source": "A0 endpoint_m paired with signed part_occlusion",
        "identity_M_equals_abs_d": identity,
        "operators": sorted(expected_operators),
        "operator_aggregation": "equal_mean_within_image",
        "component_M_numeric_crosscheck": component_audit,
    }


def _load_imagenet(
    bundle: AttributionReferenceBundle, support: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    science_path = bundle.path("A2", "results/imagenet/per_image_idsds.parquet")
    selected_methods = tuple(method for method in PRIMARY_METHODS if method != "endpoint_m")
    science = _read_filtered_parquet(
        science_path,
        columns=(
            "dataset",
            "model",
            "method",
            "image_id",
            "spearman",
            "effects",
            "decaf_M",
            "deletion_target_sha256",
        ),
        expression=pads.field("method").isin(selected_methods),
    )
    science["dataset"] = "imagenet"
    science = _restrict_to_support(
        science, support, dataset="imagenet", label="ImageNet primary quality"
    )
    existing = science.loc[
        science["method"].astype(str).isin(selected_methods),
        ["dataset", "model", "method", "image_id", "spearman"],
    ].copy()
    decaf5 = science.loc[
        science["method"].eq("decaf_5"),
        [
            "model",
            "image_id",
            "effects",
            "decaf_M",
            "deletion_target_sha256",
        ],
    ].copy()
    scores = _array_stack(decaf5["decaf_M"], width=16, label="ImageNet decaf_M")
    effects = _array_stack(decaf5["effects"], width=16, label="ImageNet effects")
    endpoint = decaf5.loc[:, ["model", "image_id"]].copy()
    endpoint.insert(0, "dataset", "imagenet")
    endpoint["method"] = "endpoint_m"
    endpoint["spearman"] = row_spearman(scores, effects)
    quality = pd.concat([existing, endpoint], ignore_index=True)

    base = decaf5.set_index(["model", "image_id"])["decaf_M"]
    schedule_audit: dict[str, Any] = {}
    for method in ("decaf_3", "decaf_9"):
        compare = science.loc[science["method"].eq(method)].set_index(["model", "image_id"])[
            "decaf_M"
        ]
        compare = compare.reindex(base.index)
        if compare.isna().any():
            raise AttributionReferenceError(f"ImageNet {method} M is incomplete")
        statistics = _difference_statistics(
            _array_stack(base.reset_index(drop=True), width=16, label="ImageNet DECAF-5 M"),
            _array_stack(compare.reset_index(drop=True), width=16, label=f"ImageNet {method} M"),
        )
        if statistics["max_abs_difference"] != 0.0:
            raise AttributionReferenceError(f"ImageNet DECAF schedule M differs for {method}")
        schedule_audit[method] = statistics

    deletion = pd.read_parquet(
        bundle.path("A2", "results/imagenet/deletion_targets.parquet"),
        columns=(
            "model",
            "image_id",
            "deletion_effects",
            "finite_complete",
            "member_path",
        ),
    )
    deletion = _restrict_to_support(
        deletion, support, dataset="imagenet", label="ImageNet deletion targets"
    )
    records = decaf5.merge(
        deletion,
        on=["model", "image_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(records) != len(decaf5):
        raise AttributionReferenceError(
            "ImageNet endpoint and deletion-target rows are not exactly paired"
        )
    target_effects = _array_stack(
        records["deletion_effects"], width=16, label="ImageNet deletion effects"
    )
    stored_effects = _array_stack(records["effects"], width=16, label="ImageNet quality effects")
    target_alignment = _difference_statistics(stored_effects, target_effects)
    if target_alignment["max_abs_difference"] != 0.0:
        raise AttributionReferenceError(
            "ImageNet quality effects differ from shared deletion targets"
        )
    return quality, {
        "source": "A2 stored DECAF-5 decaf_M and signed deletion effects",
        "deletion_contract": "exact one-to-one model/image target alignment",
        "deletion_effect_alignment": target_alignment,
        "decaf_M_schedule_identity": schedule_audit,
        "rows": len(records),
    }


def _validate_bootstrap_reproduction(
    bundle: AttributionReferenceBundle, generated: pd.DataFrame
) -> dict[str, Any]:
    sealed = pd.read_parquet(bundle.path("A3", "endpoint_m/bootstrap_with_m.parquet"))
    keys = ["dataset", "statistic", "level", "model", "method", "replicate"]
    paired_methods = {f"{left}-minus-{right}" for left, right in PRIMARY_PAIRS}
    expected = sealed.loc[
        sealed["dataset"].astype(str).isin(("funnybirds", "imagenet"))
        & (
            (
                ~sealed["statistic"].astype(str).eq("paired_difference")
                & sealed["method"].astype(str).isin(PRIMARY_METHODS)
            )
            | (
                sealed["statistic"].astype(str).eq("paired_difference")
                & sealed["method"].astype(str).isin(paired_methods)
            )
        ),
        [*keys, "value"],
    ]
    compared = generated.merge(
        expected,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_generated", "_sealed"),
    )
    if len(compared) != len(generated) or len(expected) != len(generated):
        raise AttributionReferenceError("generated bootstrap key inventory differs from sealed A3")
    difference = np.abs(
        compared["value_generated"].to_numpy(dtype=np.float64)
        - compared["value_sealed"].to_numpy(dtype=np.float64)
    )
    maximum = float(difference.max(initial=0.0))
    if maximum > 1.0e-15:
        raise AttributionReferenceError(
            f"historical bootstrap reproduction failed: max diff {maximum}"
        )
    return {
        "compared_rows": len(compared),
        "max_abs_difference": maximum,
        "bitwise_equal_fraction": float(
            np.mean(
                compared["value_generated"].to_numpy(dtype=np.float64)
                == compared["value_sealed"].to_numpy(dtype=np.float64)
            )
        ),
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "model_resampling": False,
        "shared_image_indices_across_methods": True,
    }


def _validate_pairwise_reproduction(
    bundle: AttributionReferenceBundle, generated: pd.DataFrame
) -> dict[str, Any]:
    sealed = pd.read_csv(bundle.path("A3", "endpoint_m/pairwise_differences_with_m.csv"))
    keys = ["dataset", "left_method", "right_method"]
    expected_pairs = {
        (dataset, left, right)
        for dataset in ("funnybirds", "imagenet")
        for left, right in PRIMARY_PAIRS
    }
    observed_pairs = {tuple(value) for value in generated[keys].itertuples(index=False, name=None)}
    if observed_pairs != expected_pairs:
        raise AttributionReferenceError(f"pairwise direction inventory drifted: {observed_pairs}")
    joined = generated.merge(
        sealed, on=keys, how="inner", validate="one_to_one", suffixes=("_new", "_old")
    )
    if len(joined) != 12:
        raise AttributionReferenceError("sealed A3 does not contain all 12 pairs")
    numeric = (
        "mean_paired_difference",
        "median_paired_difference",
        "ci90_low",
        "ci90_high",
        "ci95_low",
        "ci95_high",
        "bootstrap_probability_difference_gt_zero",
    )
    maximum = 0.0
    for column in numeric:
        difference = np.abs(
            joined[f"{column}_new"].to_numpy(dtype=np.float64)
            - joined[f"{column}_old"].to_numpy(dtype=np.float64)
        )
        maximum = max(maximum, float(difference.max(initial=0.0)))
    exact = (
        "model_win_count",
        "model_tie_count",
        "n_models",
        "n_paired_image_clusters",
        "n_per_model_json",
        "bootstrap_replicates",
        "pairing",
        "sign_convention",
        "median_aggregation",
    )
    for column in exact:
        if not joined[f"{column}_new"].astype(str).equals(joined[f"{column}_old"].astype(str)):
            raise AttributionReferenceError(f"pairwise field drifted: {column}")
    if maximum > 1.0e-15:
        raise AttributionReferenceError(
            f"pairwise numerical reproduction failed: max diff {maximum}"
        )
    return {
        "rows": len(joined),
        "pairs_per_dataset": 6,
        "max_abs_difference": maximum,
        "direction": "left_minus_right",
    }


def _validate_method_reproduction(
    bundle: AttributionReferenceBundle, summary: pd.DataFrame
) -> dict[str, Any]:
    sealed = pd.read_csv(bundle.path("A3", "endpoint_m/method_results_with_m.csv"))
    generated = summary.loc[
        summary["level"].eq("dataset_macro") & summary["method"].astype(str).isin(PRIMARY_METHODS)
    ].copy()
    joined = generated.merge(
        sealed,
        on=["dataset", "method"],
        how="inner",
        validate="one_to_one",
        suffixes=("_new", "_old"),
    )
    if len(joined) != 14:
        raise AttributionReferenceError(
            "computed method summary does not cover all A3 primary rows"
        )
    maximum = 0.0
    for column in (
        "mean",
        "median",
        "sem",
        "ci90_low",
        "ci90_high",
        "ci95_low",
        "ci95_high",
    ):
        difference = np.abs(
            joined[f"{column}_new"].to_numpy(dtype=np.float64)
            - joined[f"{column}_old"].to_numpy(dtype=np.float64)
        )
        maximum = max(maximum, float(difference.max(initial=0.0)))
    if maximum > 1.0e-15:
        raise AttributionReferenceError(f"method-summary reproduction failed: max diff {maximum}")
    for column in ("n_models", "bootstrap_replicates"):
        if not np.array_equal(
            joined[f"{column}_new"].to_numpy(dtype=np.int64),
            joined[f"{column}_old"].to_numpy(dtype=np.int64),
        ):
            raise AttributionReferenceError(f"method-summary contract field drifted: {column}")
    for column in ("n_images_per_model_json", "model_aggregation"):
        if not joined[f"{column}_new"].astype(str).equals(joined[f"{column}_old"].astype(str)):
            raise AttributionReferenceError(f"method-summary contract field drifted: {column}")
    sealed_models = pd.read_csv(bundle.path("A3", "endpoint_m/per_model_long_with_m.csv")).loc[
        lambda frame: frame["level"].astype(str).eq("model")
    ]
    generated_models = summary.loc[
        summary["level"].astype(str).eq("model")
        & summary["method"].astype(str).isin(PRIMARY_METHODS)
    ]
    model_keys = ["dataset", "level", "model", "method"]
    model_joined = generated_models.merge(
        sealed_models,
        on=model_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_new", "_old"),
    )
    if len(generated_models) != 42 or len(sealed_models) != 42 or len(model_joined) != 42:
        raise AttributionReferenceError("computed per-model summary does not cover all 42 A3 rows")
    model_maximum = 0.0
    for column in (
        "mean",
        "median",
        "sem",
        "ci90_low",
        "ci90_high",
        "ci95_low",
        "ci95_high",
    ):
        difference = np.abs(
            model_joined[f"{column}_new"].to_numpy(dtype=np.float64)
            - model_joined[f"{column}_old"].to_numpy(dtype=np.float64)
        )
        model_maximum = max(model_maximum, float(difference.max(initial=0.0)))
    for column in ("n_images", "bootstrap_replicates"):
        if not np.array_equal(
            model_joined[f"{column}_new"].to_numpy(dtype=np.int64),
            model_joined[f"{column}_old"].to_numpy(dtype=np.int64),
        ):
            raise AttributionReferenceError(f"per-model contract field drifted: {column}")
    if model_maximum > 1.0e-15:
        raise AttributionReferenceError(f"per-model reproduction failed: max diff {model_maximum}")
    return {
        "rows": len(joined),
        "max_abs_difference": maximum,
        "per_model_rows": len(model_joined),
        "per_model_max_abs_difference": model_maximum,
    }


def _table8_from_summary(bundle: AttributionReferenceBundle, summary: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 8 from reproduced model summaries and cross-check A3."""

    selected = summary.loc[
        summary["level"].astype(str).eq("model")
        & summary["method"].astype(str).isin(TABLE8_LABELS),
        ["dataset", "model", "method", "mean", "n_images"],
    ].copy()
    if len(selected) != 42 or selected.duplicated(["dataset", "model", "method"]).any():
        raise AttributionReferenceError("Table 8 requires exactly 42 unique model/method rows")
    support = selected.groupby(["dataset", "model"], sort=True)["n_images"].agg(
        ["first", "nunique"]
    )
    if not bool(support["nunique"].eq(1).all()):
        raise AttributionReferenceError("Table 8 method supports differ within a model")
    wide = selected.pivot(index=["dataset", "model"], columns="method", values="mean").rename(
        columns=TABLE8_LABELS
    )
    wide = wide[list(TABLE8_LABELS.values())].reset_index()
    wide["n_images"] = [
        int(support.loc[(row.dataset, row.model), "first"]) for row in wide.itertuples(index=False)
    ]
    wide.columns.name = None
    wide = wide.sort_values(["dataset", "model"], kind="stable").reset_index(drop=True)
    sealed = pd.read_csv(bundle.path("A3", "endpoint_m/per_model_with_m.csv"))
    sealed = sealed.sort_values(["dataset", "model"], kind="stable").reset_index(drop=True)
    if list(wide.columns) != list(sealed.columns) or len(wide) != len(sealed):
        raise AttributionReferenceError("Table 8 sealed cross-check schema drifted")
    for column in TABLE8_LABELS.values():
        if (
            np.max(
                np.abs(
                    wide[column].to_numpy(dtype=np.float64)
                    - sealed[column].to_numpy(dtype=np.float64)
                ),
                initial=0.0,
            )
            > 1.0e-15
        ):
            raise AttributionReferenceError(f"Table 8 sealed cross-check drifted: {column}")
    if not wide[["dataset", "model", "n_images"]].equals(sealed[["dataset", "model", "n_images"]]):
        raise AttributionReferenceError("Table 8 sealed key/support cross-check drifted")
    return wide


def _validate_partimagenet_legacy(
    bundle: AttributionReferenceBundle,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_parquet(
        bundle.path("A1", "common_support/partimagenet_common_support_manifest.parquet")
    )
    _require_columns(
        manifest,
        ("dataset", "model", "image_id", "included", "exclusion_reason"),
        "PartImageNet support manifest",
    )
    if len(manifest) != 4_096 or manifest.duplicated(["model", "image_id"]).any():
        raise AttributionReferenceError(
            "PartImageNet candidate inventory must contain 4096 unique rows"
        )
    included = manifest.loc[manifest["included"].astype(bool), ["model", "image_id"]].copy()
    included["model"] = included["model"].astype(str)
    included["image_id"] = included["image_id"].astype(str)
    counts = included.groupby("model", sort=True).size().to_dict()
    if counts != PARTIMAGENET_SUPPORT_COUNTS or len(included) != 3_586:
        raise AttributionReferenceError(f"PartImageNet common support drifted: {counts}")
    exclusions = (
        manifest.loc[~manifest["included"].astype(bool), "exclusion_reason"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    expected_exclusions = {
        "correct_classification_not_evidenced_by_formal_outputs": 267,
        (
            "incomplete_primary_methods[part_lime_1000:"
            "missing_part_attribution|missing_heldout_quality]"
        ): 243,
    }
    if exclusions != expected_exclusions:
        raise AttributionReferenceError(f"PartImageNet exclusion reasons drifted: {exclusions}")

    summary = pd.read_csv(bundle.path("A1", "common_support/common_support_method_summary.csv"))
    summary = summary.loc[
        summary["analysis_scope"].astype(str).eq("partimagenet_deep")
        & summary["dataset"].astype(str).eq("partimagenet")
    ].copy()
    methods = tuple(sorted(summary["method"].astype(str).unique()))
    metrics = tuple(sorted(summary["metric"].astype(str).unique()))
    expected_operator_set = {"background_texture", "telea_dilate3"}

    heldout_expression = (
        (pads.field("dataset") == "partimagenet")
        & pads.field("method").isin(methods)
        & (pads.field("track") == "main")
        & (pads.field("reference") == "gaussian_blur_k31_sigma12")
    )
    heldout = _read_filtered_parquet(
        bundle.path("A0", "formal/heldout_quality.parquet"),
        columns=(
            "dataset",
            "model",
            "method",
            "image_id",
            "operator",
            "metric",
            "value",
        ),
        expression=heldout_expression,
    )
    heldout["model"] = heldout["model"].astype(str)
    heldout["image_id"] = heldout["image_id"].astype(str)
    heldout = heldout.merge(included, on=["model", "image_id"], how="inner", validate="many_to_one")
    heldout["value"] = pd.to_numeric(heldout["value"], errors="coerce")
    if not np.isfinite(heldout["value"].to_numpy(dtype=np.float64)).all():
        raise AttributionReferenceError("PartImageNet heldout quality contains non-finite values")
    operator_groups = heldout.groupby(["model", "method", "image_id", "metric"], sort=False)[
        "operator"
    ]
    operator_count = operator_groups.size()
    operator_sets = operator_groups.agg(lambda values: set(values.astype(str)))
    if not bool((operator_count == 2).all()) or not all(
        value == expected_operator_set for value in operator_sets
    ):
        raise AttributionReferenceError(
            "PartImageNet quality lacks either historical heldout operator"
        )
    averaged = (
        heldout.groupby(["model", "method", "image_id", "metric"], sort=True)["value"]
        .mean()
        .reset_index()
    )
    for model, expected_count in PARTIMAGENET_SUPPORT_COUNTS.items():
        expected_ids = set(included.loc[included["model"].eq(model), "image_id"].astype(str))
        for method in methods:
            for metric in metrics:
                ids = set(
                    averaged.loc[
                        averaged["model"].eq(model)
                        & averaged["method"].eq(method)
                        & averaged["metric"].eq(metric),
                        "image_id",
                    ].astype(str)
                )
                if ids != expected_ids or len(ids) != expected_count:
                    raise AttributionReferenceError(
                        f"PartImageNet exact support failed for {model}/{method}/{metric}"
                    )

    maximum = 0.0
    for row in summary.itertuples(index=False):
        model_means: list[float] = []
        for model in PARTIMAGENET_SUPPORT_COUNTS:
            values = averaged.loc[
                averaged["model"].eq(model)
                & averaged["method"].eq(row.method)
                & averaged["metric"].eq(row.metric),
                "value",
            ].to_numpy(dtype=np.float64)
            model_means.append(float(values.mean()))
        recomputed = float(np.mean(model_means))
        maximum = max(maximum, abs(recomputed - float(row.mean)))
    if maximum > 5.0e-10:
        raise AttributionReferenceError(
            f"PartImageNet raw reaggregation drifted: max diff {maximum}"
        )

    part_expression = (
        (pads.field("dataset") == "partimagenet")
        & pads.field("method").isin(methods)
        & (pads.field("track") == "main")
        & (pads.field("reference") == "gaussian_blur_k31_sigma12")
    )
    parts = _read_filtered_parquet(
        bundle.path("A0", "formal/part_attribution.parquet"),
        columns=("model", "method", "image_id", "part_group", "attribution_score"),
        expression=part_expression,
    )
    parts["model"] = parts["model"].astype(str)
    parts["image_id"] = parts["image_id"].astype(str)
    parts = parts.merge(included, on=["model", "image_id"], how="inner", validate="many_to_one")
    if not np.isfinite(
        pd.to_numeric(parts["attribution_score"], errors="coerce").to_numpy(dtype=np.float64)
    ).all():
        raise AttributionReferenceError("PartImageNet part attribution contains non-finite values")
    observed_part_keys = parts[["model", "method", "image_id"]].drop_duplicates()
    for model, expected_count in PARTIMAGENET_SUPPORT_COUNTS.items():
        expected_ids = set(included.loc[included["model"].eq(model), "image_id"].astype(str))
        for method in methods:
            ids = set(
                observed_part_keys.loc[
                    observed_part_keys["model"].eq(model) & observed_part_keys["method"].eq(method),
                    "image_id",
                ].astype(str)
            )
            if ids != expected_ids or len(ids) != expected_count:
                raise AttributionReferenceError(
                    f"PartImageNet part support failed for {model}/{method}"
                )
    return summary, {
        "candidate_rows": len(manifest),
        "included_rows": len(included),
        "counts_per_model": counts,
        "exclusion_reason_counts": exclusions,
        "methods": list(methods),
        "metrics": list(metrics),
        "operators": sorted(expected_operator_set),
        "raw_reaggregation_max_abs_difference": maximum,
        "support_gate": "exact image_id pivot for every model/method/metric",
    }


def _derive_partimagenet_raw_support(
    bundle: AttributionReferenceBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Derive common support from raw completeness, never from included alone."""

    manifest = pd.read_parquet(
        bundle.path(
            "A1",
            "common_support/partimagenet_common_support_manifest.parquet",
        )
    )
    _require_columns(
        manifest,
        (
            "model",
            "image_id",
            "number_of_parts",
            "correct_classification_evidenced",
            "included",
            "exclusion_reason",
        ),
        "PartImageNet support manifest",
    )
    manifest["model"] = manifest["model"].astype(str)
    manifest["image_id"] = manifest["image_id"].astype(str)
    manifest["number_of_parts"] = pd.to_numeric(manifest["number_of_parts"], errors="coerce")
    if (
        len(manifest) != 4_096
        or manifest.duplicated(["model", "image_id"]).any()
        or set(manifest["model"]) != set(PARTIMAGENET_SUPPORT_COUNTS)
        or manifest["number_of_parts"].isna().any()
        or not manifest["number_of_parts"].between(2, 5).all()
    ):
        raise AttributionReferenceError("PartImageNet candidate inventory drifted")
    candidates = manifest[["model", "image_id", "number_of_parts"]]
    base_expression = (
        (pads.field("dataset") == "partimagenet")
        & pads.field("method").isin(PARTIMAGENET_METHODS)
        & (pads.field("track") == "main")
        & (pads.field("reference") == "gaussian_blur_k31_sigma12")
    )
    parts = _read_filtered_parquet(
        bundle.path("A0", "formal/part_attribution.parquet"),
        columns=(
            "model",
            "method",
            "image_id",
            "target_class",
            "part_group",
            "attribution_score",
        ),
        expression=base_expression,
    )
    for column in ("model", "method", "image_id", "part_group"):
        parts[column] = parts[column].astype(str)
    parts = parts.merge(
        candidates,
        on=["model", "image_id"],
        how="inner",
        validate="many_to_one",
    )
    parts["attribution_score"] = pd.to_numeric(parts["attribution_score"], errors="coerce")
    part_keys = ["model", "method", "image_id", "part_group"]
    if (
        parts.duplicated(part_keys).any()
        or not np.isfinite(parts["attribution_score"].to_numpy(dtype=np.float64)).all()
    ):
        raise AttributionReferenceError("PartImageNet part rows are invalid")
    part_vectors = (
        parts.groupby(["model", "method", "image_id"], sort=True)
        .agg(
            row_count=("part_group", "size"),
            unique_parts=("part_group", "nunique"),
            expected_parts=("number_of_parts", "first"),
            vector=(
                "part_group",
                lambda values: json.dumps(sorted(values.astype(str).tolist())),
            ),
        )
        .reset_index()
    )
    part_vectors["complete"] = part_vectors["row_count"].eq(
        part_vectors["expected_parts"]
    ) & part_vectors["unique_parts"].eq(part_vectors["expected_parts"])
    part_images = (
        part_vectors.groupby(["model", "image_id"], sort=True)
        .agg(
            methods=("method", "nunique"),
            all_complete=("complete", "all"),
            vector_count=("vector", "nunique"),
        )
        .reset_index()
    )
    valid_parts = part_images.loc[
        part_images["methods"].eq(len(PARTIMAGENET_METHODS))
        & part_images["all_complete"]
        & part_images["vector_count"].eq(1),
        ["model", "image_id"],
    ]

    heldout = _read_filtered_parquet(
        bundle.path("A0", "formal/heldout_quality.parquet"),
        columns=(
            "model",
            "method",
            "image_id",
            "operator",
            "metric",
            "value",
        ),
        expression=base_expression,
    )
    for column in ("model", "method", "image_id", "operator", "metric"):
        heldout[column] = heldout[column].astype(str)
    heldout = heldout.merge(
        candidates,
        on=["model", "image_id"],
        how="inner",
        validate="many_to_one",
    )
    heldout["value"] = pd.to_numeric(heldout["value"], errors="coerce")
    heldout_keys = ["model", "method", "image_id", "metric", "operator"]
    operators = {"background_texture", "telea_dilate3"}
    if (
        heldout.duplicated(heldout_keys).any()
        or set(heldout["operator"]) != operators
        or set(heldout["metric"]) != set(PARTIMAGENET_METRICS)
        or not np.isfinite(heldout["value"].to_numpy(dtype=np.float64)).all()
    ):
        raise AttributionReferenceError("PartImageNet heldout rows are invalid")
    operator_groups = (
        heldout.groupby(["model", "method", "image_id", "metric"], sort=True)["operator"]
        .agg(row_count="size", operator_count="nunique")
        .reset_index()
    )
    operator_groups["complete"] = operator_groups["row_count"].eq(len(operators)) & operator_groups[
        "operator_count"
    ].eq(len(operators))
    heldout_images = (
        operator_groups.groupby(["model", "image_id"], sort=True)
        .agg(
            groups=("complete", "size"),
            methods=("method", "nunique"),
            all_complete=("complete", "all"),
        )
        .reset_index()
    )
    valid_heldout = heldout_images.loc[
        heldout_images["groups"].eq(len(PARTIMAGENET_METHODS) * len(PARTIMAGENET_METRICS))
        & heldout_images["methods"].eq(len(PARTIMAGENET_METHODS))
        & heldout_images["all_complete"],
        ["model", "image_id"],
    ]
    classification = manifest.loc[
        manifest["correct_classification_evidenced"].astype(bool),
        ["model", "image_id"],
    ]
    derived = classification.merge(
        valid_parts,
        on=["model", "image_id"],
        how="inner",
        validate="one_to_one",
    ).merge(
        valid_heldout,
        on=["model", "image_id"],
        how="inner",
        validate="one_to_one",
    )
    included = manifest.loc[manifest["included"].astype(bool), ["model", "image_id"]].copy()
    comparison = derived.merge(
        included,
        on=["model", "image_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        raise AttributionReferenceError(
            "PartImageNet included support is not derivable from raw completeness"
        )
    counts = included.groupby("model", sort=True).size().to_dict()
    if counts != PARTIMAGENET_SUPPORT_COUNTS or len(included) != 3_586:
        raise AttributionReferenceError(f"PartImageNet support drifted: {counts}")
    included_contract = manifest.loc[
        manifest["included"].astype(bool),
        ["model", "image_id", "number_of_parts"],
    ]
    part_observations = int(included_contract["number_of_parts"].sum())
    part_distribution = {
        int(key): int(value)
        for key, value in included_contract["number_of_parts"].value_counts().sort_index().items()
    }
    if part_observations != 10_937 or part_distribution != {
        2: 1_136,
        3: 1_289,
        4: 1_007,
        5: 154,
    }:
        raise AttributionReferenceError("PartImageNet semantic part inventory drifted")
    included_parts = parts.merge(
        included,
        on=["model", "image_id"],
        how="inner",
        validate="many_to_one",
    )
    if included_parts.groupby("method", sort=True).size().to_dict() != {
        method: part_observations for method in PARTIMAGENET_METHODS
    }:
        raise AttributionReferenceError(
            "PartImageNet semantic vectors are incomplete on derived support"
        )
    included_heldout = heldout.merge(
        included,
        on=["model", "image_id"],
        how="inner",
        validate="many_to_one",
    )
    expected_heldout = (
        len(included) * len(PARTIMAGENET_METHODS) * len(PARTIMAGENET_METRICS) * len(operators)
    )
    if len(included_heldout) != expected_heldout:
        raise AttributionReferenceError("PartImageNet heldout support drifted")
    averaged = (
        included_heldout.groupby(["model", "method", "image_id", "metric"], sort=True)["value"]
        .mean()
        .reset_index()
    )
    audit = {
        "candidate_rows": len(manifest),
        "included_rows": len(included),
        "counts_per_model": counts,
        "part_observations_per_method": part_observations,
        "part_count_distribution": part_distribution,
        "operators": sorted(operators),
    }
    return manifest, included, averaged, included_parts, audit


def _regenerate_partimagenet_summary(
    bundle: AttributionReferenceBundle,
    *,
    included: pd.DataFrame,
    averaged: pd.DataFrame,
    part_observations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute all 75 Table 11 rows from A0 points and A1 macro draws."""

    sealed = pd.read_csv(bundle.path("A1", "common_support/common_support_method_summary.csv"))
    sealed = sealed.loc[
        sealed["analysis_scope"].astype(str).eq("partimagenet_deep")
        & sealed["dataset"].astype(str).eq("partimagenet")
    ].copy()
    methods = tuple(sorted(sealed["method"].astype(str).unique()))
    metrics = tuple(sorted(sealed["metric"].astype(str).unique()))
    if (
        len(sealed) != 75
        or sealed.duplicated(["method", "metric"]).any()
        or methods != PARTIMAGENET_METHODS
        or metrics != PARTIMAGENET_METRICS
    ):
        raise AttributionReferenceError(
            "PartImageNet summary must contain exactly 15 methods x 5 metrics"
        )
    model_points = (
        averaged.groupby(["model", "method", "metric"], sort=True)["value"]
        .agg(raw_mean="mean", raw_median="median")
        .reset_index()
    )
    if len(model_points) != (
        len(PARTIMAGENET_SUPPORT_COUNTS) * len(PARTIMAGENET_METHODS) * len(PARTIMAGENET_METRICS)
    ):
        raise AttributionReferenceError("PartImageNet raw model summary inventory drifted")
    points = (
        model_points.groupby(["method", "metric"], sort=True)[["raw_mean", "raw_median"]]
        .mean()
        .reset_index()
    )
    bootstrap = pd.read_parquet(
        bundle.path("A1", "common_support/common_support_bootstrap.parquet")
    )
    bootstrap = bootstrap.loc[
        bootstrap["analysis_scope"].astype(str).eq("partimagenet_deep")
        & bootstrap["dataset"].astype(str).eq("partimagenet")
    ].copy()
    bootstrap["method"] = bootstrap["method"].astype(str)
    bootstrap["metric"] = bootstrap["metric"].astype(str)
    bootstrap["level"] = bootstrap["level"].astype(str)
    bootstrap["replicate"] = pd.to_numeric(bootstrap["replicate"], errors="coerce")
    bootstrap["value"] = pd.to_numeric(bootstrap["value"], errors="coerce")
    keys = ["level", "model", "method", "metric", "replicate"]
    if (
        len(bootstrap) != 375_000
        or bootstrap.duplicated(keys).any()
        or set(bootstrap["method"]) != set(PARTIMAGENET_METHODS)
        or set(bootstrap["metric"]) != set(PARTIMAGENET_METRICS)
        or bootstrap["level"].value_counts().to_dict()
        != {"model": 300_000, "dataset_macro": 75_000}
        or not np.isfinite(bootstrap["value"].to_numpy(dtype=np.float64)).all()
    ):
        raise AttributionReferenceError("PartImageNet bootstrap inventory drifted")
    coverage = bootstrap.groupby(
        ["level", "model", "method", "metric"],
        dropna=False,
        sort=True,
    )["replicate"].agg(rows="size", unique="nunique", minimum="min", maximum="max")
    if not bool(
        coverage["rows"].eq(BOOTSTRAP_REPLICATES).all()
        and coverage["unique"].eq(BOOTSTRAP_REPLICATES).all()
        and coverage["minimum"].eq(0).all()
        and coverage["maximum"].eq(BOOTSTRAP_REPLICATES - 1).all()
    ):
        raise AttributionReferenceError("PartImageNet bootstrap replicate coverage drifted")
    model_draws = bootstrap.loc[bootstrap["level"].eq("model")]
    if set(model_draws["model"].astype(str)) != set(PARTIMAGENET_SUPPORT_COUNTS):
        raise AttributionReferenceError("PartImageNet bootstrap model inventory drifted")
    recomputed_macro = (
        model_draws.pivot(
            index=["method", "metric", "replicate"],
            columns="model",
            values="value",
        )
        .mean(axis=1)
        .rename("recomputed_macro")
        .reset_index()
    )
    macro_draws = bootstrap.loc[
        bootstrap["level"].eq("dataset_macro"),
        ["method", "metric", "replicate", "value"],
    ]
    macro_joined = macro_draws.merge(
        recomputed_macro,
        on=["method", "metric", "replicate"],
        how="inner",
        validate="one_to_one",
    )
    macro_difference = float(
        np.max(
            np.abs(
                macro_joined["value"].to_numpy(dtype=np.float64)
                - macro_joined["recomputed_macro"].to_numpy(dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    if len(macro_joined) != 75_000 or macro_difference > 5.0e-16:
        raise AttributionReferenceError(
            "PartImageNet macro draws are not equal-model fixed-model averages"
        )
    uncertainty = (
        macro_draws.groupby(["method", "metric"], sort=True)["value"]
        .agg(
            standard_error=lambda values: float(np.std(values, ddof=1)),
            ci90_low=lambda values: float(np.quantile(values, 0.05)),
            ci90_high=lambda values: float(np.quantile(values, 0.95)),
            ci95_low=lambda values: float(np.quantile(values, 0.025)),
            ci95_high=lambda values: float(np.quantile(values, 0.975)),
        )
        .reset_index()
    )
    generated = points.merge(
        uncertainty,
        on=["method", "metric"],
        how="inner",
        validate="one_to_one",
    ).rename(columns={"raw_mean": "mean", "raw_median": "median"})
    generated = generated.assign(
        analysis_scope="partimagenet_deep",
        support_set="strict_all_candidates",
        dataset="partimagenet",
        subset="deep",
        number_of_models=4,
        number_of_common_images_total=len(included),
        minimum_common_images_per_model=min(PARTIMAGENET_SUPPORT_COUNTS.values()),
        maximum_common_images_per_model=max(PARTIMAGENET_SUPPORT_COUNTS.values()),
        common_images_per_model_json=json.dumps(PARTIMAGENET_SUPPORT_COUNTS, sort_keys=True),
        number_of_part_observations=part_observations,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        model_aggregation="equal_weight_macro_average",
        operator_aggregation="equal_mean_within_image",
        operators_json=json.dumps(["background_texture", "telea_dilate3"]),
    )
    generated = (
        generated[list(sealed.columns)]
        .sort_values(["method", "metric"], kind="stable")
        .reset_index(drop=True)
    )
    joined = generated.merge(
        sealed,
        on=["method", "metric"],
        how="inner",
        validate="one_to_one",
        suffixes=("_generated", "_sealed"),
    )
    if len(joined) != 75:
        raise AttributionReferenceError("PartImageNet generated summary inventory drifted")
    maximum = 0.0
    numeric = (
        "mean",
        "median",
        "standard_error",
        "ci90_low",
        "ci90_high",
        "ci95_low",
        "ci95_high",
    )
    for column in numeric:
        difference = np.abs(
            joined[f"{column}_generated"].to_numpy(dtype=np.float64)
            - joined[f"{column}_sealed"].to_numpy(dtype=np.float64)
        )
        maximum = max(maximum, float(difference.max(initial=0.0)))
    if maximum > 5.0e-10:
        raise AttributionReferenceError(
            f"PartImageNet raw/bootstrap reaggregation drifted: {maximum}"
        )
    exact_columns = [
        column for column in sealed.columns if column not in {"method", "metric", *numeric}
    ]
    for column in exact_columns:
        if (
            not joined[f"{column}_generated"]
            .astype(str)
            .equals(joined[f"{column}_sealed"].astype(str))
        ):
            raise AttributionReferenceError(
                f"PartImageNet summary contract field drifted: {column}"
            )
    return generated, {
        "summary_rows": len(generated),
        "bootstrap_rows": len(bootstrap),
        "bootstrap_macro_alignment_max_abs_difference": macro_difference,
        "raw_bootstrap_reaggregation_max_abs_difference": maximum,
    }


def _validate_partimagenet_components(
    bundle: AttributionReferenceBundle,
    *,
    included: pd.DataFrame,
    included_parts: pd.DataFrame,
    part_observations: int,
) -> dict[str, Any]:
    """Validate the complete M/E/C/F/Abs vectors used by all DECAF schedules."""

    components = _read_filtered_parquet(
        bundle.path("A0", "formal/decaf_components.parquet"),
        columns=(
            "model",
            "method",
            "image_id",
            "target_class",
            "part_group",
            "M",
            "E",
            "C",
            "F",
            "Abs",
        ),
        expression=(
            (pads.field("dataset") == "partimagenet")
            & pads.field("method").isin(("decaf_3", "decaf_5", "decaf_9"))
            & (pads.field("track") == "main")
            & (pads.field("reference") == "gaussian_blur_k31_sigma12")
        ),
    )
    for column in ("model", "method", "image_id", "part_group"):
        components[column] = components[column].astype(str)
    components = components.merge(
        included,
        on=["model", "image_id"],
        how="inner",
        validate="many_to_one",
    )
    keys = ["model", "method", "image_id", "part_group"]
    values = ["M", "E", "C", "F", "Abs"]
    if (
        len(components) != 3 * part_observations
        or components.duplicated(keys).any()
        or set(components["method"]) != {"decaf_3", "decaf_5", "decaf_9"}
        or not np.isfinite(components[values].to_numpy(dtype=np.float64)).all()
        or bool((components[values] < 0).any().any())
    ):
        raise AttributionReferenceError("PartImageNet DECAF component inventory drifted")
    expected = included_parts.loc[
        included_parts["method"].isin(("decaf_3", "decaf_5", "decaf_9")),
        [*keys, "target_class", "attribution_score"],
    ]
    joined = components.merge(
        expected,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_component", "_part"),
    )
    if len(joined) != len(expected):
        raise AttributionReferenceError("PartImageNet DECAF component keys drifted")
    if not np.array_equal(
        joined["target_class_component"].to_numpy(dtype=np.int64),
        joined["target_class_part"].to_numpy(dtype=np.int64),
    ):
        raise AttributionReferenceError("PartImageNet DECAF target classes drifted")
    e_difference = float(
        np.max(
            np.abs(
                joined["E"].to_numpy(dtype=np.float64)
                - joined["attribution_score"].to_numpy(dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    abs_difference = float(
        np.max(
            np.abs(
                joined["Abs"].to_numpy(dtype=np.float64)
                - joined[["E", "C", "F"]].sum(axis=1).to_numpy(dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    if e_difference != 0.0 or abs_difference > 2.0e-7:
        raise AttributionReferenceError("PartImageNet DECAF component identities drifted")
    return {
        "decaf_component_rows": len(components),
        "decaf_e_identity_max_abs_difference": e_difference,
        "decaf_abs_composition_max_abs_difference": abs_difference,
    }


def _validate_partimagenet(
    bundle: AttributionReferenceBundle,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Regenerate all formal PartImageNet rows from raw and bootstrap inputs."""

    manifest, included, averaged, included_parts, raw_audit = _derive_partimagenet_raw_support(
        bundle
    )
    exclusions = (
        manifest.loc[~manifest["included"].astype(bool), "exclusion_reason"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    expected_exclusions = {
        "correct_classification_not_evidenced_by_formal_outputs": 267,
        (
            "incomplete_primary_methods[part_lime_1000:"
            "missing_part_attribution|missing_heldout_quality]"
        ): 243,
    }
    if exclusions != expected_exclusions:
        raise AttributionReferenceError(f"PartImageNet exclusion reasons drifted: {exclusions}")
    part_observations = int(raw_audit["part_observations_per_method"])
    summary, summary_audit = _regenerate_partimagenet_summary(
        bundle,
        included=included,
        averaged=averaged,
        part_observations=part_observations,
    )
    component_audit = _validate_partimagenet_components(
        bundle,
        included=included,
        included_parts=included_parts,
        part_observations=part_observations,
    )
    return summary, {
        **raw_audit,
        **summary_audit,
        **component_audit,
        "exclusion_reason_counts": exclusions,
        "methods": list(PARTIMAGENET_METHODS),
        "metrics": list(PARTIMAGENET_METRICS),
        "support_gate": (
            "classification evidence AND exact 15-method semantic part vectors "
            "AND exact 15-method x 5-metric x 2-operator heldout pivot"
        ),
    }


def _validate_timing(
    bundle: AttributionReferenceBundle,
) -> tuple[pd.DataFrame, dict[str, float]]:
    raw = pd.read_csv(bundle.path("A1", "dinov2_g_timing/dinov2_g_timing_raw.csv"))
    summary = pd.read_csv(bundle.path("A1", "dinov2_g_timing/dinov2_g_timing_summary.csv"))
    raw = raw.loc[raw["status"].astype(str).eq("completed")].copy()
    if len(raw) != 24:
        raise AttributionReferenceError(
            f"DINOv2-g timing requires 24 completed repeats, got {len(raw)}"
        )
    for row in summary.itertuples(index=False):
        selected = raw.loc[raw["method"].astype(str).eq(str(row.method))]
        if len(selected) != int(row.valid_repeats) or len(selected) != 3:
            raise AttributionReferenceError(f"DINOv2-g repeat inventory drifted for {row.method}")
        for column in (
            "wall_seconds_per_image",
            "cuda_seconds_per_image",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "forward_calls_per_image",
            "forward_rows_per_image",
            "backward_calls_per_image",
        ):
            recomputed = float(np.median(selected[column].to_numpy(dtype=np.float64)))
            if recomputed != float(getattr(row, column)):
                raise AttributionReferenceError(
                    f"DINOv2-g median timing drifted for {row.method}/{column}"
                )
    indexed = summary.set_index("method")
    ratios = {
        "ig32_over_decaf5_wall_time": float(
            indexed.loc["ig_32", "wall_seconds_per_image"]
            / indexed.loc["decaf_5", "wall_seconds_per_image"]
        ),
        "ig32_over_decaf5_peak_memory": float(
            indexed.loc["ig_32", "peak_allocated_bytes"]
            / indexed.loc["decaf_5", "peak_allocated_bytes"]
        ),
    }
    sealed_ratios = json.loads(
        bundle.path("A1", "dinov2_g_timing/dinov2_g_exact_ratios.json").read_text(encoding="utf-8")
    )
    mapping = {
        "ig32_over_decaf5_wall_time": "ig32_over_decaf5_wall_time",
        "ig32_over_decaf5_peak_memory": "ig32_over_decaf5_peak_memory",
    }
    for name, sealed_name in mapping.items():
        if abs(ratios[name] - float(sealed_ratios[sealed_name])) > 1.0e-15:
            raise AttributionReferenceError(f"DINOv2-g ratio drifted: {name}")
    return summary, ratios


def _validate_full50k(
    bundle: AttributionReferenceBundle,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    a2 = (
        pd.read_csv(bundle.path("A2", "results/imagenet/full50k_scale_check.csv"))
        .sort_values(["method", "level", "model"], kind="stable")
        .reset_index(drop=True)
    )
    a3 = pd.read_csv(bundle.path("A3", "endpoint_m/full50k_with_m.csv"))
    non_m = (
        a3.loc[~a3["method"].astype(str).eq("endpoint_m")]
        .sort_values(["method", "level", "model"], kind="stable")
        .reset_index(drop=True)
    )
    if len(a2) != 12 or not a2.equals(non_m):
        raise AttributionReferenceError(
            "A3 non-M full50k rows do not exactly reproduce declared A2 rows"
        )
    endpoint = a3.loc[a3["method"].astype(str).eq("endpoint_m")].copy()
    if len(endpoint) != 4:
        raise AttributionReferenceError("A3 full50k Endpoint M inventory drifted")
    frame = (
        pd.concat([a2, endpoint], ignore_index=True)
        .sort_values(["method", "level", "model"], kind="stable")
        .reset_index(drop=True)
    )
    methods = {"decaf_5", "ig_32", "ig_u_32", "endpoint_m"}
    if set(frame["method"].astype(str)) != methods:
        raise AttributionReferenceError("full50k method inventory drifted")
    for method in methods:
        selected = frame.loc[
            frame["method"].astype(str).eq(method) & frame["level"].astype(str).eq("model")
        ]
        counts = {str(row.model): int(row.n_images) for row in selected.itertuples(index=False)}
        if counts != FULL50K_SUPPORT_COUNTS:
            raise AttributionReferenceError(f"full50k support drifted for {method}: {counts}")
        macro = frame.loc[
            frame["method"].astype(str).eq(method) & frame["level"].astype(str).eq("dataset_macro")
        ]
        if len(macro) != 1:
            raise AttributionReferenceError(f"full50k macro missing for {method}")
        recorded = json.loads(str(macro.iloc[0]["n_images_per_model_json"]))
        if recorded != FULL50K_SUPPORT_COUNTS:
            raise AttributionReferenceError(f"full50k macro support JSON drifted for {method}")
    return frame, {
        "methods": sorted(methods),
        "counts_per_model": FULL50K_SUPPORT_COUNTS,
        "support_gate": "identical exact model image counts for all four methods",
    }


def _validated_method_results(bundle: AttributionReferenceBundle) -> pd.DataFrame:
    """Use A2 as the primary 13-method source and append only validated A3 M."""

    a2 = (
        pd.read_csv(bundle.path("A2", "results/statistics/method_results.csv"))
        .sort_values(["dataset", "method"], kind="stable")
        .reset_index(drop=True)
    )
    a3 = pd.read_csv(bundle.path("A3", "endpoint_m/method_results_with_m.csv"))
    non_m = (
        a3.loc[~a3["method"].astype(str).eq("endpoint_m")]
        .sort_values(["dataset", "method"], kind="stable")
        .reset_index(drop=True)
    )
    if (
        len(a2) != 26
        or a2["method"].nunique() != 13
        or list(a2.columns) != list(a3.columns)
        or not a2.equals(non_m)
    ):
        raise AttributionReferenceError(
            "A3 non-M method rows do not exactly reproduce declared A2 rows"
        )
    endpoint = a3.loc[a3["method"].astype(str).eq("endpoint_m")].copy()
    if len(endpoint) != 2 or set(endpoint["dataset"].astype(str)) != {
        "funnybirds",
        "imagenet",
    }:
        raise AttributionReferenceError("A3 Endpoint M method inventory drifted")
    return (
        pd.concat([a2, endpoint], ignore_index=True)
        .sort_values(["dataset", "method"], kind="stable")
        .reset_index(drop=True)
    )


FORMAL_TABLE_SOURCES: dict[int, tuple[tuple[str, str], ...]] = {
    2: (
        ("A2", "results/statistics/method_results.csv"),
        ("A2", "results/compute/timing_summary.csv"),
        ("A2", "results/compute/memory_summary.csv"),
        ("A2", "results/compute/query_counts.csv"),
        ("A3", "endpoint_m/method_results_with_m.csv"),
    ),
    3: (
        ("A1", "dinov2_g_timing/dinov2_g_quality_timing_join.csv"),
        ("A1", "dinov2_g_timing/dinov2_g_timing_summary.csv"),
    ),
    4: (("A3", "endpoint_m/pairwise_differences_with_m.csv"),),
    6: (
        ("A2", "results/statistics/method_results.csv"),
        ("A3", "endpoint_m/method_results_with_m.csv"),
    ),
    7: (("A3", "endpoint_m/pairwise_differences_with_m.csv"),),
    8: (
        ("A3", "endpoint_m/per_model_long_with_m.csv"),
        ("A3", "endpoint_m/per_model_with_m.csv"),
    ),
    9: (
        ("A2", "results/imagenet/full50k_scale_check.csv"),
        ("A3", "endpoint_m/full50k_with_m.csv"),
    ),
    10: (
        ("A2", "results/compute/timing_summary.csv"),
        ("A2", "results/compute/memory_summary.csv"),
        ("A2", "results/compute/query_counts.csv"),
    ),
    11: (
        ("A1", "common_support/common_support_method_summary.csv"),
        ("A0", "formal/part_attribution.parquet"),
        ("A0", "formal/heldout_quality.parquet"),
    ),
}


def _join_compute_tables(bundle: AttributionReferenceBundle) -> pd.DataFrame:
    members = (
        ("timing", "results/compute/timing_summary.csv"),
        ("memory", "results/compute/memory_summary.csv"),
        ("queries", "results/compute/query_counts.csv"),
    )
    frames = [(label, pd.read_csv(bundle.path("A2", member))) for label, member in members]
    common_keys = [
        column
        for column in ("dataset", "model", "method")
        if all(column in frame.columns for _, frame in frames)
    ]
    if "method" not in common_keys:
        raise AttributionReferenceError("ImageNet compute summaries have no common method key")
    result: pd.DataFrame | None = None
    for label, frame in frames:
        if frame.duplicated(common_keys).any():
            raise AttributionReferenceError(f"{label} compute summary has duplicate join keys")
        renamed = frame.rename(
            columns={
                column: f"{label}_{column}" for column in frame.columns if column not in common_keys
            }
        )
        result = (
            renamed
            if result is None
            else result.merge(
                renamed,
                on=common_keys,
                how="outer",
                validate="one_to_one",
            )
        )
    assert result is not None
    return result.sort_values(common_keys, kind="stable").reset_index(drop=True)


def _build_formal_tables(
    bundle: AttributionReferenceBundle,
    *,
    summary: pd.DataFrame,
    partimagenet_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    full50k: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    method_results = _validated_method_results(bundle)
    compute = _join_compute_tables(bundle)
    compute_numeric = compute.select_dtypes(include=[np.number]).columns.tolist()
    compute_by_method = (
        compute.groupby("method", as_index=False, sort=True)[compute_numeric]
        .mean()
        .rename(columns={column: f"imagenet_compute_{column}" for column in compute_numeric})
    )
    imagenet = method_results.loc[method_results["dataset"].astype(str).eq("imagenet")].merge(
        compute_by_method, on="method", how="left", validate="one_to_one"
    )
    funnybirds = method_results.loc[method_results["dataset"].astype(str).eq("funnybirds")]
    table_02 = pd.concat([funnybirds, imagenet], ignore_index=True, sort=False)
    table_03 = pd.read_csv(bundle.path("A1", "dinov2_g_timing/dinov2_g_quality_timing_join.csv"))
    table_08 = _table8_from_summary(bundle, summary)
    return {
        2: table_02,
        3: table_03,
        4: pairwise.copy(),
        6: method_results.copy(),
        7: pairwise.copy(),
        8: table_08,
        9: full50k.copy(),
        10: compute,
        11: partimagenet_summary.copy(),
    }


def _assert_headlines(
    summary: pd.DataFrame,
    *,
    ratios: Mapping[str, float],
    partimagenet_support: int,
) -> dict[str, Any]:
    macro = summary.loc[summary["level"].astype(str).eq("dataset_macro")]
    indexed = macro.set_index(["dataset", "method"])["mean"]
    actual = {
        "funnybirds_decaf_5": float(indexed.loc[("funnybirds", "decaf_5")]),
        "imagenet_decaf_5": float(indexed.loc[("imagenet", "decaf_5")]),
        "funnybirds_endpoint_m": float(indexed.loc[("funnybirds", "endpoint_m")]),
        "imagenet_endpoint_m": float(indexed.loc[("imagenet", "endpoint_m")]),
        "ig32_over_decaf5_wall_time": float(ratios["ig32_over_decaf5_wall_time"]),
        "ig32_over_decaf5_peak_memory": float(ratios["ig32_over_decaf5_peak_memory"]),
        "partimagenet_common_support": float(partimagenet_support),
    }
    assertions: dict[str, Any] = {}
    for name, (expected, tolerance) in HEADLINE_TARGETS.items():
        difference = abs(actual[name] - expected)
        if difference > tolerance:
            raise AttributionReferenceError(
                f"Attribution headline drifted: {name}: {actual[name]} != {expected}"
            )
        assertions[name] = {
            "status": "passed",
            "actual": actual[name],
            "expected": expected,
            "absolute_difference": difference,
            "tolerance": tolerance,
        }
    return {
        "schema_version": 1,
        "assertion_count": len(assertions),
        "status": "passed",
        "assertions": assertions,
    }


def _validate_historical_code(bundle: AttributionReferenceBundle) -> dict[str, Any]:
    path = bundle.path("A3", "code/endpoint_m.py")
    text = path.read_text(encoding="utf-8")
    required_fragments = (
        "stable_seed(_BOOTSTRAP_SEED, replicate, dataset, model)",
        "same_resampled_images_within_each_model",
        '"sign_convention": "left_minus_right"',
        '"median_aggregation": "equal_weight_mean_of_within_model_medians"',
        '"background_texture", "telea_dilate3"',
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise AttributionReferenceError(f"sealed A3 historical implementation drifted: {missing}")
    return {
        "sha256": sha256_file(path),
        "required_contract_fragments": len(required_fragments),
        "status": "passed",
    }


def analyze_reference(context: RunContext) -> dict[str, Any]:
    """Materialize A0--A3 and reproduce formal Attribution results on CPU."""

    from decaf.experiments.attribution.analyze import atomic_csv
    from decaf.experiments.attribution.evaluate import atomic_parquet

    bundle = materialize_attribution_references(context)
    support = _strict_support(bundle)
    funnybirds, funnybirds_audit = _load_funnybirds(bundle, support)
    imagenet, imagenet_audit = _load_imagenet(bundle, support)
    quality = pd.concat([funnybirds, imagenet], ignore_index=True)
    validate_exact_inventory(
        quality,
        expected_counts=PRIMARY_SUPPORT_COUNTS,
        methods=PRIMARY_METHODS,
    )

    bootstrap = bootstrap_replicates(quality)
    summary = summarize_reference_quality(quality, bootstrap)
    pairwise = summarize_reference_pairwise(quality, bootstrap)
    bootstrap_audit = _validate_bootstrap_reproduction(bundle, bootstrap)
    method_audit = _validate_method_reproduction(bundle, summary)
    pairwise_audit = _validate_pairwise_reproduction(bundle, pairwise)
    partimagenet_summary, partimagenet_audit = _validate_partimagenet(bundle)
    _, timing_ratios = _validate_timing(bundle)
    full50k, full50k_audit = _validate_full50k(bundle)
    historical_code_audit = _validate_historical_code(bundle)
    headlines = _assert_headlines(
        summary,
        ratios=timing_ratios,
        partimagenet_support=partimagenet_audit["included_rows"],
    )

    metrics = context.path / "metrics"
    atomic_parquet(quality, metrics / "reference_primary_quality.parquet")
    atomic_parquet(bootstrap, metrics / "bootstrap_with_m.parquet")
    atomic_csv(summary, metrics / "reference_quality_summary.csv")
    atomic_csv(
        pd.read_csv(bundle.path("A3", "endpoint_m/method_results_with_m.csv")),
        metrics / "method_results.csv",
    )
    atomic_csv(
        pd.read_csv(bundle.path("A3", "endpoint_m/per_model_with_m.csv")),
        metrics / "per_model_results.csv",
    )
    atomic_csv(pairwise, metrics / "pairwise_differences.csv")
    atomic_csv(
        pd.read_csv(bundle.path("A1", "dinov2_g_timing/dinov2_g_timing_summary.csv")),
        metrics / "timing_summary.csv",
    )
    atomic_json(metrics / "attribution_headlines.json", headlines)
    atomic_json(
        metrics / "endpoint_m" / "source_audit.json",
        {
            "schema_version": 1,
            "passed": True,
            "generated_in_stage": "analyze",
            "inference_performed": False,
            "source_mode": "sealed_reference_replay",
            "funnybirds": funnybirds_audit,
            "imagenet": imagenet_audit,
        },
    )
    audit = {
        "schema_version": 1,
        "status": "passed",
        "inference_performed": False,
        "reference_receipt": str(bundle.receipt_path.relative_to(context.path)),
        "primary_support_counts": {
            f"{dataset}/{model}": count
            for (dataset, model), count in PRIMARY_SUPPORT_COUNTS.items()
        },
        "bootstrap": bootstrap_audit,
        "method_results": method_audit,
        "pairwise": pairwise_audit,
        "partimagenet": partimagenet_audit,
        "full50k": full50k_audit,
        "timing_ratios": timing_ratios,
        "historical_code": historical_code_audit,
        "headline_assertions": headlines["assertion_count"],
    }
    atomic_json(metrics / "attribution_reference_audit.json", audit)

    table_root = metrics / "formal_tables"
    tables = _build_formal_tables(
        bundle,
        summary=summary,
        partimagenet_summary=partimagenet_summary,
        pairwise=pairwise,
        full50k=full50k,
    )
    table_receipts: list[dict[str, Any]] = []
    for number, frame in sorted(tables.items()):
        path = table_root / f"table_{number:02d}.csv"
        atomic_csv(frame, path)
        table_receipts.append(
            {
                "table": number,
                "path": path.relative_to(context.path).as_posix(),
                "sha256": sha256_file(path),
                "rows": len(frame),
                "sources": [
                    {"run_id": run_id, "member": member}
                    for run_id, member in FORMAL_TABLE_SOURCES[number]
                ],
            }
        )
    atomic_json(
        metrics / "formal_table_inputs.json",
        {
            "schema_version": 1,
            "mapping": "exact_registered_visual_manifest_inputs",
            "tables": table_receipts,
        },
    )
    return {
        "source_mode": "sealed_reference_replay",
        "inference_performed": False,
        "materialized_run_ids": list(REFERENCE_RUN_IDS),
        "primary_quality_rows": len(quality),
        "bootstrap_rows": len(bootstrap),
        "pairwise_rows": len(pairwise),
        "partimagenet_common_support": partimagenet_audit["included_rows"],
        "headline_assertions": headlines["assertion_count"],
        "formal_table_count": len(tables),
    }


__all__ = [
    "AttributionReferenceBundle",
    "AttributionReferenceError",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "FORMAL_TABLE_SOURCES",
    "HEADLINE_TARGETS",
    "PARTIMAGENET_SUPPORT_COUNTS",
    "PRIMARY_METHODS",
    "PRIMARY_PAIRS",
    "PRIMARY_SUPPORT_COUNTS",
    "analyze_reference",
    "bootstrap_replicates",
    "materialize_attribution_references",
    "stable_seed",
    "summarize_reference_pairwise",
    "summarize_reference_quality",
    "validate_exact_inventory",
    "validate_materialized_attribution_references",
]
