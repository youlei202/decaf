"""Aggregate exact attribution slices and run current-core plus current-E2E checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.experiments.common import atomic_json
from tools.crossgen.compare_core import (
    BOUNDARY_ABS,
    HARD_MISMATCH_ABS,
    TIER_A_ATOL,
    TIER_A_RTOL,
    TIER_B_ABS,
    compare_record,
)
from tools.crossgen.current_attribution_export import (
    ENDPOINT_METHODS,
    MODELS,
    TARGET_IDENTITIES,
)
from tools.crossgen.legacy_attribution_export import ENDPOINT_EPSILON, METHODS
from tools.crossgen.schema import (
    NEUTRAL_COLUMNS,
    read_trajectory_record,
    sha256_file,
    write_trajectory_record,
)

DEFAULT_ROOT = Path("/work/Users/leiyo/decaf_cross_generation_equivalence/v2")
A2_FUNNY_SUPPORT = Path(
    "/work/Users/leiyo/decaf_idsds_funnybirds_v1_results/manifests/"
    "funnybirds_common_support.parquet"
)
A2_FUNNY_QUALITY = Path(
    "/work/Users/leiyo/decaf_idsds_funnybirds_v1_results/funnybirds/"
    "reused_quality.parquet"
)
SUMMARY_NAMES = ("M", "E", "C", "F", "Abs")
VECTOR_METRICS = ("score", *SUMMARY_NAMES)
UNIT_KEYS = ("dataset", "model", "method", "image_id", "factor_or_part_id")
IMAGE_KEYS = ("dataset", "model", "method", "image_id")
EXPECTED_UNITS = 1476
EXPECTED_IMAGES = 144
COUNTERFACTUAL_IDENTITIES = {
    "normalized_zero_4x4_patch_deletion": {
        "reference": "normalized_zero",
        "intervention_operator": "endpoint_part_deletion",
    },
    "gaussian_blur_k31_sigma12": {
        "reference": "locked_gaussian_blur_k31_sigma12_raw_rgb",
        "intervention_operator": "endpoint_part_deletion",
    },
}


def _legacy_path(root: Path, dataset: str, model_id: str) -> Path:
    return root / "trajectories" / f"attribution__{dataset}__{model_id}.parquet"


def _current_path(root: Path, dataset: str, model_id: str) -> Path:
    return root / "trajectories" / (
        f"attribution_current__{dataset}__{model_id}.parquet"
    )


def _fraction(values: pd.Series) -> float:
    return float(values.astype(bool).mean()) if len(values) else float("nan")


def _stats(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("comparison statistics require non-empty finite values")
    return {
        "median_absolute_error": float(np.median(array)),
        "p95_absolute_error": float(np.percentile(array, 95)),
        "maximum_absolute_error": float(np.max(array)),
    }


def _dominant(e: float, c: float, f: float) -> str:
    values = {"E": float(e), "C": float(c), "F": float(f)}
    maximum = max(values.values())
    return "|".join(name for name in ("E", "C", "F") if values[name] == maximum)


def _metadata(raw: Any) -> dict[str, Any]:
    payload = json.loads(str(raw))
    if not isinstance(payload, dict):
        raise TypeError("attribution metadata must be a JSON object")
    return payload


def _one(series: pd.Series, *, name: str) -> Any:
    values = series.tolist()
    if not values:
        raise ValueError(f"{name} is empty")
    first = values[0]
    if all(isinstance(value, (float, int, np.number)) for value in values):
        array = np.asarray(values, dtype=np.float64)
        if not np.allclose(array, array[0], atol=1.0e-12, rtol=1.0e-10):
            raise ValueError(f"{name} changes within one unit")
    elif any(value != first for value in values[1:]):
        raise ValueError(f"{name} changes within one unit")
    return first


def combine_legacy_records(root: Path) -> tuple[pd.DataFrame, Path]:
    """Combine six disjoint legacy records into one atomic neutral record."""

    frames: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    seen_units: set[str] = set()
    for dataset, model_ids in MODELS.items():
        for model_id in model_ids:
            path = _legacy_path(root, dataset, model_id)
            frame = read_trajectory_record(path)
            units = set(frame["unit_id"].astype(str))
            overlap = sorted(seen_units & units)
            if overlap:
                raise ValueError(
                    f"legacy unit IDs overlap across model records: {overlap[:3]}"
                )
            seen_units.update(units)
            prefix = f"attribution::{dataset}::{model_id}::"
            if not all(value.startswith(prefix) for value in units):
                raise ValueError(f"legacy identity prefix differs: {dataset}/{model_id}")
            frames.append(frame)
            source_paths.append(path)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["unit_id", "stage_index"]).any():
        raise ValueError("combined neutral record has duplicate unit/stage rows")
    if len(seen_units) != EXPECTED_UNITS:
        raise ValueError(
            f"combined neutral has {len(seen_units)} units, expected {EXPECTED_UNITS}"
        )
    output = root / "trajectories/attribution.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(".attribution.part.parquet")
    temporary.unlink(missing_ok=True)
    try:
        write_trajectory_record(combined.loc[:, list(NEUTRAL_COLUMNS)], temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(
        root / "manifests/attribution_aggregate.json",
        {
            "schema_version": 1,
            "experiment_family": "attribution",
            "source_records": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in source_paths
            ],
            "row_count": int(len(combined)),
            "unit_count": int(len(seen_units)),
            "unit_definition": "dataset/model/method/image/factor_or_part",
            "unit_ids_unique_across_six_records": True,
            "output": str(output),
            "output_sha256": sha256_file(output),
        },
    )
    return combined, output


def _funny_spearman(root: Path, neutral: pd.DataFrame) -> pd.DataFrame:
    support = pd.read_parquet(A2_FUNNY_SUPPORT)
    funny_rows = neutral["unit_id"].astype(str).str.startswith(
        "attribution::funnybirds::"
    )
    image_ids = sorted(
        set(neutral.loc[funny_rows, "sample_or_pair_id"].astype(str))
    )
    selected_support = support[
        support["model"].isin(MODELS["funnybirds"])
        & support["image_id"].astype(str).isin(image_ids)
    ].copy()
    if (
        len(selected_support) != len(MODELS["funnybirds"]) * 8
        or not selected_support["included"].astype(bool).all()
        or not selected_support["correctly_classified"].astype(bool).all()
    ):
        raise ValueError("A2 FunnyBirds common-support gate is incomplete")
    for observed in selected_support["observed_methods"].astype(str):
        if any(method not in observed for method in METHODS):
            raise ValueError("A2 FunnyBirds support lacks a DECAF method")

    quality = pd.read_parquet(A2_FUNNY_QUALITY)
    quality = quality[
        quality["dataset"].eq("funnybirds")
        & quality["model"].isin(MODELS["funnybirds"])
        & quality["method"].isin(METHODS)
        & quality["image_id"].astype(str).isin(image_ids)
    ].copy()
    keys = list(IMAGE_KEYS)
    expected = len(MODELS["funnybirds"]) * len(METHODS) * 8
    if (
        len(quality) != expected
        or quality.duplicated(keys).any()
        or not quality["finite_complete"].astype(bool).all()
        or not quality["correctly_classified"].astype(bool).all()
        or not np.isfinite(quality["spearman"].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("A2 reused FunnyBirds Spearman rows are incomplete")
    atomic_json(
        root / "manifests/attribution_funnybirds_a2_gate.json",
        {
            "schema_version": 1,
            "common_support": str(A2_FUNNY_SUPPORT),
            "common_support_sha256": sha256_file(A2_FUNNY_SUPPORT),
            "reused_quality": str(A2_FUNNY_QUALITY),
            "reused_quality_sha256": sha256_file(A2_FUNNY_QUALITY),
            "models": list(MODELS["funnybirds"]),
            "image_ids": image_ids,
            "methods": list(METHODS),
            "rows": int(len(quality)),
            "all_included": True,
        },
    )
    return quality.loc[:, [*keys, "spearman"]].rename(
        columns={"spearman": "historical_spearman"}
    )


def historical_units(
    neutral: pd.DataFrame,
    funny_spearman: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse neutral stages to one strict sealed row per part/patch."""

    rows: list[dict[str, Any]] = []
    image_spearman: dict[tuple[str, ...], float] = {}
    for unit_id, group in neutral.groupby("unit_id", sort=True):
        metadata = _metadata(_one(group["metadata_json"], name="metadata_json"))
        parts = str(unit_id).split("::")
        if len(parts) != 6 or parts[0] != "attribution":
            raise ValueError(f"malformed attribution unit ID: {unit_id}")
        _, dataset, model, method, parsed_image_id, parsed_factor = parts
        image_id = str(_one(group["sample_or_pair_id"], name="sample_or_pair_id"))
        factor = str(_one(group["factor_or_part_id"], name="factor_or_part_id"))
        if parsed_image_id != image_id or parsed_factor != factor:
            raise ValueError(
                "attribution unit ID identity differs from columns: "
                f"{unit_id!r} vs image={image_id!r}, factor={factor!r}"
            )
        if dataset not in MODELS or model not in MODELS[dataset] or method not in METHODS:
            raise ValueError(f"attribution unit ID has unsupported identity: {unit_id}")
        image_key = (dataset, model, method, image_id)
        sealed_spearman = metadata.get("sealed_spearman")
        if sealed_spearman is not None:
            value = float(sealed_spearman)
            if image_key in image_spearman and not np.isclose(
                image_spearman[image_key],
                value,
                atol=1.0e-12,
                rtol=1.0e-10,
            ):
                raise ValueError("sealed Spearman changes across vector elements")
            image_spearman[image_key] = value
        counterfactual_map = str(
            _one(group["counterfactual_map"], name="counterfactual_map")
        )
        if counterfactual_map not in COUNTERFACTUAL_IDENTITIES:
            raise ValueError(
                f"unsupported historical counterfactual map: {counterfactual_map}"
            )
        counterfactual_identity = COUNTERFACTUAL_IDENTITIES[counterfactual_map]
        row = {
            "unit_id": str(unit_id),
            "dataset": dataset,
            "model": model,
            "method": method,
            "image_id": image_id,
            "factor_or_part_id": factor,
            "historical_checkpoint_sha256": str(
                _one(group["checkpoint_sha256"], name="checkpoint_sha256")
            ),
            "historical_target": int(metadata["target"]),
            "historical_counterfactual_map": counterfactual_map,
            "historical_reference": counterfactual_identity["reference"],
            "historical_intervention_operator": counterfactual_identity[
                "intervention_operator"
            ],
            "historical_endpoint_d": float(metadata["historical_endpoint_d"]),
            "historical_score": float(
                _one(group["historical_E"], name="historical_E")
            ),
        }
        for name in SUMMARY_NAMES:
            row[f"historical_{name}"] = float(
                _one(group[f"historical_{name}"], name=f"historical_{name}")
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_UNITS or result.duplicated(list(UNIT_KEYS)).any():
        raise ValueError("historical part/patch inventory is not one-to-one")

    for row in funny_spearman.itertuples(index=False):
        image_spearman[
            (str(row.dataset), str(row.model), str(row.method), str(row.image_id))
        ] = float(row.historical_spearman)
    expected_keys = set(
        map(tuple, result.loc[:, list(IMAGE_KEYS)].drop_duplicates().to_numpy())
    )
    if set(image_spearman) != expected_keys:
        missing = sorted(expected_keys - set(image_spearman))
        extra = sorted(set(image_spearman) - expected_keys)
        raise ValueError(
            f"historical Spearman identity differs: missing={missing}, extra={extra}"
        )
    result["historical_spearman"] = [
        image_spearman[tuple(row)]
        for row in result.loc[:, list(IMAGE_KEYS)].itertuples(index=False, name=None)
    ]
    return result


def _current_checkpoint(raw: Any) -> str:
    assets = json.loads(str(raw))
    if (
        not isinstance(assets, list)
        or len(assets) != 1
        or not isinstance(assets[0], dict)
        or not isinstance(assets[0].get("sha256"), str)
    ):
        raise ValueError("current checkpoint asset identity is not singular")
    return str(assets[0]["sha256"])


def expand_current(current_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Expand current image rows to one row per named vector element."""

    rows: list[dict[str, Any]] = []
    seen_images: set[tuple[str, ...]] = set()
    identity_to_map = {
        (value["reference"], value["intervention_operator"]): key
        for key, value in COUNTERFACTUAL_IDENTITIES.items()
    }
    for frame in current_frames:
        for image in frame.itertuples(index=False):
            dataset = str(image.dataset)
            model = str(image.model)
            method = str(image.method)
            image_id = str(image.image_id)
            image_key = (dataset, model, method, image_id)
            if image_key in seen_images:
                raise ValueError(f"duplicate current image/method row: {image_key}")
            seen_images.add(image_key)
            names = tuple(map(str, image.part_names))
            vectors = {
                "score": np.asarray(image.patch_scores, dtype=np.float64),
                **{
                    name: np.asarray(
                        getattr(image, f"decaf_{name}"),
                        dtype=np.float64,
                    )
                    for name in SUMMARY_NAMES
                },
            }
            endpoint = np.asarray(image.endpoint_effects, dtype=np.float64)
            reference = str(image.reference)
            intervention_operator = str(image.intervention_operator)
            endpoint_method = ENDPOINT_METHODS.get(dataset)
            if endpoint_method is None:
                raise ValueError(f"unsupported current dataset: {dataset}")
            expected_identity = TARGET_IDENTITIES[dataset][endpoint_method]
            if (
                reference != expected_identity["reference"]
                or intervention_operator
                != expected_identity["intervention_operator"]
            ):
                raise ValueError(
                    f"current endpoint provenance differs: {image_key}, "
                    f"reference={reference!r}, "
                    f"intervention_operator={intervention_operator!r}"
                )
            identity_pair = (reference, intervention_operator)
            if identity_pair not in identity_to_map:
                raise ValueError(f"current endpoint identity is unmapped: {identity_pair}")
            counterfactual_map = identity_to_map[identity_pair]
            if (
                not names
                or len(set(names)) != len(names)
                or any(value.shape != (len(names),) for value in vectors.values())
                or endpoint.shape != (len(names),)
                or not all(np.isfinite(value).all() for value in vectors.values())
                or not np.isfinite(endpoint).all()
                or not bool(image.finite_complete)
                or not bool(image.numeric_audit_passed)
            ):
                raise ValueError(f"current vector contract failed: {image_key}")
            if not np.array_equal(vectors["score"], vectors["E"]):
                raise ValueError("current primary score differs from unsigned E")
            checkpoint = _current_checkpoint(image.checkpoint_assets_json)
            for vector_index, factor in enumerate(names):
                rows.append(
                    {
                        "unit_id": (
                            f"attribution::{dataset}::{model}::{method}::"
                            f"{image_id}::{factor}"
                        ),
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "image_id": image_id,
                        "factor_or_part_id": factor,
                        "vector_index": vector_index,
                        "current_checkpoint_sha256": checkpoint,
                        "current_target": int(image.target_class),
                        "current_counterfactual_map": counterfactual_map,
                        "current_reference": reference,
                        "current_intervention_operator": intervention_operator,
                        "current_endpoint_d": float(endpoint[vector_index]),
                        "current_spearman": float(image.spearman),
                        "current_input_domain": str(image.input_domain),
                        "current_preprocess_inside_forward": bool(
                            image.model_preprocess_inside_forward
                        ),
                        **{
                            f"current_{name}": float(value[vector_index])
                            for name, value in vectors.items()
                        },
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_UNITS or result.duplicated(list(UNIT_KEYS)).any():
        raise ValueError("current part/patch expansion is not one-to-one")
    if len(seen_images) != EXPECTED_IMAGES:
        raise ValueError(
            f"current inventory has {len(seen_images)} image rows, "
            f"expected {EXPECTED_IMAGES}"
        )
    return result


def compare_e2e_frames(
    historical: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare strict vector elements and separately score each image Spearman."""

    comparison = historical.merge(
        current,
        on=["unit_id", *UNIT_KEYS],
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != EXPECTED_UNITS:
        raise ValueError(
            f"current/historical join has {len(comparison)} units, "
            f"expected {EXPECTED_UNITS}"
        )
    comparison["checkpoint_match"] = (
        comparison["current_checkpoint_sha256"]
        == comparison["historical_checkpoint_sha256"]
    )
    comparison["target_match"] = (
        comparison["current_target"].astype(int)
        == comparison["historical_target"].astype(int)
    )
    comparison["counterfactual_map_match"] = (
        comparison["current_counterfactual_map"]
        == comparison["historical_counterfactual_map"]
    )
    comparison["reference_match"] = (
        comparison["current_reference"] == comparison["historical_reference"]
    )
    comparison["intervention_operator_match"] = (
        comparison["current_intervention_operator"]
        == comparison["historical_intervention_operator"]
    )
    comparison["identity_match"] = (
        comparison["checkpoint_match"]
        & comparison["target_match"]
        & comparison["counterfactual_map_match"]
        & comparison["reference_match"]
        & comparison["intervention_operator_match"]
    )
    for name in VECTOR_METRICS:
        comparison[f"abs_error_{name}"] = np.abs(
            comparison[f"current_{name}"] - comparison[f"historical_{name}"]
        )
        comparison[f"signed_error_{name}"] = (
            comparison[f"current_{name}"] - comparison[f"historical_{name}"]
        )

    comparison["historical_gate"] = (
        comparison["historical_M"] >= ENDPOINT_EPSILON
    )
    comparison["current_gate"] = comparison["current_M"] >= ENDPOINT_EPSILON
    comparison["historical_orientation"] = np.where(
        comparison["historical_gate"],
        np.sign(comparison["historical_endpoint_d"]).astype(int),
        0,
    )
    comparison["current_orientation"] = np.where(
        comparison["current_gate"],
        np.sign(comparison["current_endpoint_d"]).astype(int),
        0,
    )
    comparison["boundary"] = (
        np.abs(comparison["historical_M"] - ENDPOINT_EPSILON) <= BOUNDARY_ABS
    )
    comparison["gate_match"] = (
        comparison["current_gate"] == comparison["historical_gate"]
    )
    comparison["orientation_match"] = (
        comparison["current_orientation"] == comparison["historical_orientation"]
    )
    comparison["historical_dominant"] = comparison.apply(
        lambda row: _dominant(
            row["historical_E"],
            row["historical_C"],
            row["historical_F"],
        ),
        axis=1,
    )
    comparison["current_dominant"] = comparison.apply(
        lambda row: _dominant(
            row["current_E"],
            row["current_C"],
            row["current_F"],
        ),
        axis=1,
    )
    comparison["dominant_match"] = (
        comparison["current_dominant"] == comparison["historical_dominant"]
    )
    close = pd.DataFrame(
        {
            name: np.isclose(
                comparison[f"current_{name}"],
                comparison[f"historical_{name}"],
                atol=TIER_A_ATOL,
                rtol=TIER_A_RTOL,
            )
            for name in VECTOR_METRICS
        }
    )
    comparison["tier_a_pass"] = close.all(axis=1)
    maximum_error = comparison[
        [f"abs_error_{name}" for name in VECTOR_METRICS]
    ].max(axis=1)
    comparison["tier_b_pass"] = (
        (maximum_error <= TIER_B_ABS)
        & (
            comparison["boundary"]
            | (comparison["gate_match"] & comparison["orientation_match"])
        )
        & comparison["dominant_match"]
    )
    comparison["tier"] = np.where(
        comparison["tier_a_pass"],
        "A",
        np.where(comparison["tier_b_pass"], "B", "FAIL"),
    )
    comparison["hard_mismatch"] = (
        (maximum_error > HARD_MISMATCH_ABS)
        | (
            ~comparison["boundary"]
            & (~comparison["gate_match"] | ~comparison["orientation_match"])
        )
        | ~comparison["dominant_match"]
        | ~comparison["identity_match"]
    )
    comparison = comparison.sort_values(list(UNIT_KEYS), kind="stable").reset_index(
        drop=True
    )

    spearman_columns = [
        *IMAGE_KEYS,
        "historical_spearman",
        "current_spearman",
    ]
    spearman = comparison.loc[:, spearman_columns].drop_duplicates(
        list(IMAGE_KEYS)
    )
    if len(spearman) != EXPECTED_IMAGES:
        raise ValueError("per-image Spearman inventory is incomplete")
    spearman["signed_error"] = (
        spearman["current_spearman"] - spearman["historical_spearman"]
    )
    spearman["absolute_error"] = np.abs(spearman["signed_error"])
    spearman["tier_a_pass"] = np.isclose(
        spearman["current_spearman"],
        spearman["historical_spearman"],
        atol=TIER_A_ATOL,
        rtol=TIER_A_RTOL,
    )
    spearman["tier_b_pass"] = spearman["absolute_error"] <= TIER_B_ABS
    spearman["tier"] = np.where(
        spearman["tier_a_pass"],
        "A",
        np.where(spearman["tier_b_pass"], "B", "FAIL"),
    )
    spearman["hard_mismatch"] = (
        spearman["absolute_error"] > HARD_MISMATCH_ABS
    )
    spearman = spearman.sort_values(list(IMAGE_KEYS), kind="stable").reset_index(
        drop=True
    )

    non_boundary = comparison.loc[~comparison["boundary"]]
    tier_a = comparison["tier"].eq("A")
    tier_b = comparison["tier"].eq("B")
    spearman_a = spearman["tier"].eq("A")
    spearman_b = spearman["tier"].eq("B")
    summary = {
        "schema_version": 1,
        "experiment_family": "attribution",
        "unit_definition": "one named part/patch vector element",
        "unit_count": int(len(comparison)),
        "image_method_count": int(len(spearman)),
        "models": 6,
        "datasets": list(MODELS),
        "methods": list(METHODS),
        "samples_per_dataset_model": 8,
        "tolerances": {
            "tier_a_atol": TIER_A_ATOL,
            "tier_a_rtol": TIER_A_RTOL,
            "tier_b_absolute": TIER_B_ABS,
            "hard_mismatch_absolute": HARD_MISMATCH_ABS,
            "boundary_absolute": BOUNDARY_ABS,
        },
        "tier_a_fraction": _fraction(tier_a),
        "tier_b_fraction": _fraction(tier_b),
        "tier_a_or_b_fraction": _fraction(tier_a | tier_b),
        "non_boundary_tier_a_or_b_fraction": _fraction(
            non_boundary["tier"].isin(["A", "B"])
        ),
        "hard_mismatch_fraction": _fraction(comparison["hard_mismatch"]),
        "gate_agreement": _fraction(non_boundary["gate_match"]),
        "orientation_agreement": _fraction(non_boundary["orientation_match"]),
        "dominant_mechanism_agreement": _fraction(comparison["dominant_match"]),
        "identity_agreement": _fraction(comparison["identity_match"]),
        "counterfactual_map_agreement": _fraction(
            comparison["counterfactual_map_match"]
        ),
        "reference_agreement": _fraction(comparison["reference_match"]),
        "intervention_operator_agreement": _fraction(
            comparison["intervention_operator_match"]
        ),
        "metrics": {
            name: _stats(comparison[f"abs_error_{name}"])
            for name in VECTOR_METRICS
        },
        "mean_signed_error": {
            name: float(comparison[f"signed_error_{name}"].mean())
            for name in VECTOR_METRICS
        },
        "spearman": {
            **_stats(spearman["absolute_error"]),
            "mean_signed_error": float(spearman["signed_error"].mean()),
            "tier_a_fraction": _fraction(spearman_a),
            "tier_b_fraction": _fraction(spearman_b),
            "tier_a_or_b_fraction": _fraction(spearman_a | spearman_b),
            "hard_mismatch_fraction": _fraction(spearman["hard_mismatch"]),
        },
    }
    return comparison, spearman, summary


def _core_acceptance(summary: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "non_boundary_tier_a_or_b_at_least_95pct": (
            summary["non_boundary_tier_a_or_b_fraction"] >= 0.95
        ),
        "hard_mismatch_at_most_5pct": summary["hard_mismatch_fraction"] <= 0.05,
        "gate_at_least_99pct": summary["gate_agreement"] >= 0.99,
        "orientation_at_least_99pct": summary["orientation_agreement"] >= 0.99,
        "dominant_at_least_95pct": (
            summary["dominant_mechanism_agreement"] >= 0.95
        ),
        "identity_exact": summary["identity_agreement"] == 1.0,
        "numeric_identity_exact": summary["numeric_identity_fraction"] == 1.0,
    }


def _atomic_csv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _run_core(neutral_path: Path, root: Path) -> dict[str, Any]:
    output = root / "comparisons/attribution_core.csv"
    summary = root / "comparisons/attribution_core_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(".attribution_core.part.csv")
    temporary_summary = summary.with_name(".attribution_core_summary.part.json")
    temporary_output.unlink(missing_ok=True)
    temporary_summary.unlink(missing_ok=True)
    try:
        result = compare_record(
            neutral_path,
            temporary_output,
            summary_output=temporary_summary,
        )
        temporary_output.replace(output)
        temporary_summary.replace(summary)
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
    result["comparison_path"] = output
    result["summary_path"] = summary
    return result


def run_aggregate(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Build all requested aggregate artifacts from six completed model slices."""

    neutral, neutral_path = combine_legacy_records(root)
    funny_spearman = _funny_spearman(root, neutral)
    historical = historical_units(neutral, funny_spearman)
    current_frames = [
        pd.read_parquet(_current_path(root, dataset, model_id))
        for dataset, model_ids in MODELS.items()
        for model_id in model_ids
    ]
    current = expand_current(current_frames)
    core = _run_core(neutral_path, root)
    comparison, spearman, summary = compare_e2e_frames(historical, current)

    comparison_path = root / "comparisons/attribution.csv"
    spearman_path = root / "comparisons/attribution_spearman.csv"
    summary_path = root / "comparisons/attribution_summary.json"
    _atomic_csv(comparison, comparison_path)
    _atomic_csv(spearman, spearman_path)
    core_acceptance = _core_acceptance(core["summary"])
    summary["comparison"] = str(comparison_path)
    summary["comparison_sha256"] = sha256_file(comparison_path)
    summary["spearman_comparison"] = str(spearman_path)
    summary["spearman_comparison_sha256"] = sha256_file(spearman_path)
    summary["neutral_trajectory"] = str(neutral_path)
    summary["neutral_trajectory_sha256"] = sha256_file(neutral_path)
    summary["current_core"] = {
        "unit_count": core["summary"]["unit_count"],
        "tier_a_fraction": core["summary"]["tier_a_fraction"],
        "tier_b_fraction": core["summary"]["tier_b_fraction"],
        "tier_a_or_b_fraction": core["summary"]["tier_a_or_b_fraction"],
        "hard_mismatch_fraction": core["summary"]["hard_mismatch_fraction"],
        "gate_agreement": core["summary"]["gate_agreement"],
        "orientation_agreement": core["summary"]["orientation_agreement"],
        "dominant_mechanism_agreement": core["summary"][
            "dominant_mechanism_agreement"
        ],
        "identity_agreement": core["summary"]["identity_agreement"],
        "numeric_identity_fraction": core["summary"]["numeric_identity_fraction"],
        "acceptance": core_acceptance,
    }
    summary["acceptance"] = {
        "non_boundary_tier_a_or_b_at_least_95pct": (
            summary["non_boundary_tier_a_or_b_fraction"] >= 0.95
        ),
        "hard_mismatch_at_most_5pct": summary["hard_mismatch_fraction"] <= 0.05,
        "gate_at_least_99pct": summary["gate_agreement"] >= 0.99,
        "orientation_at_least_99pct": summary["orientation_agreement"] >= 0.99,
        "dominant_at_least_95pct": (
            summary["dominant_mechanism_agreement"] >= 0.95
        ),
        "identity_exact": summary["identity_agreement"] == 1.0,
        "spearman_tier_a_or_b_at_least_95pct": (
            summary["spearman"]["tier_a_or_b_fraction"] >= 0.95
        ),
        "spearman_hard_mismatch_at_most_5pct": (
            summary["spearman"]["hard_mismatch_fraction"] <= 0.05
        ),
        "no_systematic_vector_bias": max(
            abs(value) for value in summary["mean_signed_error"].values()
        )
        <= TIER_B_ABS,
        "no_systematic_spearman_bias": (
            abs(summary["spearman"]["mean_signed_error"]) <= TIER_B_ABS
        ),
    }
    summary["status"] = (
        "PASS_CORE_AND_E2E"
        if all(summary["acceptance"].values()) and all(core_acceptance.values())
        else "FAIL_NUMERICAL"
    )
    atomic_json(summary_path, summary)
    return {
        "trajectory": str(neutral_path),
        "core_comparison": str(core["comparison_path"]),
        "core_summary": str(core["summary_path"]),
        "e2e_comparison": str(comparison_path),
        "spearman_comparison": str(spearman_path),
        "e2e_summary": str(summary_path),
        "status": summary["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(run_aggregate(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
