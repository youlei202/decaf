"""Export one exact eight-image attribution slice through the current B200 runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.core.manifests import sha256_file
from decaf.experiments.attribution.endpoint import row_spearman
from decaf.experiments.attribution.gpu_runtime import evaluate_member
from decaf.experiments.common import RunContext, atomic_json, load_profile
from tools.crossgen.legacy_attribution_export import METHODS

DEFAULT_RUNTIME_RUN = Path(
    "/work/Users/leiyo/decaf_b200_verification/runs/attribution_main"
)
MODELS = {
    "imagenet1k_idsds": ("resnet50", "vgg16", "vit_base_patch16_224"),
    "funnybirds": (
        "funnybirds_resnet50",
        "funnybirds_vgg16",
        "funnybirds_vit_b_16",
    ),
}
PRIMARY_SCOPES = {
    "imagenet1k_idsds": "smoke_idsds_primary",
    "funnybirds": "smoke_funnybirds_primary",
}
TARGET_METHODS = {
    "imagenet1k_idsds": ("__deletion_targets__",),
    "funnybirds": (
        "__part_deletion_targets__",
        "__heldout_background_texture__",
        "__heldout_telea_dilate3__",
    ),
}
ENDPOINT_METHODS = {
    "imagenet1k_idsds": "__deletion_targets__",
    "funnybirds": "__part_deletion_targets__",
}
TARGET_IDENTITIES = {
    "imagenet1k_idsds": {
        "__deletion_targets__": {
            "reference": "normalized_zero",
            "intervention_operator": "endpoint_part_deletion",
        },
    },
    "funnybirds": {
        "__part_deletion_targets__": {
            "reference": "locked_gaussian_blur_k31_sigma12_raw_rgb",
            "intervention_operator": "endpoint_part_deletion",
        },
        "__heldout_background_texture__": {
            "reference": "locked_gaussian_blur_k31_sigma12_raw_rgb",
            "intervention_operator": "background_texture",
        },
        "__heldout_telea_dilate3__": {
            "reference": "locked_gaussian_blur_k31_sigma12_raw_rgb",
            "intervention_operator": "telea_dilate3",
        },
    },
}
SUMMARY_NAMES = ("M", "E", "C", "F", "Abs")


def _vector(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _read_selection(
    path: Path, *, dataset: str, model_id: str
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("dataset") != dataset
        or payload.get("model_id") != model_id
        or payload.get("selection") != "first_eight_in_frozen_candidate_order"
    ):
        raise ValueError("fixed sample selection identity differs")
    image_ids = payload.get("image_ids")
    targets = payload.get("targets")
    if (
        not isinstance(image_ids, list)
        or not isinstance(targets, list)
        or len(image_ids) != 8
        or len(targets) != 8
        or len(set(map(str, image_ids))) != 8
    ):
        raise ValueError("fixed sample selection must contain eight unique IDs/targets")
    return payload


def _load_jobs(runtime_run: Path) -> list[dict[str, Any]]:
    path = runtime_run / "manifests/jobs.jsonl"
    jobs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not jobs or not all(isinstance(job, dict) for job in jobs):
        raise ValueError("current attribution job manifest is empty or malformed")
    return jobs


def _quality_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    model_id: str,
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for method in methods:
        matches = [
            dict(job)
            for job in jobs
            if job.get("dataset") == dataset
            and job.get("model_id") == model_id
            and job.get("scope") == PRIMARY_SCOPES[dataset]
            and job.get("kind") == "quality"
            and job.get("method_id") == method
            and int(job.get("image_start", -1)) == 0
            and int(job.get("image_stop", -1)) == 8
            and int(job.get("image_count", -1)) == 8
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one current quality job for {model_id}/{method}")
        selected.append(matches[0])
    return selected


def _target_jobs(
    jobs: Sequence[Mapping[str, Any]],
    quality_jobs: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> dict[str, dict[str, Any]]:
    by_id = {str(job["member_id"]): dict(job) for job in jobs}
    dependencies = {
        str(dependency["method_id"]): str(dependency["member_id"])
        for job in quality_jobs
        for dependency in job.get("depends_on", [])
        if dependency.get("method_id") in TARGET_METHODS[dataset]
    }
    if set(dependencies) != set(TARGET_METHODS[dataset]):
        raise ValueError("current target dependencies are incomplete")
    missing = sorted(set(dependencies.values()) - set(by_id))
    if missing:
        raise ValueError(f"current target jobs are absent: {missing}")
    return {
        method_id: by_id[member_id]
        for method_id, member_id in dependencies.items()
    }


def _bind_quality(
    frame: pd.DataFrame,
    target_frames: Mapping[str, pd.DataFrame],
    selection: Mapping[str, Any],
    *,
    dataset: str,
) -> pd.DataFrame:
    expected_ids = list(map(str, selection["image_ids"]))
    expected_targets = list(map(int, selection["targets"]))
    if frame["image_id"].astype(str).tolist() != expected_ids:
        raise ValueError("quality selection order differs from the fixed manifest")
    if frame["target_class"].astype(int).tolist() != expected_targets:
        raise ValueError("quality target classes differ from the fixed manifest")
    targets: dict[str, dict[str, np.ndarray]] = {}
    for method_id in TARGET_METHODS[dataset]:
        target_frame = target_frames[method_id]
        if target_frame["image_id"].astype(str).tolist() != expected_ids:
            raise ValueError(f"target selection order differs for {method_id}")
        required_provenance = {"reference", "intervention_operator"}
        missing_provenance = required_provenance - set(target_frame.columns)
        if missing_provenance:
            raise ValueError(
                f"target provenance columns absent for {method_id}: "
                f"{sorted(missing_provenance)}"
            )
        expected_identity = TARGET_IDENTITIES[dataset][method_id]
        for name in sorted(required_provenance):
            observed = target_frame[name].astype(str).tolist()
            expected = [expected_identity[name]] * len(expected_ids)
            if observed != expected:
                raise ValueError(
                    f"target {name} differs for {method_id}: "
                    f"observed={sorted(set(observed))}, "
                    f"expected={expected_identity[name]!r}"
                )
        targets[method_id] = {
            str(row.image_id): _vector(row.target_effects, name=f"{method_id} target")
            for row in target_frame.itertuples(index=False)
        }
    endpoint_method = ENDPOINT_METHODS[dataset]
    endpoint_by_id = targets[endpoint_method]
    endpoint_frame = target_frames[endpoint_method]
    result = frame.copy()
    endpoints: list[np.ndarray] = []
    quality_targets: list[np.ndarray] = []
    quality: list[float] = []
    background: list[np.ndarray] = []
    telea: list[np.ndarray] = []
    background_quality: list[float] = []
    telea_quality: list[float] = []
    for row in result.itertuples(index=False):
        patch = _vector(row.patch_scores, name="patch_scores")
        endpoint = endpoint_by_id[str(row.image_id)]
        if endpoint.shape != patch.shape:
            raise ValueError(f"endpoint shape differs for {row.image_id}")
        endpoints.append(endpoint)
        if dataset == "funnybirds":
            first = targets["__heldout_background_texture__"][str(row.image_id)]
            second = targets["__heldout_telea_dilate3__"][str(row.image_id)]
            if first.shape != patch.shape or second.shape != patch.shape:
                raise ValueError(f"held-out target shape differs for {row.image_id}")
            first_quality = float(row_spearman(patch, first)[0])
            second_quality = float(row_spearman(patch, second)[0])
            background.append(first)
            telea.append(second)
            background_quality.append(first_quality)
            telea_quality.append(second_quality)
            quality_targets.append((first + second) / 2.0)
            quality.append((first_quality + second_quality) / 2.0)
        else:
            quality_targets.append(endpoint.copy())
            quality.append(float(row_spearman(patch, endpoint)[0]))
    result["endpoint_effects"] = endpoints
    result["quality_target_effects"] = quality_targets
    result["decaf_M"] = [np.abs(value) for value in endpoints]
    result["spearman"] = quality
    # These values are copied from the evaluated target frame after exact,
    # dataset-specific validation. They are evidence, not inferred labels.
    result["reference"] = endpoint_frame["reference"].astype(str).tolist()
    result["intervention_operator"] = (
        endpoint_frame["intervention_operator"].astype(str).tolist()
    )
    if dataset == "funnybirds":
        result["heldout_background_texture_effects"] = background
        result["heldout_telea_dilate3_effects"] = telea
        result["spearman_background_texture"] = background_quality
        result["spearman_telea_dilate3"] = telea_quality
        result["quality_aggregation"] = "equal_mean_within_image"
    for row in result.itertuples(index=False):
        patch = _vector(row.patch_scores, name="patch_scores")
        components = {
            name: _vector(getattr(row, f"decaf_{name}"), name=f"decaf_{name}")
            for name in SUMMARY_NAMES
        }
        if any(value.shape != patch.shape for value in components.values()):
            raise ValueError("current component vector shape differs from patch scores")
        if not np.allclose(patch, components["E"], atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("current primary DECAF score is not the unsigned E vector")
        if not np.allclose(
            components["Abs"],
            components["E"] + components["C"] + components["F"],
            atol=5.0e-6,
            rtol=5.0e-6,
        ):
            raise ValueError("current Abs identity failed")
        if not np.allclose(
            components["M"],
            np.abs(_vector(row.endpoint_effects, name="endpoint_effects")),
            atol=1.0e-12,
            rtol=1.0e-12,
        ):
            raise ValueError("current endpoint M identity failed")
    return result


def _atomic_parquet(frame: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.part{output.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def export_current_attribution(
    *,
    dataset: str,
    model_id: str,
    sample_manifest: Path,
    output: Path,
    receipt: Path,
    runtime_run: Path = DEFAULT_RUNTIME_RUN,
    methods: Sequence[str] = METHODS,
) -> dict[str, Any]:
    """Run one model through current public runtime members."""

    selected_methods = tuple(methods)
    if dataset not in MODELS or model_id not in MODELS[dataset]:
        raise ValueError(f"unsupported dataset/model pair: {dataset}/{model_id}")
    if (
        not selected_methods
        or len(set(selected_methods)) != len(selected_methods)
        or any(method not in METHODS for method in selected_methods)
    ):
        raise ValueError(f"methods must be a unique subset of {METHODS}")
    selection = _read_selection(
        sample_manifest,
        dataset=dataset,
        model_id=model_id,
    )
    runtime_selection = (
        runtime_run
        / "manifests/fixed_samples"
        / f"{dataset}--{model_id}.json"
    )
    if sha256_file(runtime_selection) != sha256_file(sample_manifest):
        raise ValueError("explicit selection does not match the current runtime selection")
    config_path = runtime_run / "config.yaml"
    config = load_profile("attribution", "smoke", explicit=config_path)
    context = RunContext(
        experiment="attribution",
        profile="smoke",
        stage="compute",
        path=runtime_run.resolve(),
        config=config,
        workers=1,
        resume=True,
    )
    jobs = _load_jobs(runtime_run)
    quality_jobs = _quality_jobs(
        jobs,
        dataset=dataset,
        model_id=model_id,
        methods=selected_methods,
    )
    target_frames = {
        method_id: evaluate_member(job, context)
        for method_id, job in _target_jobs(
            jobs,
            quality_jobs,
            dataset=dataset,
        ).items()
    }
    frames = [
        _bind_quality(
            evaluate_member(job, context),
            target_frames,
            selection,
            dataset=dataset,
        )
        for job in quality_jobs
    ]
    result = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "model", "method", "image_id"]
    expected_rows = 8 * len(selected_methods)
    if len(result) != expected_rows or result.duplicated(keys).any():
        raise AssertionError(
            f"current attribution produced {len(result)} rows, expected {expected_rows}"
        )
    result = result.sort_values(keys, kind="stable").reset_index(drop=True)
    if set(result["runtime_cuda_matmul_allow_tf32"].astype(bool)) != {False}:
        raise ValueError("current CUDA matmul TF32 contract differs")
    if set(result["runtime_cudnn_allow_tf32"].astype(bool)) != {False}:
        raise ValueError("current cuDNN TF32 contract differs")
    _atomic_parquet(result, output)
    checkpoint_assets = {
        str(value) for value in result["checkpoint_assets_json"].astype(str)
    }
    if len(checkpoint_assets) != 1:
        raise ValueError("current checkpoint assets changed within one model export")
    references = set(result["reference"].astype(str))
    intervention_operators = set(result["intervention_operator"].astype(str))
    if len(references) != 1 or len(intervention_operators) != 1:
        raise ValueError("current endpoint provenance changed within one model export")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "experiment_family": "attribution",
        "executor": "decaf.experiments.attribution.gpu_runtime.evaluate_member",
        "dataset": dataset,
        "model_id": model_id,
        "methods": list(selected_methods),
        "sample_count": 8,
        "image_ids": list(map(str, selection["image_ids"])),
        "targets": list(map(int, selection["targets"])),
        "fixed_sample_manifest": str(sample_manifest.resolve()),
        "fixed_sample_manifest_sha256": sha256_file(sample_manifest),
        "runtime_run": str(runtime_run.resolve()),
        "runtime_config_sha256": sha256_file(config_path),
        "runtime_jobs_sha256": sha256_file(runtime_run / "manifests/jobs.jsonl"),
        "repository_commit": commit,
        "checkpoint_assets_json": next(iter(checkpoint_assets)),
        "reference": next(iter(references)),
        "intervention_operator": next(iter(intervention_operators)),
        "runtime_cuda_matmul_allow_tf32": False,
        "runtime_cudnn_allow_tf32": False,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "rows": int(len(result)),
    }
    atomic_json(receipt, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(MODELS),
        default="imagenet1k_idsds",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--runtime-run", type=Path, default=DEFAULT_RUNTIME_RUN)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = export_current_attribution(
        dataset=args.dataset,
        model_id=args.model,
        sample_manifest=args.sample_manifest,
        output=args.output,
        receipt=args.receipt,
        runtime_run=args.runtime_run,
        methods=args.methods,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
