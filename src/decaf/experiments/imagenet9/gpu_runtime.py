"""Real single-B200 ImageNet-9 smoke execution and fingerprint collection.

This module is dormant unless ``DECAF_B200_VERIFY=1``.  All data, mapping,
cache, and checkpoint paths are selected through named environment variables;
there is no network-enabled fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.core.manifests import read_json
from decaf.core.receipts import finalize_global_receipt, load_member_receipt, write_member_receipt
from decaf.experiments.common import RunContext, atomic_json, atomic_text, load_profile
from decaf.experiments.imagenet9.baselines import baseline_plan
from decaf.experiments.imagenet9.gpu_methods import paired_saliency_scores, reveal_sequence
from decaf.experiments.imagenet9.gpu_models import (
    MAPPING_SHA256,
    PAIRED_MANIFEST_SHA256,
    B200Assets,
    canonical_tensor_identity,
    load_model,
    load_official_mapping,
    preprocess_paths,
    probability_model,
    resolve_b200_assets,
)
from decaf.experiments.imagenet9.pairs import normalize_wide_manifest
from decaf.paper.reference import sha256_file

B200_GATE_ENV = "DECAF_B200_VERIFY"
REQUIRED_PAIR_TYPES = ("same_rand", "same_next")
REQUIRED_REVEAL_PATHS = ("blend", "patch_A", "patch_B")
REQUIRED_METHODS = (
    "input_x_gradient",
    "integrated_gradients",
    "smoothgrad",
    "occlusion",
    "rise",
)


def b200_enabled(
    config: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the explicit real-GPU gate is exactly enabled."""

    env = os.environ if environment is None else environment
    variable = B200_GATE_ENV
    if config is not None:
        smoke = config.get("b200_smoke")
        if not isinstance(smoke, Mapping):
            return False
        variable = str(smoke.get("activation_environment", variable))
    return env.get(variable) == "1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("b200_smoke")
    if not isinstance(raw, Mapping):
        raise KeyError("ImageNet-9 smoke config has no b200_smoke settings")
    settings = dict(raw)
    if int(settings.get("source_pairs", -1)) != 16:
        raise ValueError("ImageNet-9 B200 smoke must use exactly 16 fixed source pairs/model")
    if tuple(map(str, settings.get("pair_types", ()))) != REQUIRED_PAIR_TYPES:
        raise ValueError("ImageNet-9 B200 smoke must cover Same-Rand and Same-Next")
    if tuple(map(str, settings.get("reveal_paths", ()))) != REQUIRED_REVEAL_PATHS:
        raise ValueError("ImageNet-9 B200 smoke must use blend and nested patch orders A/B")
    if tuple(map(str, settings.get("methods", ()))) != REQUIRED_METHODS:
        raise ValueError("ImageNet-9 B200 smoke must retain all five attribution baselines")
    alpha = tuple(map(float, settings.get("alpha", ())))
    if alpha != tuple(float(value) for value in np.linspace(0.0, 1.0, 9)):
        raise ValueError("ImageNet-9 B200 smoke must use the registered nine-stage alpha grid")
    if int(settings.get("integrated_gradients_steps", -1)) != 16:
        raise ValueError("IG must use exactly 16 steps")
    if int(settings.get("smoothgrad_samples", -1)) != 16:
        raise ValueError("SmoothGrad must use exactly 16 samples")
    if np.prod(tuple(map(int, settings.get("occlusion_grid", ())))) != 49:
        raise ValueError("Occlusion must use 49 cells")
    if int(settings.get("rise_masks", -1)) != 256:
        raise ValueError("RISE must use exactly 256 masks")
    baseline_plan(list(REQUIRED_METHODS))
    return settings


def _selected_source_pairs(assets: B200Assets, settings: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(assets.pair_manifest)
    required = {
        "pair_id",
        "true_in9_class",
        "mixed_same_path",
        "mixed_rand_path",
        "mixed_next_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"real paired manifest is missing columns: {missing}")
    candidates = frame.copy()
    if "split" in candidates and (candidates["split"].astype(str) == "deep_split").sum() >= 16:
        candidates = candidates[candidates["split"].astype(str) == "deep_split"].copy()
    seed = int(settings["selection_seed"])
    candidates["_selection_key"] = candidates["pair_id"].astype(str).map(
        lambda pair_id: hashlib.sha256(f"{seed}|{pair_id}".encode()).hexdigest()
    )
    candidates = candidates.sort_values(
        ["true_in9_class", "_selection_key", "pair_id"], kind="stable"
    )
    by_class = {
        class_id: group.to_dict("records")
        for class_id, group in candidates.groupby("true_in9_class", sort=True)
    }
    if set(map(int, by_class)) != set(range(9)):
        raise ValueError("fixed B200 support must cover all nine ImageNet-9 classes")
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < int(settings["source_pairs"]):
        progress = False
        for class_id in range(9):
            rows = by_class[class_id]
            if depth < len(rows):
                selected.append(rows[depth])
                progress = True
                if len(selected) == int(settings["source_pairs"]):
                    break
        if not progress:
            raise RuntimeError("paired manifest cannot supply 16 stratified fixed source pairs")
        depth += 1
    source = pd.DataFrame(selected).drop(columns=["_selection_key"], errors="ignore")
    if len(source) != 16 or source["pair_id"].astype(str).duplicated().any():
        raise AssertionError("fixed source-pair selection is not exactly 16 unique rows")
    return source.reset_index(drop=True)


def selected_pair_frame(assets: B200Assets, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Return 16 fixed source rows expanded to 32 typed Same-Rand/Next pairs."""

    source = _selected_source_pairs(assets, settings)
    expanded = normalize_wide_manifest(source, dataset_root=assets.dataset_root, expected_rows=16)
    if len(expanded) != 32 or expanded["source_pair_id"].nunique() != 16:
        raise AssertionError("B200 source support did not expand to 32 typed paired rows")
    counts = expanded["pair_type"].value_counts().to_dict()
    if counts != {"same_rand": 16, "same_next": 16}:
        raise AssertionError(f"B200 pair-type support differs: {counts}")
    for column in ("original_path", "counterfactual_path"):
        for relative in expanded[column].astype(str):
            path = (assets.dataset_root / relative).resolve(strict=True)
            try:
                path.relative_to(assets.dataset_root)
            except ValueError as error:
                raise ValueError(f"prepared image escapes dataset root: {relative}") from error
            if not path.is_file():
                raise FileNotFoundError(path)
    return expanded


def _support_sha256(pairs: pd.DataFrame) -> str:
    columns = (
        "pair_id",
        "source_pair_id",
        "source_row_index",
        "pair_type",
        "original_path",
        "counterfactual_path",
        "class_id",
    )
    return _canonical_sha256(pairs.loc[:, columns].to_dict("records"))


def _member_spec_sha256(member: Mapping[str, Any]) -> str:
    return _canonical_sha256(member)


def build_b200_plan(
    config: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the exact real-data, real-checkpoint single-B200 member plan."""

    settings = _settings(config)
    assets = resolve_b200_assets(config, environment)
    pairs = selected_pair_frame(assets, settings)
    config_sha256 = _canonical_sha256(
        {key: value for key, value in config.items() if key != "_source"}
    )
    support_sha256 = _support_sha256(pairs)
    members: list[dict[str, Any]] = []
    for asset in assets.models:
        common = {
            "resource": "single_cuda",
            "required": True,
            "model_id": asset.model_id,
            "architecture": asset.architecture,
            "output_classes": asset.output_classes,
            "configuration_sha256": config_sha256,
            "paired_support_sha256": support_sha256,
            "dataset_manifest_sha256": PAIRED_MANIFEST_SHA256,
            "mapping_sha256": MAPPING_SHA256,
            "checkpoint_sha256": asset.checkpoint_sha256,
        }
        for path in REQUIRED_REVEAL_PATHS:
            member_id = f"scan__{asset.model_id}__{path}"
            members.append(
                {
                    "member_id": member_id,
                    "phase": "decaf_scan",
                    "reveal_path": path,
                    "output": f"raw/scans/{member_id}.parquet",
                    "receipt": f"receipts/members/{member_id}.json",
                    **common,
                }
            )
        for method in REQUIRED_METHODS:
            member_id = f"baseline__{asset.model_id}__{method}"
            members.append(
                {
                    "member_id": member_id,
                    "phase": "saliency_baseline",
                    "method_id": method,
                    "output": f"raw/baselines/{member_id}.parquet",
                    "receipt": f"receipts/members/{member_id}.json",
                    **common,
                }
            )
    identifiers = [str(member["member_id"]) for member in members]
    outputs = [str(member["output"]) for member in members]
    receipts = [str(member["receipt"]) for member in members]
    if (
        len(members) != 24
        or len(identifiers) != len(set(identifiers))
        or len(outputs) != len(set(outputs))
        or len(receipts) != len(set(receipts))
    ):
        raise AssertionError("ImageNet-9 B200 member universe is duplicated or incomplete")
    return {
        "schema_version": 2,
        "experiment": "imagenet9",
        "profile": "smoke",
        "verification_mode": "single_b200_real_cuda",
        "configuration_sha256": config_sha256,
        "paired_support_sha256": support_sha256,
        "execution_class": "real_cuda",
        "gpu_execution_verified": False,
        "counts": {
            "models": 3,
            "source_pairs_per_model": 16,
            "expanded_pairs_per_model": 32,
            "same_rand_pairs_per_model": 16,
            "same_next_pairs_per_model": 16,
            "reveal_paths": 3,
            "decaf_methods": 1,
            "attribution_methods": 5,
            "methods_total": 6,
            "scan_members": 9,
            "baseline_members": 15,
            "members": 24,
        },
        "scientific_contract": {
            "official_probability_mapping": "softmax_1000_once_then_sum_mapped_mass",
            "mapped_mass_renormalized": False,
            "second_softmax": False,
            "fine_tuned_probability_adapter": "direct_9_way_softmax_once",
            "variable_size_preprocessing": "per_image_before_batch_coalescing",
            "paired_randomness": "shared_within_each_factual_counterfactual_pair",
            "reveal_paths": list(REQUIRED_REVEAL_PATHS),
            "attribution_methods": baseline_plan(list(REQUIRED_METHODS)),
        },
        "dataset": {
            "manifest_sha256": PAIRED_MANIFEST_SHA256,
            "mapping_sha256": MAPPING_SHA256,
            "support_sha256": support_sha256,
        },
        "models": [asset.record() for asset in assets.models],
        "members": members,
    }


def _prepared_manifests(
    config: Mapping[str, Any], assets: B200Assets, pairs: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = _settings(config)
    data = {
        "schema_version": 2,
        "status": "verified_external_real_images",
        "verification_mode": "single_b200_real_cuda",
        "dataset_root_environment": config["data"]["root_environment"],
        "dataset_root": str(assets.dataset_root),
        "paired_manifest": str(assets.pair_manifest),
        "paired_manifest_sha256": PAIRED_MANIFEST_SHA256,
        "mapping": str(assets.mapping),
        "mapping_sha256": MAPPING_SHA256,
        "fixed_source_pairs": 16,
        "expanded_pairs": 32,
        "pair_type_counts": {"same_next": 16, "same_rand": 16},
        "paired_support_sha256": _support_sha256(pairs),
        "input_size": int(settings["input_size"]),
        "preprocessing": "resize_shorter_side_256_then_center_crop_224_per_image",
        "restricted_images_copied": False,
    }
    checkpoints = {
        "schema_version": 2,
        "status": "verified_external_offline_checkpoints",
        "verification_mode": "single_b200_real_cuda",
        "network_fallback": False,
        "weight_cache_root": str(assets.weight_cache_root),
        "checkpoint_root": str(assets.checkpoint_root),
        "items": [asset.record() for asset in assets.models],
    }
    return data, checkpoints


def prepare_b200(context: RunContext) -> dict[str, Any]:
    """Validate and bind the fixed real support and all three checkpoints."""

    settings = _settings(context.config)
    assets = resolve_b200_assets(context.config)
    pairs = selected_pair_frame(assets, settings)
    plan = build_b200_plan(context.config)
    data, checkpoints = _prepared_manifests(context.config, assets, pairs)
    atomic_text(context.path / "manifests" / "pairs.csv", pairs.to_csv(index=False))
    atomic_json(context.path / "manifests" / "data.json", data)
    atomic_json(context.path / "manifests" / "checkpoints.json", checkpoints)
    atomic_json(context.path / "manifests" / "plan.json", plan)
    jobs_text = "".join(
        json.dumps(member, sort_keys=True, separators=(",", ":")) + "\n"
        for member in plan["members"]
    )
    atomic_text(context.path / "manifests" / "jobs.jsonl", jobs_text)
    return {
        "models": 3,
        "fixed_source_pairs_per_model": 16,
        "expanded_pairs_per_model": 32,
        "planned_members": 24,
        "source": "real_backgrounds_challenge",
        "gpu_inference_executed": False,
    }


def _prepared_run_bindings(context: RunContext, plan: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "config_file_sha256": context.path / "config.yaml",
        "plan_manifest_sha256": context.path / "manifests" / "plan.json",
        "jobs_manifest_sha256": context.path / "manifests" / "jobs.jsonl",
        "data_manifest_sha256": context.path / "manifests" / "data.json",
        "checkpoint_manifest_sha256": context.path / "manifests" / "checkpoints.json",
        "pair_manifest_sha256": context.path / "manifests" / "pairs.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"prepared ImageNet-9 B200 bindings are missing: {missing}")
    persisted_plan = read_json(paths["plan_manifest_sha256"])
    if persisted_plan != dict(plan):
        raise ValueError("prepared ImageNet-9 B200 plan differs from the active asset plan")
    jobs = [
        json.loads(line)
        for line in paths["jobs_manifest_sha256"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if jobs != plan["members"]:
        raise ValueError("prepared ImageNet-9 B200 jobs differ from the plan")
    return {name: sha256_file(path) for name, path in paths.items()}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.part{path.suffix}")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _require_single_b200(device: str = "cuda:0") -> tuple[Any, str]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("ImageNet-9 real smoke requires persistent GPU PyTorch") from error
    if str(device) != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "ImageNet-9 verification requires exactly one visible CUDA device cuda:0"
        )
    selected = torch.device(device)
    name = str(torch.cuda.get_device_name(selected))
    if "B200" not in name:
        raise RuntimeError(f"ImageNet-9 verification requires an NVIDIA B200, received {name}")
    return torch, name


def _artifact_record(context: RunContext, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    relative = resolved.relative_to(context.path)
    return {
        "path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _receipt_reusable(
    context: RunContext,
    member: Mapping[str, Any],
    *,
    run_bindings: Mapping[str, str],
) -> bool:
    path = context.path / str(member["receipt"])
    if not path.is_file():
        return False
    try:
        receipt = load_member_receipt(path)
    except (OSError, TypeError, ValueError):
        return False
    details = receipt.get("details", {})
    if (
        receipt.get("status") != "completed"
        or details.get("member_spec_sha256") != _member_spec_sha256(member)
        or details.get("run_bindings") != dict(run_bindings)
        or details.get("checkpoint_sha256") != member["checkpoint_sha256"]
    ):
        return False
    artifacts = details.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        return False
    artifact = artifacts[0]
    output = context.path / str(artifact.get("path", ""))
    return (
        output.is_file()
        and output.stat().st_size == int(artifact.get("bytes", -1))
        and sha256_file(output) == artifact.get("sha256")
        and artifact.get("path") == member["output"]
    )


def _pair_batches(pairs: pd.DataFrame, batch_size: int) -> Sequence[pd.DataFrame]:
    return tuple(
        pairs.iloc[start : start + batch_size].reset_index(drop=True)
        for start in range(0, len(pairs), batch_size)
    )


def _preprocessed_endpoints(
    batch: pd.DataFrame, *, assets: B200Assets, settings: Mapping[str, Any], torch: Any
) -> tuple[Any, Any, Any]:
    size = int(settings["input_size"])
    plus = preprocess_paths(
        batch["original_path"].astype(str).tolist(), dataset_root=assets.dataset_root, size=size
    )
    minus = preprocess_paths(
        batch["counterfactual_path"].astype(str).tolist(),
        dataset_root=assets.dataset_root,
        size=size,
    )
    targets = batch["class_id"].to_numpy(dtype=np.int64)
    return (
        torch.from_numpy(plus).to(device="cuda:0", dtype=torch.float32),
        torch.from_numpy(minus).to(device="cuda:0", dtype=torch.float32),
        torch.from_numpy(targets).to(device="cuda:0", dtype=torch.long),
    )


def _paired_stage_probabilities(
    model: Any,
    plus_stages: Any,
    minus_stages: Any,
    *,
    inference_batch_size: int,
    torch: Any,
) -> tuple[Any, Any]:
    """Evaluate the single shared midpoint once and broadcast its probabilities.

    CUDA convolution kernels need not return bit-identical values for duplicate
    rows occupying different batch lanes.  Alpha zero is one shared scientific
    state, not two approximately equal states, so it has exactly one forward
    evaluation.  All revealed stages continue to evaluate both branches.
    """

    if plus_stages.shape != minus_stages.shape or plus_stages.ndim != 5:
        raise ValueError("paired reveal stages must be equal-shape SBCHW tensors")
    stages, pairs = map(int, plus_stages.shape[:2])
    if stages < 2 or pairs < 1 or inference_batch_size < 1:
        raise ValueError("paired stage evaluation requires stages, pairs, and batch capacity")
    if not torch.equal(plus_stages[0], minus_stages[0]):
        raise AssertionError("alpha-zero reveal tensors must be one exact shared midpoint")

    shared_inputs = plus_stages[0]
    revealed_inputs = torch.cat((plus_stages[1:], minus_stages[1:]), dim=1).reshape(
        -1, *plus_stages.shape[2:]
    )
    inputs = torch.cat((shared_inputs, revealed_inputs), dim=0)
    probability_parts = []
    with torch.inference_mode():
        for chunk in inputs.split(int(inference_batch_size)):
            probability_parts.append(model(chunk))
    probabilities = torch.cat(probability_parts, dim=0)
    if probabilities.ndim != 2 or tuple(probabilities.shape) != (
        pairs + (stages - 1) * 2 * pairs,
        9,
    ):
        raise ValueError("probability model returned the wrong paired-stage shape")
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("paired-stage probabilities contain NaN/Inf")
    shared_probabilities = probabilities[:pairs]
    revealed_probabilities = probabilities[pairs:].reshape(stages - 1, 2 * pairs, 9)
    plus_probabilities = torch.cat(
        (shared_probabilities.unsqueeze(0), revealed_probabilities[:, :pairs]), dim=0
    )
    minus_probabilities = torch.cat(
        (shared_probabilities.unsqueeze(0), revealed_probabilities[:, pairs:]), dim=0
    )
    if not torch.equal(plus_probabilities[0], minus_probabilities[0]):
        raise AssertionError("shared-midpoint probability broadcast lost exact identity")
    return plus_probabilities, minus_probabilities


def _scan_frame(
    member: Mapping[str, Any],
    pairs: pd.DataFrame,
    *,
    assets: B200Assets,
    settings: Mapping[str, Any],
    model: Any,
    torch: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positions = tuple(map(float, settings["alpha"]))
    pair_batch = max(1, int(settings["attribution_pair_batch_size"]))
    inference_batch = int(settings["inference_batch_size"])
    for batch in _pair_batches(pairs, pair_batch):
        plus, minus, targets = _preprocessed_endpoints(
            batch, assets=assets, settings=settings, torch=torch
        )
        plus_stages, minus_stages = reveal_sequence(
            plus,
            minus,
            pair_ids=batch["pair_id"].astype(str).tolist(),
            path=str(member["reveal_path"]),
            alpha=positions,
            blur_sigma=float(settings["blur_sigma"]),
            patch_grid=tuple(map(int, settings["patch_grid"])),
            patch_seed=int(settings["patch_seed"]),
        )
        plus_probabilities, minus_probabilities = _paired_stage_probabilities(
            model,
            plus_stages,
            minus_stages,
            inference_batch_size=inference_batch,
            torch=torch,
        )
        target_index = targets[None, :, None].expand(len(positions), -1, -1)
        plus_scores = plus_probabilities.gather(2, target_index).squeeze(2)
        minus_scores = minus_probabilities.gather(2, target_index).squeeze(2)
        responses = (plus_scores - minus_scores).detach().double().cpu().numpy()
        if not np.isfinite(responses).all() or not np.array_equal(
            responses[0], np.zeros_like(responses[0])
        ):
            raise FloatingPointError("shared-midpoint scan is non-finite or does not start at zero")
        for pair_index, pair in enumerate(batch.to_dict("records")):
            for stage_index, position in enumerate(positions):
                rows.append(
                    {
                        "pair_id": str(pair["pair_id"]),
                        "pair_type": str(pair["pair_type"]),
                        "model_id": str(member["model_id"]),
                        "reveal_path": str(member["reveal_path"]),
                        "stage_index": stage_index,
                        "alpha": position,
                        "response": float(responses[stage_index, pair_index]),
                    }
                )
        del plus, minus, targets, plus_stages, minus_stages
        del plus_probabilities, minus_probabilities
    frame = pd.DataFrame(rows)
    expected = len(pairs) * len(positions)
    if len(frame) != expected:
        raise AssertionError(f"scan member produced {len(frame)} rows, expected {expected}")
    return frame


def _baseline_frame(
    member: Mapping[str, Any],
    pairs: pd.DataFrame,
    *,
    assets: B200Assets,
    settings: Mapping[str, Any],
    model: Any,
    torch: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pair_batch = max(1, int(settings["attribution_pair_batch_size"]))
    for batch in _pair_batches(pairs, pair_batch):
        plus, minus, targets = _preprocessed_endpoints(
            batch, assets=assets, settings=settings, torch=torch
        )
        values = paired_saliency_scores(
            str(member["method_id"]),
            model,
            plus,
            minus,
            targets,
            pair_ids=batch["pair_id"].astype(str).tolist(),
            settings=settings,
        )
        for pair, value in zip(batch.to_dict("records"), values, strict=True):
            rows.append(
                {
                    "pair_id": str(pair["pair_id"]),
                    "pair_type": str(pair["pair_type"]),
                    "model_id": str(member["model_id"]),
                    "method_id": str(member["method_id"]),
                    "score": float(value),
                }
            )
        del plus, minus, targets
    frame = pd.DataFrame(rows)
    if len(frame) != len(pairs) or not np.isfinite(frame["score"].to_numpy()).all():
        raise FloatingPointError("baseline member did not produce one finite score per pair")
    return frame


def _write_terminal_member(
    context: RunContext,
    member: Mapping[str, Any],
    *,
    status: str,
    run_bindings: Mapping[str, str],
    artifacts: Sequence[dict[str, Any]] = (),
    error: str | None = None,
) -> None:
    write_member_receipt(
        context.path / str(member["receipt"]),
        str(member["member_id"]),
        status,
        details={
            "phase": member["phase"],
            "output": member["output"],
            "member_spec_sha256": _member_spec_sha256(member),
            "run_bindings": dict(run_bindings),
            "checkpoint_sha256": member["checkpoint_sha256"],
            "dataset_manifest_sha256": PAIRED_MANIFEST_SHA256,
            "mapping_sha256": MAPPING_SHA256,
            "gpu_inference_executed": status == "completed",
            "artifacts": list(artifacts),
        },
        error=error,
    )


def _aggregate_outputs(context: RunContext, plan: Mapping[str, Any]) -> dict[str, Any]:
    from decaf.experiments.imagenet9.evaluate import (
        evaluate_response_frame,
        validate_response_frame,
    )

    scan_frames = [
        pd.read_parquet(context.path / str(member["output"]))
        for member in plan["members"]
        if member["phase"] == "decaf_scan"
    ]
    baseline_frames = [
        pd.read_parquet(context.path / str(member["output"]))
        for member in plan["members"]
        if member["phase"] == "saliency_baseline"
    ]
    responses = validate_response_frame(pd.concat(scan_frames, ignore_index=True))
    baselines = pd.concat(baseline_frames, ignore_index=True)
    expected_baseline_columns = {"pair_id", "pair_type", "model_id", "method_id", "score"}
    if not expected_baseline_columns <= set(baselines) or len(baselines) != 3 * 5 * 32:
        raise ValueError("aggregated B200 baseline schema or row count differs")
    scores = evaluate_response_frame(
        responses,
        epsilon=float(_settings(context.config)["epsilon"]),
    )
    if len(scores) != 3 * 3 * 32:
        raise ValueError("aggregated B200 DECAF score row count differs")
    atomic_text(context.path / "raw" / "response_paths.csv", responses.to_csv(index=False))
    atomic_text(context.path / "metrics" / "decaf_scores.csv", scores.to_csv(index=False))
    atomic_text(context.path / "metrics" / "baseline_scores.csv", baselines.to_csv(index=False))
    artifacts = {
        relative: sha256_file(context.path / relative)
        for relative in (
            "raw/response_paths.csv",
            "metrics/decaf_scores.csv",
            "metrics/baseline_scores.csv",
        )
    }
    atomic_json(
        context.path / "receipts" / "imagenet9_b200_compute.json",
        {
            "schema_version": 2,
            "status": "completed",
            "verification_mode": "single_b200_real_cuda",
            "gpu_inference_verified": True,
            "artifacts": artifacts,
        },
    )
    return {
        "trajectory_count": len(scores),
        "baseline_rows": len(baselines),
        "aggregate_artifacts": artifacts,
    }


def compute_b200(context: RunContext) -> dict[str, Any]:
    """Run all 24 members sequentially on one real B200 with hash resume."""

    torch, device_name = _require_single_b200()
    plan = build_b200_plan(context.config)
    validate_b200_prepare_resume(context)
    bindings = _prepared_run_bindings(context, plan)
    settings = _settings(context.config)
    assets = resolve_b200_assets(context.config)
    pairs = selected_pair_frame(assets, settings)
    mapping = load_official_mapping(assets.mapping, torch)
    receipts: dict[str, Mapping[str, Any]] = {}
    reused = 0
    completed = 0
    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    try:
        for asset in assets.models:
            members = [member for member in plan["members"] if member["model_id"] == asset.model_id]
            pending = [
                member
                for member in members
                if not (
                    context.resume
                    and _receipt_reusable(context, member, run_bindings=bindings)
                )
            ]
            if not pending:
                for member in members:
                    receipts[str(member["member_id"])] = load_member_receipt(
                        context.path / str(member["receipt"])
                    )
                    reused += 1
                continue
            base_model, load_record = load_model(asset, device="cuda:0")
            model = probability_model(
                base_model,
                mapping_matrix=mapping,
                output_classes=asset.output_classes,
            ).to(device=torch.device("cuda:0"))
            try:
                for member in members:
                    member_id = str(member["member_id"])
                    receipt_path = context.path / str(member["receipt"])
                    if context.resume and _receipt_reusable(
                        context, member, run_bindings=bindings
                    ):
                        receipts[member_id] = load_member_receipt(receipt_path)
                        reused += 1
                        continue
                    _write_terminal_member(
                        context,
                        member,
                        status="running",
                        run_bindings=bindings,
                    )
                    try:
                        if member["phase"] == "decaf_scan":
                            frame = _scan_frame(
                                member,
                                pairs,
                                assets=assets,
                                settings=settings,
                                model=model,
                                torch=torch,
                            )
                        else:
                            frame = _baseline_frame(
                                member,
                                pairs,
                                assets=assets,
                                settings=settings,
                                model=model,
                                torch=torch,
                            )
                        output = _atomic_parquet(context.path / str(member["output"]), frame)
                        artifact = _artifact_record(context, output)
                        _write_terminal_member(
                            context,
                            member,
                            status="completed",
                            run_bindings=bindings,
                            artifacts=(artifact,),
                        )
                        completed += 1
                    except Exception as error:
                        _write_terminal_member(
                            context,
                            member,
                            status="failed",
                            run_bindings=bindings,
                            error=f"{type(error).__name__}: {error}",
                        )
                        raise
                    receipts[member_id] = load_member_receipt(receipt_path)
            finally:
                del model, base_model
                torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda:0"))
    except BaseException:
        for member in plan["members"]:
            path = context.path / str(member["receipt"])
            if path.is_file() and str(member["member_id"]) not in receipts:
                try:
                    receipts[str(member["member_id"])] = load_member_receipt(path)
                except (OSError, TypeError, ValueError):
                    pass
        finalize_global_receipt(
            context.path / "receipts" / "compute_members.json",
            context.path.name,
            receipts,
            expected_members=[str(member["member_id"]) for member in plan["members"]],
            details={"run_bindings": bindings, "verification_mode": "single_b200_real_cuda"},
        )
        raise
    finalize_global_receipt(
        context.path / "receipts" / "compute_members.json",
        context.path.name,
        receipts,
        expected_members=[str(member["member_id"]) for member in plan["members"]],
        details={
            "run_bindings": bindings,
            "verification_mode": "single_b200_real_cuda",
            "completed": completed,
            "reused": reused,
        },
    )
    aggregate = _aggregate_outputs(context, plan)
    return {
        **aggregate,
        "completed_member_receipts": len(receipts),
        "members_executed": completed,
        "members_reused": reused,
        "source": "real_backgrounds_challenge_cuda",
        "gpu_inference_executed": True,
        "gpu_name": device_name,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(torch.device("cuda:0"))),
    }


def validate_b200_prepare_resume(context: RunContext) -> dict[str, Any]:
    """Rehash every prepared input before a completed prepare stage is skipped."""

    settings = _settings(context.config)
    assets = resolve_b200_assets(context.config)
    expected_pairs = selected_pair_frame(assets, settings)
    expected_plan = build_b200_plan(context.config)
    expected_data, expected_checkpoints = _prepared_manifests(
        context.config, assets, expected_pairs
    )
    paths = {
        "plan": context.path / "manifests" / "plan.json",
        "data": context.path / "manifests" / "data.json",
        "checkpoints": context.path / "manifests" / "checkpoints.json",
        "pairs": context.path / "manifests" / "pairs.csv",
        "jobs": context.path / "manifests" / "jobs.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"prepared ImageNet-9 B200 files are missing: {missing}")
    if read_json(paths["plan"]) != expected_plan:
        raise ValueError("prepared ImageNet-9 B200 plan changed before resume")
    if (
        read_json(paths["data"]) != expected_data
        or read_json(paths["checkpoints"]) != expected_checkpoints
    ):
        raise ValueError("prepared ImageNet-9 B200 asset manifests changed before resume")
    observed_pairs = pd.read_csv(paths["pairs"])
    if _support_sha256(observed_pairs) != _support_sha256(expected_pairs):
        raise ValueError("prepared ImageNet-9 B200 fixed pair support changed before resume")
    jobs = [json.loads(line) for line in paths["jobs"].read_text(encoding="utf-8").splitlines()]
    if jobs != expected_plan["members"]:
        raise ValueError("prepared ImageNet-9 B200 jobs changed before resume")
    return {"prepared_assets_rehashed": True, "fixed_source_pairs": 16}


def validate_b200_compute_resume(context: RunContext) -> dict[str, Any]:
    """Require all member and aggregate hashes before skipping compute."""

    validate_b200_prepare_resume(context)
    plan = build_b200_plan(context.config)
    bindings = _prepared_run_bindings(context, plan)
    invalid = [
        str(member["member_id"])
        for member in plan["members"]
        if not _receipt_reusable(context, member, run_bindings=bindings)
    ]
    if invalid:
        raise ValueError(f"ImageNet-9 B200 member receipts are not reusable: {invalid}")
    global_receipt = read_json(context.path / "receipts" / "compute_members.json")
    if (
        global_receipt.get("status") != "completed"
        or not global_receipt.get("all_processes_exited")
        or set(global_receipt.get("members", {}))
        != {str(member["member_id"]) for member in plan["members"]}
        or global_receipt.get("details", {}).get("run_bindings") != bindings
    ):
        raise ValueError("ImageNet-9 B200 global member receipt is not reusable")
    aggregate = read_json(context.path / "receipts" / "imagenet9_b200_compute.json")
    if aggregate.get("status") != "completed" or not aggregate.get("gpu_inference_verified"):
        raise ValueError("ImageNet-9 B200 aggregate compute receipt is incomplete")
    for relative, digest in aggregate.get("artifacts", {}).items():
        path = context.path / str(relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"ImageNet-9 B200 aggregate artifact changed: {relative}")
    return {"member_hashes_validated": 24, "aggregate_hashes_validated": 3}


def write_downstream_receipt(
    context: RunContext, stage: str, relative_paths: Sequence[str]
) -> None:
    """Bind analyze/paper artifacts so full-stage resume does not trust timestamps."""

    artifacts = {}
    for relative in relative_paths:
        path = context.path / relative
        if not path.is_file():
            raise FileNotFoundError(f"ImageNet-9 B200 {stage} artifact is missing: {relative}")
        artifacts[relative] = sha256_file(path)
    atomic_json(
        context.path / "receipts" / f"imagenet9_b200_{stage}.json",
        {
            "schema_version": 2,
            "status": "completed",
            "stage": stage,
            "artifacts": artifacts,
        },
    )


def validate_downstream_resume(context: RunContext, stage: str) -> dict[str, Any]:
    receipt = read_json(context.path / "receipts" / f"imagenet9_b200_{stage}.json")
    if receipt.get("status") != "completed" or receipt.get("stage") != stage:
        raise ValueError(f"ImageNet-9 B200 {stage} receipt is incomplete")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError(f"ImageNet-9 B200 {stage} receipt has no artifacts")
    for relative, digest in artifacts.items():
        path = context.path / str(relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"ImageNet-9 B200 {stage} artifact changed: {relative}")
    return {"artifact_hashes_validated": len(artifacts)}


def b200_method_plan(config: Mapping[str, Any]) -> list[dict[str, object]]:
    """Return the five frozen baseline rows used by Table 1 smoke paper-data."""

    _settings(config)
    return baseline_plan(list(REQUIRED_METHODS))


def validate_checkpoint_fingerprint_records(records: Any) -> list[dict[str, Any]]:
    """Validate the strict three-case schema consumed by the global verifier."""

    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("ImageNet-9 fingerprint collector must return exactly three cases")
    kinds: set[str] = set()
    widths: list[int] = []
    result: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("ImageNet-9 fingerprint case must be an object")
        record = dict(raw)
        required = {
            "family",
            "case_id",
            "model_id",
            "checkpoints",
            "sample_ids",
            "preprocessed_tensor",
            "target_class",
            "logits",
            "probabilities",
            "precision",
            "device",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"ImageNet-9 fingerprint case is missing fields: {missing}")
        if record["family"] != "imagenet9" or not str(record["case_id"]):
            raise ValueError("ImageNet-9 fingerprint family/case identity is invalid")
        kind = str(record.get("model_kind", ""))
        architecture_family = str(record.get("architecture_family", ""))
        kinds.add(f"{kind}:{architecture_family}")
        checkpoints = record["checkpoints"]
        if not isinstance(checkpoints, list) or len(checkpoints) != 1:
            raise ValueError("each ImageNet-9 fingerprint must bind one checkpoint")
        checkpoint = checkpoints[0]
        if not isinstance(checkpoint, Mapping):
            raise TypeError("ImageNet-9 fingerprint checkpoint must be an object")
        path = Path(str(checkpoint.get("path", "")))
        digest = str(checkpoint.get("sha256", ""))
        size = int(checkpoint.get("bytes", -1))
        if (
            not path.is_absolute()
            or len(digest) != 64
            or size < 1
            or not path.is_file()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise ValueError("ImageNet-9 fingerprint checkpoint identity is incomplete")
        sample_ids = record["sample_ids"]
        if not isinstance(sample_ids, list) or len(sample_ids) != 1:
            raise ValueError("ImageNet-9 fingerprint must use one fixed sample")
        tensor = record["preprocessed_tensor"]
        if (
            not isinstance(tensor, Mapping)
            or len(str(tensor.get("sha256", ""))) != 64
            or tensor.get("dtype") != "float32"
            or tensor.get("shape") != [1, 3, 224, 224]
            or tensor.get("byte_order") != "little-endian"
            or tensor.get("layout") != "C-contiguous"
        ):
            raise ValueError("ImageNet-9 preprocessed tensor identity is invalid")
        logits = np.asarray(record["logits"], dtype=np.float64)
        probabilities = np.asarray(record["probabilities"], dtype=np.float64)
        target = int(record["target_class"])
        if (
            logits.ndim != 2
            or logits.shape[0] != 1
            or logits.shape != probabilities.shape
            or logits.shape[1] not in {9, 1000}
            or not 0 <= target < 9
            or not np.isfinite(logits).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError("ImageNet-9 fingerprint logits/probabilities are invalid")
        if record["precision"] != "float32" or record["device"] != "cuda:0":
            raise ValueError("ImageNet-9 fingerprint precision/device must be float32/cuda:0")
        widths.append(int(logits.shape[1]))
        result.append(record)
    expected_kinds = {
        "off_the_shelf:cnn",
        "fine_tuned:cnn",
        "fine_tuned:transformer",
    }
    if kinds != expected_kinds or sorted(widths) != [9, 9, 1000]:
        raise ValueError("ImageNet-9 fingerprints do not cover 1k/CNN/transformer exactly")
    if len({str(record["case_id"]) for record in result}) != 3:
        raise ValueError("ImageNet-9 fingerprint case IDs must be unique")
    return result


def collect_checkpoint_fingerprints(device: str = "cuda:0") -> list[dict[str, Any]]:
    """Collect exactly three real-CUDA forward fingerprints from offline bytes."""

    torch, device_name = _require_single_b200(device)
    config = load_profile("imagenet9", "smoke")
    settings = _settings(config)
    assets = resolve_b200_assets(config)
    pairs = selected_pair_frame(assets, settings)
    sample = pairs.iloc[0]
    preprocessed = preprocess_paths(
        [str(sample["original_path"])],
        dataset_root=assets.dataset_root,
        size=int(settings["input_size"]),
    )
    tensor_identity = canonical_tensor_identity(preprocessed)
    inputs = torch.from_numpy(preprocessed).to(device=device, dtype=torch.float32)
    mapping = load_official_mapping(assets.mapping, torch)
    records: list[dict[str, Any]] = []
    for asset in assets.models:
        model, load_record = load_model(asset, device=device)
        adapter = probability_model(
            model,
            mapping_matrix=mapping,
            output_classes=asset.output_classes,
        ).to(device=torch.device(device))
        try:
            with torch.inference_mode():
                logits_tensor = adapter.normalized_logits(inputs).float()
                raw_probabilities = torch.softmax(logits_tensor, dim=-1)
                mapped_probabilities = adapter(inputs)
            if tuple(mapped_probabilities.shape) != (1, 9) or not bool(
                torch.isfinite(mapped_probabilities).all()
            ):
                raise FloatingPointError("ImageNet-9 fingerprint mapping produced invalid output")
            torch.cuda.synchronize(torch.device(device))
            case_suffix = (
                "off_the_shelf"
                if asset.kind == "off_the_shelf"
                else f"finetuned_{asset.architecture_family}"
            )
            records.append(
                {
                    "schema_version": 2,
                    "family": "imagenet9",
                    "case_id": f"imagenet9_{case_suffix}",
                    "model_id": asset.model_id,
                    "model_kind": asset.kind,
                    "architecture": asset.architecture,
                    "architecture_family": asset.architecture_family,
                    "checkpoints": [
                        {
                            "identity": asset.model_id,
                            "path": str(asset.checkpoint),
                            "sha256": asset.checkpoint_sha256,
                            "bytes": asset.checkpoint_bytes,
                        }
                    ],
                    "sample_ids": [f"{sample['pair_id']}:mixed_same"],
                    "preprocessed_tensor": {
                        **tensor_identity,
                        "definition": (
                            "per-image resize-short-side-256, center-crop-224, RGB/255 NCHW"
                        ),
                        "source_pair_manifest_sha256": PAIRED_MANIFEST_SHA256,
                    },
                    "target_class": int(sample["class_id"]),
                    "logits": logits_tensor.detach().cpu().numpy().tolist(),
                    "probabilities": raw_probabilities.detach().cpu().numpy().tolist(),
                    "imagenet9_probabilities": mapped_probabilities.detach().cpu().numpy().tolist(),
                    "probability_adapter": {
                        "softmax_count": 1,
                        "official_mapping_sha256": MAPPING_SHA256,
                        "mapped_mass_renormalized": False,
                        "direct_nine_way": asset.output_classes == 9,
                    },
                    "precision": "float32",
                    "device": str(device),
                    "device_name": device_name,
                    "checkpoint_load": load_record,
                }
            )
        finally:
            del adapter, model
            torch.cuda.empty_cache()
    return validate_checkpoint_fingerprint_records(records)


__all__ = [
    "b200_enabled",
    "b200_method_plan",
    "build_b200_plan",
    "collect_checkpoint_fingerprints",
    "compute_b200",
    "prepare_b200",
    "selected_pair_frame",
    "validate_b200_compute_resume",
    "validate_b200_prepare_resume",
    "validate_checkpoint_fingerprint_records",
    "validate_downstream_resume",
    "write_downstream_receipt",
]
