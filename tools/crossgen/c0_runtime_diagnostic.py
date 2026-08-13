"""Diagnostics-only C0 runtime attribution for one sealed historical unit.

This command never publishes a neutral trajectory or changes an acceptance
threshold.  It writes one JSON file under an explicitly supplied diagnostics
path and compares explicit TF32-off/on executions against the sealed aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.crossgen.legacy_controlled_export import (
    C0_CANDIDATES,
    C0_ENDPOINT_EPSILON,
    C0_NOISE_SEEDS,
    C0_PROTOCOL_FAMILY,
    C0_PROTOCOL_VALUE,
    DEFAULT_C0_CONFIG,
    DEFAULT_C0_RESULTS_ROOT,
    DEFAULT_C0_SOURCE_CONFIG,
    DEFAULT_HISTORICAL_ROOT,
    _c0_endpoint_row,
    _c0_endpoint_rows,
    _c0_registered_assets,
    _c0_replay_prefix_count,
    _delivery_expected_sha,
    _historical_c0_modules,
    _regular_file,
)
from tools.crossgen.schema import sha256_file

C0_FORMAL_AUDIT_FIELDS = (
    "endpoint_abs",
    "auc_abs_info",
    "auc_align_info",
    "auc_opp_info",
    "auc_null_info",
)
C0_EXTENDED_ALPHA_FIELDS = ("auc_align_alpha",)


def _diagnostic_error_scopes(
    sealed: dict[str, float], recomputed: dict[str, float]
) -> dict[str, Any]:
    """Separate the registered common-information gate from extra diagnostics."""

    missing = sorted((set(C0_FORMAL_AUDIT_FIELDS) | set(C0_EXTENDED_ALPHA_FIELDS)) - set(sealed))
    missing += sorted(
        (set(C0_FORMAL_AUDIT_FIELDS) | set(C0_EXTENDED_ALPHA_FIELDS)) - set(recomputed)
    )
    if missing:
        raise KeyError(f"C0 diagnostic error scope is missing fields: {missing}")
    all_errors = {name: abs(recomputed[name] - sealed[name]) for name in sealed}
    formal = {name: all_errors[name] for name in C0_FORMAL_AUDIT_FIELDS}
    extended = {name: all_errors[name] for name in C0_EXTENDED_ALPHA_FIELDS}
    return {
        "absolute_errors": all_errors,
        "formal_common_absolute_errors": formal,
        "formal_common_maximum_absolute_error": max(formal.values()),
        "extended_alpha_absolute_errors": extended,
        "extended_alpha_maximum_absolute_error": max(extended.values()),
    }


def _tensor_sha256(tensor: Any) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def _numeric_state(torch: Any, device: Any) -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "device": str(device),
        "device_name": str(torch.cuda.get_device_name(device)),
        "device_capability": list(map(int, torch.cuda.get_device_capability(device))),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _set_tf32_mode(torch: Any, *, enabled: bool) -> None:
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _sealed_values(
    summary_path: Path, *, model_id: str, base_id: int, factor: str
) -> dict[str, float]:
    frame = pd.read_parquet(
        summary_path,
        filters=[("base_id", "==", base_id), ("factor", "==", factor)],
        use_threads=False,
    )
    frame = frame.loc[
        frame["model_id"].astype(str).eq(model_id)
        & frame["family"].astype(str).eq(C0_PROTOCOL_FAMILY)
        & np.isclose(
            frame["protocol_value"].astype(float),
            C0_PROTOCOL_VALUE,
            atol=0.0,
            rtol=0.0,
        )
    ]
    if len(frame) != 1:
        raise ValueError("diagnostic sealed summary identity is not unique")
    row = frame.iloc[0]
    if int(row["n_repeats_averaged"]) != 6:
        raise ValueError("diagnostic sealed summary is not a six-repeat aggregate")
    return {
        "endpoint_abs": float(row["endpoint_abs"]),
        "auc_abs_info": float(row["auc_abs_info"]),
        "auc_align_info": float(row["auc_align_info"]),
        "auc_opp_info": float(row["auc_opp_info"]),
        "auc_null_info": float(row["auc_null_info"]),
        "auc_align_alpha": float(row["auc_align_alpha"]),
    }


def run_diagnostic(
    *,
    historical_root: Path,
    results_root: Path,
    config_path: Path,
    source_config_path: Path,
    output: Path,
    device_name: str,
    model_id: str,
    base_id: int,
    overwrite: bool,
) -> dict[str, Any]:
    import torch

    if output.exists() and not overwrite:
        raise FileExistsError(f"diagnostic output exists; pass --overwrite: {output}")
    selection_matches = [
        item
        for item in C0_CANDIDATES
        if str(item["model_id"]) == model_id and int(item["base_id"]) == base_id
    ]
    if len(selection_matches) != 1:
        raise ValueError("diagnostic target must be one registered C0 selection")
    selection = selection_matches[0]
    task = str(selection["task"])
    factor = str(selection["factor"])

    history = historical_root.expanduser().resolve()
    results = results_root.expanduser().resolve()
    effective_config = _regular_file(config_path.expanduser().resolve(), "C0 effective config")
    source_config = _regular_file(
        source_config_path.expanduser().resolve(), "C0 authoritative source config"
    )
    assets = _c0_registered_assets(results, effective_config, source_config)
    (
        main_common,
        main_sweep,
        model_adapter,
        v11_channels,
        geometry_class,
        runtime_provenance,
    ) = _historical_c0_modules(history, assets["package_path"])
    config = main_common.load_main_config(effective_config)
    if main_common.main_output_root(config) != results:
        raise ValueError("diagnostic config does not resolve to the sealed results")

    target = torch.device(device_name)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C0 TF32 diagnostic requires one CUDA device")
    torch.cuda.set_device(target)
    initial_numeric_state = _numeric_state(torch, target)

    configured_models = {model.model_id: model for model in main_common.main_models(config)}
    if model_id not in configured_models:
        raise ValueError(f"diagnostic model is outside the sealed config: {model_id}")
    model_spec = configured_models[model_id]
    delivery_inputs = assets["delivery_inputs"]
    manifest_path = _regular_file(
        results / str(config["models"]["manifest"]), "C0 model manifest"
    )
    _delivery_expected_sha(delivery_inputs, manifest_path, "model manifest")
    manifest_frame = pd.read_csv(manifest_path)
    manifest_rows = manifest_frame.loc[manifest_frame["model_id"].astype(str).eq(model_id)]
    if len(manifest_rows) != 1:
        raise ValueError("diagnostic model manifest identity is not unique")
    manifest = manifest_rows.iloc[0]
    checkpoint_path = _regular_file(Path(str(manifest["checkpoint_path"])), "C0 checkpoint")
    checkpoint_sha256 = str(manifest["checkpoint_sha256"])
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("diagnostic checkpoint SHA-256 changed")
    probability_path = _regular_file(
        Path(str(manifest["probability_cache_path"])), "C0 probability cache"
    )
    if sha256_file(probability_path) != str(manifest["probability_cache_sha256"]):
        raise ValueError("diagnostic probability cache SHA-256 changed")

    job = results / "jobs" / model_id
    audit = json.loads(_regular_file(job / "audit.json", "C0 model audit").read_text())
    historical_batch_size = int(audit["final_batch_size"])
    summary_path = _regular_file(job / "sample_auc_selected.parquet", "C0 sample summary")
    _delivery_expected_sha(delivery_inputs, summary_path, "sealed sample summary")
    sealed = _sealed_values(summary_path, model_id=model_id, base_id=base_id, factor=factor)

    dynamic_path = _regular_file(
        results / "data" / f"{task}_dynamic_base_ids.npy", "C0 dynamic IDs"
    )
    endpoint_path = _regular_file(
        results / "data" / f"{task}_endpoint_base_ids.npy", "C0 endpoint IDs"
    )
    dynamic_ids = np.load(dynamic_path, allow_pickle=False).astype(np.int64, copy=False)
    endpoint_ids = np.load(endpoint_path, allow_pickle=False).astype(np.int64, copy=False)
    positions = np.flatnonzero(dynamic_ids == base_id)
    if positions.size != 1:
        raise ValueError("diagnostic base ID is absent or repeated in the dynamic lock")
    position = int(positions[0])

    processed_path = _regular_file(
        main_common.resolve_path(config, str(config["frozen_inputs"]["processed_images"])),
        "C0 processed images",
    )
    geometry_path = _regular_file(
        main_common.resolve_path(config, str(config["frozen_inputs"]["geometry"])),
        "C0 covariance geometry",
    )
    geometry = geometry_class.from_geometry_archive(geometry_path)
    information = main_sweep._load_information_maps(results)
    alpha = np.asarray(main_common.alpha_grid(config), dtype=np.float64)
    points = [
        point
        for point in main_common.protocol_points(config)
        if point.family == C0_PROTOCOL_FAMILY
        and np.isclose(float(point.value), C0_PROTOCOL_VALUE, atol=0.0, rtol=0.0)
    ]
    if len(points) != 1:
        raise ValueError("diagnostic linear lambda=0 point is not unique")
    point = points[0]
    protocol_index = information.protocol_index(point)
    common_inverse = information.alpha_of_c[protocol_index]
    common_grid = information.c_grid
    full_window = tuple(map(float, config["integration"]["full_window"]))
    alpha_budget_grid = np.arange(0.0, 1.0000001, 0.02, dtype=np.float64)

    variants, _ = main_sweep._load_variants(
        config, results, model_spec, manifest, endpoint_ids, dynamic_ids
    )
    variant_indices: dict[str, int] = {}
    for map_seed in ("20260882", "20260883"):
        matches = [
            index
            for index, variant in enumerate(variants)
            if str(variant.factor) == factor and str(variant.cf_map_seed) == map_seed
        ]
        if len(matches) != 1:
            raise ValueError(f"diagnostic map identity is not unique: {map_seed}")
        variant_indices[map_seed] = matches[0]
    stack_size = len(variants) + 1
    replay_count = _c0_replay_prefix_count(
        dynamic_count=int(dynamic_ids.size),
        stack_size=stack_size,
        historical_batch_size=historical_batch_size,
        selected_positions=np.asarray([position], dtype=np.int64),
    )
    endpoint_deltas = np.stack(
        [variant.dynamic_endpoint_delta[:replay_count] for variant in variants], axis=1
    ).astype(np.float64, copy=False)
    endpoint_table = _c0_endpoint_rows(assets["endpoint_path"])
    endpoint_rows: dict[str, Any] = {}
    for map_seed, variant_index in variant_indices.items():
        endpoint_row = _c0_endpoint_row(
            endpoint_table, selection, map_seed, check_expected_state=False
        )
        endpoint_rows[map_seed] = endpoint_row
        if int(endpoint_row["counterfactual_id"]) != int(
            variants[variant_index].counterfactual_ids[position]
        ):
            raise ValueError("diagnostic counterfactual identity changed")
        endpoint_deltas[position, variant_index] = float(endpoint_row["delta_endpoint"])

    tensors = [
        model_adapter.processed_images_to_nchw(
            processed_path, dynamic_ids[:replay_count], device=target
        )
    ]
    tensors.extend(
        model_adapter.processed_images_to_nchw(
            processed_path, variant.counterfactual_ids[:replay_count], device=target
        )
        for variant in variants
    )
    clean_stack = torch.stack(tensors, dim=1)
    del tensors
    loaded = model_adapter.load_validated_checkpoint(
        checkpoint_path,
        task=str(model_spec.task),
        architecture=str(model_spec.architecture),
        seed=int(model_spec.seed),
        device=target,
    )
    if loaded.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("diagnostic checkpoint identity changed during load")
    probability_cache = np.load(probability_path, allow_pickle=False)

    mode_results: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        for mode_name, enabled in (("tf32_off", False), ("tf32_on", True)):
            _set_tf32_mode(torch, enabled=enabled)
            torch.cuda.synchronize(target)
            scorer = main_sweep.BatchBackoff(historical_batch_size, [])
            clean_scores = scorer.score(loaded.model, clean_stack, device=target)
            clean_selected = {
                "factual_live": float(clean_scores[position, 0]),
                "factual_cache": float(probability_cache[base_id]),
                "factual_abs_error": abs(
                    float(clean_scores[position, 0]) - float(probability_cache[base_id])
                ),
                "maps": {},
            }
            for map_seed, variant_index in variant_indices.items():
                cf_id = int(variants[variant_index].counterfactual_ids[position])
                live = float(clean_scores[position, variant_index + 1])
                cached = float(probability_cache[cf_id])
                clean_selected["maps"][map_seed] = {
                    "counterfactual_id": cf_id,
                    "counterfactual_live": live,
                    "counterfactual_cache": cached,
                    "counterfactual_abs_error": abs(live - cached),
                }

            sums = {name: 0.0 for name in ("abs", "align", "opp", "null")}
            alpha_align_sum = 0.0
            curves: list[dict[str, Any]] = []
            noise_records: list[dict[str, Any]] = []
            for noise_seed in C0_NOISE_SEEDS:
                shape = (int(dynamic_ids.size), int(geometry.ambient_dimension))
                standard = main_sweep._standard_normal(shape, noise_seed, target)
                isotropic = main_sweep._standard_normal(
                    shape, noise_seed + 1_000_003, target
                )
                eta_covariance, eta_isotropic = v11_channels.linear_base_noises_torch(
                    standard, isotropic, geometry.covariance
                )
                noise = main_sweep._noise_for_point(
                    point,
                    standard=standard,
                    eta_covariance=eta_covariance,
                    eta_isotropic=eta_isotropic,
                    covariance=geometry.covariance,
                )
                retained_noise = noise[:replay_count].clone()
                selected_noise = retained_noise[position]
                noise_records.append(
                    {
                        "noise_seed": int(noise_seed),
                        "standard_sha256": _tensor_sha256(standard),
                        "eta_covariance_sha256": _tensor_sha256(eta_covariance),
                        "retained_noise_sha256": _tensor_sha256(retained_noise),
                        "selected_noise_sha256": _tensor_sha256(selected_noise),
                        "selected_noise_first8": selected_noise[:8]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(float)
                        .tolist(),
                        "selected_noise_mean": float(selected_noise.float().mean().item()),
                        "selected_noise_std": float(selected_noise.float().std().item()),
                    }
                )
                delta = main_sweep._delta_curves(
                    loaded.model,
                    clean_stack,
                    endpoint_deltas,
                    retained_noise,
                    geometry.mean,
                    alpha,
                    scorer,
                    target,
                )
                for map_seed, variant_index in variant_indices.items():
                    components = main_sweep._component_matrices(
                        endpoint_deltas[:, variant_index],
                        delta[variant_index],
                        C0_ENDPOINT_EPSILON,
                    )
                    aucs = main_sweep._auc_components(
                        alpha, components, common_inverse, common_grid, full_window
                    )
                    for name in sums:
                        sums[name] += float(aucs[name][position])
                    alpha_aucs = main_sweep._auc_components(
                        alpha,
                        components,
                        alpha_budget_grid,
                        alpha_budget_grid,
                        full_window,
                    )
                    alpha_align_sum += float(alpha_aucs["align"][position])
                    curves.append(
                        {
                            "noise_seed": int(noise_seed),
                            "cf_map_seed": map_seed,
                            "variant_index": int(variant_index),
                            "endpoint_delta": float(endpoint_deltas[position, variant_index]),
                            "response_curve": delta[variant_index, position]
                            .astype(float)
                            .tolist(),
                            "auc_abs_info": float(aucs["abs"][position]),
                            "auc_align_info": float(aucs["align"][position]),
                            "auc_opp_info": float(aucs["opp"][position]),
                            "auc_null_info": float(aucs["null"][position]),
                            "auc_align_alpha": float(alpha_aucs["align"][position]),
                        }
                    )
                del standard, isotropic, eta_covariance, eta_isotropic, noise, retained_noise
            recomputed = {
                "endpoint_abs": float(
                    np.mean(
                        [abs(float(row["delta_endpoint"])) for row in endpoint_rows.values()],
                        dtype=np.float64,
                    )
                ),
                "auc_abs_info": sums["abs"] / 6.0,
                "auc_align_info": sums["align"] / 6.0,
                "auc_opp_info": sums["opp"] / 6.0,
                "auc_null_info": sums["null"] / 6.0,
                "auc_align_alpha": alpha_align_sum / 6.0,
            }
            error_scopes = _diagnostic_error_scopes(sealed, recomputed)
            mode_results[mode_name] = {
                "numeric_state": _numeric_state(torch, target),
                "clean_endpoint_comparison": clean_selected,
                "batch_attempts": list(map(int, scorer.attempts)),
                "noise_records": noise_records,
                "curves": curves,
                "recomputed": recomputed,
                **error_scopes,
            }
            torch.cuda.synchronize(target)
    finally:
        torch.set_float32_matmul_precision(initial_numeric_state["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = initial_numeric_state[
            "cuda_matmul_allow_tf32"
        ]
        torch.backends.cudnn.allow_tf32 = initial_numeric_state["cudnn_allow_tf32"]

    result = {
        "artifact_type": "c0_runtime_tf32_diagnostic_only",
        "acceptance_effect": "none",
        "target": {
            "model_id": model_id,
            "base_id": base_id,
            "task": task,
            "factor": factor,
            "dynamic_position": position,
            "replay_dynamic_prefix": replay_count,
            "stack_size": stack_size,
            "historical_batch_size": historical_batch_size,
            "noise_seeds": list(C0_NOISE_SEEDS),
            "protocol_family": C0_PROTOCOL_FAMILY,
            "protocol_value": C0_PROTOCOL_VALUE,
            "alpha": alpha.astype(float).tolist(),
        },
        "initial_numeric_state": initial_numeric_state,
        "sealed": sealed,
        "modes": mode_results,
        "provenance": {
            "package": str(assets["package_path"]),
            "package_sha256": sha256_file(assets["package_path"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "probability_cache": str(probability_path),
            "probability_cache_sha256": sha256_file(probability_path),
            "sample_summary": str(summary_path),
            "sample_summary_sha256": sha256_file(summary_path),
            "historical_runtime": runtime_provenance,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_C0_RESULTS_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_C0_CONFIG)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_C0_SOURCE_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-id", default="context_gate__resnet18__seed_3101")
    parser.add_argument("--base-id", type=int, default=75310)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_diagnostic(
        historical_root=args.historical_root,
        results_root=args.results_root,
        config_path=args.config,
        source_config_path=args.source_config,
        output=args.output,
        device_name=args.device,
        model_id=args.model_id,
        base_id=args.base_id,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "runtime_seconds": result["runtime_seconds"],
                "formal_common_maximum_absolute_errors": {
                    name: mode["formal_common_maximum_absolute_error"]
                    for name, mode in result["modes"].items()
                },
                "extended_alpha_maximum_absolute_errors": {
                    name: mode["extended_alpha_maximum_absolute_error"]
                    for name, mode in result["modes"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
