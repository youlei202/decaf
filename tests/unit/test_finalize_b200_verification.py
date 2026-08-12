"""Focused tests for fail-closed single-B200 evidence finalization."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from io import StringIO
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[2] / "scripts" / "reproduce" / ("finalize_b200_verification.py")
    spec = importlib.util.spec_from_file_location("finalize_b200_verification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity() -> dict[str, object]:
    return {
        "repository_commit": "a" * 40,
        "repository_tree": "b" * 40,
        "tracked_worktree_clean": True,
    }


def _stage_receipt(root: Path, run: str, stage: str, details: object = None) -> None:
    _write_json(
        root / "runs" / run / "receipts" / f"{stage}.json",
        {
            "schema_version": 1,
            "stage": stage,
            "status": "completed",
            "details": details or {},
        },
    )


def _member_receipts(
    root: Path,
    run: str,
    count: int,
    *,
    queue: bool = False,
    mixed_resume: bool = False,
) -> None:
    members: dict[str, object] = {}
    events = []
    for index in range(count):
        member_id = f"member-{index:03d}"
        relative = f"raw/{member_id}.bin"
        output = root / "runs" / run / relative
        _write_text(output, f"{run}:{member_id}\n")
        _write_json(
            root / "runs" / run / "receipts" / "members" / f"{member_id}.json",
            {
                "schema_version": 1,
                "kind": "member",
                "member_id": member_id,
                "optional": False,
                "status": "completed",
                "error": None,
                "details": {
                    "artifacts": [
                        {
                            "path": relative,
                            "bytes": output.stat().st_size,
                            "sha256": _sha256(output),
                        }
                    ]
                },
            },
        )
        members[member_id] = {"optional": False, "status": "completed"}
        if mixed_resume and index < 2:
            events.append({"member_id": member_id, "event": "resume_skip", "device": 0})
        else:
            events.extend(
                [
                    {"member_id": member_id, "event": "start", "device": 0},
                    {"member_id": member_id, "event": "completed", "device": 0},
                ]
            )
    details: dict[str, object] = {}
    if queue:
        details = {
            "backend": "gpu",
            "endpoint_m_stage": "analyze",
            "member_count": count,
            "plan_contract_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "data_binding_manifest_sha256": "3" * 64,
            "checkpoint_binding_manifest_sha256": "4" * 64,
            "scheduler": "single_gpu_dynamic_queue",
            "visible_device": "cuda:0",
            "exclusive_member_concurrency": 1,
            "dynamic_refill": True,
            "duplicate_execution": False,
            "queue_events": events,
            "failures": {},
            "multi_gpu_real_execution": "NOT_TESTED_SINGLE_GPU_NODE",
        }
    _write_json(
        root / "runs" / run / "receipts" / "compute_members.json",
        {
            "schema_version": 1,
            "kind": "global",
            "run_id": run,
            "status": "completed",
            "all_processes_exited": True,
            "member_count": count,
            "members": members,
            "details": details,
        },
    )


def _run_receipt(
    root: Path,
    run: str,
    experiment: str,
    profile: str,
    index: int,
    *,
    member_count: int,
    queue: bool = False,
) -> None:
    _write_json(
        root / "runs" / run / "run.json",
        {
            "schema_version": 1,
            "experiment": experiment,
            "profile": profile,
            "requested_stage": "all",
            "run_id": run,
            "status": "completed",
            "completed_stages": ["prepare", "compute", "analyze", "paper"],
            "started_at": f"2026-08-12T1{index}:00:00Z",
            "finished_at": f"2026-08-12T1{index}:30:00Z",
        },
    )
    _write_json(
        root / "runs" / run / "environment.json",
        {"available_cpus": 48, "python": "3.11.9", "platform": "Linux-test"},
    )
    for stage in ("prepare", "compute", "analyze", "paper"):
        details: dict[str, object] = {}
        if queue and stage == "compute":
            details = {
                "backend": "gpu",
                "scheduler": "single_gpu_dynamic_queue",
                "completed_members": member_count,
            }
        _stage_receipt(root, run, stage, details)
    _member_receipts(root, run, member_count, queue=queue)


def _controlled(root: Path) -> None:
    _run_receipt(root, "controlled", "controlled", "smoke", 0, member_count=5)
    output = StringIO()
    columns = [
        "experiment",
        "model_id",
        "metric",
        "value",
        "n_values",
        "gpu_verification",
        "architecture",
        "family",
        "case_id",
        "expected_behavior",
    ]
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    cases = (
        ("case-active-r", "active", "resnet18"),
        ("case-active-v", "active", "small_vit"),
        ("case-null", "null", "small_vit"),
        ("case-aligned", "aligned", "resnet18"),
        ("case-opposed", "opposed", "small_vit"),
    )
    for case_id, behavior, architecture in cases:
        sample_ids = (
            {
                "endpoint_cf_ids": list(range(8)),
                "endpoint_fact_ids": list(range(8, 16)),
                "swap_cf_ids": list(range(16, 24)),
                "swap_fact_ids": list(range(24, 32)),
            }
            if behavior == "opposed"
            else {
                "factual_ids": list(range(8)),
                "counterfactual_ids": list(range(8, 16)),
            }
        )
        _write_json(
            root / "runs/controlled/raw/b200" / f"{case_id}.json",
            {
                "case_id": case_id,
                "expected_behavior": behavior,
                "sample_ids": sample_ids,
                "observed_behaviors": {behavior: 8},
                "execution": {
                    "backend": "cuda",
                    "gpu_verification": "passed",
                    "device": {
                        "resolved": "cuda:0",
                        "count_visible": 1,
                        "name": "NVIDIA B200",
                    },
                },
                "numeric_audit": {
                    "passed": True,
                    "finite_model_scores": True,
                    "nonnegative_ecf": True,
                    "pointwise_conservation": {"passed": True},
                    "integrated_conservation": {"passed": True},
                    "tiny_endpoint_swap": {"passed": True},
                },
            },
        )
        for metric in ("E", "C", "F", "Abs", "Net", "M"):
            writer.writerow(
                {
                    "experiment": "controlled",
                    "model_id": case_id,
                    "metric": metric,
                    "value": "0.25",
                    "n_values": "8",
                    "gpu_verification": "passed",
                    "architecture": architecture,
                    "family": "smoke",
                    "case_id": case_id,
                    "expected_behavior": behavior,
                }
            )
    metrics = root / "runs/controlled/metrics/controlled_smoke_metrics.csv"
    _write_text(metrics, output.getvalue())
    _write_json(
        root / "runs/controlled/metrics/controlled_smoke_summary.json",
        {
            "schema_version": 1,
            "status": "completed",
            "scope": "real_cuda_single_b200_shard",
            "gpu_real_shard_verification": "passed",
            "rows": 30,
            "metrics_sha256": _sha256(metrics),
        },
    )


def _imagenet9(root: Path) -> None:
    _run_receipt(root, "imagenet9", "imagenet9", "smoke", 1, member_count=24)
    _write_json(
        root / "runs/imagenet9/manifests/plan.json",
        {
            "verification_mode": "single_b200_real_cuda",
            "execution_class": "real_cuda",
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
            },
            "models": [{"model_id": f"model-{index}"} for index in range(3)],
            "members": [{"member_id": f"member-{index:03d}"} for index in range(24)],
        },
    )
    aggregate: dict[str, str] = {}
    for name in ("raw/response_paths.csv", "metrics/decaf_scores.csv", "metrics/baselines.csv"):
        path = root / "runs/imagenet9" / name
        _write_text(path, "value\n1\n")
        aggregate[name] = _sha256(path)
    _write_json(
        root / "runs/imagenet9/receipts/imagenet9_b200_compute.json",
        {
            "status": "completed",
            "verification_mode": "single_b200_real_cuda",
            "gpu_inference_verified": True,
            "artifacts": aggregate,
        },
    )
    _write_json(
        root / "runs/imagenet9/metrics/summary.json",
        {
            "gpu_inference_verified": True,
            "source_mode": "computed_run",
            "model_count": 3,
            "score_rows": 288,
            "baseline_rows": 480,
            "reveal_paths": ["blend", "patch_A", "patch_B"],
            "baseline_methods": [
                "input_x_gradient",
                "integrated_gradients",
                "occlusion",
                "rise",
                "smoothgrad",
            ],
        },
    )
    _stage_receipt(
        root,
        "imagenet9",
        "compute",
        {
            "gpu_inference_executed": True,
            "gpu_name": "NVIDIA B200",
            "source": "real_backgrounds_challenge_cuda",
            "completed_member_receipts": 24,
            "trajectory_count": 288,
            "baseline_rows": 480,
        },
    )
    global_path = root / "runs/imagenet9/receipts/compute_members.json"
    global_receipt = json.loads(global_path.read_text(encoding="utf-8"))
    global_receipt["details"] = {"verification_mode": "single_b200_real_cuda"}
    _write_json(global_path, global_receipt)


ATTRIBUTION_SCOPES = {
    "attribution_main": (
        "smoke_idsds_deletion_targets",
        "smoke_idsds_primary",
        "smoke_funnybirds_deletion_targets",
        "smoke_funnybirds_heldout_targets",
        "smoke_funnybirds_primary",
    ),
    "dinov2_g": ("smoke_dinov2_g_quality", "smoke_dinov2_g_timing"),
    "partimagenet": (
        "smoke_partimagenet_deletion_targets",
        "smoke_partimagenet_heldout_targets",
        "smoke_partimagenet_boundary",
    ),
}


def _attribution(root: Path, run: str, profile: str, index: int) -> None:
    count = {"attribution_main": 72, "dinov2_g": 16, "partimagenet": 8}[run]
    _run_receipt(
        root,
        run,
        "attribution",
        profile,
        index,
        member_count=count,
        queue=True,
    )
    _write_json(
        root / "runs" / run / "manifests/plan.json",
        {
            "experiment": "attribution",
            "scope_names": list(ATTRIBUTION_SCOPES[run]),
            "member_count": count,
            "expected_member_count": count,
            "endpoint_m_stage": "analyze",
            "config_sha256": "2" * 64,
            "plan_contract_sha256": "1" * 64,
            "members": [
                {
                    "member_id": f"member-{member_index:03d}",
                    "scope": ATTRIBUTION_SCOPES[run][member_index % len(ATTRIBUTION_SCOPES[run])],
                    "dataset": "imagenet1k_idsds",
                    "model_id": "resnet50",
                    "method_id": "decaf_5",
                    "kind": "quality",
                    "image_start": 0,
                    "image_stop": 1,
                    "output_path": f"raw/member-{member_index:03d}.bin",
                    "receipt_path": f"receipts/members/member-{member_index:03d}.json",
                    "job_sha256": f"{member_index + 10:064x}",
                }
                for member_index in range(count)
            ],
        },
    )
    for member_index in range(count):
        member_id = f"member-{member_index:03d}"
        output = root / "runs" / run / "raw" / f"{member_id}.bin"
        receipt_path = root / "runs" / run / "receipts/members" / f"{member_id}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["details"] = {
            "output_path": f"raw/{member_id}.bin",
            "output_sha256": _sha256(output),
            "rows": 1,
            "columns": ["image_id"],
            "scope": ATTRIBUTION_SCOPES[run][member_index % len(ATTRIBUTION_SCOPES[run])],
            "dataset": "imagenet1k_idsds",
            "model_id": "resnet50",
            "method_id": "decaf_5",
            "job_sha256": f"{member_index + 10:064x}",
            "config_sha256": "2" * 64,
            "plan_contract_sha256": "1" * 64,
            "input_manifest_sha256": "5" * 64,
            "checkpoint_bytes_sha256": "6" * 64,
            "data_binding_manifest_sha256": "3" * 64,
            "checkpoint_binding_manifest_sha256": "4" * 64,
        }
        receipt["started_at"] = "2026-08-12T18:00:00Z"
        receipt["finished_at"] = "2026-08-12T18:01:00Z"
        _write_json(receipt_path, receipt)
    global_path = root / "runs" / run / "receipts/compute_members.json"
    global_receipt = json.loads(global_path.read_text(encoding="utf-8"))
    _write_json(global_path, global_receipt)
    _stage_receipt(
        root,
        run,
        "compute",
        {
            "backend": "gpu",
            "scheduler": "single_gpu_dynamic_queue",
            "completed_members": count,
            "resumed_members": 0,
            "failed_members": 0,
            "member_count": count,
        },
    )
    inventory = [
        {
            "member_id": f"member-{member_index:03d}",
            "output_path": f"raw/member-{member_index:03d}.bin",
            "output_sha256": _sha256(
                root / "runs" / run / "raw" / f"member-{member_index:03d}.bin"
            ),
            "receipt_path": f"receipts/members/member-{member_index:03d}.json",
        }
        for member_index in range(count)
    ]
    _write_json(
        root / "runs" / run / "receipts/resume/compute.json",
        {
            "schema_version": 1,
            "stage": "compute",
            "status": "completed",
            "validation_started_at": "2026-08-13T00:00:00Z",
            "validation_finished_at": "2026-08-13T00:01:00Z",
            "member_count": count,
            "resumed_members": count,
            "reexecuted": 0,
            "source_compute_receipt_sha256": _sha256(root / "runs" / run / "receipts/compute.json"),
            "artifact_inventory_sha256": hashlib.sha256(
                json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )


def _covertype(root: Path) -> None:
    _run_receipt(root, "covertype", "covertype", "smoke", 5, member_count=2)
    _write_json(
        root / "runs/covertype/metrics/analysis_summary.json",
        {
            "source_mode": "computed_run",
            "all_decaf_identities_passed": True,
            "model_count": 2,
            "module_c_models": 1,
            "module_f_models": 1,
        },
    )
    _write_text(
        root / "runs/covertype/metrics/model_results.csv",
        (
            "E,C,F,preserve_rate,invert_rate,null_context_prediction_change_rate,"
            "decaf_identity_passed\n0.3,0.2,0.1,0.9,0.1,0.2,True\n"
        ),
    )


def _resume_and_scheduler(root: Path) -> None:
    _write_json(
        root / "verification/controlled_resume.json",
        {
            "status": "passed",
            "family": "controlled",
            "scope": "real_cuda_single_b200_shard",
            "members": {
                "expected": 5,
                "completed_before_resume": 5,
                "skipped_by_resume_validation": 5,
                "unchanged_after_resume": 5,
                "reexecuted": 0,
            },
            "checks": {"all_members_skipped": True, "artifacts_unchanged": True},
            "evidence": {
                "unchanged": True,
                "pre_resume_sha256": "c" * 64,
                "post_resume_sha256": "c" * 64,
            },
        },
    )
    _write_json(
        root / "verification/imagenet9_resume.json",
        {
            "status": "passed",
            "family": "imagenet9",
            "scope": "real_cuda_single_b200_shard",
            "members": {
                "expected": 24,
                "completed_before_resume": 24,
                "skipped_by_resume_validation": 24,
                "unchanged_after_resume": 24,
                "reexecuted": 0,
            },
            "checks": {"all_members_skipped": True, "artifacts_unchanged": True},
            "evidence": {
                "unchanged": True,
                "pre_resume_sha256": "9" * 64,
                "post_resume_sha256": "9" * 64,
            },
        },
    )
    scheduler_checks = {
        "heterogeneous_member_queue": True,
        "no_duplicate_execution": True,
        "dynamic_refill_gpu0": True,
        "unique_output_paths": True,
        "unique_receipts": True,
        "member_failure_isolation": True,
        "global_receipt_finalization": True,
    }
    _write_json(
        root / "verification/single_gpu_scheduler.json",
        {
            "status": "passed",
            "scheduler": "single_gpu_dynamic_queue",
            "checks": scheduler_checks,
            "multi_gpu_scheduler_static_plan": "PASS",
            "multi_gpu_scheduler_real_execution": "NOT_TESTED_SINGLE_GPU_NODE",
        },
    )
    _write_json(
        root / "runs/resume_test/run.json",
        {
            "status": "completed",
            "experiment": "attribution",
            "profile": "smoke-resume",
            "requested_stage": "compute",
            "completed_stages": ["compute"],
        },
    )
    _write_json(
        root / "runs/resume_test/manifests/plan.json",
        {
            "scope_names": ["resume_idsds_deletion_targets", "resume_idsds_primary"],
            "member_count": 5,
            "config_sha256": "2" * 64,
            "plan_contract_sha256": "1" * 64,
            "members": [
                {
                    "member_id": f"member-{index:03d}",
                    "scope": (
                        "resume_idsds_deletion_targets" if index == 0 else "resume_idsds_primary"
                    ),
                    "dataset": "imagenet1k_idsds",
                    "model_id": "resnet50",
                    "method_id": "decaf_5",
                    "kind": "quality",
                    "image_start": 0,
                    "image_stop": 1,
                    "output_path": f"raw/member-{index:03d}.bin",
                    "receipt_path": f"receipts/members/member-{index:03d}.json",
                    "job_sha256": f"{index + 10:064x}",
                }
                for index in range(5)
            ],
        },
    )
    _member_receipts(root, "resume_test", 5, queue=True, mixed_resume=True)
    for index in range(5):
        member_id = f"member-{index:03d}"
        output = root / "runs/resume_test/raw" / f"{member_id}.bin"
        receipt_path = root / "runs/resume_test/receipts/members" / f"{member_id}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["details"] = {
            "output_path": f"raw/{member_id}.bin",
            "output_sha256": _sha256(output),
            "rows": 1,
            "columns": ["image_id"],
            "scope": "resume_idsds_deletion_targets" if index == 0 else "resume_idsds_primary",
            "dataset": "imagenet1k_idsds",
            "model_id": "resnet50",
            "method_id": "decaf_5",
            "job_sha256": f"{index + 10:064x}",
            "config_sha256": "2" * 64,
            "plan_contract_sha256": "1" * 64,
            "input_manifest_sha256": "5" * 64,
            "checkpoint_bytes_sha256": "6" * 64,
            "data_binding_manifest_sha256": "3" * 64,
            "checkpoint_binding_manifest_sha256": "4" * 64,
        }
        receipt["started_at"] = "2026-08-12T18:00:00Z"
        receipt["finished_at"] = "2026-08-12T18:01:00Z"
        _write_json(receipt_path, receipt)
    _stage_receipt(
        root,
        "resume_test",
        "compute",
        {
            "backend": "gpu",
            "scheduler": "single_gpu_dynamic_queue",
            "member_count": 5,
            "completed_members": 3,
            "resumed_members": 2,
            "failed_members": 0,
        },
    )
    _write_json(
        root / "verification/single_gpu_resume_fault_injection.json",
        {
            "status": "passed",
            "signal": "SIGTERM",
            "interrupted_global_status": "partial",
            "final_global_status": "completed",
            "completed_before_signal": 2,
            "running_receipts_after_signal": 0,
            "completed_members_skipped_on_resume": 2,
            "incomplete_members_completed_on_resume": 3,
            "checks": {
                "normal_sigterm_used": True,
                "terminalized_without_running": True,
                "completed_members_skipped": True,
                "incomplete_members_finished": True,
                "final_status_completed": True,
            },
        },
    )


def _fingerprints(root: Path) -> None:
    checkpoint = root / "fixtures/checkpoint.bin"
    _write_text(checkpoint, "checkpoint bytes\n")
    checkpoint_two = root / "fixtures/checkpoint-two.bin"
    _write_text(checkpoint_two, "second checkpoint bytes\n")
    common_checkpoint = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }
    cases: list[dict[str, object]] = []
    for architecture, model_id in (
        ("resnet18", "object_shape__resnet18__seed_3101"),
        ("small_vit", "object_shape__small_vit__seed_3101"),
    ):
        cases.append(
            {
                "family": "controlled",
                "case_id": f"controlled__base_{architecture}_object_shape",
                "model_id": model_id,
                "architecture": architecture,
                "checkpoints": [{**common_checkpoint, "identity": model_id}],
                "sample_ids": [1, 2],
                "preprocessed_tensor": {
                    "sha256": "d" * 64,
                    "dtype": "float32",
                    "shape": [2, 3, 32, 32],
                    "byte_order": "little-endian",
                    "layout": "C-contiguous",
                    "bytes": 2 * 3 * 32 * 32 * 4,
                },
                "target_class": [1, 1],
                "logits": [[1.0, 0.0], [1.0, 0.0]],
                "probabilities": [[0.75, 0.25], [0.75, 0.25]],
                "precision": "float32",
                "device": "cuda:0",
                "device_details": {
                    "requested": "cuda:0",
                    "resolved": "cuda:0",
                    "count_visible": 1,
                    "name": "NVIDIA B200",
                },
            }
        )
    imagenet_cases = (
        ("imagenet9_off_the_shelf", "tv_resnet18_imagenet1k_v1", "off_the_shelf", "cnn", 1000),
        ("imagenet9_finetuned_cnn", "ft_resnet50_original_s7101", "fine_tuned", "cnn", 9),
        (
            "imagenet9_finetuned_transformer",
            "ft_vit_b_16_original_s7101",
            "fine_tuned",
            "transformer",
            9,
        ),
    )
    for case_id, model_id, kind, architecture_family, width in imagenet_cases:
        cases.append(
            {
                "family": "imagenet9",
                "case_id": case_id,
                "model_id": model_id,
                "model_kind": kind,
                "architecture_family": architecture_family,
                "checkpoints": [{**common_checkpoint, "identity": model_id}],
                "sample_ids": ["pair:mixed_same"],
                "preprocessed_tensor": {
                    "sha256": "d" * 64,
                    "dtype": "float32",
                    "shape": [1, 3, 224, 224],
                    "byte_order": "little-endian",
                    "layout": "C-contiguous",
                    "bytes": 3 * 224 * 224 * 4,
                },
                "target_class": 0,
                "logits": [[0.0] * width],
                "probabilities": [[1.0 / width] * width],
                "imagenet9_probabilities": [[1.0 / 9] * 9],
                "probability_adapter": {
                    "direct_nine_way": kind != "off_the_shelf",
                    "mapped_mass_renormalized": False,
                    "official_mapping_sha256": "e" * 64,
                    "softmax_count": 1,
                },
                "precision": "float32",
                "device": "cuda:0",
                "device_name": "NVIDIA B200",
            }
        )
    attribution_cases = (
        ("funnybirds_resnet50", "funnybirds", "funnybirds_resnet", 50, 256),
        ("funnybirds_vgg16", "funnybirds", "funnybirds_vgg", 50, 256),
        ("funnybirds_vit_b_16", "funnybirds", "funnybirds_vit", 50, 224),
        ("resnet50", "imagenet1k_idsds", "idsds_resnet50", 1000, 224),
        ("vgg16", "imagenet1k_idsds", "idsds_vgg16", 1000, 224),
        (
            "vit_base_patch16_224",
            "imagenet1k_idsds",
            "idsds_vit_base_patch16_224",
            1000,
            224,
        ),
        ("dinov2_vit_g_14", "imagenet1k_idsds", "dinov2_vitg14_backbone", 1000, 224),
    )
    for model_id, dataset, checkpoint_id, width, size in attribution_cases:
        checkpoint_ids = (
            [checkpoint_id, "dinov2_vitg14_linear_head"]
            if model_id == "dinov2_vit_g_14"
            else [checkpoint_id]
        )
        cases.append(
            {
                "family": "attribution",
                "case_id": f"attribution/{dataset}/{model_id}",
                "model_id": model_id,
                "dataset": dataset,
                "checkpoints": [
                    {
                        **(
                            common_checkpoint
                            if position == 0
                            else {
                                "path": str(checkpoint_two),
                                "bytes": checkpoint_two.stat().st_size,
                                "sha256": _sha256(checkpoint_two),
                            }
                        ),
                        "checkpoint_id": identity,
                    }
                    for position, identity in enumerate(checkpoint_ids)
                ],
                "sample_ids": ["sample-0"],
                "preprocessed_tensor": {
                    "sha256": "d" * 64,
                    "dtype": "torch.float32",
                    "shape": [1, 3, size, size],
                    "byte_order": "little",
                    "layout": "contiguous_c_order",
                    "contiguous": True,
                    "contiguous_bytes": 3 * size * size * 4,
                },
                "target_class": 0,
                "logits": [[0.0] * width],
                "probabilities": [[1.0 / width] * width],
                "precision": "fp32",
                "device": "cuda:0",
                "device_name": "NVIDIA B200",
                "cuda_synchronized": True,
            }
        )
    case_ids = [str(case["case_id"]) for case in cases]
    payload_path = root / "verification/checkpoint_fingerprints.json"
    payload = {
        "schema_version": 1,
        "status": "passed",
        "repository": _identity(),
        "environment": {
            "python": "3.10.20",
            "libraries": {
                name: "1.0" for name in ("torch", "torchvision", "timm", "captum", "numpy")
            },
            "cuda_runtime": "12.8",
            "device_index": 0,
            "device_name": "NVIDIA B200",
            "device_total_memory_bytes": 192_000_000_000,
        },
        "tensor_hash_contract": {
            "algorithm": "sha256",
            "source": "C-contiguous tensor bytes after CPU conversion",
            "byte_order": "little-endian",
            "shape_and_dtype_recorded_separately": True,
        },
        "coverage": {"controlled": 2, "imagenet9": 3, "attribution": 7},
        "case_count": 12,
        "cases": cases,
    }
    _write_json(payload_path, payload)
    case_set = hashlib.sha256(
        json.dumps(case_ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "status": "passed",
        "checkpoint_fingerprints_path": "checkpoint_fingerprints.json",
        "checkpoint_fingerprints_sha256": _sha256(payload_path),
        "case_set_sha256": case_set,
        "coverage": {"controlled": 2, "imagenet9": 3, "attribution": 7},
        "case_count": 12,
        "device": {
            "python": "3.10.20",
            "device_name": "NVIDIA B200",
            "device_index": 0,
            "device_total_memory_bytes": 192_000_000_000,
            "cuda_runtime": "12.8",
            "libraries": {
                name: "1.0" for name in ("torch", "torchvision", "timm", "captum", "numpy")
            },
        },
        "checks": {
            "exact_case_coverage": True,
            "checkpoint_bytes_verified": True,
            "preprocessed_tensor_hashes_recorded": True,
            "finite_logits": True,
            "normalized_probabilities": True,
            "single_b200": True,
        },
    }
    _write_json(root / "verification/checkpoint_fingerprint_report.json", report)
    _write_json(
        root / "verification/checkpoint_fingerprint_verification.json",
        {
            "status": "passed",
            "mode": "checkpoint-fingerprint",
            "gpu_real_shard_verification": "checkpoint_fingerprints_passed",
            "steps": {"checkpoint_fingerprint": report},
            **_identity(),
        },
    )


def _audit_reports(root: Path) -> None:
    old_identity = {
        "repository_commit": "e" * 40,
        "repository_tree": "f" * 40,
        "tracked_worktree_clean": True,
    }
    pre_gpu = {
        "unit": {"unit": {"status": "passed"}},
        "full_plan": {"full_plan": {"status": "passed"}},
        "repository_audit": {"repository_audit": {"passed": True}},
    }
    for mode, steps in pre_gpu.items():
        _write_json(
            root / f"verification/pre_gpu/{mode}/cpu_verification.json",
            {
                "status": "passed",
                "mode": mode.replace("_", "-"),
                "steps": steps,
                **old_identity,
            },
        )

    audit = {"passed": True, "finding_count": 0, "findings": [], "root": "."}
    _write_json(root / "verification/repository_audit.json", audit)
    families = {
        family: {"status": "passed"}
        for family in ("controlled", "imagenet9", "attribution", "covertype")
    }
    quality_checks = {
        name: {"status": "passed"}
        for name in ("ruff_check", "ruff_format", "static_imports", "shell_syntax")
    }
    final_steps = {
        "quality": {"quality": {"status": "passed", "checks": quality_checks}},
        "unit": {"unit": {"status": "passed"}},
        "full_plan": {"full_plan": {"status": "passed", "families": families}},
        "repository_audit": {"repository_audit": audit},
    }
    for mode, steps in final_steps.items():
        _write_json(
            root / f"verification/final_audit/{mode}/cpu_verification.json",
            {
                "status": "passed",
                "mode": mode.replace("_", "-"),
                "steps": steps,
                **_identity(),
            },
        )
    pytest_log = root / "verification/final_audit/full_pytest.log"
    _write_text(pytest_log, "321 passed in 12.34s\n")
    _write_json(
        root / "verification/final_audit/full_pytest.json",
        {
            "schema_version": 1,
            "status": "passed",
            "command": ["python", "-m", "pytest"],
            "exit_code": 0,
            "started_at": "2026-08-12T19:00:00Z",
            "finished_at": "2026-08-12T19:01:00Z",
            "elapsed_seconds": 60.0,
            "passed_tests": 321,
            "environment_contract": {
                "mode": "cpu_oracle_with_pinned_real_assets",
                "b200_gate_removed": True,
                "inherited_b200_variables_removed": [
                    "DECAF_B200_VERIFY",
                    "DECAF_ALLOW_NON_B200_TEST",
                    "DECAF_RESUME_TEST_MEMBER_DELAY_SECONDS",
                ],
                "assets": {
                    "covertype_archive": {"sha256": "1" * 64, "size_bytes": 10},
                    "idsds_manifest": {"sha256": "2" * 64, "size_bytes": 20},
                    "reference_run_archives": [
                        {"sha256": f"{index + 3:064x}", "size_bytes": 30 + index}
                        for index in range(9)
                    ],
                },
            },
            "output_log": {
                "path": "verification/final_audit/full_pytest.log",
                "streams": "stdout+stderr",
                "size_bytes": pytest_log.stat().st_size,
                "sha256": _sha256(pytest_log),
            },
            **_identity(),
        },
    )


def _cpu_restructuring(root: Path) -> None:
    _write_json(
        root / "verification/cpu_restructuring_status.json",
        {
            "status": "completed_with_documented_historical_gap",
            "analysis_replay_status": "passed",
            "cpu_tests_status": "passed",
            "quality_status": "passed",
            "static_plan_status": "passed",
            "reference_provenance_verification_status": "passed",
            "historical_bundle_verification_status": "passed",
            "source_snapshot_recovery_status": "repaired_and_verified",
            "public_payload_portability_status": "passed",
            "reference_runs_inventoried": 9,
            "paper_assets_mapped_count": 28,
            "figures_regenerated_count": 11,
            "figures_source_missing_recorded_count": 1,
            "tables_regenerated_count": 16,
            "historical_repository_modified_by_restructure": False,
            "new_repository_commit": "e" * 40,
            "source_status_sha256": "f" * 64,
            "historical_source_gaps": [{"asset_id": "figure_01"}],
        },
    )


def _analysis(root: Path) -> None:
    verification = root / "verification"
    records: list[dict[str, object]] = []

    def record(relative: str, role: str, text: str) -> None:
        path = verification / relative
        _write_text(path, text)
        records.append(
            {
                "portable_path": f"verification_root/{relative}",
                "source_root": "verification_root",
                "relative_path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "role": role,
            }
        )

    for index in range(28):
        record(f"paper_outputs/generated/asset-{index:02d}.tex", "generated_tex", "% data\n")
    for index in range(27):
        record(f"paper_outputs/canonical/asset-{index:02d}.csv", "canonical_csv", "x\n1\n")
    record(
        "paper_outputs/receipts/replay_receipt.json",
        "replay_receipt",
        json.dumps({"runs": [{}] * 9, "inputs": [{}] * 72}) + "\n",
    )
    record(
        "paper_outputs/receipts/family_replay_receipt.json",
        "family_replay_receipt",
        json.dumps({"status": "completed", "family_count": 4}) + "\n",
    )
    record(
        "paper_outputs/receipts/canonical_receipt.json",
        "canonical_receipt",
        json.dumps({"status": "completed", "artifact_count": 27}) + "\n",
    )
    headline = {
        "status": "passed",
        "assertion_count": 27,
        "verified_count": 27,
        "source_missing_count": 0,
    }
    headline_path = verification / "headline_assertions.json"
    _write_json(headline_path, headline)
    records.append(
        {
            "portable_path": "verification_root/headline_assertions.json",
            "source_root": "verification_root",
            "relative_path": "headline_assertions.json",
            "sha256": _sha256(headline_path),
            "size_bytes": headline_path.stat().st_size,
            "role": "headline_assertions",
        }
    )
    output = StringIO()
    writer = csv.DictWriter(
        output, fieldnames=("asset_id", "kind", "comparison_status"), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerow(
        {
            "asset_id": "figure_01",
            "kind": "figure",
            "comparison_status": "source_missing_recorded",
        }
    )
    for index in range(2, 13):
        writer.writerow(
            {
                "asset_id": f"figure_{index:02d}",
                "kind": "figure",
                "comparison_status": "regenerated_semantic_geometry",
            }
        )
    for index in range(1, 17):
        writer.writerow(
            {
                "asset_id": f"table_{index:02d}",
                "kind": "table",
                "comparison_status": "regenerated_semantic_table",
            }
        )
    diff_path = verification / "paper_artifact_diff.csv"
    _write_text(diff_path, output.getvalue())
    records.append(
        {
            "portable_path": "verification_root/paper_artifact_diff.csv",
            "source_root": "verification_root",
            "relative_path": "paper_artifact_diff.csv",
            "sha256": _sha256(diff_path),
            "size_bytes": diff_path.stat().st_size,
            "role": "paper_artifact_diff",
        }
    )
    records.sort(key=lambda row: str(row["portable_path"]))
    role_hash = {str(row["role"]): row["sha256"] for row in records}
    analysis = {
        "schema_version": 2,
        "status": "passed",
        "reference_runs_verified": 9,
        "inputs_materialized": 72,
        "family_replays_completed": 4,
        "canonical_assets_materialized": 27,
        "paper_assets_mapped": 28,
        "figure_assets_emitted": 12,
        "figures_regenerated": 11,
        "figures_source_missing_recorded": 1,
        "source_missing_recorded": ["figure_01"],
        "tables_regenerated": 16,
        "headline_assertion_count": 27,
        "headline_assertions_status": "passed",
        "model_inference_performed": False,
        "paper_outputs_root": "verification_root/paper_outputs",
        "artifact_inventory_count": 60,
        "artifact_inventory": records,
        "artifact_inventory_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "replay_receipt_sha256": role_hash["replay_receipt"],
        "family_replay_receipt_sha256": role_hash["family_replay_receipt"],
        "canonical_receipt_sha256": role_hash["canonical_receipt"],
        "headline_assertions_sha256": role_hash["headline_assertions"],
        "paper_artifact_diff_sha256": role_hash["paper_artifact_diff"],
        **_identity(),
    }
    _write_json(verification / "analysis_replay.json", analysis)
    _write_json(
        verification / "cpu_verification.json",
        {
            "schema_version": 1,
            "status": "passed",
            "mode": "analysis-replay",
            "steps": {"analysis_replay": analysis},
            **_identity(),
        },
    )


def _tmux_logs(root: Path) -> None:
    operations = " ".join(
        (
            "checkpoint-fingerprint",
            "runs/controlled",
            "runs/imagenet9",
            "runs/attribution_main",
            "runs/dinov2_g",
            "runs/partimagenet",
            "runs/resume_test",
        )
    )
    _write_text(root / "logs/controller.log", operations + "\n")
    _write_text(root / "logs/gpu.log", operations + "\n")
    _write_text(root / "logs/covertype_cpu.log", "runs/covertype\n")
    _write_text(root / "logs/monitor.log", "monitor active\n")
    _write_text(root / "resource_utilization/gpu.csv", "0,NVIDIA B200\n")
    _write_text(root / "resource_utilization/vmstat.log", "vmstat\n")
    _write_text(root / "resource_utilization/diskstats.log", "diskstats\n")
    _write_text(root / "resource_utilization/process_memory.log", "memory\n")


def _fixture(root: Path) -> None:
    root.mkdir()
    _tmux_logs(root)
    _cpu_restructuring(root)
    _fingerprints(root)
    _controlled(root)
    _imagenet9(root)
    _attribution(root, "attribution_main", "smoke", 2)
    _attribution(root, "dinov2_g", "large-model-smoke", 3)
    _attribution(root, "partimagenet", "boundary-smoke", 4)
    _covertype(root)
    _resume_and_scheduler(root)
    _audit_reports(root)
    _analysis(root)


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, module) -> None:
    monkeypatch.setattr(
        module,
        "_repository_identity",
        lambda _repository: module.RepositoryIdentity(
            commit="a" * 40,
            tree="b" * 40,
            branch="gpu-verification-v1",
        ),
    )
    monkeypatch.setattr(module, "_tmux_session_active", lambda _session: True)
    monkeypatch.setattr(
        module,
        "_host_environment",
        lambda: {
            "platform": "Linux-test",
            "available_cpu_count": 48,
            "host_memory_bytes": 256_000_000_000,
        },
    )
    monkeypatch.setattr(module, "_now", lambda: "2026-08-12T20:00:00Z")


def test_finalizer_emits_portable_honest_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path / "b200-verification"
    _fixture(root)
    _patch_runtime(monkeypatch, module)

    status = module.finalize_b200_verification(
        repository=tmp_path / "repository",
        verification_root=root,
    )

    assert status["status"] == "passed"
    assert all(status["acceptance_gates"].values())
    assert status["paper_results"]["figures"] == {
        "mapped": 12,
        "total": 12,
        "regenerated": 11,
        "source_missing_recorded": 1,
        "source_missing_assets": ["figure_01"],
    }
    assert status["paper_results"]["tables"]["mapped"] == 16
    assert status["paper_numeric_source"] == "sealed_historical_outputs"
    assert status["full_paper_scale_compute_rerun"] is False
    for run, expected in (("attribution_main", 72), ("dinov2_g", 16), ("partimagenet", 8)):
        shard = status["representative_shards"][run]
        assert shard["immediate_resume_skipped"] == expected
        assert shard["immediate_resume_reexecuted"] == 0
    assert status["multi_gpu_scheduler"] == {
        "static_plan": "PASS",
        "real_execution": "NOT_TESTED_SINGLE_GPU_NODE",
    }
    report = (root / "B200_VERIFICATION_REPORT.md").read_text(encoding="utf-8")
    assert "Reference paper results were regenerated from sealed historical outputs." in report
    assert "Full paper-scale computation was not rerun." in report
    provenance_text = (root / "provenance/B200_PROVENANCE.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in provenance_text
    assert not any(0x4E00 <= ord(character) <= 0x9FFF for character in provenance_text)


def test_gpu_queue_requires_one_ordered_terminal_path_per_member() -> None:
    module = _module()
    receipt = {
        "member_count": 3,
        "members": {
            "executed": {"status": "completed"},
            "resumed-a": {"status": "completed"},
            "resumed-b": {"status": "completed"},
        },
        "details": {
            "backend": "gpu",
            "endpoint_m_stage": "analyze",
            "member_count": 3,
            "plan_contract_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "data_binding_manifest_sha256": "3" * 64,
            "checkpoint_binding_manifest_sha256": "4" * 64,
            "scheduler": "single_gpu_dynamic_queue",
            "visible_device": "cuda:0",
            "exclusive_member_concurrency": 1,
            "dynamic_refill": True,
            "duplicate_execution": False,
            "multi_gpu_real_execution": "NOT_TESTED_SINGLE_GPU_NODE",
            "failures": {},
            "queue_events": [
                {"member_id": "executed", "event": "start", "device": 0},
                {"member_id": "executed", "event": "completed", "device": 0},
                {"member_id": "resumed-a", "event": "resume_skip", "device": 0},
                {"member_id": "resumed-b", "event": "resume_skip", "device": 0},
            ],
        },
    }

    assert module._validate_gpu_queue(receipt, "test", require_resume_mix=True) == (2, 1)

    duplicate_terminal = json.loads(json.dumps(receipt))
    duplicate_terminal["details"]["queue_events"].append(
        {"member_id": "executed", "event": "completed", "device": 0}
    )
    with pytest.raises(module.FinalizationError, match="exactly one terminal"):
        module._validate_gpu_queue(duplicate_terminal, "test")

    missing_start = json.loads(json.dumps(receipt))
    del missing_start["details"]["queue_events"][0]
    with pytest.raises(module.FinalizationError, match="ordered start and terminal"):
        module._validate_gpu_queue(missing_start, "test")

    resumed_with_start = json.loads(json.dumps(receipt))
    resumed_with_start["details"]["queue_events"].insert(
        2, {"member_id": "resumed-a", "event": "start", "device": 0}
    )
    with pytest.raises(module.FinalizationError, match="must not have a start"):
        module._validate_gpu_queue(resumed_with_start, "test")


def test_nonrenormalized_imagenet9_probability_mass_is_validated() -> None:
    module = _module()
    observed = module._validate_subprobabilities(
        [[0.1, 0.2, 0.3]], "mapped ImageNet-9 probabilities"
    )
    assert observed == [[0.1, 0.2, 0.3]]
    with pytest.raises(module.FinalizationError, match="sub-probability"):
        module._validate_subprobabilities([[0.5, 0.6]], "mapped ImageNet-9 probabilities")


def test_large_model_timing_members_expect_one_aggregate_row(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "b200-verification"
    _fixture(root)
    plan_path = root / "runs/dinov2_g/manifests/plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    timing_job = plan["members"][0]
    timing_job["kind"] = "large_model_timing"
    timing_job["image_stop"] = 8
    _write_json(plan_path, plan)

    global_receipt = json.loads(
        (root / "runs/dinov2_g/receipts/compute_members.json").read_text(encoding="utf-8")
    )
    spec = next(spec for spec in module.RUN_SPECS if spec.key == "dinov2_g")
    module._validate_attribution_member_bindings(module.Evidence(root), spec, plan, global_receipt)


def test_finalizer_rejects_tampered_analysis_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path / "b200-verification"
    _fixture(root)
    _patch_runtime(monkeypatch, module)
    analysis_path = root / "verification/analysis_replay.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["figures_regenerated"] = 12
    _write_json(analysis_path, analysis)
    wrapper_path = root / "verification/cpu_verification.json"
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    wrapper["steps"]["analysis_replay"] = analysis
    _write_json(wrapper_path, wrapper)

    with pytest.raises(module.FinalizationError, match="figures_regenerated"):
        module.finalize_b200_verification(
            repository=tmp_path / "repository",
            verification_root=root,
        )

    assert not (root / "B200_VERIFICATION_STATUS.json").exists()
    assert not (root / "B200_VERIFICATION_REPORT.md").exists()
    assert not (root / "provenance/B200_PROVENANCE.json").exists()


def test_finalizer_rejects_analysis_wrapper_from_another_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path / "b200-verification"
    _fixture(root)
    _patch_runtime(monkeypatch, module)
    wrapper_path = root / "verification/cpu_verification.json"
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    wrapper["steps"]["analysis_replay"]["reference_runs_verified"] = 8
    _write_json(wrapper_path, wrapper)

    with pytest.raises(module.FinalizationError, match="CLI wrapper differs"):
        module.finalize_b200_verification(
            repository=tmp_path / "repository",
            verification_root=root,
        )
