#!/usr/bin/env python3
"""Fail-closed finalization of the portable single-B200 verification evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any

REQUIRED_STAGES = ("prepare", "compute", "analyze", "paper")
REQUIRED_FINGERPRINT_COVERAGE = {"controlled": 2, "imagenet9": 3, "attribution": 7}
REQUIRED_FINGERPRINT_CHECKS = {
    "exact_case_coverage",
    "checkpoint_bytes_verified",
    "preprocessed_tensor_hashes_recorded",
    "finite_logits",
    "normalized_probabilities",
    "single_b200",
}
REQUIRED_FINGERPRINT_LIBRARIES = {"torch", "torchvision", "timm", "captum", "numpy"}
REQUIRED_CONTROLLED_FINGERPRINTS = {
    (
        "controlled__base_resnet18_object_shape",
        "object_shape__resnet18__seed_3101",
        "resnet18",
    ),
    (
        "controlled__base_small_vit_object_shape",
        "object_shape__small_vit__seed_3101",
        "small_vit",
    ),
}
REQUIRED_IMAGENET9_FINGERPRINTS = {
    (
        "imagenet9_off_the_shelf",
        "tv_resnet18_imagenet1k_v1",
        "off_the_shelf",
        "cnn",
    ),
    (
        "imagenet9_finetuned_cnn",
        "ft_resnet50_original_s7101",
        "fine_tuned",
        "cnn",
    ),
    (
        "imagenet9_finetuned_transformer",
        "ft_vit_b_16_original_s7101",
        "fine_tuned",
        "transformer",
    ),
}
REQUIRED_ATTRIBUTION_FINGERPRINTS = {
    ("attribution/funnybirds/funnybirds_resnet50", "funnybirds_resnet50", "funnybirds"),
    ("attribution/funnybirds/funnybirds_vgg16", "funnybirds_vgg16", "funnybirds"),
    ("attribution/funnybirds/funnybirds_vit_b_16", "funnybirds_vit_b_16", "funnybirds"),
    ("attribution/imagenet1k_idsds/resnet50", "resnet50", "imagenet1k_idsds"),
    ("attribution/imagenet1k_idsds/vgg16", "vgg16", "imagenet1k_idsds"),
    (
        "attribution/imagenet1k_idsds/vit_base_patch16_224",
        "vit_base_patch16_224",
        "imagenet1k_idsds",
    ),
    (
        "attribution/imagenet1k_idsds/dinov2_vit_g_14",
        "dinov2_vit_g_14",
        "imagenet1k_idsds",
    ),
}
REQUIRED_ATTRIBUTION_CHECKPOINTS = {
    ("funnybirds_resnet50", "funnybirds"): ("funnybirds_resnet",),
    ("funnybirds_vgg16", "funnybirds"): ("funnybirds_vgg",),
    ("funnybirds_vit_b_16", "funnybirds"): ("funnybirds_vit",),
    ("resnet50", "imagenet1k_idsds"): ("idsds_resnet50",),
    ("vgg16", "imagenet1k_idsds"): ("idsds_vgg16",),
    ("vit_base_patch16_224", "imagenet1k_idsds"): ("idsds_vit_base_patch16_224",),
    ("dinov2_vit_g_14", "imagenet1k_idsds"): (
        "dinov2_vitg14_backbone",
        "dinov2_vitg14_linear_head",
    ),
}
REQUIRED_ATTRIBUTION_MEMBER_COUNTS = {
    "attribution_main": 72,
    "dinov2_g": 16,
    "partimagenet": 8,
    "resume_test": 5,
}
REQUIRED_FULL_PYTEST_COMMAND = ("python", "-m", "pytest")
REQUIRED_FULL_PYTEST_ENVIRONMENT_MODE = "cpu_oracle_with_pinned_real_assets"
REQUIRED_FULL_PYTEST_ASSETS = {
    "covertype_archive",
    "idsds_manifest",
    "reference_run_archives",
}
REQUIRED_ANALYSIS_VALUES: dict[str, Any] = {
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
}
REQUIRED_SCHEDULER_CHECKS = {
    "heterogeneous_member_queue",
    "no_duplicate_execution",
    "dynamic_refill_gpu0",
    "unique_output_paths",
    "unique_receipts",
    "member_failure_isolation",
    "global_receipt_finalization",
}
REQUIRED_FAULT_CHECKS = {
    "normal_sigterm_used",
    "terminalized_without_running",
    "completed_members_skipped",
    "incomplete_members_finished",
    "final_status_completed",
}
PRIVATE_PATH_FRAGMENTS = (
    "/" + "work" + "/" + "Users" + "/",
    "/" + "home" + "/",
    "/" + "Users" + "/",
    "/" + "mnt" + "/",
    "/" + "tmp" + "/",
    "C:" + "\\" + "Users" + "\\",
)


class FinalizationError(RuntimeError):
    """Raised when required evidence is absent, stale, or contradictory."""


@dataclass(frozen=True)
class RepositoryIdentity:
    commit: str
    tree: str
    branch: str


@dataclass(frozen=True)
class RunSpec:
    key: str
    experiment: str
    profile: str
    scopes: tuple[str, ...] = ()
    gpu: bool = True


RUN_SPECS = (
    RunSpec("controlled", "controlled", "smoke"),
    RunSpec("imagenet9", "imagenet9", "smoke"),
    RunSpec(
        "attribution_main",
        "attribution",
        "smoke",
        (
            "smoke_idsds_deletion_targets",
            "smoke_idsds_primary",
            "smoke_funnybirds_deletion_targets",
            "smoke_funnybirds_heldout_targets",
            "smoke_funnybirds_primary",
        ),
    ),
    RunSpec(
        "dinov2_g",
        "attribution",
        "large-model-smoke",
        ("smoke_dinov2_g_quality", "smoke_dinov2_g_timing"),
    ),
    RunSpec(
        "partimagenet",
        "attribution",
        "boundary-smoke",
        (
            "smoke_partimagenet_deletion_targets",
            "smoke_partimagenet_heldout_targets",
            "smoke_partimagenet_boundary",
        ),
    ),
    RunSpec("covertype", "covertype", "smoke", gpu=False),
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise FinalizationError(f"unsafe evidence path: {value!r}")
    return value


class Evidence:
    """Read verification evidence while building a portable hash inventory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise FinalizationError(f"verification root is not a directory: {self.root}")
        self._records: dict[str, dict[str, Any]] = {}

    def file(self, relative: str) -> Path:
        relative = _safe_relative(relative)
        path = self.root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise FinalizationError(
                f"required evidence is missing, empty, or a symlink: {relative}"
            )
        try:
            path.resolve().relative_to(self.root)
        except ValueError as error:
            raise FinalizationError(f"evidence escapes verification root: {relative}") from error
        self._records[relative] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        return path

    def json(self, relative: str) -> dict[str, Any]:
        path = self.file(relative)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FinalizationError(f"evidence is not a UTF-8 JSON document: {relative}") from error
        if not isinstance(payload, dict):
            raise FinalizationError(f"evidence JSON is not an object: {relative}")
        return payload

    def text(self, relative: str) -> str:
        path = self.file(relative)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise FinalizationError(f"evidence is not UTF-8 text: {relative}") from error

    def inventory(self) -> list[dict[str, Any]]:
        return [self._records[key] for key in sorted(self._records)]


def _command(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        arguments,
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _repository_identity(repository: Path) -> RepositoryIdentity:
    if not (repository / ".git").exists():
        raise FinalizationError("repository is not a Git worktree")
    status = _command(
        repository,
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise FinalizationError("final verification requires a clean repository worktree")
    identity = RepositoryIdentity(
        commit=_command(repository, "git", "rev-parse", "HEAD"),
        tree=_command(repository, "git", "rev-parse", "HEAD^{tree}"),
        branch=_command(repository, "git", "branch", "--show-current"),
    )
    if identity.branch != "gpu-verification-v1":
        raise FinalizationError("final verification must run on the gpu-verification-v1 branch")
    if not all(len(value) in {40, 64} for value in (identity.commit, identity.tree)):
        raise FinalizationError("repository commit/tree identity is malformed")
    return identity


def _tmux_session_active(session: str) -> bool:
    process = subprocess.run(
        ("tmux", "has-session", "-t", session),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process.returncode == 0


def _host_environment() -> dict[str, Any]:
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 0
    memory_bytes = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            memory_bytes = int(line.split()[1]) * 1024
            break
    if cpu_count < 1 or memory_bytes < 1:
        raise FinalizationError("host CPU or memory inventory is unavailable")
    return {
        "platform": platform.platform(),
        "available_cpu_count": cpu_count,
        "host_memory_bytes": memory_bytes,
    }


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalizationError(f"{label} must be an object")
    return value


def _require_all_true(value: object, label: str, required: set[str] | None = None) -> None:
    checks = _require_mapping(value, label)
    if not checks:
        raise FinalizationError(f"{label} is empty")
    if required is not None and not required.issubset(checks):
        raise FinalizationError(f"{label} omits checks: {sorted(required - set(checks))}")
    failed = sorted(key for key, passed in checks.items() if passed is not True)
    if failed:
        raise FinalizationError(f"{label} contains non-passing checks: {failed}")


def _numeric_matrix(value: object, label: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise FinalizationError(f"{label} must be a non-empty matrix")
    result: list[list[float]] = []
    width: int | None = None
    for raw_row in value:
        if not isinstance(raw_row, list) or not raw_row:
            raise FinalizationError(f"{label} must be a non-empty rectangular matrix")
        row: list[float] = []
        for raw_value in raw_row:
            if isinstance(raw_value, bool):
                raise FinalizationError(f"{label} contains a boolean value")
            try:
                number = float(raw_value)
            except (TypeError, ValueError, OverflowError) as error:
                raise FinalizationError(f"{label} contains a non-numeric value") from error
            if not math.isfinite(number):
                raise FinalizationError(f"{label} contains a non-finite value")
            row.append(number)
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise FinalizationError(f"{label} is not rectangular")
        result.append(row)
    return result


def _validate_probabilities(value: object, label: str) -> list[list[float]]:
    probabilities = _numeric_matrix(value, label)
    for row in probabilities:
        if any(number < 0.0 or number > 1.0 for number in row) or not math.isclose(
            sum(row), 1.0, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise FinalizationError(f"{label} is not row-normalized")
    return probabilities


def _validate_subprobabilities(value: object, label: str) -> list[list[float]]:
    """Validate a non-renormalized probability projection with mass at most one."""

    probabilities = _numeric_matrix(value, label)
    for row in probabilities:
        total = sum(row)
        if (
            any(number < 0.0 or number > 1.0 for number in row)
            or total <= 0.0
            or total > 1.0 + 1.0e-6
        ):
            raise FinalizationError(f"{label} is not a valid sub-probability mass")
    return probabilities


def _utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise FinalizationError(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise FinalizationError(f"{label} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):  # noqa: UP017
        raise FinalizationError(f"{label} is not normalized to UTC")
    return parsed


def _require_bound(report: Mapping[str, Any], identity: RepositoryIdentity, label: str) -> None:
    if report.get("repository_commit") != identity.commit:
        raise FinalizationError(f"{label} is not bound to the final repository commit")
    if report.get("repository_tree") != identity.tree:
        raise FinalizationError(f"{label} is not bound to the final repository tree")
    if report.get("tracked_worktree_clean") is not True:
        raise FinalizationError(f"{label} was not produced from a clean tracked worktree")


def _validate_stage_receipts(evidence: Evidence, run: str) -> None:
    for stage in REQUIRED_STAGES:
        receipt = evidence.json(f"runs/{run}/receipts/{stage}.json")
        if receipt.get("stage") != stage or receipt.get("status") != "completed":
            raise FinalizationError(f"{run} {stage} stage has not completed")


def _validate_artifact(
    evidence: Evidence,
    run: str,
    relative: object,
    digest: object,
    size: object | None = None,
) -> None:
    if not isinstance(relative, str) or not _is_sha256(digest):
        raise FinalizationError(f"{run} member artifact contract is malformed")
    path = evidence.file(f"runs/{run}/{_safe_relative(relative)}")
    if _sha256(path) != digest:
        raise FinalizationError(f"{run} member artifact hash differs: {relative}")
    if size is not None and path.stat().st_size != size:
        raise FinalizationError(f"{run} member artifact size differs: {relative}")


def _validate_member_outputs(
    evidence: Evidence, run: str, global_receipt: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    members = _require_mapping(global_receipt.get("members"), f"{run} global members")
    if global_receipt.get("member_count") != len(members) or not members:
        raise FinalizationError(f"{run} global member inventory is incomplete")
    if any(
        not isinstance(value, Mapping) or value.get("status") != "completed"
        for value in members.values()
    ):
        raise FinalizationError(f"{run} has a non-completed member")
    receipts_root = evidence.root / "runs" / run / "receipts" / "members"
    receipt_paths = sorted(receipts_root.rglob("*.json")) if receipts_root.is_dir() else []
    by_id: dict[str, dict[str, Any]] = {}
    for path in receipt_paths:
        relative = path.relative_to(evidence.root).as_posix()
        receipt = evidence.json(relative)
        member_id = receipt.get("member_id")
        if (
            not isinstance(member_id, str)
            or member_id in by_id
            or receipt.get("kind") != "member"
            or receipt.get("status") != "completed"
            or receipt.get("error") is not None
        ):
            raise FinalizationError(f"{run} member receipt is malformed: {relative}")
        details = _require_mapping(receipt.get("details"), f"{run} member details")
        validated = 0
        artifacts = details.get("artifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                record = _require_mapping(artifact, f"{run} member artifact")
                _validate_artifact(
                    evidence,
                    run,
                    record.get("path"),
                    record.get("sha256"),
                    record.get("bytes"),
                )
                validated += 1
        if "output_path" in details:
            _validate_artifact(
                evidence,
                run,
                details.get("output_path"),
                details.get("output_sha256"),
            )
            validated += 1
        if "artifact" in details:
            _validate_artifact(
                evidence,
                run,
                details.get("artifact"),
                details.get("artifact_sha256"),
                details.get("artifact_size_bytes"),
            )
            validated += 1
        if validated != 1:
            raise FinalizationError(
                f"{run} member must bind exactly one persisted output: {member_id}"
            )
        by_id[member_id] = receipt
    if set(by_id) != set(members):
        raise FinalizationError(f"{run} member receipts differ from the global inventory")
    return by_id


def _validate_global_receipt(evidence: Evidence, run: str) -> dict[str, Any]:
    receipt = evidence.json(f"runs/{run}/receipts/compute_members.json")
    if (
        receipt.get("kind") != "global"
        or receipt.get("status") != "completed"
        or receipt.get("all_processes_exited") is not True
    ):
        raise FinalizationError(f"{run} global compute receipt is not terminal and complete")
    _validate_member_outputs(evidence, run, receipt)
    return receipt


def _validate_controlled(evidence: Evidence) -> dict[str, Any]:
    summary = evidence.json("runs/controlled/metrics/controlled_smoke_summary.json")
    if (
        summary.get("status") != "completed"
        or summary.get("scope") != "real_cuda_single_b200_shard"
        or summary.get("gpu_real_shard_verification") != "passed"
        or not _is_sha256(summary.get("metrics_sha256"))
    ):
        raise FinalizationError("Controlled summary does not verify a real B200 shard")
    metrics_relative = "runs/controlled/metrics/controlled_smoke_metrics.csv"
    metrics_path = evidence.file(metrics_relative)
    if _sha256(metrics_path) != summary.get("metrics_sha256"):
        raise FinalizationError("Controlled metrics hash differs from its summary")
    metrics_text = metrics_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(StringIO(metrics_text)))
    if len(rows) != summary.get("rows"):
        raise FinalizationError("Controlled metrics row count differs from its summary")
    required_behaviors = {"active", "null", "aligned", "opposed"}
    if {row.get("expected_behavior") for row in rows} != required_behaviors:
        raise FinalizationError("Controlled behavior coverage is incomplete")
    if {row.get("architecture") for row in rows} != {"resnet18", "small_vit"}:
        raise FinalizationError("Controlled architecture coverage is incomplete")
    for row in rows:
        try:
            value = float(str(row.get("value")))
        except ValueError as error:
            raise FinalizationError("Controlled metrics contain a non-numeric value") from error
        if not math.isfinite(value) or row.get("gpu_verification") != "passed":
            raise FinalizationError("Controlled metrics contain unverified/non-finite values")
        if row.get("metric") in {"E", "C", "F"} and value < 0.0:
            raise FinalizationError("Controlled E/C/F metrics must be nonnegative")
    observed: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((evidence.root / "runs/controlled/raw").rglob("*.json")):
        relative = path.relative_to(evidence.root).as_posix()
        payload = evidence.json(relative)
        behavior = payload.get("expected_behavior")
        counts = _require_mapping(
            payload.get("observed_behaviors"), "Controlled observed behaviors"
        )
        execution = _require_mapping(payload.get("execution"), "Controlled execution")
        device = _require_mapping(execution.get("device"), "Controlled CUDA device")
        numeric = _require_mapping(payload.get("numeric_audit"), "Controlled numeric audit")
        if (
            execution.get("backend") != "cuda"
            or execution.get("gpu_verification") != "passed"
            or device.get("resolved") != "cuda:0"
            or device.get("count_visible") != 1
            or "B200" not in str(device.get("name"))
            or numeric.get("passed") is not True
            or numeric.get("finite_model_scores") is not True
            or numeric.get("nonnegative_ecf") is not True
        ):
            raise FinalizationError("Controlled raw CUDA/numeric audit has not passed")
        for audit_name in (
            "pointwise_conservation",
            "integrated_conservation",
            "tiny_endpoint_swap",
        ):
            audit = _require_mapping(numeric.get(audit_name), f"Controlled {audit_name} audit")
            if audit.get("passed") is not True:
                raise FinalizationError(f"Controlled {audit_name} audit has not passed")
        if behavior not in required_behaviors:
            raise FinalizationError("Controlled behavior record is unknown")
        observed.setdefault(str(behavior), []).append(dict(counts))
        sample_ids = _require_mapping(payload.get("sample_ids"), "Controlled sample IDs")
        expected_id_keys = (
            {"endpoint_cf_ids", "endpoint_fact_ids", "swap_cf_ids", "swap_fact_ids"}
            if behavior == "opposed"
            else {"counterfactual_ids", "factual_ids"}
        )
        if (
            not expected_id_keys.issubset(sample_ids)
            or any(
                not isinstance(sample_ids[key], list) or len(sample_ids[key]) != 8
                for key in expected_id_keys
            )
            or counts.get(str(behavior)) != 8
        ):
            raise FinalizationError(
                f"Controlled {behavior} behavior was not observed for all samples"
            )
    if (
        set(observed) != required_behaviors
        or len(observed["active"]) != 2
        or any(len(observed[name]) != 1 for name in ("null", "aligned", "opposed"))
    ):
        raise FinalizationError("Controlled raw behavior evidence is incomplete")
    return {"member_count": 5, "scope": summary["scope"]}


def _validate_imagenet9(evidence: Evidence, global_receipt: Mapping[str, Any]) -> dict[str, Any]:
    plan = evidence.json("runs/imagenet9/manifests/plan.json")
    counts = _require_mapping(plan.get("counts"), "ImageNet-9 scientific counts")
    required_counts = {
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
    }
    scientific = _require_mapping(plan.get("scientific_contract"), "ImageNet-9 scientific contract")
    if (
        plan.get("verification_mode") != "single_b200_real_cuda"
        or plan.get("execution_class") != "real_cuda"
        or any(counts.get(key) != value for key, value in required_counts.items())
        or len(plan.get("members", [])) != 24
        or len(plan.get("models", [])) != 3
        or scientific.get("official_probability_mapping")
        != "softmax_1000_once_then_sum_mapped_mass"
        or scientific.get("mapped_mass_renormalized") is not False
        or scientific.get("second_softmax") is not False
        or scientific.get("fine_tuned_probability_adapter") != "direct_9_way_softmax_once"
        or scientific.get("variable_size_preprocessing") != "per_image_before_batch_coalescing"
        or scientific.get("paired_randomness") != "shared_within_each_factual_counterfactual_pair"
    ):
        raise FinalizationError("ImageNet-9 B200 plan contract differs")
    details = _require_mapping(global_receipt.get("details"), "ImageNet-9 global details")
    if details.get("verification_mode") != "single_b200_real_cuda":
        raise FinalizationError("ImageNet-9 global receipt is not a real CUDA receipt")
    aggregate = evidence.json("runs/imagenet9/receipts/imagenet9_b200_compute.json")
    if (
        aggregate.get("status") != "completed"
        or aggregate.get("verification_mode") != "single_b200_real_cuda"
        or aggregate.get("gpu_inference_verified") is not True
    ):
        raise FinalizationError("ImageNet-9 aggregate compute receipt has not passed")
    artifacts = _require_mapping(aggregate.get("artifacts"), "ImageNet-9 artifacts")
    if len(artifacts) != 3:
        raise FinalizationError("ImageNet-9 aggregate must bind exactly three artifacts")
    for relative, digest in artifacts.items():
        _validate_artifact(evidence, "imagenet9", relative, digest)
    summary = evidence.json("runs/imagenet9/metrics/summary.json")
    if (
        summary.get("gpu_inference_verified") is not True
        or summary.get("source_mode") != "computed_run"
        or summary.get("model_count") != 3
        or summary.get("score_rows") != 288
        or summary.get("baseline_rows") != 480
        or summary.get("reveal_paths") != ["blend", "patch_A", "patch_B"]
        or summary.get("baseline_methods")
        != [
            "input_x_gradient",
            "integrated_gradients",
            "occlusion",
            "rise",
            "smoothgrad",
        ]
    ):
        raise FinalizationError("ImageNet-9 analysis did not preserve GPU verification")
    compute = evidence.json("runs/imagenet9/receipts/compute.json")
    compute_details = _require_mapping(compute.get("details"), "ImageNet-9 compute details")
    if (
        compute_details.get("gpu_inference_executed") is not True
        or "B200" not in str(compute_details.get("gpu_name"))
        or compute_details.get("source") != "real_backgrounds_challenge_cuda"
        or compute_details.get("completed_member_receipts") != 24
        or compute_details.get("trajectory_count") != 288
        or compute_details.get("baseline_rows") != 480
    ):
        raise FinalizationError("ImageNet-9 compute stage does not prove real B200 work")
    return {"member_count": 24, "scope": "single_b200_real_cuda"}


def _validate_gpu_queue(
    receipt: Mapping[str, Any], label: str, *, require_resume_mix: bool = False
) -> tuple[int, int]:
    details = _require_mapping(receipt.get("details"), f"{label} queue details")
    failures = _require_mapping(details.get("failures"), f"{label} queue failures")
    events = details.get("queue_events")
    if (
        details.get("backend") != "gpu"
        or details.get("scheduler") != "single_gpu_dynamic_queue"
        or details.get("visible_device") != "cuda:0"
        or details.get("exclusive_member_concurrency") != 1
        or details.get("dynamic_refill") is not True
        or details.get("duplicate_execution") is not False
        or details.get("member_count") != receipt.get("member_count")
        or details.get("endpoint_m_stage") != "analyze"
        or not _is_sha256(details.get("plan_contract_sha256"))
        or not _is_sha256(details.get("config_sha256"))
        or not _is_sha256(details.get("data_binding_manifest_sha256"))
        or not _is_sha256(details.get("checkpoint_binding_manifest_sha256"))
        or details.get("multi_gpu_real_execution") != "NOT_TESTED_SINGLE_GPU_NODE"
        or failures
        or not isinstance(events, list)
    ):
        raise FinalizationError(f"{label} single-GPU queue contract differs")
    members = _require_mapping(receipt.get("members"), f"{label} queue members")
    if receipt.get("member_count") != len(members) or not members:
        raise FinalizationError(f"{label} queue member inventory differs")
    by_member: dict[str, list[str]] = {str(member_id): [] for member_id in members}
    allowed_events = {"start", "completed", "resume_skip"}
    for raw in events:
        event = _require_mapping(raw, f"{label} queue event")
        member_id = event.get("member_id")
        event_name = event.get("event")
        if (
            not isinstance(member_id, str)
            or member_id not in by_member
            or event_name not in allowed_events
            or event.get("device") != 0
        ):
            raise FinalizationError(f"{label} queue event inventory differs")
        by_member[member_id].append(str(event_name))

    counts: Counter[str] = Counter()
    for member_id, member_events in by_member.items():
        terminals = [event for event in member_events if event in {"completed", "resume_skip"}]
        if len(terminals) != 1:
            raise FinalizationError(
                f"{label} member must have exactly one terminal event: {member_id}"
            )
        terminal = terminals[0]
        if terminal == "resume_skip":
            if member_events != ["resume_skip"]:
                raise FinalizationError(
                    f"{label} resumed member must not have a start event: {member_id}"
                )
        elif member_events != ["start", "completed"]:
            raise FinalizationError(
                f"{label} executed member must have one ordered start and terminal: {member_id}"
            )
        counts[terminal] += 1
    if counts["resume_skip"] + counts["completed"] != len(members):
        raise FinalizationError(f"{label} queue terminal inventory differs")
    if require_resume_mix and (counts["resume_skip"] < 2 or counts["completed"] < 1):
        raise FinalizationError("fault-injection resume did not skip and complete members")
    return counts["resume_skip"], counts["completed"]


def _validate_attribution_member_bindings(
    evidence: Evidence,
    spec: RunSpec,
    plan: Mapping[str, Any],
    global_receipt: Mapping[str, Any],
) -> None:
    members = plan.get("members")
    if (
        not isinstance(members, list)
        or len(members) != REQUIRED_ATTRIBUTION_MEMBER_COUNTS[spec.key]
    ):
        raise FinalizationError(f"{spec.key} attribution member count differs")
    jobs: dict[str, Mapping[str, Any]] = {}
    outputs: set[str] = set()
    receipts: set[str] = set()
    for raw in members:
        job = _require_mapping(raw, f"{spec.key} attribution job")
        member_id = job.get("member_id")
        output_path = job.get("output_path")
        receipt_path = job.get("receipt_path")
        if (
            not isinstance(member_id, str)
            or member_id in jobs
            or not isinstance(output_path, str)
            or not isinstance(receipt_path, str)
            or output_path in outputs
            or receipt_path in receipts
            or not _is_sha256(job.get("job_sha256"))
            or job.get("scope") not in spec.scopes
            or not isinstance(job.get("image_start"), int)
            or not isinstance(job.get("image_stop"), int)
            or job.get("image_stop", 0) <= job.get("image_start", -1)
        ):
            raise FinalizationError(f"{spec.key} attribution job inventory is malformed")
        jobs[member_id] = job
        outputs.add(output_path)
        receipts.add(receipt_path)
    if set(global_receipt.get("members", {})) != set(jobs):
        raise FinalizationError(f"{spec.key} global member inventory differs from its plan")
    global_details = _require_mapping(
        global_receipt.get("details"), f"{spec.key} attribution global details"
    )
    if global_details.get("plan_contract_sha256") != plan.get(
        "plan_contract_sha256"
    ) or global_details.get("config_sha256") != plan.get("config_sha256"):
        raise FinalizationError(f"{spec.key} global receipt is not bound to its plan")
    persisted = _validate_member_outputs(evidence, spec.key, global_receipt)
    for member_id, job in jobs.items():
        receipt = persisted[member_id]
        details = _require_mapping(receipt.get("details"), f"{spec.key} member details")
        expected_rows = (
            1
            if job.get("kind") in {"timing", "large_model_timing"}
            else job["image_stop"] - job["image_start"]
        )
        if (
            details.get("output_path") != job["output_path"]
            or details.get("scope") != job.get("scope")
            or details.get("dataset") != job.get("dataset")
            or details.get("model_id") != job.get("model_id")
            or details.get("method_id") != job.get("method_id")
            or details.get("job_sha256") != job.get("job_sha256")
            or details.get("config_sha256") != plan.get("config_sha256")
            or details.get("plan_contract_sha256") != plan.get("plan_contract_sha256")
            or not _is_sha256(details.get("input_manifest_sha256"))
            or not _is_sha256(details.get("checkpoint_bytes_sha256"))
            or details.get("data_binding_manifest_sha256")
            != global_details.get("data_binding_manifest_sha256")
            or details.get("checkpoint_binding_manifest_sha256")
            != global_details.get("checkpoint_binding_manifest_sha256")
            or details.get("rows") != expected_rows
            or not isinstance(details.get("columns"), list)
            or not details.get("columns")
        ):
            raise FinalizationError(f"{spec.key} member lineage differs: {member_id}")


def _validate_attribution_resume(
    evidence: Evidence,
    spec: RunSpec,
    global_receipt: Mapping[str, Any],
) -> tuple[int, int]:
    plan = evidence.json(f"runs/{spec.key}/manifests/plan.json")
    skipped, completed = _validate_gpu_queue(global_receipt, spec.key)
    compute = evidence.json(f"runs/{spec.key}/receipts/compute.json")
    details = _require_mapping(compute.get("details"), f"{spec.key} compute details")
    expected = REQUIRED_ATTRIBUTION_MEMBER_COUNTS[spec.key]
    if (
        compute.get("status") != "completed"
        or details.get("backend") != "gpu"
        or details.get("scheduler") != "single_gpu_dynamic_queue"
        or details.get("member_count") != expected
        or details.get("failed_members") != 0
        or details.get("completed_members") != completed
        or details.get("resumed_members") != skipped
        or completed + skipped != expected
    ):
        raise FinalizationError(f"{spec.key} original compute receipt differs from its queue")
    resume = evidence.json(f"runs/{spec.key}/receipts/resume/compute.json")
    resume_started = _utc_timestamp(
        resume.get("validation_started_at"), f"{spec.key} resume validation_started_at"
    )
    resume_finished = _utc_timestamp(
        resume.get("validation_finished_at"), f"{spec.key} resume validation_finished_at"
    )
    source_compute = evidence.file(f"runs/{spec.key}/receipts/compute.json")
    persisted = _validate_member_outputs(evidence, spec.key, global_receipt)
    member_finished = [
        _utc_timestamp(receipt.get("finished_at"), f"{spec.key} member finished_at")
        for receipt in persisted.values()
    ]
    inventory_records = [
        {
            "member_id": member_id,
            "output_path": receipt["details"]["output_path"],
            "output_sha256": receipt["details"]["output_sha256"],
            "receipt_path": str(
                next(
                    job["receipt_path"] for job in plan["members"] if job["member_id"] == member_id
                )
            ),
        }
        for member_id, receipt in sorted(persisted.items())
    ]
    if (
        resume.get("schema_version") != 1
        or resume.get("stage") != "compute"
        or resume.get("status") != "completed"
        or resume_finished < resume_started
        or any(resume_started <= finished for finished in member_finished)
        or resume.get("member_count") != expected
        or resume.get("resumed_members") != expected
        or resume.get("reexecuted") != 0
        or resume.get("source_compute_receipt_sha256") != _sha256(source_compute)
        or resume.get("artifact_inventory_sha256") != _canonical_sha256(inventory_records)
    ):
        raise FinalizationError(f"{spec.key} immediate resume validation receipt differs")
    return int(resume["resumed_members"]), int(resume["reexecuted"])


def _validate_attribution(
    evidence: Evidence, spec: RunSpec, global_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    plan = evidence.json(f"runs/{spec.key}/manifests/plan.json")
    if (
        plan.get("experiment") != "attribution"
        or tuple(plan.get("scope_names", ())) != spec.scopes
        or plan.get("member_count") != plan.get("expected_member_count")
        or plan.get("member_count") != global_receipt.get("member_count")
        or plan.get("member_count") != REQUIRED_ATTRIBUTION_MEMBER_COUNTS[spec.key]
        or plan.get("endpoint_m_stage") != "analyze"
    ):
        raise FinalizationError(f"{spec.key} attribution plan contract differs")
    _validate_attribution_member_bindings(evidence, spec, plan, global_receipt)
    resumed, reexecuted = _validate_attribution_resume(evidence, spec, global_receipt)
    return {
        "member_count": global_receipt["member_count"],
        "immediate_resume_skipped": resumed,
        "immediate_resume_reexecuted": reexecuted,
        "scope": "real_cuda_single_b200_shard",
    }


def _validate_covertype(evidence: Evidence) -> dict[str, Any]:
    summary = evidence.json("runs/covertype/metrics/analysis_summary.json")
    if (
        summary.get("source_mode") != "computed_run"
        or summary.get("all_decaf_identities_passed") is not True
        or not isinstance(summary.get("module_c_models"), int)
        or not isinstance(summary.get("module_f_models"), int)
        or summary.get("module_c_models", 0) < 1
        or summary.get("module_f_models", 0) < 1
    ):
        raise FinalizationError("Covertype computed-run analysis has not passed")
    rows = list(csv.DictReader(StringIO(evidence.text("runs/covertype/metrics/model_results.csv"))))
    required_columns = {
        "E",
        "C",
        "F",
        "preserve_rate",
        "invert_rate",
        "null_context_prediction_change_rate",
        "decaf_identity_passed",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise FinalizationError("Covertype canonical analysis columns are incomplete")
    if any(row.get("decaf_identity_passed") != "True" for row in rows):
        raise FinalizationError("Covertype contains a failed DECAF identity")
    return {"member_count": summary["model_count"], "scope": "real_cpu_shard"}


def _validate_runs(evidence: Evidence) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    gpu_intervals: list[tuple[str, str, str]] = []
    for spec in RUN_SPECS:
        run = evidence.json(f"runs/{spec.key}/run.json")
        if (
            run.get("status") != "completed"
            or run.get("experiment") != spec.experiment
            or run.get("profile") != spec.profile
            or run.get("requested_stage") != "all"
            or tuple(run.get("completed_stages", ())) != REQUIRED_STAGES
        ):
            raise FinalizationError(f"{spec.key} run receipt is incomplete or misidentified")
        environment = evidence.json(f"runs/{spec.key}/environment.json")
        if (
            not isinstance(environment.get("available_cpus"), int)
            or environment.get("available_cpus", 0) < 1
            or not isinstance(environment.get("python"), str)
        ):
            raise FinalizationError(f"{spec.key} environment receipt is malformed")
        _validate_stage_receipts(evidence, spec.key)
        global_receipt = _validate_global_receipt(evidence, spec.key)
        if spec.key == "controlled":
            details = _validate_controlled(evidence)
        elif spec.key == "imagenet9":
            details = _validate_imagenet9(evidence, global_receipt)
        elif spec.experiment == "attribution":
            details = _validate_attribution(evidence, spec, global_receipt)
        else:
            details = _validate_covertype(evidence)
        result[spec.key] = {"status": "PASS", **details}
        if spec.gpu:
            started = run.get("started_at")
            finished = run.get("finished_at")
            if not isinstance(started, str) or not isinstance(finished, str) or started > finished:
                raise FinalizationError(f"{spec.key} GPU run interval is malformed")
            gpu_intervals.append((spec.key, started, finished))
    for previous, current in zip(gpu_intervals, gpu_intervals[1:], strict=False):
        if previous[2] > current[1]:
            raise FinalizationError(
                f"GPU shards overlap or violate required order: {previous[0]}, {current[0]}"
            )
    return result


def _validate_controlled_resume(evidence: Evidence) -> None:
    report = evidence.json("verification/controlled_resume.json")
    members = _require_mapping(report.get("members"), "Controlled resume members")
    if (
        report.get("status") != "passed"
        or report.get("family") != "controlled"
        or report.get("scope") != "real_cuda_single_b200_shard"
        or members.get("expected") != 5
        or members.get("completed_before_resume") != 5
        or members.get("skipped_by_resume_validation") != 5
        or members.get("unchanged_after_resume") != 5
        or members.get("reexecuted") != 0
    ):
        raise FinalizationError("Controlled resume validation differs from its contract")
    _require_all_true(report.get("checks"), "Controlled resume checks")
    evidence_record = _require_mapping(report.get("evidence"), "Controlled resume evidence")
    if (
        evidence_record.get("unchanged") is not True
        or not _is_sha256(evidence_record.get("pre_resume_sha256"))
        or evidence_record.get("pre_resume_sha256") != evidence_record.get("post_resume_sha256")
    ):
        raise FinalizationError("Controlled resume evidence hashes differ")


def _validate_imagenet9_resume(evidence: Evidence) -> None:
    report = evidence.json("verification/imagenet9_resume.json")
    members = _require_mapping(report.get("members"), "ImageNet-9 resume members")
    if (
        report.get("status") != "passed"
        or report.get("family") != "imagenet9"
        or report.get("scope") != "real_cuda_single_b200_shard"
        or members.get("expected") != 24
        or members.get("completed_before_resume") != 24
        or members.get("skipped_by_resume_validation") != 24
        or members.get("unchanged_after_resume") != 24
        or members.get("reexecuted") != 0
    ):
        raise FinalizationError("ImageNet-9 resume validation differs from its contract")
    _require_all_true(report.get("checks"), "ImageNet-9 resume checks")
    evidence_record = _require_mapping(report.get("evidence"), "ImageNet-9 resume evidence")
    if (
        evidence_record.get("unchanged") is not True
        or not _is_sha256(evidence_record.get("pre_resume_sha256"))
        or evidence_record.get("pre_resume_sha256") != evidence_record.get("post_resume_sha256")
    ):
        raise FinalizationError("ImageNet-9 resume evidence hashes differ")


def _find_fact(report: Mapping[str, Any], name: str) -> Any:
    values = []
    for container in (
        report,
        report.get("members"),
        report.get("interruption"),
        report.get("resume"),
    ):
        if isinstance(container, Mapping) and name in container:
            values.append(container[name])
    if not values or any(value != values[0] for value in values[1:]):
        raise FinalizationError(f"fault report fact is missing or contradictory: {name}")
    return values[0]


def _validate_fault_injection(evidence: Evidence) -> dict[str, int]:
    report = evidence.json("verification/single_gpu_resume_fault_injection.json")
    if report.get("status") != "passed" or report.get("signal") != "SIGTERM":
        raise FinalizationError("fault injection did not pass with a normal SIGTERM")
    _require_all_true(report.get("checks"), "fault-injection checks", REQUIRED_FAULT_CHECKS)
    interrupted = _find_fact(report, "interrupted_global_status")
    final = _find_fact(report, "final_global_status")
    before = _find_fact(report, "completed_before_signal")
    running = _find_fact(report, "running_receipts_after_signal")
    skipped = _find_fact(report, "completed_members_skipped_on_resume")
    completed = _find_fact(report, "incomplete_members_completed_on_resume")
    if (
        interrupted not in {"partial", "failed"}
        or final != "completed"
        or not isinstance(before, int)
        or before < 2
        or before >= REQUIRED_ATTRIBUTION_MEMBER_COUNTS["resume_test"]
        or running != 0
        or not isinstance(skipped, int)
        or skipped != before
        or not isinstance(completed, int)
        or completed != REQUIRED_ATTRIBUTION_MEMBER_COUNTS["resume_test"] - before
    ):
        raise FinalizationError("fault-injection lifecycle facts differ from the contract")

    run = evidence.json("runs/resume_test/run.json")
    if (
        run.get("status") != "completed"
        or run.get("experiment") != "attribution"
        or run.get("profile") != "smoke-resume"
        or run.get("requested_stage") != "compute"
        or run.get("completed_stages") != ["compute"]
    ):
        raise FinalizationError("fault-injection final run is incomplete")
    plan = evidence.json("runs/resume_test/manifests/plan.json")
    expected_scopes = ("resume_idsds_deletion_targets", "resume_idsds_primary")
    global_receipt = _validate_global_receipt(evidence, "resume_test")
    if (
        tuple(plan.get("scope_names", ())) != expected_scopes
        or plan.get("member_count") != REQUIRED_ATTRIBUTION_MEMBER_COUNTS["resume_test"]
        or plan.get("member_count") != global_receipt.get("member_count")
    ):
        raise FinalizationError("fault-injection plan differs from the smoke-resume contract")
    resume_spec = RunSpec("resume_test", "attribution", "smoke-resume", expected_scopes)
    _validate_attribution_member_bindings(evidence, resume_spec, plan, global_receipt)
    observed_skips, observed_completions = _validate_gpu_queue(
        global_receipt, "fault-injection", require_resume_mix=True
    )
    if skipped != observed_skips or completed != observed_completions:
        raise FinalizationError("fault report counts differ from the final queue receipt")
    compute = evidence.json("runs/resume_test/receipts/compute.json")
    compute_details = _require_mapping(compute.get("details"), "fault-injection compute details")
    if (
        compute.get("status") != "completed"
        or compute_details.get("backend") != "gpu"
        or compute_details.get("scheduler") != "single_gpu_dynamic_queue"
        or compute_details.get("member_count") != REQUIRED_ATTRIBUTION_MEMBER_COUNTS["resume_test"]
        or compute_details.get("failed_members") != 0
        or compute_details.get("completed_members") != observed_completions
        or compute_details.get("resumed_members") != observed_skips
    ):
        raise FinalizationError("fault-injection compute receipt differs from its resumed queue")
    return {"skipped": skipped, "completed_after_resume": completed}


def _validate_scheduler(evidence: Evidence) -> None:
    report = evidence.json("verification/single_gpu_scheduler.json")
    if report.get("status") != "passed" or report.get("scheduler") != ("single_gpu_dynamic_queue"):
        raise FinalizationError("single-GPU scheduler report has not passed")
    _require_all_true(report.get("checks"), "scheduler checks", REQUIRED_SCHEDULER_CHECKS)
    if report.get("multi_gpu_scheduler_static_plan") != "PASS":
        raise FinalizationError("multi-GPU static-plan result is not PASS")
    if report.get("multi_gpu_scheduler_real_execution") != ("NOT_TESTED_SINGLE_GPU_NODE"):
        raise FinalizationError("multi-GPU real-execution boundary is misstated")


def _validate_fingerprint_tensor(record: Mapping[str, Any], family: str, sample_count: int) -> None:
    tensor = _require_mapping(record.get("preprocessed_tensor"), "fingerprint tensor")
    shape = tensor.get("shape")
    if (
        not _is_sha256(tensor.get("sha256"))
        or not isinstance(shape, list)
        or len(shape) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape
        )
        or shape[0] != sample_count
        or shape[1] != 3
    ):
        raise FinalizationError("fingerprint tensor identity/shape is malformed")
    expected_bytes = math.prod(shape) * 4
    if family == "attribution":
        valid = (
            tensor.get("dtype") == "torch.float32"
            and tensor.get("byte_order") == "little"
            and tensor.get("layout") == "contiguous_c_order"
            and tensor.get("contiguous") is True
            and tensor.get("contiguous_bytes") == expected_bytes
        )
    else:
        expected_spatial = [32, 32] if family == "controlled" else [224, 224]
        valid = (
            tensor.get("dtype") == "float32"
            and tensor.get("byte_order") == "little-endian"
            and tensor.get("layout") == "C-contiguous"
            and tensor.get("bytes") == expected_bytes
            and shape[2:] == expected_spatial
        )
    if not valid:
        raise FinalizationError("fingerprint tensor metadata differs from its family contract")


def _validate_fingerprint_checkpoints(record: Mapping[str, Any], family: str) -> None:
    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise FinalizationError("checkpoint fingerprint case has no checkpoint bytes")
    expected_ids: tuple[str, ...]
    if family == "attribution":
        key = (str(record.get("model_id")), str(record.get("dataset")))
        try:
            expected_ids = REQUIRED_ATTRIBUTION_CHECKPOINTS[key]
        except KeyError as error:
            raise FinalizationError(f"unknown attribution fingerprint model: {key}") from error
        observed_ids = tuple(str(item.get("checkpoint_id")) for item in checkpoints)
    else:
        expected_ids = (str(record.get("model_id")),)
        observed_ids = tuple(str(item.get("identity")) for item in checkpoints)
    if observed_ids != expected_ids:
        raise FinalizationError("fingerprint checkpoint identities differ")
    paths: set[str] = set()
    for raw in checkpoints:
        item = _require_mapping(raw, "fingerprint checkpoint")
        raw_path = item.get("path")
        size = item.get("bytes")
        path = Path(str(raw_path or ""))
        if (
            not isinstance(raw_path, str)
            or not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or path.stat().st_size != size
            or not _is_sha256(item.get("sha256"))
            or _sha256(path) != item.get("sha256")
            or raw_path in paths
        ):
            raise FinalizationError(f"fingerprint checkpoint bytes differ: {record.get('case_id')}")
        paths.add(raw_path)


def _validate_fingerprint_case(record: Mapping[str, Any]) -> tuple[Any, ...]:
    family = record.get("family")
    case_id = record.get("case_id")
    model_id = record.get("model_id")
    sample_ids = record.get("sample_ids")
    if (
        family not in REQUIRED_FINGERPRINT_COVERAGE
        or not isinstance(case_id, str)
        or not isinstance(model_id, str)
    ):
        raise FinalizationError("checkpoint fingerprint case identity is malformed")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise FinalizationError(f"fingerprint sample IDs are absent: {case_id}")
    if family == "controlled":
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in sample_ids
        ):
            raise FinalizationError(f"controlled fingerprint sample IDs are invalid: {case_id}")
    elif len(sample_ids) != 1 or any(
        not isinstance(value, str) or not value.strip() for value in sample_ids
    ):
        raise FinalizationError(f"image fingerprint sample IDs are invalid: {case_id}")
    if len({str(value) for value in sample_ids}) != len(sample_ids):
        raise FinalizationError(f"fingerprint sample IDs are duplicated: {case_id}")

    _validate_fingerprint_tensor(record, str(family), len(sample_ids))
    _validate_fingerprint_checkpoints(record, str(family))
    logits = _numeric_matrix(record.get("logits"), f"{case_id} logits")
    probabilities = _validate_probabilities(record.get("probabilities"), f"{case_id} probabilities")
    if len(logits) != len(sample_ids) or [len(row) for row in logits] != [
        len(row) for row in probabilities
    ]:
        raise FinalizationError(f"fingerprint output shapes differ: {case_id}")
    output_width = len(logits[0])

    if family == "controlled":
        identity = (case_id, model_id, record.get("architecture"))
        targets = record.get("target_class")
        device = _require_mapping(record.get("device_details"), "controlled device details")
        if (
            identity not in REQUIRED_CONTROLLED_FINGERPRINTS
            or record.get("precision") != "float32"
            or record.get("device") != "cuda:0"
            or output_width != 2
            or not isinstance(targets, list)
            or len(targets) != len(sample_ids)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < output_width
                for value in targets
            )
            or device.get("requested") != "cuda:0"
            or device.get("resolved") != "cuda:0"
            or device.get("count_visible") != 1
            or "B200" not in str(device.get("name"))
        ):
            raise FinalizationError(f"controlled fingerprint contract differs: {case_id}")
        return identity

    target = record.get("target_class")
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target < 0
        or target >= output_width
        or record.get("device") != "cuda:0"
        or "B200" not in str(record.get("device_name"))
    ):
        raise FinalizationError(f"fingerprint target/device contract differs: {case_id}")
    if family == "imagenet9":
        identity = (
            case_id,
            model_id,
            record.get("model_kind"),
            record.get("architecture_family"),
        )
        off_the_shelf = record.get("model_kind") == "off_the_shelf"
        mapped = (
            _validate_subprobabilities(
                record.get("imagenet9_probabilities"),
                f"{case_id} ImageNet-9 probabilities",
            )
            if off_the_shelf
            else _validate_probabilities(
                record.get("imagenet9_probabilities"),
                f"{case_id} ImageNet-9 probabilities",
            )
        )
        adapter = _require_mapping(record.get("probability_adapter"), "probability adapter")
        expected_width = 1_000 if off_the_shelf else 9
        if (
            identity not in REQUIRED_IMAGENET9_FINGERPRINTS
            or record.get("precision") != "float32"
            or output_width != expected_width
            or len(mapped) != len(sample_ids)
            or any(len(row) != 9 for row in mapped)
            or adapter.get("direct_nine_way") is not (not off_the_shelf)
            or adapter.get("mapped_mass_renormalized") is not False
            or adapter.get("softmax_count") != 1
            or not _is_sha256(adapter.get("official_mapping_sha256"))
        ):
            raise FinalizationError(f"ImageNet-9 fingerprint contract differs: {case_id}")
        return identity

    identity = (case_id, model_id, record.get("dataset"))
    expected_width = 50 if record.get("dataset") == "funnybirds" else 1_000
    if (
        identity not in REQUIRED_ATTRIBUTION_FINGERPRINTS
        or record.get("precision") != "fp32"
        or record.get("cuda_synchronized") is not True
        or output_width != expected_width
    ):
        raise FinalizationError(f"attribution fingerprint contract differs: {case_id}")
    return identity


def _validate_fingerprints(evidence: Evidence, identity: RepositoryIdentity) -> dict[str, Any]:
    report = evidence.json("verification/checkpoint_fingerprint_report.json")
    if (
        report.get("status") != "passed"
        or report.get("coverage") != REQUIRED_FINGERPRINT_COVERAGE
        or report.get("case_count") != 12
    ):
        raise FinalizationError("checkpoint fingerprint coverage differs")
    checks = _require_mapping(report.get("checks"), "checkpoint fingerprint checks")
    if set(checks) != REQUIRED_FINGERPRINT_CHECKS:
        raise FinalizationError("checkpoint fingerprint check inventory differs")
    _require_all_true(checks, "checkpoint fingerprint checks", REQUIRED_FINGERPRINT_CHECKS)
    device = _require_mapping(report.get("device"), "checkpoint fingerprint device")
    libraries = _require_mapping(device.get("libraries"), "checkpoint libraries")
    if (
        "B200" not in str(device.get("device_name"))
        or device.get("device_index") != 0
        or not isinstance(device.get("cuda_runtime"), str)
        or not device.get("cuda_runtime")
        or not isinstance(device.get("python"), str)
        or not device.get("python")
        or set(libraries) != REQUIRED_FINGERPRINT_LIBRARIES
        or any(not isinstance(libraries[name], str) or not libraries[name] for name in libraries)
    ):
        raise FinalizationError("checkpoint fingerprints do not identify the B200 runtime")
    relative = report.get("checkpoint_fingerprints_path")
    if not isinstance(relative, str):
        raise FinalizationError("checkpoint fingerprint payload path is missing")
    payload_path = evidence.file(f"verification/{_safe_relative(relative)}")
    if _sha256(payload_path) != report.get("checkpoint_fingerprints_sha256"):
        raise FinalizationError("checkpoint fingerprint payload hash differs")
    payload = evidence.json(f"verification/{relative}")
    payload_repository = _require_mapping(
        payload.get("repository"), "checkpoint fingerprint repository"
    )
    _require_bound(payload_repository, identity, "checkpoint fingerprint payload")
    tensor_contract = _require_mapping(
        payload.get("tensor_hash_contract"), "checkpoint tensor hash contract"
    )
    cases = payload.get("cases")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "passed"
        or payload.get("coverage") != REQUIRED_FINGERPRINT_COVERAGE
        or payload.get("case_count") != 12
        or payload.get("environment") != report.get("device")
        or tensor_contract.get("algorithm") != "sha256"
        or tensor_contract.get("source") != "C-contiguous tensor bytes after CPU conversion"
        or tensor_contract.get("byte_order") != "little-endian"
        or tensor_contract.get("shape_and_dtype_recorded_separately") is not True
        or not isinstance(cases, list)
        or len(cases) != 12
    ):
        raise FinalizationError("checkpoint fingerprint payload coverage differs")
    case_ids: list[str] = []
    coverage: Counter[str] = Counter()
    observed_controlled: set[tuple[Any, ...]] = set()
    observed_imagenet9: set[tuple[Any, ...]] = set()
    observed_attribution: set[tuple[Any, ...]] = set()
    for case in cases:
        record = _require_mapping(case, "checkpoint fingerprint case")
        family = str(record.get("family"))
        case_identity = _validate_fingerprint_case(record)
        if family == "controlled":
            observed_controlled.add(case_identity)
        elif family == "imagenet9":
            observed_imagenet9.add(case_identity)
        else:
            observed_attribution.add(case_identity)
        case_ids.append(str(record["case_id"]))
        coverage[family] += 1
    if len(case_ids) != len(set(case_ids)) or dict(coverage) != REQUIRED_FINGERPRINT_COVERAGE:
        raise FinalizationError("fingerprint case identities are duplicated or incomplete")
    if (
        observed_controlled != REQUIRED_CONTROLLED_FINGERPRINTS
        or observed_imagenet9 != REQUIRED_IMAGENET9_FINGERPRINTS
        or observed_attribution != REQUIRED_ATTRIBUTION_FINGERPRINTS
    ):
        raise FinalizationError("fingerprint exact model/family identities differ")
    if _canonical_sha256(case_ids) != report.get("case_set_sha256"):
        raise FinalizationError("fingerprint case-set digest differs")
    wrapper = evidence.json("verification/checkpoint_fingerprint_verification.json")
    if (
        wrapper.get("status") != "passed"
        or wrapper.get("mode") != "checkpoint-fingerprint"
        or wrapper.get("gpu_real_shard_verification") != "checkpoint_fingerprints_passed"
        or wrapper.get("steps", {}).get("checkpoint_fingerprint") != report
    ):
        raise FinalizationError("checkpoint fingerprint wrapper differs from its report")
    _require_bound(wrapper, identity, "checkpoint fingerprint wrapper")
    return {
        "case_count": 12,
        "coverage": dict(REQUIRED_FINGERPRINT_COVERAGE),
        "device": {
            "name": device["device_name"],
            "cuda_runtime": device["cuda_runtime"],
            "pytorch": libraries["torch"],
            "total_memory_bytes": device.get("device_total_memory_bytes"),
        },
    }


def _validate_pre_gpu(evidence: Evidence) -> dict[str, str]:
    reports: dict[str, dict[str, Any]] = {}
    for mode in ("unit", "full_plan", "repository_audit"):
        report = evidence.json(f"verification/pre_gpu/{mode}/cpu_verification.json")
        expected_mode = mode.replace("_", "-")
        if report.get("status") != "passed" or report.get("mode") != expected_mode:
            raise FinalizationError(f"pre-GPU {expected_mode} verification has not passed")
        if report.get("tracked_worktree_clean") is not True:
            raise FinalizationError(f"pre-GPU {expected_mode} report was not clean")
        reports[mode] = report
    identities = {
        (report.get("repository_commit"), report.get("repository_tree"))
        for report in reports.values()
    }
    if len(identities) != 1:
        raise FinalizationError("pre-GPU reports were not produced from one repository state")
    if reports["unit"].get("steps", {}).get("unit", {}).get("status") != "passed":
        raise FinalizationError("pre-GPU unit/regression step has not passed")
    full_plan = reports["full_plan"].get("steps", {}).get("full_plan", {})
    if full_plan.get("status") != "passed":
        raise FinalizationError("pre-GPU full-plan step has not passed")
    audit_step = reports["repository_audit"].get("steps", {}).get("repository_audit", {})
    if audit_step.get("passed") is not True:
        raise FinalizationError("pre-GPU repository audit has not passed")
    commit, tree = identities.pop()
    if not isinstance(commit, str) or not isinstance(tree, str):
        raise FinalizationError("pre-GPU repository identity is malformed")
    return {"status": "PASS", "repository_commit": commit, "repository_tree": tree}


def _validate_cpu_restructuring(evidence: Evidence) -> dict[str, Any]:
    report = evidence.json("verification/cpu_restructuring_status.json")
    expected = {
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
    }
    for key, expected_value in expected.items():
        if report.get(key) != expected_value:
            raise FinalizationError(
                "prior CPU restructuring field differs: "
                f"{key}={report.get(key)!r}, expected {expected_value!r}"
            )
    commit = report.get("new_repository_commit")
    if not isinstance(commit, str) or len(commit) not in {40, 64}:
        raise FinalizationError("prior CPU restructuring repository commit is malformed")
    if not _is_sha256(report.get("source_status_sha256")):
        raise FinalizationError("prior CPU restructuring source-status digest is malformed")
    gaps = report.get("historical_source_gaps")
    if (
        not isinstance(gaps, list)
        or len(gaps) != 1
        or not isinstance(gaps[0], Mapping)
        or gaps[0].get("asset_id") != "figure_01"
    ):
        raise FinalizationError("prior CPU restructuring source-gap record differs")
    return {
        "status": "PASS",
        "source_status": report["status"],
        "repository_commit": commit,
        "source_status_sha256": report["source_status_sha256"],
        "documented_source_gap": "figure_01",
    }


def _validate_full_pytest(evidence: Evidence, identity: RepositoryIdentity) -> dict[str, Any]:
    receipt = evidence.json("verification/final_audit/full_pytest.json")
    _require_bound(receipt, identity, "final full pytest receipt")
    command = receipt.get("command")
    started = _utc_timestamp(receipt.get("started_at"), "full pytest started_at")
    finished = _utc_timestamp(receipt.get("finished_at"), "full pytest finished_at")
    log = _require_mapping(receipt.get("output_log"), "full pytest output log")
    environment = _require_mapping(
        receipt.get("environment_contract"), "full pytest environment contract"
    )
    assets = _require_mapping(environment.get("assets"), "full pytest pinned assets")
    relative = log.get("path")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "passed"
        or command != list(REQUIRED_FULL_PYTEST_COMMAND)
        or receipt.get("exit_code") != 0
        or finished < started
        or not isinstance(receipt.get("elapsed_seconds"), (int, float))
        or isinstance(receipt.get("elapsed_seconds"), bool)
        or not math.isfinite(float(receipt.get("elapsed_seconds")))
        or float(receipt.get("elapsed_seconds")) < 0.0
        or not isinstance(receipt.get("passed_tests"), int)
        or isinstance(receipt.get("passed_tests"), bool)
        or receipt.get("passed_tests", 0) < 1
        or environment.get("mode") != REQUIRED_FULL_PYTEST_ENVIRONMENT_MODE
        or environment.get("b200_gate_removed") is not True
        or set(assets) != REQUIRED_FULL_PYTEST_ASSETS
        or not isinstance(assets.get("reference_run_archives"), list)
        or len(assets.get("reference_run_archives", [])) != 9
        or not isinstance(relative, str)
        or relative != "verification/final_audit/full_pytest.log"
        or log.get("streams") != "stdout+stderr"
        or not _is_sha256(log.get("sha256"))
        or not isinstance(log.get("size_bytes"), int)
        or isinstance(log.get("size_bytes"), bool)
        or log.get("size_bytes", 0) < 1
    ):
        raise FinalizationError("final full pytest receipt contract differs")
    asset_records = [assets["covertype_archive"], assets["idsds_manifest"]]
    asset_records.extend(assets["reference_run_archives"])
    if any(
        not isinstance(record, Mapping)
        or not _is_sha256(record.get("sha256"))
        or not isinstance(record.get("size_bytes"), int)
        or isinstance(record.get("size_bytes"), bool)
        or record.get("size_bytes", 0) < 1
        for record in asset_records
    ):
        raise FinalizationError("full pytest pinned asset inventory differs")
    output_path = evidence.file(_safe_relative(relative))
    if output_path.stat().st_size != log.get("size_bytes") or _sha256(output_path) != log.get(
        "sha256"
    ):
        raise FinalizationError("final full pytest output log bytes differ")
    try:
        output = output_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise FinalizationError("final full pytest output is not UTF-8") from error
    summaries = [int(value) for value in re.findall(r"(?m)(\d+) passed(?:,|\s+in\b)", output)]
    if (
        not summaries
        or summaries[-1] != receipt.get("passed_tests")
        or re.search(r"(?m)\b\d+ (?:failed|error|errors)\b", output) is not None
    ):
        raise FinalizationError("final full pytest log does not prove the declared passing suite")
    return {
        "status": "PASS",
        "command": list(REQUIRED_FULL_PYTEST_COMMAND),
        "passed_tests": receipt["passed_tests"],
        "output_sha256": log["sha256"],
    }


def _validate_final_audits(evidence: Evidence, identity: RepositoryIdentity) -> dict[str, str]:
    reports: dict[str, dict[str, Any]] = {}
    for mode in ("quality", "unit", "full_plan", "repository_audit"):
        report = evidence.json(f"verification/final_audit/{mode}/cpu_verification.json")
        expected_mode = mode.replace("_", "-")
        if report.get("status") != "passed" or report.get("mode") != expected_mode:
            raise FinalizationError(f"final {expected_mode} verification has not passed")
        _require_bound(report, identity, f"final {expected_mode} report")
        reports[mode] = report
    quality = reports["quality"].get("steps", {}).get("quality", {})
    if quality.get("status") != "passed":
        raise FinalizationError("final quality gate has not passed")
    quality_checks = _require_mapping(quality.get("checks"), "final quality checks")
    for name in ("ruff_check", "ruff_format", "static_imports", "shell_syntax"):
        if (
            not isinstance(quality_checks.get(name), Mapping)
            or quality_checks[name].get("status") != "passed"
        ):
            raise FinalizationError(f"final quality report omits passing {name}")
    if reports["unit"].get("steps", {}).get("unit", {}).get("status") != "passed":
        raise FinalizationError("final unit/regression gate has not passed")
    full_plan = reports["full_plan"].get("steps", {}).get("full_plan", {})
    families = _require_mapping(full_plan.get("families"), "final full-plan families")
    if full_plan.get("status") != "passed" or set(families) != {
        "controlled",
        "imagenet9",
        "attribution",
        "covertype",
    }:
        raise FinalizationError("final static full-plan family inventory differs")
    if any(
        not isinstance(value, Mapping) or value.get("status") != "passed"
        for value in families.values()
    ):
        raise FinalizationError("a final static full-plan family has not passed")
    audit_step = reports["repository_audit"].get("steps", {}).get("repository_audit", {})
    standalone_audit = evidence.json("verification/repository_audit.json")
    if audit_step.get("passed") is not True or dict(audit_step) != standalone_audit:
        raise FinalizationError("final standalone and embedded repository audits differ")
    return {mode: "PASS" for mode in reports}


def _validate_analysis_inventory(evidence: Evidence, analysis: Mapping[str, Any]) -> dict[str, str]:
    inventory = analysis.get("artifact_inventory")
    if not isinstance(inventory, list) or len(inventory) != 60:
        raise FinalizationError("analysis replay inventory is not exactly 60 files")
    if _canonical_sha256(inventory) != analysis.get("artifact_inventory_sha256"):
        raise FinalizationError("analysis replay inventory digest differs")
    paths: set[str] = set()
    roles: Counter[str] = Counter()
    role_hashes: dict[str, str] = {}
    for raw in inventory:
        record = _require_mapping(raw, "analysis inventory record")
        relative = record.get("relative_path")
        if (
            not isinstance(relative, str)
            or record.get("source_root") != "verification_root"
            or record.get("portable_path") != f"verification_root/{relative}"
            or not _is_sha256(record.get("sha256"))
        ):
            raise FinalizationError("analysis inventory contains a malformed portable record")
        relative = _safe_relative(relative)
        path = evidence.file(f"verification/{relative}")
        if path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256"):
            raise FinalizationError(f"sealed analysis artifact bytes differ: {relative}")
        if relative in paths:
            raise FinalizationError(f"analysis inventory contains a duplicate path: {relative}")
        paths.add(relative)
        role = str(record.get("role"))
        roles[role] += 1
        if role in {
            "replay_receipt",
            "family_replay_receipt",
            "canonical_receipt",
            "headline_assertions",
            "paper_artifact_diff",
        }:
            if role in role_hashes:
                raise FinalizationError(f"analysis inventory duplicates singleton role: {role}")
            role_hashes[role] = str(record["sha256"])
    expected_roles = {
        "generated_tex": 28,
        "canonical_csv": 27,
        "replay_receipt": 1,
        "family_replay_receipt": 1,
        "canonical_receipt": 1,
        "headline_assertions": 1,
        "paper_artifact_diff": 1,
    }
    if dict(roles) != expected_roles:
        raise FinalizationError(f"analysis inventory role counts differ: {dict(roles)}")
    for role, report_key in (
        ("replay_receipt", "replay_receipt_sha256"),
        ("family_replay_receipt", "family_replay_receipt_sha256"),
        ("canonical_receipt", "canonical_receipt_sha256"),
        ("headline_assertions", "headline_assertions_sha256"),
        ("paper_artifact_diff", "paper_artifact_diff_sha256"),
    ):
        if role_hashes[role] != analysis.get(report_key):
            raise FinalizationError(f"analysis report digest differs for {role}")
    return role_hashes


def _validate_analysis(evidence: Evidence, identity: RepositoryIdentity) -> dict[str, Any]:
    analysis = evidence.json("verification/analysis_replay.json")
    _require_bound(analysis, identity, "analysis replay")
    wrapper = evidence.json("verification/cpu_verification.json")
    if (
        wrapper.get("status") != "passed"
        or wrapper.get("mode") != "analysis-replay"
        or wrapper.get("steps", {}).get("analysis_replay") != analysis
    ):
        raise FinalizationError("analysis-replay CLI wrapper differs from its report")
    _require_bound(wrapper, identity, "analysis-replay CLI wrapper")
    for key, expected in REQUIRED_ANALYSIS_VALUES.items():
        if analysis.get(key) != expected:
            raise FinalizationError(
                f"analysis replay field differs: {key}={analysis.get(key)!r}, expected {expected!r}"
            )
    role_hashes = _validate_analysis_inventory(evidence, analysis)
    headline = evidence.json("verification/headline_assertions.json")
    if (
        headline.get("status") != "passed"
        or headline.get("assertion_count") != 27
        or headline.get("verified_count") != 27
        or headline.get("source_missing_count") != 0
    ):
        raise FinalizationError("headline assertion receipt is incomplete")
    diff_rows = list(
        csv.DictReader(StringIO(evidence.text("verification/paper_artifact_diff.csv")))
    )
    if len(diff_rows) != 28 or len({row.get("asset_id") for row in diff_rows}) != 28:
        raise FinalizationError("paper artifact diff is not exactly 28 unique assets")
    kinds = Counter(str(row.get("kind")) for row in diff_rows)
    comparisons = Counter(str(row.get("comparison_status")) for row in diff_rows)
    source_missing = [
        row.get("asset_id")
        for row in diff_rows
        if row.get("comparison_status") == "source_missing_recorded"
    ]
    if (
        kinds != {"figure": 12, "table": 16}
        or comparisons
        != {
            "regenerated_semantic_geometry": 11,
            "source_missing_recorded": 1,
            "regenerated_semantic_table": 16,
        }
        or source_missing != ["figure_01"]
    ):
        raise FinalizationError("paper artifact diff figure/table mapping differs")

    singleton_paths = {
        str(record["role"]): str(record["relative_path"])
        for record in analysis["artifact_inventory"]
        if record["role"] in role_hashes
    }
    replay = evidence.json(f"verification/{singleton_paths['replay_receipt']}")
    family = evidence.json(f"verification/{singleton_paths['family_replay_receipt']}")
    canonical = evidence.json(f"verification/{singleton_paths['canonical_receipt']}")
    if (
        len(replay.get("runs", [])) != 9
        or len(replay.get("inputs", [])) != 72
        or family.get("status") != "completed"
        or family.get("family_count") != 4
        or canonical.get("status") != "completed"
        or canonical.get("artifact_count") != 27
    ):
        raise FinalizationError("sealed historical replay receipts are incomplete")
    return {
        "status": "PASS",
        "figures": {
            "mapped": 12,
            "total": 12,
            "regenerated": 11,
            "source_missing_recorded": 1,
            "source_missing_assets": ["figure_01"],
        },
        "tables": {"mapped": 16, "total": 16, "regenerated": 16},
        "headline_assertions": "PASS",
        "numerical_source": "sealed_historical_outputs",
    }


def _validate_tmux_evidence(evidence: Evidence, session: str) -> None:
    if not _tmux_session_active(session):
        raise FinalizationError(f"required tmux session is not active: {session}")
    controller = evidence.text("logs/controller.log")
    gpu = evidence.text("logs/gpu.log")
    covertype = evidence.text("logs/covertype_cpu.log")
    evidence.text("logs/monitor.log")
    combined = controller + gpu
    required_gpu_tokens = [
        "checkpoint-fingerprint",
        "runs/controlled",
        "runs/imagenet9",
        "runs/attribution_main",
        "runs/dinov2_g",
        "runs/partimagenet",
        "runs/resume_test",
    ]
    if any(token not in combined for token in required_gpu_tokens):
        raise FinalizationError("tmux logs do not cover every required GPU operation")
    if "runs/covertype" not in covertype:
        raise FinalizationError("tmux Covertype log does not cover the CPU shard")
    gpu_monitor = evidence.text("resource_utilization/gpu.csv")
    if "NVIDIA B200" not in gpu_monitor:
        raise FinalizationError("resource monitor does not identify an NVIDIA B200")
    for relative in (
        "resource_utilization/vmstat.log",
        "resource_utilization/diskstats.log",
        "resource_utilization/process_memory.log",
    ):
        evidence.file(relative)


def _now() -> str:
    now = datetime.now(timezone.utc)  # noqa: UP017 -- GPU runtime uses Python 3.10.
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def _build_report(status: Mapping[str, Any]) -> str:
    repository = status["repository"]
    machine = status["machine"]
    fingerprints = status["checkpoint_fingerprints"]
    shards = status["representative_shards"]
    paper = status["paper_results"]
    return f"""# DECAF Single-B200 Verification Report

Status: PASS

## Repository and machine

- Commit: `{repository["commit"]}`
- Tree: `{repository["tree"]}`
- Branch: `{repository["branch"]}`
- GPU: {machine["gpu"]["name"]}
- CUDA runtime: {machine["gpu"]["cuda_runtime"]}
- PyTorch: {machine["gpu"]["pytorch"]}
- Available CPUs: {machine["host"]["available_cpu_count"]}
- Host memory bytes: {machine["host"]["host_memory_bytes"]}

## Checkpoint fingerprints

PASS: {fingerprints["case_count"]}/12 cases were verified ({fingerprints["coverage"]}).

## Representative execution shards

- Controlled real B200 shard: {shards["controlled"]["status"]}
- ImageNet-9 real B200 shard: {shards["imagenet9"]["status"]}
- Attribution main real B200 shard: {shards["attribution_main"]["status"]}
- DINOv2-g real B200 shard: {shards["dinov2_g"]["status"]}
- PartImageNet boundary real B200 shard: {shards["partimagenet"]["status"]}
- Covertype CPU shard: {shards["covertype"]["status"]}

## Scheduling and resume

- Single-GPU scheduler: PASS
- Resume reuse: PASS
- SIGTERM fault injection and resume: PASS
- Multi-GPU static plan: PASS
- Real multi-GPU execution: NOT_TESTED_SINGLE_GPU_NODE

## Paper replay

- Figures mapped: {paper["figures"]["mapped"]}/{paper["figures"]["total"]}
- Figures regenerated from data: {paper["figures"]["regenerated"]}/12
- Figure 1: source_missing (recorded)
- Tables regenerated and mapped: {paper["tables"]["mapped"]}/{paper["tables"]["total"]}
- Headline assertions: PASS

Reference paper results were regenerated from sealed historical outputs.
Representative real B200 shards validated the refactored compute paths.
Full paper-scale computation was not rerun.
Real multi-GPU scheduling was not exercised on this single-GPU node.

The real B200 shards are compute-path validation only. They are not the source of the
paper's numerical claims or formal timing results.
"""


def _assert_portable_english(text: str, label: str) -> None:
    matched = next((fragment for fragment in PRIVATE_PATH_FRAGMENTS if fragment in text), None)
    if matched is not None:
        raise FinalizationError(f"{label} contains a private absolute path fragment")
    if any(
        0x3400 <= ord(character) <= 0x4DBF
        or 0x4E00 <= ord(character) <= 0x9FFF
        or 0xF900 <= ord(character) <= 0xFAFF
        for character in text
    ):
        raise FinalizationError(f"{label} contains CJK text")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def finalize_b200_verification(
    *, repository: Path, verification_root: Path, tmux_session: str = "decaf-b200-verify"
) -> dict[str, Any]:
    """Validate all final gates, then atomically emit the three portable documents."""

    repository = repository.resolve()
    identity = _repository_identity(repository)
    evidence = Evidence(verification_root)
    _validate_tmux_evidence(evidence, tmux_session)
    cpu_restructuring = _validate_cpu_restructuring(evidence)
    pre_gpu = _validate_pre_gpu(evidence)
    fingerprints = _validate_fingerprints(evidence, identity)
    shards = _validate_runs(evidence)
    _validate_controlled_resume(evidence)
    _validate_imagenet9_resume(evidence)
    fault = _validate_fault_injection(evidence)
    _validate_scheduler(evidence)
    final_audits = _validate_final_audits(evidence, identity)
    full_pytest = _validate_full_pytest(evidence, identity)
    final_audits["full_pytest"] = "PASS"
    paper = _validate_analysis(evidence, identity)
    host = _host_environment()

    acceptance_gates = {
        "substantive_work_in_tmux": True,
        "cpu_restructuring_status": True,
        "single_b200_detected": True,
        "checkpoint_fingerprints": True,
        "controlled_real_b200_shard": True,
        "imagenet9_real_b200_shard": True,
        "attribution_main_real_b200_shard": True,
        "dinov2_g_real_b200_shard": True,
        "partimagenet_boundary_real_b200_shard": True,
        "covertype_cpu_shard": True,
        "resume_reruns_skip_completed_members": True,
        "imagenet9_resume_rerun_skipped_completed_members": True,
        "sigterm_resume_fault_injection": True,
        "single_gpu_scheduler": True,
        "multi_gpu_static_plan": True,
        "analysis_replay": True,
        "figures_12_of_12_mapped": True,
        "tables_16_of_16_mapped": True,
        "headline_assertions": True,
        "repository_audit": True,
        "full_pytest": True,
        "unit_regression_tests": True,
        "quality_checks": True,
        "full_plan": True,
    }
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "completed_at": _now(),
        "repository": {
            "commit": identity.commit,
            "tree": identity.tree,
            "branch": identity.branch,
            "tracked_worktree_clean": True,
        },
        "machine": {"host": host, "gpu": fingerprints["device"]},
        "checkpoint_fingerprints": {
            "status": "PASS",
            "case_count": fingerprints["case_count"],
            "coverage": fingerprints["coverage"],
        },
        "representative_shards": shards,
        "resume": {
            "status": "PASS",
            "controlled_hash_stable": True,
            "imagenet9_hash_stable": True,
            "fault_injection": fault,
        },
        "single_gpu_scheduler": {"status": "PASS", "device": "cuda:0"},
        "multi_gpu_scheduler": {
            "static_plan": "PASS",
            "real_execution": "NOT_TESTED_SINGLE_GPU_NODE",
        },
        "paper_results": paper,
        "paper_numeric_source": "sealed_historical_outputs",
        "full_paper_scale_compute_rerun": False,
        "cpu_restructuring": cpu_restructuring,
        "pre_gpu_readiness": pre_gpu,
        "final_audits": final_audits,
        "full_pytest": full_pytest,
        "acceptance_gates": acceptance_gates,
        "scope_statements": [
            "Reference paper results were regenerated from sealed historical outputs.",
            "Representative real B200 shards validated the refactored compute paths.",
            "Full paper-scale computation was not rerun.",
            "Real multi-GPU scheduling was not exercised on this single-GPU node.",
        ],
    }
    report_text = _build_report(status)
    evidence_inventory = evidence.inventory()
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "repository": status["repository"],
        "verification_scope": {
            "paper_numeric_source": "sealed_historical_outputs",
            "representative_real_b200_shards": True,
            "full_paper_scale_compute_rerun": False,
            "multi_gpu_static_plan": "PASS",
            "multi_gpu_real_execution": "NOT_TESTED_SINGLE_GPU_NODE",
        },
        "paper_assets": paper,
        "evidence_file_count": len(evidence_inventory),
        "evidence_inventory_sha256": _canonical_sha256(evidence_inventory),
        "evidence_files": evidence_inventory,
    }
    status_text = _json_text(status)
    provenance_text = _json_text(provenance)
    for label, text in (
        ("B200 verification status", status_text),
        ("B200 verification report", report_text),
        ("B200 provenance", provenance_text),
    ):
        _assert_portable_english(text, label)

    root = evidence.root
    _atomic_text(root / "B200_VERIFICATION_STATUS.json", status_text)
    _atomic_text(root / "B200_VERIFICATION_REPORT.md", report_text)
    _atomic_text(root / "provenance" / "B200_PROVENANCE.json", provenance_text)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--verification-root", required=True, type=Path)
    parser.add_argument("--tmux-session", default="decaf-b200-verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = finalize_b200_verification(
        repository=args.repository,
        verification_root=args.verification_root,
        tmux_session=args.tmux_session,
    )
    print(f"b200_verification_status={status['status']}")
    print(f"repository_commit={status['repository']['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FinalizationError",
    "RepositoryIdentity",
    "build_parser",
    "finalize_b200_verification",
    "main",
]
