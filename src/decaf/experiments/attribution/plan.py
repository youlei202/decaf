"""Deterministic, location-independent plans for attribution profiles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from decaf.experiments.attribution.methods import (
    BOUNDARY_COMPUTE_METHODS,
    FULL50K_METHODS,
    FUNNYBIRDS_PRIMARY_METHODS,
    FUNNYBIRDS_SUPPLEMENT_METHODS,
    LARGE_MODEL_METHODS,
    MAIN_METHODS,
    VERIFY_BOUNDARY_METHODS,
    VERIFY_LARGE_MODEL_METHODS,
    VERIFY_MAIN_METHODS,
    VERIFY_RESUME_METHODS,
    get_method,
)
from decaf.experiments.attribution.models import (
    ALIGNED_ARCHITECTURES,
    BOUNDARY_MODELS,
    FUNNYBIRDS_MODELS,
    IDSDS_MODELS,
    LARGE_MODEL,
    checkpoint_coverage,
    supports_dataset,
)

PRIMARY_MANIFEST_SHA256 = "3f6f9bad1c631f3eb95e8e2ae2fb171dd86470deaed7f3c93259feea952c0e79"
FULL_MANIFEST_SHA256 = "bad5c0fe0df455bce6a5172233cd0cd549c562ee006e549bf2d7142fd0184fd7"
TIMING_MANIFEST_SHA256 = "6c4f6d6b5bbe83d6d5ac9b9558dc97f844a2f2f2d4c77d1fc0db18a1505f8bd4"
FUNNYBIRDS_SUPPORT_SHA256 = "19ae0c0766857ebb7f0ea09ae24f28ad97b322f2cd9a45325184ade43b7ed1ac"
PARTIMAGENET_MANIFEST_SHA256 = "d1198f5a06bd4ef9656473a047fe4e01ddabf2a76f5868dfa7ee6579ae710657"
FUNNYBIRDS_STUDY_MANIFEST_SHA256 = (
    "bc4d1c647fd0f5ab6611bacfa5a558e15b246916cee037ca80cc6b056d890f2c"
)
DELETION_TARGET_METHOD = "__deletion_targets__"
FUNNYBIRDS_DELETION_TARGET_METHOD = "__part_deletion_targets__"
FUNNYBIRDS_HELDOUT_METHODS = (
    "__heldout_background_texture__",
    "__heldout_telea_dilate3__",
)
TARGET_METHODS = frozenset(
    (
        DELETION_TARGET_METHOD,
        FUNNYBIRDS_DELETION_TARGET_METHOD,
        *FUNNYBIRDS_HELDOUT_METHODS,
    )
)


@dataclass(frozen=True, slots=True)
class ScopeSpec:
    """Cartesian member contract for one scientific scope."""

    name: str
    dataset: str
    models: tuple[str, ...]
    methods: tuple[str, ...]
    shards: int
    repeats: int
    images: int
    manifest_sha256: str | None
    kind: str
    expected_members: int

    @property
    def member_count(self) -> int:
        return len(self.models) * len(self.methods) * self.shards * self.repeats


SCOPES: dict[str, ScopeSpec] = {
    "oracle": ScopeSpec(
        "oracle",
        "oracle",
        ("oracle_linear",),
        ("decaf_5",),
        1,
        1,
        8,
        None,
        "cpu_score_oracle",
        1,
    ),
    "smoke_idsds_deletion_targets": ScopeSpec(
        "smoke_idsds_deletion_targets",
        "imagenet1k_idsds",
        IDSDS_MODELS,
        (DELETION_TARGET_METHOD,),
        1,
        1,
        8,
        PRIMARY_MANIFEST_SHA256,
        "shared_deletion_targets",
        3,
    ),
    "smoke_idsds_primary": ScopeSpec(
        "smoke_idsds_primary",
        "imagenet1k_idsds",
        IDSDS_MODELS,
        VERIFY_MAIN_METHODS,
        1,
        1,
        8,
        PRIMARY_MANIFEST_SHA256,
        "quality",
        30,
    ),
    "smoke_funnybirds_deletion_targets": ScopeSpec(
        "smoke_funnybirds_deletion_targets",
        "funnybirds",
        FUNNYBIRDS_MODELS,
        (FUNNYBIRDS_DELETION_TARGET_METHOD,),
        1,
        1,
        8,
        FUNNYBIRDS_STUDY_MANIFEST_SHA256,
        "shared_part_deletion_targets",
        3,
    ),
    "smoke_funnybirds_heldout_targets": ScopeSpec(
        "smoke_funnybirds_heldout_targets",
        "funnybirds",
        FUNNYBIRDS_MODELS,
        FUNNYBIRDS_HELDOUT_METHODS,
        1,
        1,
        8,
        FUNNYBIRDS_STUDY_MANIFEST_SHA256,
        "shared_heldout_targets",
        6,
    ),
    "smoke_funnybirds_primary": ScopeSpec(
        "smoke_funnybirds_primary",
        "funnybirds",
        FUNNYBIRDS_MODELS,
        VERIFY_MAIN_METHODS,
        1,
        1,
        8,
        FUNNYBIRDS_STUDY_MANIFEST_SHA256,
        "quality",
        30,
    ),
    "resume_idsds_deletion_targets": ScopeSpec(
        "resume_idsds_deletion_targets",
        "imagenet1k_idsds",
        ("resnet50",),
        (DELETION_TARGET_METHOD,),
        1,
        1,
        8,
        PRIMARY_MANIFEST_SHA256,
        "shared_deletion_targets",
        1,
    ),
    "resume_idsds_primary": ScopeSpec(
        "resume_idsds_primary",
        "imagenet1k_idsds",
        ("resnet50",),
        VERIFY_RESUME_METHODS,
        1,
        1,
        8,
        PRIMARY_MANIFEST_SHA256,
        "quality",
        4,
    ),
    "smoke_dinov2_g_quality": ScopeSpec(
        "smoke_dinov2_g_quality",
        "imagenet1k_idsds",
        (LARGE_MODEL,),
        VERIFY_LARGE_MODEL_METHODS,
        1,
        1,
        8,
        PRIMARY_MANIFEST_SHA256,
        "large_model_quality",
        8,
    ),
    "smoke_dinov2_g_timing": ScopeSpec(
        "smoke_dinov2_g_timing",
        "imagenet1k_idsds",
        (LARGE_MODEL,),
        VERIFY_LARGE_MODEL_METHODS,
        1,
        1,
        8,
        PRIMARY_MANIFEST_SHA256,
        "large_model_timing",
        8,
    ),
    "smoke_partimagenet_deletion_targets": ScopeSpec(
        "smoke_partimagenet_deletion_targets",
        "partimagenet",
        ("resnet50",),
        (FUNNYBIRDS_DELETION_TARGET_METHOD,),
        1,
        1,
        8,
        PARTIMAGENET_MANIFEST_SHA256,
        "shared_part_deletion_targets",
        1,
    ),
    "smoke_partimagenet_heldout_targets": ScopeSpec(
        "smoke_partimagenet_heldout_targets",
        "partimagenet",
        ("resnet50",),
        FUNNYBIRDS_HELDOUT_METHODS,
        1,
        1,
        8,
        PARTIMAGENET_MANIFEST_SHA256,
        "shared_heldout_targets",
        2,
    ),
    "smoke_partimagenet_boundary": ScopeSpec(
        "smoke_partimagenet_boundary",
        "partimagenet",
        ("resnet50",),
        VERIFY_BOUNDARY_METHODS,
        1,
        1,
        8,
        PARTIMAGENET_MANIFEST_SHA256,
        "boundary_quality",
        5,
    ),
    "idsds_deletion_targets": ScopeSpec(
        "idsds_deletion_targets",
        "imagenet1k_idsds",
        IDSDS_MODELS,
        (DELETION_TARGET_METHOD,),
        20,
        1,
        10_000,
        PRIMARY_MANIFEST_SHA256,
        "shared_deletion_targets",
        60,
    ),
    "idsds_primary": ScopeSpec(
        "idsds_primary",
        "imagenet1k_idsds",
        IDSDS_MODELS,
        MAIN_METHODS,
        20,
        1,
        10_000,
        PRIMARY_MANIFEST_SHA256,
        "quality",
        780,
    ),
    "funnybirds_supplement": ScopeSpec(
        "funnybirds_supplement",
        "funnybirds",
        FUNNYBIRDS_MODELS,
        FUNNYBIRDS_SUPPLEMENT_METHODS,
        1,
        1,
        1_484,
        FUNNYBIRDS_SUPPORT_SHA256,
        "quality_supplement",
        6,
    ),
    "funnybirds_deletion_targets": ScopeSpec(
        "funnybirds_deletion_targets",
        "funnybirds",
        FUNNYBIRDS_MODELS,
        (FUNNYBIRDS_DELETION_TARGET_METHOD,),
        1,
        1,
        1_500,
        FUNNYBIRDS_SUPPORT_SHA256,
        "shared_part_deletion_targets",
        3,
    ),
    "funnybirds_heldout_targets": ScopeSpec(
        "funnybirds_heldout_targets",
        "funnybirds",
        FUNNYBIRDS_MODELS,
        FUNNYBIRDS_HELDOUT_METHODS,
        1,
        1,
        1_500,
        FUNNYBIRDS_SUPPORT_SHA256,
        "shared_heldout_targets",
        6,
    ),
    "funnybirds_primary": ScopeSpec(
        "funnybirds_primary",
        "funnybirds",
        FUNNYBIRDS_MODELS,
        FUNNYBIRDS_PRIMARY_METHODS,
        1,
        1,
        1_500,
        FUNNYBIRDS_SUPPORT_SHA256,
        "quality",
        33,
    ),
    "idsds_full50k_deletion_targets": ScopeSpec(
        "idsds_full50k_deletion_targets",
        "imagenet1k_idsds",
        IDSDS_MODELS,
        (DELETION_TARGET_METHOD,),
        100,
        1,
        50_000,
        FULL_MANIFEST_SHA256,
        "shared_deletion_targets",
        300,
    ),
    "idsds_full50k": ScopeSpec(
        "idsds_full50k",
        "imagenet1k_idsds",
        IDSDS_MODELS,
        FULL50K_METHODS,
        100,
        1,
        50_000,
        FULL_MANIFEST_SHA256,
        "scale_check",
        900,
    ),
    "idsds_timing": ScopeSpec(
        "idsds_timing",
        "imagenet1k_idsds",
        IDSDS_MODELS,
        MAIN_METHODS,
        1,
        3,
        1_024,
        TIMING_MANIFEST_SHA256,
        "timing",
        117,
    ),
    "dinov2_g_quality": ScopeSpec(
        "dinov2_g_quality",
        "partimagenet",
        (LARGE_MODEL,),
        LARGE_MODEL_METHODS,
        4,
        1,
        238,
        PARTIMAGENET_MANIFEST_SHA256,
        "large_model_quality",
        32,
    ),
    "dinov2_g_timing": ScopeSpec(
        "dinov2_g_timing",
        "partimagenet",
        (LARGE_MODEL,),
        LARGE_MODEL_METHODS,
        1,
        3,
        238,
        PARTIMAGENET_MANIFEST_SHA256,
        "large_model_timing",
        24,
    ),
    "partimagenet_deletion_targets": ScopeSpec(
        "partimagenet_deletion_targets",
        "partimagenet",
        BOUNDARY_MODELS,
        (FUNNYBIRDS_DELETION_TARGET_METHOD,),
        16,
        1,
        4_096,
        PARTIMAGENET_MANIFEST_SHA256,
        "shared_part_deletion_targets",
        64,
    ),
    "partimagenet_heldout_targets": ScopeSpec(
        "partimagenet_heldout_targets",
        "partimagenet",
        BOUNDARY_MODELS,
        FUNNYBIRDS_HELDOUT_METHODS,
        16,
        1,
        4_096,
        PARTIMAGENET_MANIFEST_SHA256,
        "shared_heldout_targets",
        128,
    ),
    "partimagenet_boundary": ScopeSpec(
        "partimagenet_boundary",
        "partimagenet",
        BOUNDARY_MODELS,
        BOUNDARY_COMPUTE_METHODS,
        16,
        1,
        4_096,
        PARTIMAGENET_MANIFEST_SHA256,
        "boundary_quality",
        896,
    ),
}

PROFILE_SCOPES: dict[str, tuple[str, ...]] = {
    "smoke": ("oracle",),
    "main": (
        "idsds_deletion_targets",
        "idsds_primary",
        "funnybirds_deletion_targets",
        "funnybirds_heldout_targets",
        "funnybirds_primary",
        "funnybirds_supplement",
    ),
    "paper": (
        "idsds_deletion_targets",
        "idsds_primary",
        "funnybirds_deletion_targets",
        "funnybirds_heldout_targets",
        "funnybirds_primary",
        "funnybirds_supplement",
        "idsds_full50k_deletion_targets",
        "idsds_full50k",
        "idsds_timing",
        "dinov2_g_quality",
        "dinov2_g_timing",
        "partimagenet_deletion_targets",
        "partimagenet_heldout_targets",
        "partimagenet_boundary",
    ),
    "large-model": ("dinov2_g_quality", "dinov2_g_timing"),
    "boundary": (
        "partimagenet_deletion_targets",
        "partimagenet_heldout_targets",
        "partimagenet_boundary",
    ),
}

# These deliberately small profiles exercise the real offline CUDA path.  They
# are separate from the paper-scale contracts above so the existing static plan
# remains byte-for-byte interpretable as a full-compute declaration.
VERIFICATION_PROFILE_SCOPES: dict[str, tuple[str, ...]] = {
    "smoke-b200": (
        "smoke_idsds_deletion_targets",
        "smoke_idsds_primary",
        "smoke_funnybirds_deletion_targets",
        "smoke_funnybirds_heldout_targets",
        "smoke_funnybirds_primary",
    ),
    "large-model-smoke": (
        "smoke_dinov2_g_quality",
        "smoke_dinov2_g_timing",
    ),
    "boundary-smoke": (
        "smoke_partimagenet_deletion_targets",
        "smoke_partimagenet_heldout_targets",
        "smoke_partimagenet_boundary",
    ),
    "smoke-resume": (
        "resume_idsds_deletion_targets",
        "resume_idsds_primary",
    ),
}

PROFILE_MEMBER_COUNTS = {
    profile: sum(SCOPES[name].expected_members for name in names)
    for profile, names in PROFILE_SCOPES.items()
}
VERIFICATION_PROFILE_MEMBER_COUNTS = {
    profile: sum(SCOPES[name].expected_members for name in names)
    for profile, names in VERIFICATION_PROFILE_SCOPES.items()
}


def _profile_key(config: Mapping[str, Any]) -> str:
    """Resolve the opt-in real-GPU variant while preserving default smoke."""

    profile = str(config.get("profile", ""))
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise TypeError("attribution execution configuration must be a mapping")
    if profile == "smoke" and execution.get("verification_profile") == "single_b200":
        return "smoke-b200"
    return profile


def canonical_sha256(payload: Any) -> str:
    """Hash a location-independent JSON contract."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_job_sha256(job: Mapping[str, Any]) -> str:
    """Hash every canonical job field except the hash itself."""

    return canonical_sha256({key: value for key, value in job.items() if key != "job_sha256"})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _dependency_specs(
    member: Mapping[str, Any],
    target_jobs: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
) -> list[dict[str, str]]:
    scope = str(member["scope"])
    model = str(member["model_id"])
    shard = int(member["shard"])
    targets: tuple[tuple[str, str], ...]
    if scope == "idsds_primary":
        targets = (("idsds_deletion_targets", DELETION_TARGET_METHOD),)
    elif scope == "smoke_idsds_primary":
        targets = (("smoke_idsds_deletion_targets", DELETION_TARGET_METHOD),)
    elif scope == "resume_idsds_primary":
        targets = (("resume_idsds_deletion_targets", DELETION_TARGET_METHOD),)
    elif scope == "idsds_full50k":
        targets = (("idsds_full50k_deletion_targets", DELETION_TARGET_METHOD),)
    elif scope in {"funnybirds_primary", "funnybirds_supplement"}:
        targets = (
            ("funnybirds_deletion_targets", FUNNYBIRDS_DELETION_TARGET_METHOD),
            ("funnybirds_heldout_targets", "__heldout_background_texture__"),
            ("funnybirds_heldout_targets", "__heldout_telea_dilate3__"),
        )
    elif scope == "smoke_funnybirds_primary":
        targets = (
            ("smoke_funnybirds_deletion_targets", FUNNYBIRDS_DELETION_TARGET_METHOD),
            ("smoke_funnybirds_heldout_targets", "__heldout_background_texture__"),
            ("smoke_funnybirds_heldout_targets", "__heldout_telea_dilate3__"),
        )
    elif scope == "partimagenet_boundary":
        targets = (
            ("partimagenet_deletion_targets", FUNNYBIRDS_DELETION_TARGET_METHOD),
            ("partimagenet_heldout_targets", "__heldout_background_texture__"),
            ("partimagenet_heldout_targets", "__heldout_telea_dilate3__"),
        )
    elif scope == "smoke_partimagenet_boundary":
        targets = (
            ("smoke_partimagenet_deletion_targets", FUNNYBIRDS_DELETION_TARGET_METHOD),
            ("smoke_partimagenet_heldout_targets", "__heldout_background_texture__"),
            ("smoke_partimagenet_heldout_targets", "__heldout_telea_dilate3__"),
        )
    else:
        return []
    dependencies: list[dict[str, str]] = []
    for scope_name, target_method in targets:
        target = target_jobs.get((scope_name, model, shard, target_method))
        if target is None:
            raise AssertionError(
                f"missing target dependency for {member['member_id']}: {scope_name}/{target_method}"
            )
        dependencies.append(
            {
                "member_id": str(target["member_id"]),
                "scope": str(target["scope"]),
                "method_id": str(target["method_id"]),
                "job_sha256": str(target["job_sha256"]),
                "output_path": str(target["output_path"]),
                "receipt_path": str(target["receipt_path"]),
                "relationship": "shared_deletion_or_heldout_target_shard",
            }
        )
    return dependencies


def _bind_members(
    members: list[dict[str, Any]],
    *,
    config_sha256: str,
    plan_contract_sha256: str,
) -> None:
    coverage = checkpoint_coverage(tuple(sorted({str(member["model_id"]) for member in members})))
    for member in members:
        model_id = str(member["model_id"])
        member["config_sha256"] = config_sha256
        member["plan_contract_sha256"] = plan_contract_sha256
        member["checkpoint_contract_sha256"] = canonical_sha256(
            {
                "model_id": model_id,
                "checkpoint_ids": list(coverage[model_id]),
            }
        )
        member["input_contract_sha256"] = canonical_sha256(
            {
                "dataset": member["dataset"],
                "dataset_manifest_sha256": member["dataset_manifest_sha256"],
                "model_id": model_id,
                "image_start": member["image_start"],
                "image_stop": member["image_stop"],
                "image_count": member["image_count"],
            }
        )
        member["output_schema"] = (
            "attribution_timing_member_v1"
            if member["kind"] in {"timing", "large_model_timing"}
            else "attribution_image_member_v1"
        )
        member["depends_on"] = []
        member["job_sha256"] = canonical_job_sha256(member)
    target_jobs = {
        (
            str(job["scope"]),
            str(job["model_id"]),
            int(job["shard"]),
            str(job["method_id"]),
        ): job
        for job in members
        if str(job["method_id"]) in TARGET_METHODS
    }
    for member in members:
        member["depends_on"] = _dependency_specs(member, target_jobs)
        member["job_sha256"] = canonical_job_sha256(member)


def _model_image_count(scope: ScopeSpec, model: str) -> int:
    if scope.name in {
        "funnybirds_supplement",
        "funnybirds_deletion_targets",
        "funnybirds_heldout_targets",
        "funnybirds_primary",
    }:
        counts = {
            "funnybirds_resnet50": 499,
            "funnybirds_vgg16": 497,
            "funnybirds_vit_b_16": 488,
        }
        return counts[model]
    if scope.name in {"idsds_deletion_targets", "idsds_primary"}:
        counts = {
            "resnet50": 7_663,
            "vgg16": 7_189,
            "vit_base_patch16_224": 8_285,
        }
        return counts[model]
    if scope.name in {"idsds_full50k_deletion_targets", "idsds_full50k"}:
        counts = {
            "resnet50": 38_460,
            "vgg16": 36_042,
            "vit_base_patch16_224": 41_374,
        }
        return counts[model]
    if scope.name in {
        "partimagenet_deletion_targets",
        "partimagenet_heldout_targets",
        "partimagenet_boundary",
    }:
        return 1_024
    return scope.images


def _member(scope: ScopeSpec, model: str, method: str, shard: int, repeat: int) -> dict[str, Any]:
    model_images = _model_image_count(scope, model)
    base_count, remainder = divmod(model_images, scope.shards)
    image_start = shard * base_count + min(shard, remainder)
    image_count = base_count + int(shard < remainder)
    image_stop = image_start + image_count
    suffix = f"shard-{shard:05d}"
    if scope.repeats > 1:
        suffix += f"--repeat-{repeat:02d}"
    member_id = f"{scope.name}--{model}--{method}--{suffix}"
    base = f"{scope.name}/{model}/{method}/{suffix}"
    return {
        "schema_version": 1,
        "member_id": member_id,
        "scope": scope.name,
        "kind": scope.kind,
        "dataset": scope.dataset,
        "model_id": model,
        "method_id": method,
        "shard": shard,
        "repeat": repeat,
        "scope_image_count": scope.images,
        "model_image_count": model_images,
        "image_start": image_start,
        "image_stop": image_stop,
        "image_count": image_count,
        "dataset_manifest_sha256": scope.manifest_sha256,
        "output_path": f"raw/members/{base}.parquet",
        "receipt_path": f"receipts/members/{base}.json",
    }


def _scope_members(scope: ScopeSpec) -> list[dict[str, Any]]:
    return [
        _member(scope, model, method, shard, repeat)
        for model in scope.models
        for method in scope.methods
        for shard in range(scope.shards)
        for repeat in range(scope.repeats)
    ]


def _scope_dict(scope: ScopeSpec) -> dict[str, Any]:
    payload = asdict(scope)
    payload["member_count"] = scope.member_count
    return payload


def build_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build and strictly validate one static experiment plan."""

    profile = str(config.get("profile", ""))
    profile_key = _profile_key(config)
    all_profile_scopes = {**PROFILE_SCOPES, **VERIFICATION_PROFILE_SCOPES}
    all_member_counts = {**PROFILE_MEMBER_COUNTS, **VERIFICATION_PROFILE_MEMBER_COUNTS}
    if profile_key not in all_profile_scopes:
        raise ValueError(f"unknown attribution profile: {profile}")
    plan_config = config.get("plan", {})
    if plan_config is not None and not isinstance(plan_config, Mapping):
        raise TypeError("attribution plan configuration must be a mapping")
    configured = tuple((plan_config or {}).get("include", all_profile_scopes[profile_key]))
    if configured != all_profile_scopes[profile_key]:
        raise ValueError(
            f"profile {profile_key} scope contract drifted: "
            f"{configured} != {all_profile_scopes[profile_key]}"
        )
    selected = tuple(SCOPES[name] for name in configured)
    members = [member for scope in selected for member in _scope_members(scope)]
    sanitized_config = {key: value for key, value in config.items() if key != "_source"}
    plan_contract = {
        "schema_version": 1,
        "experiment": "attribution",
        "profile": profile,
        "profile_key": profile_key,
        "scope_names": list(configured),
        "scopes": [_scope_dict(scope) for scope in selected],
        "endpoint_m_stage": "analyze",
    }
    config_sha256 = canonical_sha256(sanitized_config)
    plan_contract_sha256 = canonical_sha256(plan_contract)
    _bind_members(
        members,
        config_sha256=config_sha256,
        plan_contract_sha256=plan_contract_sha256,
    )
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise TypeError("attribution execution configuration must be a mapping")
    formal = profile_key != "smoke"
    plan: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "attribution",
        "profile": profile,
        "profile_key": profile_key,
        "config_sha256": config_sha256,
        "plan_contract": plan_contract,
        "plan_contract_sha256": plan_contract_sha256,
        "endpoint_m_stage": "analyze",
        "execution_contract": {
            "backend": (
                "external_gpu_worker" if formal else str(execution.get("backend", "oracle"))
            ),
            "execution_claimed": False,
            "requires_gpu": formal,
            "adapter_configured": bool(execution.get("adapter")),
            "dataset_root_env": str(execution.get("dataset_root_env", "DECAF_DATA_ROOT")),
            "checkpoint_root_env": str(execution.get("checkpoint_root_env", "DECAF_CACHE_ROOT")),
            "fail_closed_assertions": [
                "adapter_callable",
                "cuda_available",
                "dataset_manifest_bytes_sha256",
                "checkpoint_bytes_sha256",
                "member_job_sha256",
                "exact_job_keys_and_image_range",
                "dependency_output_sha256",
            ],
            "ready_for_execution": (
                not formal
                or (bool(execution.get("adapter")) and bool(execution.get("requires_gpu")))
            ),
        },
        "aligned_architectures": [
            {"architecture": name, "imagenet": imagenet, "funnybirds": funnybirds}
            for name, imagenet, funnybirds in ALIGNED_ARCHITECTURES
        ],
        "scope_names": list(configured),
        "scopes": [_scope_dict(scope) for scope in selected],
        "members": members,
        "member_count": len(members),
        "expected_member_count": all_member_counts[profile_key],
        "contracts": {
            "idsds_primary_members": 780,
            "idsds_full50k_members": 900,
            "idsds_timing_members": 117,
            "funnybirds_supplement_members": 6,
            "funnybirds_primary_members": 33,
            "funnybirds_deletion_target_members": 3,
            "funnybirds_heldout_target_members": 6,
            "large_model_quality_members": 32,
            "large_model_timing_members": 24,
            "boundary_compute_members": 896,
            "boundary_target_members": 192,
            "boundary_analysis_methods": 15,
            "large_model_methods": 8,
            "aligned_architectures": 3,
        },
        "available_profiles": list((*PROFILE_SCOPES, *VERIFICATION_PROFILE_SCOPES)),
    }
    plan["audit"] = validate_plan(plan, raise_on_error=True)
    return plan


def validate_plan(plan: Mapping[str, Any], *, raise_on_error: bool = False) -> dict[str, Any]:
    """Audit counts, compatibility, hashes, and collision-free member paths."""

    errors: list[str] = []
    profile = str(plan.get("profile", ""))
    profile_key = str(plan.get("profile_key", profile))
    all_profile_scopes = {**PROFILE_SCOPES, **VERIFICATION_PROFILE_SCOPES}
    all_member_counts = {**PROFILE_MEMBER_COUNTS, **VERIFICATION_PROFILE_MEMBER_COUNTS}
    members_value = plan.get("members")
    members = members_value if isinstance(members_value, list) else []
    if profile_key not in all_profile_scopes:
        errors.append(f"unknown profile:{profile_key}")
    expected_total = all_member_counts.get(profile_key)
    if expected_total is not None and len(members) != expected_total:
        errors.append(f"member_count:{len(members)}!={expected_total}")
    if plan.get("endpoint_m_stage") != "analyze":
        errors.append("endpoint_m_stage")
    if any(member.get("method_id") == "endpoint_m" for member in members):
        errors.append("endpoint_m_compute_member")
    aligned = plan.get("aligned_architectures")
    if not isinstance(aligned, list) or len(aligned) != 3:
        errors.append("aligned_architecture_count")
    plan_contract = plan.get("plan_contract")
    if not isinstance(plan_contract, Mapping) or plan.get(
        "plan_contract_sha256"
    ) != canonical_sha256(plan_contract or {}):
        errors.append("plan_contract_sha256")
    if not _is_sha256(plan.get("config_sha256")):
        errors.append("config_sha256")
    execution_contract = plan.get("execution_contract")
    if not isinstance(execution_contract, Mapping):
        errors.append("execution_contract")
    elif profile_key != "smoke" and (
        execution_contract.get("backend") != "external_gpu_worker"
        or execution_contract.get("execution_claimed") is not False
        or execution_contract.get("requires_gpu") is not True
    ):
        errors.append("formal_execution_contract")

    identifiers = [str(member.get("member_id")) for member in members]
    outputs = [str(member.get("output_path")) for member in members]
    receipts = [str(member.get("receipt_path")) for member in members]
    for label, values in (
        ("member_id", identifiers),
        ("output_path", outputs),
        ("receipt_path", receipts),
    ):
        if len(values) != len(set(values)):
            errors.append(f"duplicate_{label}")

    actual_scopes = Counter(str(member.get("scope")) for member in members)
    selected_names = tuple(plan.get("scope_names", ()))
    for name in selected_names:
        scope = SCOPES.get(name)
        if scope is None:
            errors.append(f"unknown_scope:{name}")
            continue
        if scope.member_count != scope.expected_members:
            errors.append(f"scope_contract:{name}")
        if actual_scopes[name] != scope.expected_members:
            errors.append(f"scope_member_count:{name}")

    model_ids = tuple(sorted({str(member.get("model_id")) for member in members}))
    try:
        checkpoint_coverage(model_ids)
    except (KeyError, ValueError) as error:
        errors.append(f"checkpoint_coverage:{error}")
    target_jobs = {
        (
            str(job.get("scope")),
            str(job.get("model_id")),
            int(job.get("shard", -1)),
            str(job.get("method_id")),
        ): job
        for job in members
        if str(job.get("method_id")) in TARGET_METHODS
    }
    shard_groups: dict[tuple[str, str, str, int], list[dict[str, int]]] = defaultdict(list)
    for member in members:
        model_id = str(member.get("model_id"))
        dataset = str(member.get("dataset"))
        method_id = str(member.get("method_id"))
        scope_name = str(member.get("scope"))
        try:
            if not supports_dataset(model_id, dataset):
                errors.append(f"dataset_compatibility:{model_id}:{dataset}")
            if method_id not in TARGET_METHODS:
                method = get_method(method_id)
                if method.analysis_only:
                    errors.append(f"analysis_only_member:{method_id}")
        except KeyError as error:
            errors.append(f"registry:{error}")
        digest = member.get("dataset_manifest_sha256")
        if dataset != "oracle" and not _is_sha256(digest):
            errors.append(f"dataset_hash:{member.get('member_id')}")
        for digest_name in (
            "config_sha256",
            "plan_contract_sha256",
            "checkpoint_contract_sha256",
            "input_contract_sha256",
            "job_sha256",
        ):
            if not _is_sha256(member.get(digest_name)):
                errors.append(f"{digest_name}:{member.get('member_id')}")
        if member.get("config_sha256") != plan.get("config_sha256"):
            errors.append(f"config_binding:{member.get('member_id')}")
        if member.get("plan_contract_sha256") != plan.get("plan_contract_sha256"):
            errors.append(f"plan_binding:{member.get('member_id')}")
        if member.get("job_sha256") != canonical_job_sha256(member):
            errors.append(f"job_sha256:{member.get('member_id')}")
        try:
            expected_dependencies = _dependency_specs(member, target_jobs)
        except AssertionError as error:
            errors.append(f"dependency:{error}")
        else:
            if member.get("depends_on") != expected_dependencies:
                errors.append(f"dependency_binding:{member.get('member_id')}")
        try:
            shard_record = {
                "shard": int(member["shard"]),
                "start": int(member["image_start"]),
                "stop": int(member["image_stop"]),
                "count": int(member["image_count"]),
                "total": int(member["model_image_count"]),
            }
            repeat = int(member["repeat"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"shard_schema:{member.get('member_id')}")
        else:
            if (
                shard_record["start"] < 0
                or shard_record["stop"] <= shard_record["start"]
                or shard_record["count"] != shard_record["stop"] - shard_record["start"]
                or shard_record["stop"] > shard_record["total"]
            ):
                errors.append(f"shard_bounds:{member.get('member_id')}")
            shard_groups[(scope_name, model_id, method_id, repeat)].append(shard_record)

    for key, group in shard_groups.items():
        ordered = sorted(group, key=lambda item: item["shard"])
        scope = SCOPES.get(key[0])
        expected_shards = scope.shards if scope is not None else -1
        contiguous = all(
            left["stop"] == right["start"]
            for left, right in zip(ordered, ordered[1:], strict=False)
        )
        if (
            len(ordered) != expected_shards
            or [item["shard"] for item in ordered] != list(range(expected_shards))
            or ordered[0]["start"] != 0
            or ordered[-1]["stop"] != ordered[0]["total"]
            or sum(item["count"] for item in ordered) != ordered[0]["total"]
            or not contiguous
        ):
            errors.append(f"shard_coverage:{':'.join(map(str, key))}")

    result = {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "checked_members": len(members),
        "unique_member_ids": len(set(identifiers)),
        "unique_output_paths": len(set(outputs)),
        "unique_receipt_paths": len(set(receipts)),
        "checked_shard_groups": len(shard_groups),
        "aligned_architecture_count": len(ALIGNED_ARCHITECTURES),
        "endpoint_m_generated_in_analyze": plan.get("endpoint_m_stage") == "analyze",
        "large_model_profile_present": "large-model" in PROFILE_SCOPES,
        "boundary_profile_present": "boundary" in PROFILE_SCOPES,
        "single_b200_profiles_present": set(VERIFICATION_PROFILE_SCOPES)
        == {"smoke-b200", "large-model-smoke", "boundary-smoke", "smoke-resume"},
    }
    if errors and raise_on_error:
        raise AssertionError(f"attribution plan audit failed: {result}")
    return result


__all__ = [
    "DELETION_TARGET_METHOD",
    "FUNNYBIRDS_DELETION_TARGET_METHOD",
    "FUNNYBIRDS_HELDOUT_METHODS",
    "PROFILE_MEMBER_COUNTS",
    "PROFILE_SCOPES",
    "VERIFICATION_PROFILE_MEMBER_COUNTS",
    "VERIFICATION_PROFILE_SCOPES",
    "SCOPES",
    "ScopeSpec",
    "build_plan",
    "validate_plan",
]
