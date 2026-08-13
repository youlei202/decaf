"""Fail-closed reporting and packaging for cross-generation verification V2.

The finalizer consumes completed family artifacts.  It never runs experiments;
missing or failed mandatory evidence is an error rather than a partial report.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from decaf.experiments.attribution.endpoint import row_spearman
from decaf.experiments.imagenet9.evaluate import evaluate_response_frame
from tools.crossgen import compare_core as core_comparison
from tools.crossgen.schema import read_trajectory_record

SUMMARY_NAMES = ("M", "E", "C", "F", "Abs")
MANDATORY_FAMILIES = ("controlled", "imagenet9", "attribution", "covertype")
ALLOWED_STATUSES = {
    "PASS_CORE_AND_E2E",
    "PASS_CORE",
    "PASS_CORE_WITH_METADATA_LIMIT",
    "BLOCKED",
    "FAIL_NUMERICAL",
}
PASSING_CORE_STATUSES = {
    "PASS_CORE_AND_E2E",
    "PASS_CORE",
    "PASS_CORE_WITH_METADATA_LIMIT",
}
DEFAULT_ROOT = Path("/work/Users/leiyo/decaf_cross_generation_equivalence/v2")
DEFAULT_REPOSITORY = Path("/work/Users/leiyo/GitHub/decaf")
DEFAULT_PACKAGE_DIRECTORY = Path("/work/Users/leiyo/decaf_repro_release/packages")
DEFAULT_B200_ROOT = Path("/work/Users/leiyo/decaf_b200_verification")
DEFAULT_REPLAY_ROOT = DEFAULT_ROOT / "verification"

C0_QUALIFICATION_STATUS = "STRICT_AGGREGATE_QUALIFIED_WITH_EXCLUSIONS"
C0_EXCLUSION_REASON = "UNRESOLVED_HISTORICAL_RUNTIME_METADATA"
C0_QUALIFICATION_TOLERANCE = 5.0e-4
C0_DIAGNOSTIC_RELATIVE = Path("diagnostics/c0_all_selection_aggregate_audit.json")
C0_RUNTIME_ATTRIBUTION_RELATIVE = Path("diagnostics/c0_runtime_attribution.json")
C0_RUNTIME_ATTRIBUTION_SHA256 = "c67a425eab34ad30dfd8d27ff3b2583461a00280d98c023e4d2d6b13dfa04563"
C0_PHASE_UNITS = {"c0": 6, "c1": 10, "c2": 12}
C0_EXPECTED_CANDIDATES: dict[tuple[str, int], dict[str, Any]] = {
    ("context_gate__resnet18__seed_3101", 372490): {
        "task": "context_gate",
        "architecture": "resnet18",
        "model_seed": 3101,
        "factor": "wall_color",
        "cf_map_seed": "20260882",
        "noise_seed": 20260884,
        "expected_state": "active",
    },
    ("context_gate__resnet18__seed_3101", 75310): {
        "task": "context_gate",
        "architecture": "resnet18",
        "model_seed": 3101,
        "factor": "wall_color",
        "cf_map_seed": "20260883",
        "noise_seed": 20260884,
        "expected_state": "null",
    },
    ("context_gate__small_vit__seed_3101", 170664): {
        "task": "context_gate",
        "architecture": "small_vit",
        "model_seed": 3101,
        "factor": "object_color",
        "cf_map_seed": "20260882",
        "noise_seed": 20260884,
        "expected_state": "active",
    },
    ("context_gate__small_vit__seed_3101", 222025): {
        "task": "context_gate",
        "architecture": "small_vit",
        "model_seed": 3101,
        "factor": "object_color",
        "cf_map_seed": "20260883",
        "noise_seed": 20260884,
        "expected_state": "null",
    },
    ("color_shape_xor__resnet18__seed_3101", 74606): {
        "task": "color_shape_xor",
        "architecture": "resnet18",
        "model_seed": 3101,
        "factor": "object_color",
        "cf_map_seed": "20260882",
        "noise_seed": 20260884,
        "expected_state": "active",
    },
    ("color_shape_xor__resnet18__seed_3101", 313393): {
        "task": "color_shape_xor",
        "architecture": "resnet18",
        "model_seed": 3101,
        "factor": "object_color",
        "cf_map_seed": "20260882",
        "noise_seed": 20260884,
        "expected_state": "null",
    },
    ("color_shape_xor__small_vit__seed_3101", 313393): {
        "task": "color_shape_xor",
        "architecture": "small_vit",
        "model_seed": 3101,
        "factor": "object_color",
        "cf_map_seed": "20260883",
        "noise_seed": 20260884,
        "expected_state": "active",
    },
    ("color_shape_xor__small_vit__seed_3101", 356967): {
        "task": "color_shape_xor",
        "architecture": "small_vit",
        "model_seed": 3101,
        "factor": "object_color",
        "cf_map_seed": "20260883",
        "noise_seed": 20260884,
        "expected_state": "null",
    },
}
C0_EXPECTED_EXCLUSION_ERRORS = {
    ("context_gate__resnet18__seed_3101", 75310): 0.0026714954991345565,
    ("context_gate__small_vit__seed_3101", 222025): 0.039401795786877,
}
IMAGENET9_HISTORICAL_PACKAGE_SHA256 = (
    "3bae5ac670f6731d8a7832c3f9d7051e308a3f322c6192068bc11868be3821cc"
)
IMAGENET9_PACKAGE_PREFIX = "decaf_imagenet9_v1"
IMAGENET9_MANIFEST_MEMBER = f"{IMAGENET9_PACKAGE_PREFIX}/PACKAGE_MANIFEST.json"
IMAGENET9_SOURCE_RELATIVES = {
    f"code/cmr/decaf_imagenet9_v1/{name}.py"
    for name in ("__init__", "data", "decaf", "models", "reveal", "run")
}
COVERTYPE_HISTORICAL_PACKAGE_SHA256 = (
    "e9acaf30491dcdf654fdfb691df915e19d75e9c19d0ffe2546312d0d34f87927"
)
COVERTYPE_HISTORICAL_MANIFEST_SHA256 = (
    "e007e32828e1ab7d5ff5a272b361ddfed26485b9445a4a575badbb98ed8de0da"
)
COVERTYPE_PARENT_SHIM_SHA256 = (
    "69a6d26a02481d01849e23536e7aa0dd3104e9379f2a0f51bd4eff7114791816"
)
COVERTYPE_MANIFEST_MEMBER = "PACKAGE_CONTENTS.json"
COVERTYPE_ARCHIVE_SOURCE_PREFIX = "code/src"
COVERTYPE_NAMESPACE_PREFIX = "code/src/cmr/decaf_covertype_v1/"
COVERTYPE_REQUIRED_MODULES = {
    "__init__",
    "behaviors",
    "compatibility",
    "config",
    "data",
    "decaf",
    "mechanisms",
    "models",
}
COVERTYPE_LOADED_MODULES = COVERTYPE_REQUIRED_MODULES | {"io"}
DINO_REPOSITORY_COMMIT = "aed3391d4869f82570bea0dba765b9dda0b3d359"
DINO_REPOSITORY_TREE = "54f2a7a60e14f0d3df334dd7b35b24a613042f70"
DINO_REQUIRED_GATES = {
    "single_b200_detected",
    "dinov2_g_real_b200_shard",
    "checkpoint_fingerprints",
    "repository_audit",
}
DINO_SCOPES = ("smoke_dinov2_g_quality", "smoke_dinov2_g_timing")
ATTRIBUTION_BRIDGES = {
    "funnybirds__funnybirds_resnet50": ("funnybirds", "funnybirds_resnet50"),
    "funnybirds__funnybirds_vgg16": ("funnybirds", "funnybirds_vgg16"),
    "funnybirds__funnybirds_vit_b_16": ("funnybirds", "funnybirds_vit_b_16"),
    "imagenet1k_idsds__resnet50": ("imagenet1k_idsds", "resnet50"),
    "imagenet1k_idsds__vgg16": ("imagenet1k_idsds", "vgg16"),
    "imagenet1k_idsds__vit_base_patch16_224": (
        "imagenet1k_idsds",
        "vit_base_patch16_224",
    ),
}
ATTRIBUTION_SELECTION_SHA256 = {
    "funnybirds__funnybirds_resnet50": (
        "ae5f4244446a5b80faa9b564063ef64137b7490a1181e0da2dcc5fbd40d455c1"
    ),
    "funnybirds__funnybirds_vgg16": (
        "91e1c5584f2ba0484ef1312afae8bccdb05338042be0f1133da954a53e4e93bf"
    ),
    "funnybirds__funnybirds_vit_b_16": (
        "a5c60ad9caa456850d375b5a315061e4b70f65b09a9f068ada0414622c5d6fb8"
    ),
    "imagenet1k_idsds__resnet50": (
        "9aaaf047c0280519abdbfd500c282bb7d8ad48f0beb59edce21edc6d1fe08c1a"
    ),
    "imagenet1k_idsds__vgg16": (
        "776ace1858dc0bf2fb8235ae8056d37b2b32b26d96bed4944f25845785594ccc"
    ),
    "imagenet1k_idsds__vit_base_patch16_224": (
        "9c65f08b447442d62c716e0d34aa4bbc905598fb387dc702c29bd313176a334f"
    ),
}
ATTRIBUTION_A0_SOURCE_TREE_SHA256 = (
    "46c53e874e685d95eb7bd06649ae747fd8d903b4805bebfb63a6b78abe33cfa9"
)
ATTRIBUTION_A0_DEPLOYMENT_SHA256 = (
    "f866a79876eaaff8ca81652394e2b9383665f869f9f8fd2061e49be1732d7703"
)
ATTRIBUTION_A0_PLAN_SHA256 = (
    "6d93f2ee4a0e52a338c74f755f0bec7b0ff1b2a8e1b909ab16a1a8e617e0531a"
)
ATTRIBUTION_A0_PLAN_RECEIPT_SHA256 = (
    "3789c430ebe7a1a05c973dfdc0f52b2850bff667e9fdfe45e1e56fbf039b6f05"
)
ATTRIBUTION_A0_PACKAGE_SHA256 = (
    "7ecad798213d41662749625692618c615135776009ece187fd6a72adf067d420"
)
ATTRIBUTION_A0_HELDOUT_QUALITY_MEMBER = "formal/heldout_quality.parquet"
ATTRIBUTION_A0_HELDOUT_QUALITY_SHA256 = (
    "b1817e81879c15a99845ee671800da2aa4e8e0ac2af223ea5ae5ceda20ff10a8"
)
ATTRIBUTION_A2_FUNNY_QUALITY_MEMBER = "results/funnybirds/reused_quality.parquet"
ATTRIBUTION_A2_FUNNY_QUALITY_SHA256 = (
    "bbcf4473d95dc124b13a6529f0d871e18d11199c26274cb2378801725a2de009"
)
ATTRIBUTION_A0_REQUIRED_MODULES = {"methods", "models", "worker"}
ATTRIBUTION_A0_LOADED_MODULES = {
    "__init__",
    "config",
    "contracts",
    "data",
    "evaluation",
    "io",
    "jobs",
    "methods",
    "models",
    "scheduler",
    "worker",
}
ATTRIBUTION_A0_ANCHORS = {
    "cmr/decaf_reference_locked_v1/__init__.py": (
        "5f7fd80c768cfa51c55ef3e96a20e5ab2ff4c77853f0ee62061d2582961d6227"
    ),
    "cmr/decaf_imagenet9_v1/decaf.py": (
        "13c27814992ff86c4c6b59afc6bedb178ec73059f237b6d96c14d29dc35aeb41"
    ),
}
ATTRIBUTION_A2_PACKAGE_SHA256 = (
    "f68ed1fec48b39403fb677492283066f853722f466ce703edd5b468d59cc93a4"
)
ATTRIBUTION_A2_MANIFEST_SHA256 = (
    "6689282bef7fb97ca0a77174dff6c259ad3e348180a16d3de5cac50f15d40be5"
)
ATTRIBUTION_A2_PAYLOAD_TREE_SHA256 = (
    "5a1f0bc9215b4c75f139400165c4450995e6189a92fc73196b5b91c007291598"
)
ATTRIBUTION_A2_PARENT_SHIM_SHA256 = (
    "f1dd0f83231221587126d2960e30052c3511b0e56a7fe1c0f5836e8f49a7f909"
)
ATTRIBUTION_A2_REQUIRED_MODULES = {
    "attribution",
    "data",
    "models",
    "contracts",
    "decomposition",
    "idsds",
}
ATTRIBUTION_A2_MANIFEST_MEMBER = "PACKAGE_MANIFEST.json"
ATTRIBUTION_A2_SOURCE_PREFIX = "code_snapshot/src"
ATTRIBUTION_A2_NAMESPACE = "cmr.decaf_idsds_funnybirds_v1"
ATTRIBUTION_ENDPOINT_EPSILON = 0.02
ATTRIBUTION_SPEARMAN_COLUMNS = {
    "dataset",
    "model",
    "method",
    "image_id",
    "historical_spearman",
    "current_spearman",
    "signed_error",
    "absolute_error",
    "tier_a_pass",
    "tier_b_pass",
    "tier",
    "hard_mismatch",
}
ATTRIBUTION_SPEARMAN_IDENTITY_COLUMNS = ("dataset", "model", "method", "image_id")
ATTRIBUTION_VECTOR_METRICS = ("score", *SUMMARY_NAMES)
IMAGENET9_E2E_MODEL_BINDINGS = {
    "ft_resnet50_original_s7101": {
        "historical_model_id": "ft_resnet50_original_s7101",
        "checkpoint_sha256": (
            "fd88804bae846b971fbcac05236c82b2fc385a3ce1357d1aabd5b87dd5134130"
        ),
        "sealed_sample_sha256": (
            "85cc3d8e63321342e1cacf5e9042ee7ef4d6ba7a8f2fbdf579cde2e506dd8fdf"
        ),
    },
    "ft_vit_b_16_original_s7101": {
        "historical_model_id": "ft_vit_b_16_original_s7101",
        "checkpoint_sha256": (
            "55d9c142dab8b4936971421c97939d95eab6e718c18d65b977545eab37fa95ef"
        ),
        "sealed_sample_sha256": (
            "4069a212bb0ea6b1c04a489dc022aabb958d7d47307903e1a7cacbc10cf2167c"
        ),
    },
    "tv_resnet18_imagenet1k_v1": {
        "historical_model_id": "tv_resnet18",
        "checkpoint_sha256": (
            "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
        ),
        "sealed_sample_sha256": (
            "e264fbe9abe62c2b75eb95114c9defe09853a7235ee0b4ffc97a683fca62c3b9"
        ),
    },
}
IMAGENET9_PAIR_TYPES = {"same_next", "same_rand"}
IMAGENET9_REVEAL_PATHS = {"blend", "patch_A", "patch_B"}
HISTORICAL_SNAPSHOT_ROOT = Path("provenance/historical_source_snapshots")
HISTORICAL_SNAPSHOT_RELATIVES = {
    "attribution_a0_deployment_receipt": (
        HISTORICAL_SNAPSHOT_ROOT / "attribution_a0/deployment_receipt.json"
    ),
    "attribution_a0_formal_plan": (
        HISTORICAL_SNAPSHOT_ROOT / "attribution_a0/formal_jobs.jsonl"
    ),
    "attribution_a0_formal_plan_receipt": (
        HISTORICAL_SNAPSHOT_ROOT / "attribution_a0/formal_jobs.jsonl.receipt.json"
    ),
    "attribution_a2_package_manifest": (
        HISTORICAL_SNAPSHOT_ROOT / "attribution_a2/PACKAGE_MANIFEST.json"
    ),
    "covertype_package_manifest": (
        HISTORICAL_SNAPSHOT_ROOT / "covertype/PACKAGE_CONTENTS.json"
    ),
}

CORE_COLUMNS = {
    "unit_id",
    "boundary",
    "tier",
    "hard_mismatch",
    "gate_match",
    "orientation_match",
    "dominant_match",
    "identity_match",
    *(f"abs_error_{name}" for name in SUMMARY_NAMES),
    *(f"signed_error_{name}" for name in SUMMARY_NAMES),
    *(f"historical_{name}" for name in SUMMARY_NAMES),
}
PACKAGE_DIRECTORIES = {
    "comparisons",
    "manifests",
    "provenance",
    "readiness",
    "trajectories",
}
PACKAGE_ROOT_FILES = {
    "CROSS_GENERATION_EQUIVALENCE_REPORT_V2.md",
    "CROSS_GENERATION_EQUIVALENCE_STATUS_V2.json",
}
PACKAGE_EXACT_FILES = {
    C0_DIAGNOSTIC_RELATIVE.as_posix(),
    C0_RUNTIME_ATTRIBUTION_RELATIVE.as_posix(),
    "verification/analysis_replay.json",
    "verification/headline_assertions.json",
    "verification/cpu_verification.json",
    "verification/paper_artifact_diff.csv",
    "verification/paper_outputs/receipts/canonical_receipt.json",
    "verification/paper_outputs/receipts/family_replay_receipt.json",
    "verification/paper_outputs/receipts/replay_receipt.json",
}
PACKAGE_SUFFIXES = {
    ".bundle",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".ndjson",
    ".parquet",
    ".patch",
    ".sha256",
    ".txt",
}
PACKAGE_PYTHON_SUBTREES = (
    PurePosixPath("provenance/historical_sources/attribution_idsds"),
    PurePosixPath("provenance/historical_sources/covertype"),
)
FORBIDDEN_PACKAGE_PARTS = {
    ".cache",
    "__pycache__",
    "checkpoints",
    "datasets",
    "logs",
    "private",
    "smoke",
    "venv",
}
MAX_PACKAGE_MEMBER_BYTES = 64 * 1024 * 1024
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
UTC_TIMEZONE = timezone.utc  # noqa: UP017 -- Python 3.10 compatibility


class FinalizationError(RuntimeError):
    """Raised when final evidence fails the V2 contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FinalizationError(f"missing mandatory {label}: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizationError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise FinalizationError(f"{label} must encode a JSON object: {path}")
    return payload


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    _require_file(path, label)
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise FinalizationError(f"invalid {label}: {path}: {error}") from error
    if frame.empty:
        raise FinalizationError(f"{label} is empty: {path}")
    return frame


def _bool_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    if column not in frame:
        raise FinalizationError(f"{label} lacks {column!r}")
    values = frame[column]
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise FinalizationError(f"{label}.{column} contains non-boolean values")
    return normalized.eq("true")


def _finite_numeric(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    if column not in frame:
        raise FinalizationError(f"{label} lacks {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FinalizationError(f"{label}.{column} contains non-finite values")
    return values


def aggregate_unit_comparisons(frames: Sequence[pd.DataFrame], *, label: str) -> dict[str, Any]:
    """Aggregate exact unit-weighted comparison statistics."""

    if not frames:
        raise FinalizationError(f"no comparison frames supplied for {label}")
    normalized: list[pd.DataFrame] = []
    for index, raw in enumerate(frames):
        frame = raw.copy()
        missing = sorted(CORE_COLUMNS.difference(frame.columns))
        if missing:
            raise FinalizationError(f"{label}[{index}] lacks columns: {', '.join(missing)}")
        if frame.empty:
            raise FinalizationError(f"{label}[{index}] is empty")
        for column in (
            "boundary",
            "hard_mismatch",
            "gate_match",
            "orientation_match",
            "dominant_match",
            "identity_match",
        ):
            frame[column] = _bool_series(frame, column, f"{label}[{index}]")
        if not frame["tier"].astype(str).isin({"A", "B", "FAIL"}).all():
            raise FinalizationError(f"{label}[{index}].tier contains invalid values")
        for name in SUMMARY_NAMES:
            frame[f"abs_error_{name}"] = _finite_numeric(
                frame, f"abs_error_{name}", f"{label}[{index}]"
            )
            frame[f"signed_error_{name}"] = _finite_numeric(
                frame, f"signed_error_{name}", f"{label}[{index}]"
            )
            frame[f"historical_{name}"] = _finite_numeric(
                frame, f"historical_{name}", f"{label}[{index}]"
            )
        normalized.append(frame)
    combined = pd.concat(normalized, ignore_index=True)
    if combined["unit_id"].astype(str).duplicated().any():
        duplicates = sorted(
            combined.loc[combined["unit_id"].astype(str).duplicated(keep=False), "unit_id"]
            .astype(str)
            .unique()
        )
        raise FinalizationError(f"duplicate {label} unit IDs: {duplicates[:5]}")

    non_boundary = ~combined["boundary"]
    non_boundary_count = int(non_boundary.sum())
    if non_boundary_count == 0:
        raise FinalizationError(f"{label} contains no non-boundary units")
    tier = combined["tier"].astype(str)
    all_errors = np.concatenate(
        [combined[f"abs_error_{name}"].to_numpy(dtype=np.float64) for name in SUMMARY_NAMES]
    )
    metrics: dict[str, Any] = {}
    for name in SUMMARY_NAMES:
        errors = combined[f"abs_error_{name}"].to_numpy(dtype=np.float64)
        signed = combined[f"signed_error_{name}"].to_numpy(dtype=np.float64)
        historical = combined[f"historical_{name}"].to_numpy(dtype=np.float64)
        agreement = np.isclose(
            errors,
            0.0,
            atol=5.0e-4 + 5.0e-3 * np.abs(historical),
            rtol=0.0,
        ) | (errors <= 2.0e-3)
        metrics[name] = {
            "median_absolute_error": float(np.median(errors)),
            "p95_absolute_error": float(np.percentile(errors, 95)),
            "maximum_absolute_error": float(np.max(errors)),
            "mean_signed_error": float(np.mean(signed)),
            "agreement": float(agreement.mean()),
        }
    return {
        "unit_count": int(len(combined)),
        "non_boundary_unit_count": non_boundary_count,
        "tier_a_count": int(tier.eq("A").sum()),
        "tier_b_count": int(tier.eq("B").sum()),
        "fail_count": int(tier.eq("FAIL").sum()),
        "tier_a_fraction": float(tier.eq("A").mean()),
        "tier_b_fraction": float(tier.eq("B").mean()),
        "tier_a_or_b_fraction": float(tier.isin({"A", "B"}).mean()),
        "non_boundary_tier_a_or_b_fraction": float(tier.loc[non_boundary].isin({"A", "B"}).mean()),
        "hard_mismatch_count": int(combined["hard_mismatch"].sum()),
        "hard_mismatch_fraction": float(combined["hard_mismatch"].mean()),
        "gate_agreement": float(combined.loc[non_boundary, "gate_match"].mean()),
        "orientation_agreement": float(combined.loc[non_boundary, "orientation_match"].mean()),
        "dominant_mechanism_agreement": float(combined["dominant_match"].mean()),
        "identity_agreement": float(combined["identity_match"].mean()),
        "median_absolute_error": float(np.median(all_errors)),
        "p95_absolute_error": float(np.percentile(all_errors, 95)),
        "maximum_absolute_error": float(np.max(all_errors)),
        "metric_summaries": metrics,
    }


def _acceptance(stats: Mapping[str, Any]) -> dict[str, bool]:
    signed = [float(stats["metric_summaries"][name]["mean_signed_error"]) for name in SUMMARY_NAMES]
    return {
        "non_boundary_tier_a_or_b_at_least_95pct": float(stats["non_boundary_tier_a_or_b_fraction"])
        >= 0.95,
        "gate_at_least_99pct": float(stats["gate_agreement"]) >= 0.99,
        "orientation_at_least_99pct": float(stats["orientation_agreement"]) >= 0.99,
        "dominant_at_least_95pct": float(stats["dominant_mechanism_agreement"]) >= 0.95,
        "identity_exact": float(stats["identity_agreement"]) == 1.0,
        "no_systematic_scientific_bias": max(map(abs, signed)) <= 0.002,
    }


def _validate_core_family(family: str, stats: Mapping[str, Any]) -> None:
    failures = [name for name, passed in _acceptance(stats).items() if not passed]
    if failures:
        raise FinalizationError(
            f"mandatory {family} current-core acceptance failed: {', '.join(failures)}"
        )


def _summary_status(summary: Mapping[str, Any], label: str) -> str:
    status = str(summary.get("status", ""))
    if status not in ALLOWED_STATUSES:
        raise FinalizationError(f"{label} has invalid or missing status: {status!r}")
    return status


def _validate_core_summary(
    summary: Mapping[str, Any], stats: Mapping[str, Any], *, label: str
) -> None:
    if int(summary.get("unit_count", -1)) != int(stats["unit_count"]):
        raise FinalizationError(f"{label} unit count differs from its CSV")
    for key in (
        "tier_a_fraction",
        "tier_b_fraction",
        "tier_a_or_b_fraction",
        "hard_mismatch_fraction",
        "gate_agreement",
        "orientation_agreement",
        "dominant_mechanism_agreement",
        "identity_agreement",
    ):
        if key not in summary:
            continue
        if not np.isclose(float(summary[key]), float(stats[key]), atol=1.0e-12, rtol=0.0):
            raise FinalizationError(f"{label}.{key} differs from its unit CSV")


def _validate_artifact_binding(
    record: Mapping[str, Any],
    expected_path: Path,
    *,
    path_key: str,
    sha_key: str,
    label: str,
) -> str:
    expected_path = _require_file(expected_path, label).resolve()
    recorded_path = record.get(path_key)
    recorded_sha256 = record.get(sha_key)
    if (
        not isinstance(recorded_path, str)
        or not Path(recorded_path).is_absolute()
        or Path(recorded_path).resolve() != expected_path
    ):
        raise FinalizationError(f"{label} has a stale or mixed {path_key} path")
    digest = sha256_file(expected_path)
    if recorded_sha256 != digest:
        raise FinalizationError(f"{label} {sha_key} differs from the actual artifact")
    return digest


def _validate_recomputed_core_comparison(
    actual: pd.DataFrame, recomputed: pd.DataFrame, *, label: str
) -> None:
    if set(actual.columns) != set(recomputed.columns):
        missing = sorted(set(recomputed.columns) - set(actual.columns))
        unexpected = sorted(set(actual.columns) - set(recomputed.columns))
        raise FinalizationError(
            f"{label} columns differ from in-memory current-core recomputation: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(actual) != len(recomputed):
        raise FinalizationError(f"{label} row count differs from current-core recomputation")
    actual = actual.sort_values("unit_id", kind="stable").reset_index(drop=True)
    recomputed = recomputed.sort_values("unit_id", kind="stable").reset_index(drop=True)
    for column in recomputed.columns:
        expected = recomputed[column]
        if expected.dtype == bool:
            observed = _bool_series(actual, column, label)
            if not np.array_equal(observed.to_numpy(dtype=bool), expected.to_numpy(dtype=bool)):
                raise FinalizationError(f"{label}.{column} differs from current-core recomputation")
        elif pd.api.types.is_numeric_dtype(expected.dtype):
            observed = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=np.float64)
            wanted = expected.to_numpy(dtype=np.float64)
            if not np.isfinite(observed).all() or not np.allclose(
                observed,
                wanted,
                atol=1.0e-14,
                rtol=1.0e-12,
            ):
                raise FinalizationError(f"{label}.{column} differs from current-core recomputation")
        elif actual[column].astype(str).tolist() != expected.astype(str).tolist():
            raise FinalizationError(f"{label}.{column} differs from current-core recomputation")


def _validate_core_artifact_chain(
    *,
    trajectory_path: Path,
    comparison_path: Path,
    comparison: pd.DataFrame,
    summary: Mapping[str, Any],
    label: str,
    selection_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trajectory_sha256 = _validate_artifact_binding(
        summary,
        trajectory_path,
        path_key="trajectory_record",
        sha_key="trajectory_record_sha256",
        label=f"{label} summary trajectory",
    )
    if selection_manifest is not None:
        manifest_sha256 = _validate_artifact_binding(
            selection_manifest,
            trajectory_path,
            path_key="trajectory_record",
            sha_key="trajectory_record_sha256",
            label=f"{label} selection-manifest trajectory",
        )
        if manifest_sha256 != trajectory_sha256:
            raise FinalizationError(f"{label} summary/manifest trajectory hashes differ")
    try:
        trajectory = read_trajectory_record(trajectory_path)
        recomputed = pd.DataFrame(
            [
                core_comparison._unit_comparison(unit)  # noqa: SLF001
                for _, unit in trajectory.groupby("unit_id", sort=True)
            ]
        )
    except Exception as error:
        raise FinalizationError(f"{label} trajectory could not be recomputed: {error}") from error
    if recomputed.empty:
        raise FinalizationError(f"{label} trajectory recomputation produced no units")
    _validate_recomputed_core_comparison(comparison, recomputed, label=f"{label} core CSV")
    return {
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": trajectory_sha256,
        "comparison_path": str(comparison_path.resolve()),
        "comparison_sha256": sha256_file(comparison_path),
        "recomputed_unit_count": int(len(recomputed)),
        "validation": "in_memory_current_core_exact_fields",
    }


def _c0_identity(record: Mapping[str, Any], label: str) -> tuple[str, int]:
    model_id = record.get("model_id")
    base_id = record.get("base_id")
    if not isinstance(model_id, str) or not model_id:
        raise FinalizationError(f"{label} has invalid model_id")
    if isinstance(base_id, bool) or not isinstance(base_id, int):
        raise FinalizationError(f"{label} has invalid base_id")
    return model_id, base_id


def _exact_c0_count(record: Mapping[str, Any], key: str, expected: int, label: str) -> None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise FinalizationError(f"{label}.{key} must be exactly {expected}, got {value!r}")


def _exact_c0_float(value: Any, expected: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizationError(f"{label} must be a JSON number")
    observed = float(value)
    if not np.isfinite(observed) or observed != expected:
        raise FinalizationError(f"{label} must be exactly {expected!r}, got {value!r}")


def _c0_records(record: Mapping[str, Any], key: str, expected: int) -> list[dict[str, Any]]:
    values = record.get(key)
    if not isinstance(values, list) or len(values) != expected:
        raise FinalizationError(
            f"C0 candidate qualification {key} must contain exactly {expected} records"
        )
    if any(not isinstance(value, dict) for value in values):
        raise FinalizationError(f"C0 candidate qualification {key} contains a non-object")
    return values


def _validate_c0_candidate_record(
    record: Mapping[str, Any],
    *,
    label: str,
    should_qualify: bool,
) -> tuple[str, int]:
    identity = _c0_identity(record, label)
    expected_metadata = C0_EXPECTED_CANDIDATES.get(identity)
    if expected_metadata is None:
        raise FinalizationError(f"{label} replaced a fixed candidate: {identity!r}")
    for key, expected in expected_metadata.items():
        if record.get(key) != expected:
            raise FinalizationError(
                f"{label}.{key} changed for {identity!r}: {record.get(key)!r} != {expected!r}"
            )
    audit = record.get("audit")
    if not isinstance(audit, dict):
        raise FinalizationError(f"{label} lacks the full candidate audit")
    if record.get("qualified") is not should_qualify or audit.get("passed") is not should_qualify:
        raise FinalizationError(f"{label} has an inconsistent qualification flag")
    maximum_error = audit.get("maximum_absolute_error")
    if isinstance(maximum_error, bool) or not isinstance(maximum_error, (int, float)):
        raise FinalizationError(f"{label}.audit.maximum_absolute_error must be a JSON number")
    maximum_error = float(maximum_error)
    if not np.isfinite(maximum_error):
        raise FinalizationError(f"{label}.audit.maximum_absolute_error is non-finite")
    _exact_c0_float(
        record.get("maximum_absolute_error"),
        maximum_error,
        f"{label}.maximum_absolute_error",
    )
    _exact_c0_float(
        record.get("qualification_tolerance"),
        C0_QUALIFICATION_TOLERANCE,
        f"{label}.qualification_tolerance",
    )
    _exact_c0_float(
        audit.get("tolerance"),
        C0_QUALIFICATION_TOLERANCE,
        f"{label}.audit.tolerance",
    )
    metric_keys = {
        "endpoint_abs",
        "auc_abs_info",
        "auc_align_info",
        "auc_opp_info",
        "auc_null_info",
    }
    absolute_errors = audit.get("absolute_errors")
    recomputed = audit.get("recomputed")
    sealed = audit.get("sealed")
    forward_layout = audit.get("forward_layout")
    if any(
        not isinstance(values, dict) or set(values) != metric_keys
        for values in (absolute_errors, recomputed, sealed)
    ):
        raise FinalizationError(f"{label} lacks the complete five-metric audit")
    numeric_errors: list[float] = []
    for key in sorted(metric_keys):
        for section_name, values in (
            ("absolute_errors", absolute_errors),
            ("recomputed", recomputed),
            ("sealed", sealed),
        ):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FinalizationError(f"{label}.audit.{section_name}.{key} is not numeric")
            value = float(value)
            if not np.isfinite(value):
                raise FinalizationError(f"{label}.audit.{section_name}.{key} is non-finite")
            if section_name == "absolute_errors":
                if value < 0.0:
                    raise FinalizationError(f"{label}.audit.absolute_errors.{key} is negative")
                numeric_errors.append(value)
    if max(numeric_errors) != maximum_error:
        raise FinalizationError(f"{label} maximum error differs from its metric errors")
    if (
        audit.get("comparison_scope") != "two counterfactual maps x three noise seeds"
        or not isinstance(forward_layout, dict)
        or any(
            isinstance(forward_layout.get(key), bool)
            or not isinstance(forward_layout.get(key), int)
            or int(forward_layout[key]) < 1
            for key in (
                "dynamic_images",
                "historical_batch_size",
                "historical_flat_state_count",
                "retained_dynamic_prefix",
                "retained_flat_state_count",
                "selected_dynamic_position",
                "stack_size",
            )
        )
    ):
        raise FinalizationError(f"{label} lacks the complete six-repeat forward-layout audit")
    if record.get("protocol") != "lambda=0.000":
        raise FinalizationError(f"{label} changed the fixed C0 protocol")
    if should_qualify and maximum_error > C0_QUALIFICATION_TOLERANCE:
        raise FinalizationError(f"{label} exceeds the strict aggregate qualification tolerance")
    if not should_qualify and maximum_error <= C0_QUALIFICATION_TOLERANCE:
        raise FinalizationError(f"{label} was excluded despite satisfying the strict tolerance")
    return identity


def validate_c0_candidate_qualification(root: Path, c0_comparison: pd.DataFrame) -> dict[str, Any]:
    """Validate the fixed 8-candidate / 6-formal-unit C0 qualification evidence."""

    manifest_path = root / "manifests/controlled_c0_selection.json"
    manifest = _read_json(manifest_path, "Controlled C0 selection manifest")
    qualification = manifest.get("candidate_qualification")
    if not isinstance(qualification, dict):
        raise FinalizationError("Controlled C0 manifest lacks candidate_qualification")
    for record, label in (
        (qualification, "C0 candidate qualification"),
        (manifest.get("sealed_aggregate_audit"), "C0 sealed aggregate audit"),
    ):
        if not isinstance(record, dict):
            raise FinalizationError(f"{label} must be an object")

    _exact_c0_count(qualification, "candidate_count", 8, "C0 candidate qualification")
    _exact_c0_count(qualification, "selected_count", 6, "C0 candidate qualification")
    _exact_c0_count(qualification, "excluded_count", 2, "C0 candidate qualification")
    if qualification.get("status") != C0_QUALIFICATION_STATUS:
        raise FinalizationError(
            "C0 candidate qualification must be 6/8 with explicit exclusions, not an 8/8 pass"
        )
    if qualification.get("qualification_fraction") != "6/8":
        raise FinalizationError("C0 candidate qualification fraction must be exactly 6/8")
    _exact_c0_float(
        qualification.get("tolerance"),
        C0_QUALIFICATION_TOLERANCE,
        "C0 candidate qualification tolerance",
    )

    selected = _c0_records(qualification, "selected", 6)
    excluded = _c0_records(qualification, "excluded", 2)
    selected_identities = {
        _validate_c0_candidate_record(
            record,
            label=f"C0 selected candidate[{index}]",
            should_qualify=True,
        )
        for index, record in enumerate(selected)
    }
    excluded_identities = {
        _validate_c0_candidate_record(
            record,
            label=f"C0 excluded candidate[{index}]",
            should_qualify=False,
        )
        for index, record in enumerate(excluded)
    }
    if len(selected_identities) != 6 or len(excluded_identities) != 2:
        raise FinalizationError("C0 candidate qualification contains duplicate identities")
    if selected_identities & excluded_identities:
        raise FinalizationError("C0 selected and excluded candidate identities overlap")
    if selected_identities | excluded_identities != set(C0_EXPECTED_CANDIDATES):
        raise FinalizationError("C0 qualification hides or replaces a fixed candidate")
    if excluded_identities != set(C0_EXPECTED_EXCLUSION_ERRORS):
        raise FinalizationError("C0 qualification does not preserve the two exact exclusions")

    for index, record in enumerate(excluded):
        identity = _c0_identity(record, f"C0 excluded candidate[{index}]")
        expected_error = C0_EXPECTED_EXCLUSION_ERRORS[identity]
        _exact_c0_float(
            record.get("maximum_absolute_error"),
            expected_error,
            f"C0 excluded candidate[{index}].maximum_absolute_error",
        )
        if record.get("reason_code") != C0_EXCLUSION_REASON:
            raise FinalizationError(
                f"C0 excluded candidate[{index}] lacks reason {C0_EXCLUSION_REASON}"
            )
        reason = record.get("reason")
        required_reason_fragments = (
            C0_EXCLUSION_REASON,
            "excluded_without_replacement",
            f"maximum_absolute_error={expected_error:.17g}",
            f"qualification_tolerance={C0_QUALIFICATION_TOLERANCE:.17g}",
        )
        if not isinstance(reason, str) or any(
            fragment not in reason for fragment in required_reason_fragments
        ):
            raise FinalizationError(f"C0 excluded candidate[{index}] has an incomplete reason")

    coverage = qualification.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("passed") is not True:
        raise FinalizationError("the six C0 formal units do not retain required coverage")
    expected_checks = {
        "two_registered_architectures",
        "at_least_two_active",
        "at_least_two_null",
        "at_least_one_mixed_E_C",
        "both_registered_counterfactual_maps",
    }
    checks = coverage.get("checks")
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in expected_checks):
        raise FinalizationError("C0 qualified-subset coverage checks are incomplete or failed")
    observed = coverage.get("observed")
    requirements = coverage.get("requirements")
    if not isinstance(observed, dict) or not isinstance(requirements, dict):
        raise FinalizationError("C0 qualified-subset coverage lacks observations/requirements")
    if observed.get("architectures") != ["resnet18", "small_vit"]:
        raise FinalizationError("C0 qualified subset changed registered architecture coverage")
    if observed.get("counterfactual_maps") != ["20260882", "20260883"]:
        raise FinalizationError("C0 qualified subset changed registered map coverage")
    _exact_c0_count(observed, "active_count", 4, "C0 coverage observations")
    _exact_c0_count(observed, "null_count", 2, "C0 coverage observations")
    mixed_count = observed.get("mixed_unit_count")
    mixed_units = observed.get("mixed_units")
    if (
        isinstance(mixed_count, bool)
        or not isinstance(mixed_count, int)
        or mixed_count < 1
        or not isinstance(mixed_units, list)
        or len(mixed_units) != mixed_count
        or len(set(map(str, mixed_units))) != mixed_count
    ):
        raise FinalizationError("C0 qualified subset did not retain auditable mixed E/C coverage")
    expected_requirements = {
        "architectures": ["resnet18", "small_vit"],
        "minimum_active": 2,
        "minimum_null": 2,
        "minimum_mixed_E_C": 1,
        "counterfactual_maps": ["20260882", "20260883"],
        "mixed_threshold_each": 1.0e-5,
    }
    if requirements != expected_requirements:
        raise FinalizationError("C0 qualified-subset coverage requirements changed")

    diagnostic_path = root / C0_DIAGNOSTIC_RELATIVE
    bound_diagnostic = qualification.get("diagnostic_path")
    if not isinstance(bound_diagnostic, str) or not Path(bound_diagnostic).is_absolute():
        raise FinalizationError("C0 qualification diagnostic path is not absolute")
    if Path(bound_diagnostic).resolve() != diagnostic_path.resolve():
        raise FinalizationError("C0 qualification manifest points to a different diagnostic")
    nested_diagnostic = qualification.get("diagnostic")
    if nested_diagnostic != {
        "path": bound_diagnostic,
        "sha256": qualification.get("diagnostic_sha256"),
    }:
        raise FinalizationError("C0 qualification nested/flat diagnostic bindings differ")
    diagnostic = _read_json(diagnostic_path, "C0 candidate qualification diagnostic")
    diagnostic_sha256 = sha256_file(diagnostic_path)
    if qualification.get("diagnostic_sha256") != diagnostic_sha256:
        raise FinalizationError("C0 qualification diagnostic SHA-256 binding failed")
    if (
        diagnostic.get("artifact_type") != "c0_fixed_candidate_qualification_diagnostic"
        or diagnostic.get("completed_all_candidates") is not True
        or diagnostic.get("status") != C0_QUALIFICATION_STATUS
    ):
        raise FinalizationError("C0 qualification diagnostic is incomplete or misclassified")
    _exact_c0_count(diagnostic, "candidate_count", 8, "C0 qualification diagnostic")
    _exact_c0_count(diagnostic, "selected_count", 6, "C0 qualification diagnostic")
    _exact_c0_count(diagnostic, "excluded_count", 2, "C0 qualification diagnostic")
    _exact_c0_float(
        diagnostic.get("tolerance"),
        C0_QUALIFICATION_TOLERANCE,
        "C0 qualification diagnostic tolerance",
    )
    _exact_c0_float(
        diagnostic.get("maximum_absolute_error"),
        max(C0_EXPECTED_EXCLUSION_ERRORS.values()),
        "C0 qualification diagnostic maximum_absolute_error",
    )
    if (
        diagnostic.get("selected") != selected
        or diagnostic.get("excluded") != excluded
        or diagnostic.get("coverage") != coverage
    ):
        raise FinalizationError("C0 manifest and diagnostic qualification records differ")
    diagnostic_audits = diagnostic.get("audits")
    if not isinstance(diagnostic_audits, list) or len(diagnostic_audits) != 8:
        raise FinalizationError("C0 diagnostic must retain all eight candidate audits")
    diagnostic_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for index, record in enumerate(diagnostic_audits):
        if not isinstance(record, dict):
            raise FinalizationError(f"C0 diagnostic audit[{index}] is not an object")
        identity = _c0_identity(record, f"C0 diagnostic audit[{index}]")
        if identity in diagnostic_by_identity:
            raise FinalizationError("C0 diagnostic repeats a candidate identity")
        diagnostic_by_identity[identity] = record
    if set(diagnostic_by_identity) != set(C0_EXPECTED_CANDIDATES):
        raise FinalizationError("C0 diagnostic hides or replaces a fixed candidate audit")
    for record in (*selected, *excluded):
        identity = _c0_identity(record, "C0 qualification record")
        diagnostic_record = diagnostic_by_identity[identity]
        if (
            diagnostic_record.get("factor") != record.get("factor")
            or diagnostic_record.get("protocol") != record.get("protocol")
            or diagnostic_record.get("audit") != record.get("audit")
        ):
            raise FinalizationError(f"C0 diagnostic audit differs for {identity!r}")

    sealed_audit = manifest["sealed_aggregate_audit"]
    _exact_c0_count(sealed_audit, "row_count", 8, "C0 sealed aggregate audit")
    _exact_c0_count(sealed_audit, "pass_count", 6, "C0 sealed aggregate audit")
    _exact_c0_count(sealed_audit, "failure_count", 2, "C0 sealed aggregate audit")
    _exact_c0_count(sealed_audit, "qualified_count", 6, "C0 sealed aggregate audit")
    _exact_c0_count(sealed_audit, "excluded_count", 2, "C0 sealed aggregate audit")
    _exact_c0_float(
        sealed_audit.get("maximum_absolute_error"),
        max(C0_EXPECTED_EXCLUSION_ERRORS.values()),
        "C0 sealed aggregate audit maximum_absolute_error",
    )
    _exact_c0_float(
        sealed_audit.get("tolerance"),
        C0_QUALIFICATION_TOLERANCE,
        "C0 sealed aggregate audit tolerance",
    )
    sealed_path_value = sealed_audit.get("path")
    if not isinstance(sealed_path_value, str) or not Path(sealed_path_value).is_absolute():
        raise FinalizationError("C0 sealed aggregate audit path is not absolute")
    sealed_path = _require_file(Path(sealed_path_value), "C0 sealed aggregate audit CSV")
    if not sealed_path.resolve().is_relative_to((root / "manifests").resolve()):
        raise FinalizationError("C0 sealed aggregate audit is outside the V2 manifest tree")
    if sealed_audit.get("sha256") != sha256_file(sealed_path):
        raise FinalizationError("C0 sealed aggregate audit SHA-256 binding failed")
    sealed_frame = _read_csv(sealed_path, "C0 sealed aggregate audit CSV")
    sealed_required = {"model_id", "base_id", "factor", "protocol", "maximum_absolute_error"}
    if missing := sorted(sealed_required.difference(sealed_frame.columns)):
        raise FinalizationError(
            f"C0 sealed aggregate audit CSV lacks columns: {', '.join(missing)}"
        )
    if len(sealed_frame) != 8:
        raise FinalizationError("C0 sealed aggregate audit CSV must contain eight candidates")
    sealed_passed = _bool_series(sealed_frame, "passed", "C0 sealed aggregate audit CSV")
    sealed_errors = _finite_numeric(
        sealed_frame,
        "maximum_absolute_error",
        "C0 sealed aggregate audit CSV",
    )
    sealed_by_identity: dict[tuple[str, int], int] = {}
    for index, row in sealed_frame.iterrows():
        base_id = row["base_id"]
        if isinstance(base_id, bool) or not float(base_id).is_integer():
            raise FinalizationError("C0 sealed aggregate audit CSV has invalid base_id")
        identity = (str(row["model_id"]), int(base_id))
        if identity in sealed_by_identity:
            raise FinalizationError("C0 sealed aggregate audit CSV repeats a candidate")
        sealed_by_identity[identity] = int(index)
    if set(sealed_by_identity) != set(C0_EXPECTED_CANDIDATES):
        raise FinalizationError("C0 sealed aggregate audit CSV hides/replaces a candidate")
    qualification_by_identity = {
        _c0_identity(record, "C0 qualification record"): record for record in (*selected, *excluded)
    }
    for identity, index in sealed_by_identity.items():
        record = qualification_by_identity[identity]
        if (
            str(sealed_frame.at[index, "factor"]) != record["factor"]
            or str(sealed_frame.at[index, "protocol"]) != record["protocol"]
            or bool(sealed_passed.iloc[index]) is not bool(record["qualified"])
            or not np.isclose(
                float(sealed_errors[index]),
                float(record["maximum_absolute_error"]),
                atol=1.0e-18,
                rtol=1.0e-13,
            )
        ):
            raise FinalizationError(f"C0 sealed aggregate audit CSV differs for {identity!r}")

    _exact_c0_count(manifest, "unit_count", 6, "Controlled C0 selection manifest")
    _exact_c0_count(manifest, "active_unit_count", 4, "Controlled C0 selection manifest")
    _exact_c0_count(manifest, "null_unit_count", 2, "Controlled C0 selection manifest")
    if manifest.get("mixed_unit_count") != mixed_count:
        raise FinalizationError("Controlled C0 manifest mixed-unit count differs from coverage")
    manifest_units = manifest.get("units")
    if (
        not isinstance(manifest_units, list)
        or len(manifest_units) != 6
        or len(set(map(str, manifest_units))) != 6
    ):
        raise FinalizationError("Controlled C0 manifest must retain six unique formal units")
    unit_ids = c0_comparison.get("unit_id")
    if unit_ids is None or len(c0_comparison) != 6:
        raise FinalizationError("Controlled C0 comparison must contain six formal units")
    if set(unit_ids.astype(str)) != set(map(str, manifest_units)):
        raise FinalizationError("Controlled C0 comparison units differ from the qualified manifest")
    if not set(map(str, mixed_units)).issubset(set(map(str, manifest_units))):
        raise FinalizationError("C0 coverage cites a mixed unit outside the formal selection")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 6:
        raise FinalizationError("Controlled C0 manifest must retain six selected sources")
    source_identities: set[tuple[str, int]] = set()
    source_units: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise FinalizationError(f"Controlled C0 source[{index}] is not an object")
        identity = _c0_identity(source, f"Controlled C0 source[{index}]")
        source_identities.add(identity)
        source_units.add(str(source.get("unit_id")))
        diagnostic_record = diagnostic_by_identity.get(identity)
        if diagnostic_record is None:
            raise FinalizationError(f"Controlled C0 source[{index}] replaced a candidate")
        if source.get("sealed_aggregate_audit") != diagnostic_record.get("audit"):
            raise FinalizationError(f"Controlled C0 source audit differs for {identity!r}")
    if source_identities != selected_identities or source_units != set(map(str, manifest_units)):
        raise FinalizationError("Controlled C0 formal sources differ from the qualified candidates")

    return {
        "status": C0_QUALIFICATION_STATUS,
        "qualification_fraction": "6/8",
        "candidate_count": 8,
        "selected_count": 6,
        "excluded_count": 2,
        "formal_unit_count": 6,
        "coverage": coverage,
        "selected": [
            {
                "model_id": identity[0],
                "base_id": identity[1],
                "maximum_absolute_error": float(record["maximum_absolute_error"]),
            }
            for identity, record in sorted(
                (_c0_identity(item, "C0 selected candidate"), item) for item in selected
            )
        ],
        "excluded": [
            {
                "model_id": identity[0],
                "base_id": identity[1],
                "maximum_absolute_error": C0_EXPECTED_EXCLUSION_ERRORS[identity],
                "reason_code": C0_EXCLUSION_REASON,
                "reason": str(record["reason"]),
            }
            for identity, record in sorted(
                (_c0_identity(item, "C0 excluded candidate"), item) for item in excluded
            )
        ],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "diagnostic_path": str(diagnostic_path.resolve()),
        "diagnostic_sha256": diagnostic_sha256,
        "sealed_aggregate_audit_path": str(sealed_path.resolve()),
        "sealed_aggregate_audit_sha256": sha256_file(sealed_path),
        "runtime_inference": {
            "observation": "historical MIG versus current full-B200 execution differs",
            "causality_proven": False,
            "historical_kernel_runtime_metadata_locked": False,
            "scope": "post-exclusion inference only",
        },
    }


def validate_c0_runtime_attribution(root: Path, qualification: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the bounded, non-causal attribution for the two C0 exclusions."""

    path = root / C0_RUNTIME_ATTRIBUTION_RELATIVE
    payload = _read_json(path, "C0 runtime attribution")
    digest = sha256_file(path)
    if digest != C0_RUNTIME_ATTRIBUTION_SHA256:
        raise FinalizationError(
            "C0 runtime attribution SHA-256 differs from the reviewed structured evidence"
        )
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type")
        != "decaf_cross_generation_controlled_c0_runtime_attribution"
    ):
        raise FinalizationError("C0 runtime attribution has an unexpected schema/type")

    selection_audit = payload.get("selection_audit")
    if not isinstance(selection_audit, dict):
        raise FinalizationError("C0 runtime attribution lacks selection_audit")
    _exact_c0_count(selection_audit, "unit_count", 8, "C0 runtime attribution selection")
    _exact_c0_count(selection_audit, "strict_pass_count", 6, "C0 runtime attribution selection")
    _exact_c0_count(selection_audit, "strict_failure_count", 2, "C0 runtime attribution selection")
    _exact_c0_float(
        selection_audit.get("tolerance"),
        C0_QUALIFICATION_TOLERANCE,
        "C0 runtime attribution selection tolerance",
    )
    _exact_c0_float(
        selection_audit.get("maximum_absolute_error"),
        max(C0_EXPECTED_EXCLUSION_ERRORS.values()),
        "C0 runtime attribution selection maximum_absolute_error",
    )
    failed_units = selection_audit.get("failed_units")
    if not isinstance(failed_units, list) or len(failed_units) != 2:
        raise FinalizationError("C0 runtime attribution must retain exactly two failed units")
    failed_by_identity: dict[tuple[str, int], Mapping[str, Any]] = {}
    for index, record in enumerate(failed_units):
        if not isinstance(record, dict):
            raise FinalizationError(f"C0 runtime attribution failed_units[{index}] is invalid")
        identity = _c0_identity(record, f"C0 runtime attribution failed_units[{index}]")
        if identity in failed_by_identity:
            raise FinalizationError("C0 runtime attribution repeats a failed-unit identity")
        failed_by_identity[identity] = record
    if set(failed_by_identity) != set(C0_EXPECTED_EXCLUSION_ERRORS):
        raise FinalizationError("C0 runtime attribution changed the two excluded identities")
    for identity, expected_error in C0_EXPECTED_EXCLUSION_ERRORS.items():
        _exact_c0_float(
            failed_by_identity[identity].get("maximum_absolute_error"),
            expected_error,
            f"C0 runtime attribution failed unit {identity!r} maximum_absolute_error",
        )

    qualified_excluded = qualification.get("excluded")
    if (
        not isinstance(qualified_excluded, list)
        or {
            (str(record.get("model_id")), int(record.get("base_id", -1))): record.get(
                "maximum_absolute_error"
            )
            for record in qualified_excluded
            if isinstance(record, dict)
        }
        != C0_EXPECTED_EXCLUSION_ERRORS
    ):
        raise FinalizationError("C0 runtime attribution and qualification exclusions differ")

    disposition = payload.get("recommended_disposition")
    if (
        not isinstance(disposition, dict)
        or disposition.get("code") != C0_EXCLUSION_REASON
        or disposition.get("strict_equivalence") is not False
    ):
        raise FinalizationError("C0 runtime attribution disposition is not fail-closed")
    action = disposition.get("action")
    if not isinstance(action, str) or any(
        fragment not in action
        for fragment in (
            "all eight predeclared units",
            "six strict passes and two failures",
            "do not loosen tolerance or remove selections",
        )
    ):
        raise FinalizationError("C0 runtime attribution disposition hides/replaces evidence")

    inference = payload.get("remaining_inference")
    if not isinstance(inference, dict) or inference.get("proven") is not False:
        raise FinalizationError("C0 runtime attribution incorrectly claims proven causality")
    hypothesis = inference.get("hypothesis")
    reason_not_proven = inference.get("reason_not_proven")
    if (
        not isinstance(hypothesis, str)
        or "MIG versus full-B200" not in hypothesis
        or "plausible contributors" not in hypothesis
        or not isinstance(reason_not_proven, str)
        or "historical driver" not in reason_not_proven
        or "cuDNN" not in reason_not_proven
        or "kernel selections were not captured" not in reason_not_proven
    ):
        raise FinalizationError("C0 runtime attribution exceeds its evidence-bound inference")
    historical_runtime = payload.get("historical_runtime")
    if not isinstance(historical_runtime, dict):
        raise FinalizationError("C0 runtime attribution lacks historical runtime evidence")
    confirmed = historical_runtime.get("confirmed")
    unrecorded = historical_runtime.get("unrecorded")
    if (
        not isinstance(confirmed, dict)
        or confirmed.get("gpu_topology") != "4 x NVIDIA B200 MIG 1g.23gb"
        or not isinstance(unrecorded, list)
        or "exact kernel path on historical MIG instances" not in unrecorded
    ):
        raise FinalizationError("C0 historical MIG/kernel metadata evidence changed")

    diagnostic_path = str((root / C0_DIAGNOSTIC_RELATIVE).resolve())
    input_evidence = payload.get("input_evidence")
    if not isinstance(input_evidence, list):
        raise FinalizationError("C0 runtime attribution lacks input_evidence")
    diagnostic_inputs = [
        record
        for record in input_evidence
        if isinstance(record, dict) and record.get("path") == diagnostic_path
    ]
    if len(diagnostic_inputs) != 1 or diagnostic_inputs[0].get("sha256") != qualification.get(
        "diagnostic_sha256"
    ):
        raise FinalizationError("C0 runtime attribution is not bound to the final diagnostic")

    return {
        "status": "EVIDENCE_BOUND_NON_CAUSAL_ATTRIBUTION",
        "path": str(path.resolve()),
        "sha256": digest,
        "reason_code": C0_EXCLUSION_REASON,
        "strict_equivalence": False,
        "selection_audit": {
            "candidate_count": 8,
            "strict_pass_count": 6,
            "strict_failure_count": 2,
            "maximum_absolute_error": max(C0_EXPECTED_EXCLUSION_ERRORS.values()),
        },
        "remaining_inference": {
            "hypothesis": hypothesis,
            "proven": False,
            "reason_not_proven": reason_not_proven,
        },
        "historical_runtime": {
            "gpu_topology": confirmed["gpu_topology"],
            "kernel_runtime_metadata_locked": False,
        },
    }


def validate_imagenet9_historical_source_binding(root: Path) -> dict[str, Any]:
    """Validate ImageNet-9 historical execution against its sealed ZIP authority."""

    bridge_path = root / "provenance/imagenet9_bridge.json"
    patch_path = root / "manifests/imagenet9_historical_patch_orders.json"
    receipt_path = root / "provenance/imagenet9_current_e2e.json"
    bridge = _read_json(bridge_path, "ImageNet-9 bridge provenance")
    patch_manifest = _read_json(patch_path, "ImageNet-9 patch-order manifest")
    receipt = _read_json(receipt_path, "ImageNet-9 current E2E receipt")
    binding = bridge.get("historical_source_binding")
    if not isinstance(binding, dict) or patch_manifest.get("historical_source_binding") != binding:
        raise FinalizationError(
            "ImageNet-9 bridge and patch-order manifest bind different historical sources"
        )
    package_value = binding.get("path")
    if not isinstance(package_value, str) or not Path(package_value).is_absolute():
        raise FinalizationError("ImageNet-9 historical package path is not absolute")
    package = _require_file(Path(package_value), "ImageNet-9 sealed historical package")
    package_sha256 = sha256_file(package)
    if (
        binding.get("sha256") != IMAGENET9_HISTORICAL_PACKAGE_SHA256
        or package_sha256 != IMAGENET9_HISTORICAL_PACKAGE_SHA256
    ):
        raise FinalizationError("ImageNet-9 sealed historical package SHA-256 changed")
    if (
        binding.get("manifest_member") != IMAGENET9_MANIFEST_MEMBER
        or binding.get("recorded_member_count") != 381
        or binding.get("source_authority")
        != "sha256-verified lightweight ZIP; not historical Git HEAD"
        or binding.get("zip_import_root") != f"{package.resolve()}/{IMAGENET9_PACKAGE_PREFIX}/code"
    ):
        raise FinalizationError("ImageNet-9 sealed historical source authority changed")

    try:
        with zipfile.ZipFile(package) as archive:
            files = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(files) != len(set(files)):
                raise FinalizationError("ImageNet-9 sealed package repeats an archive member")
            try:
                manifest_bytes = archive.read(IMAGENET9_MANIFEST_MEMBER)
                package_manifest = json.loads(manifest_bytes)
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FinalizationError("ImageNet-9 sealed package manifest is invalid") from error
            if not isinstance(package_manifest, dict):
                raise FinalizationError("ImageNet-9 sealed package manifest is not an object")
            member_list = package_manifest.get("members")
            if (
                package_manifest.get("schema_version") != 1
                or package_manifest.get("namespace") != IMAGENET9_PACKAGE_PREFIX
                or package_manifest.get("lightweight") is not True
                or package_manifest.get("source_layout") != "code/cmr"
                or package_manifest.get("recorded_member_count") != 381
                or not isinstance(member_list, list)
                or len(member_list) != 381
            ):
                raise FinalizationError("ImageNet-9 sealed package manifest contract changed")
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if binding.get("manifest_sha256") != manifest_sha256:
                raise FinalizationError("ImageNet-9 package-manifest SHA-256 binding failed")
            members: dict[str, dict[str, Any]] = {}
            expected_archive_files = {IMAGENET9_MANIFEST_MEMBER}
            for index, record in enumerate(member_list):
                if not isinstance(record, dict):
                    raise FinalizationError(f"ImageNet-9 package member[{index}] is invalid")
                relative = record.get("path")
                expected_bytes = record.get("bytes")
                expected_sha256 = record.get("sha256")
                if (
                    not isinstance(relative, str)
                    or not relative
                    or relative.startswith("/")
                    or ".." in Path(relative).parts
                    or relative in members
                    or isinstance(expected_bytes, bool)
                    or not isinstance(expected_bytes, int)
                    or expected_bytes < 0
                    or not isinstance(expected_sha256, str)
                    or len(expected_sha256) != 64
                ):
                    raise FinalizationError(f"ImageNet-9 package member[{index}] is malformed")
                archive_member = f"{IMAGENET9_PACKAGE_PREFIX}/{relative}"
                try:
                    payload = archive.read(archive_member)
                except KeyError as error:
                    raise FinalizationError(
                        f"ImageNet-9 sealed package member is missing: {relative}"
                    ) from error
                observed_sha256 = hashlib.sha256(payload).hexdigest()
                if len(payload) != expected_bytes or observed_sha256 != expected_sha256:
                    raise FinalizationError(
                        f"ImageNet-9 sealed package member identity changed: {relative}"
                    )
                members[relative] = {
                    "archive_member": archive_member,
                    "bytes": expected_bytes,
                    "sha256": expected_sha256,
                }
                expected_archive_files.add(archive_member)
            if set(files) != expected_archive_files:
                raise FinalizationError("ImageNet-9 sealed package inventory changed")
    except (OSError, zipfile.BadZipFile) as error:
        raise FinalizationError(f"ImageNet-9 sealed package is unreadable: {error}") from error

    source_members = binding.get("source_members")
    if not isinstance(source_members, dict) or not IMAGENET9_SOURCE_RELATIVES.issubset(
        source_members
    ):
        raise FinalizationError("ImageNet-9 binding lacks the six required runtime modules")
    for relative in IMAGENET9_SOURCE_RELATIVES:
        if source_members.get(relative) != members.get(relative):
            raise FinalizationError(
                f"ImageNet-9 source-member binding differs from sealed bytes: {relative}"
            )

    patch_sha256 = _validate_artifact_binding(
        bridge,
        patch_path,
        path_key="patch_order_manifest",
        sha_key="patch_order_manifest_sha256",
        label="ImageNet-9 bridge patch-order manifest",
    )
    receipt_patch_sha256 = _validate_artifact_binding(
        receipt,
        patch_path,
        path_key="patch_order_manifest",
        sha_key="patch_order_manifest_sha256",
        label="ImageNet-9 current receipt patch-order manifest",
    )
    if patch_sha256 != receipt_patch_sha256:
        raise FinalizationError("ImageNet-9 bridge/current receipt patch-order hashes differ")
    current_output = root / "trajectories/imagenet9_current_e2e_scans.parquet"
    current_output_sha256 = _validate_artifact_binding(
        receipt,
        current_output,
        path_key="output",
        sha_key="output_sha256",
        label="ImageNet-9 current E2E scan receipt",
    )
    if receipt.get("patch_order_injection") is not True:
        raise FinalizationError("ImageNet-9 current E2E receipt did not inject sealed patch orders")
    return {
        "status": "SHA256_VERIFIED_SEALED_ZIP_SOURCE",
        "bridge_path": str(bridge_path.resolve()),
        "bridge_sha256": sha256_file(bridge_path),
        "package_path": str(package.resolve()),
        "package_sha256": package_sha256,
        "manifest_member": IMAGENET9_MANIFEST_MEMBER,
        "manifest_sha256": manifest_sha256,
        "recorded_member_count": 381,
        "source_authority": binding["source_authority"],
        "required_source_members": {
            relative: source_members[relative] for relative in sorted(IMAGENET9_SOURCE_RELATIVES)
        },
        "patch_order_manifest_path": str(patch_path.resolve()),
        "patch_order_manifest_sha256": patch_sha256,
        "current_receipt_path": str(receipt_path.resolve()),
        "current_receipt_sha256": sha256_file(receipt_path),
        "current_scan_path": str(current_output.resolve()),
        "current_scan_sha256": current_output_sha256,
    }


def _validated_bound_file(
    record: Any,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> Path:
    if not isinstance(record, dict):
        raise FinalizationError(f"{label} binding is not an object")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise FinalizationError(f"{label} path is not absolute")
    path = _require_file(Path(path_value), label).resolve()
    digest = sha256_file(path)
    if (
        isinstance(record.get("bytes"), bool)
        or record.get("bytes") != path.stat().st_size
        or record.get("sha256") != digest
        or (expected_sha256 is not None and digest != expected_sha256)
    ):
        raise FinalizationError(f"{label} bytes/SHA-256 binding changed")
    return path


def _attribution_a0_source_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    if (
        binding.get("authority_kind") != "deployed_tree_with_sha256_receipts"
        or binding.get("git_head_role") != "context_only_untracked"
        or binding.get("namespace") != "cmr.decaf_reference_locked_v1"
        or binding.get("origin_verified") is not True
        or binding.get("source_tree_digest_algorithm")
        != "sha256_join_relative_py_path_colon_sha256_lf"
        or binding.get("source_python_file_count") != 381
        or binding.get("source_tree_sha256") != ATTRIBUTION_A0_SOURCE_TREE_SHA256
    ):
        raise FinalizationError("Attribution A0 source authority contract changed")
    source_value = binding.get("source_root")
    if not isinstance(source_value, str) or not Path(source_value).is_absolute():
        raise FinalizationError("Attribution A0 source root is not absolute")
    source_root = Path(source_value).resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise FinalizationError("Attribution A0 deployed source root is missing or unsafe")
    cmr_root = source_root / "cmr"
    python_files = sorted(cmr_root.rglob("*.py"))
    if any(path.is_symlink() for path in python_files):
        raise FinalizationError("Attribution A0 deployed Python tree contains a symlink")
    python_files = [
        path for path in python_files if path.is_file() and "__pycache__" not in path.parts
    ]
    material = [
        f"{path.relative_to(cmr_root).as_posix()}:{sha256_file(path)}"
        for path in python_files
    ]
    source_tree_sha256 = hashlib.sha256("\n".join(material).encode("utf-8")).hexdigest()
    if len(python_files) != 381 or source_tree_sha256 != ATTRIBUTION_A0_SOURCE_TREE_SHA256:
        raise FinalizationError("Attribution A0 381-file deployed source-tree digest changed")

    deployment_path = _validated_bound_file(
        binding.get("deployment_receipt"),
        label="Attribution A0 deployment receipt",
        expected_sha256=ATTRIBUTION_A0_DEPLOYMENT_SHA256,
    )
    plan_path = _validated_bound_file(
        binding.get("formal_plan"),
        label="Attribution A0 formal plan",
        expected_sha256=ATTRIBUTION_A0_PLAN_SHA256,
    )
    plan_receipt_path = _validated_bound_file(
        binding.get("formal_plan_receipt"),
        label="Attribution A0 formal-plan receipt",
        expected_sha256=ATTRIBUTION_A0_PLAN_RECEIPT_SHA256,
    )
    deployment = _read_json(deployment_path, "Attribution A0 deployment receipt")
    plan_receipt = _read_json(plan_receipt_path, "Attribution A0 formal-plan receipt")
    deployment_plan = deployment.get("formal_job_plan")
    deployment_receipt = deployment.get("formal_job_plan_receipt")
    rebind = deployment.get("formal_job_plan_rebind")
    if (
        deployment.get("schema_version") != 1
        or deployment.get("code_root") != str(source_root)
        or deployment.get("entrypoint") != "cmr.decaf_reference_locked_v1.run"
        or deployment.get("code_sha256") != source_tree_sha256
        or deployment.get("code_changes_required") is not False
        or deployment.get("downloads_required") is not False
        or deployment.get("installs_required") is not False
        or not isinstance(deployment_plan, dict)
        or deployment_plan.get("path") != "plans/formal_jobs.jsonl"
        or deployment_plan.get("sha256") != ATTRIBUTION_A0_PLAN_SHA256
        or deployment_plan.get("bytes") != plan_path.stat().st_size
        or not isinstance(deployment_receipt, dict)
        or deployment_receipt.get("path") != "plans/formal_jobs.jsonl.receipt.json"
        or deployment_receipt.get("sha256") != ATTRIBUTION_A0_PLAN_RECEIPT_SHA256
        or deployment_receipt.get("bytes") != plan_receipt_path.stat().st_size
    ):
        raise FinalizationError("Attribution A0 deployment receipt contract changed")
    if (
        not isinstance(rebind, dict)
        or rebind.get("schema_version") != 1
        or rebind.get("kind") != "formal_job_plan_receipt_code_rebind"
        or rebind.get("code_sha256") != source_tree_sha256
        or rebind.get("plan_sha256_unchanged") is not True
        or rebind.get("rebound_fields") != ["code_sha256"]
        or not isinstance(rebind.get("plan"), dict)
        or Path(str(rebind["plan"].get("path", ""))).resolve() != plan_path
        or rebind["plan"].get("bytes") != plan_path.stat().st_size
        or rebind["plan"].get("sha256") != ATTRIBUTION_A0_PLAN_SHA256
        or not isinstance(rebind.get("receipt"), dict)
        or Path(str(rebind["receipt"].get("path", ""))).resolve() != plan_receipt_path
        or rebind["receipt"].get("bytes") != plan_receipt_path.stat().st_size
        or rebind["receipt"].get("sha256") != ATTRIBUTION_A0_PLAN_RECEIPT_SHA256
    ):
        raise FinalizationError("Attribution A0 formal-plan rebind contract changed")
    if (
        plan_receipt.get("schema_version") != 1
        or plan_receipt.get("kind") != "decaf_reference_locked_formal_job_plan"
        or Path(str(plan_receipt.get("plan_path", ""))).resolve() != plan_path
        or plan_receipt.get("plan_sha256") != ATTRIBUTION_A0_PLAN_SHA256
        or plan_receipt.get("code_sha256") != source_tree_sha256
        or plan_receipt.get("job_count") != 9184
    ):
        raise FinalizationError("Attribution A0 formal-plan receipt contract changed")

    required_modules = binding.get("required_modules")
    if not isinstance(required_modules, dict) or set(required_modules) != (
        ATTRIBUTION_A0_REQUIRED_MODULES
    ):
        raise FinalizationError("Attribution A0 required-module inventory changed")
    namespace_root = source_root / "cmr/decaf_reference_locked_v1"
    for name, record in required_modules.items():
        expected = namespace_root / f"{name}.py"
        path = _validated_bound_file(record, label=f"Attribution A0 required module {name}")
        if path != expected.resolve():
            raise FinalizationError(f"Attribution A0 required module path changed: {name}")

    anchors = binding.get("anchor_files")
    if not isinstance(anchors, dict) or set(anchors) != set(ATTRIBUTION_A0_ANCHORS):
        raise FinalizationError("Attribution A0 anchor-file inventory changed")
    for relative, expected_sha256 in ATTRIBUTION_A0_ANCHORS.items():
        path = _validated_bound_file(
            anchors[relative],
            label=f"Attribution A0 anchor {relative}",
            expected_sha256=expected_sha256,
        )
        if path != (source_root / relative).resolve():
            raise FinalizationError(f"Attribution A0 anchor path changed: {relative}")

    expected_origins: dict[str, str] = {}
    for name in ATTRIBUTION_A0_LOADED_MODULES:
        module = (
            "cmr.decaf_reference_locked_v1"
            if name == "__init__"
            else f"cmr.decaf_reference_locked_v1.{name}"
        )
        filename = "__init__.py" if name == "__init__" else f"{name}.py"
        expected_origins[module] = str((namespace_root / filename).resolve())
    if binding.get("loaded_module_origins") != dict(sorted(expected_origins.items())):
        raise FinalizationError("Attribution A0 loaded-module origins changed")
    expected_anchor_origins = {
        "cmr.decaf_reference_locked_v1": str(
            (source_root / "cmr/decaf_reference_locked_v1/__init__.py").resolve()
        ),
        "cmr.decaf_imagenet9_v1.decaf": str(
            (source_root / "cmr/decaf_imagenet9_v1/decaf.py").resolve()
        ),
    }
    parent_origin = (source_root / "cmr/__init__.py").resolve()
    _require_file(parent_origin, "Attribution A0 parent package")
    if (
        binding.get("loaded_anchor_origins") != dict(sorted(expected_anchor_origins.items()))
        or binding.get("parent_package_origin") != str(parent_origin)
    ):
        raise FinalizationError("Attribution A0 anchor/parent origins changed")

    return {
        "status": "DEPLOYED_TREE_AND_RECEIPTS_SHA256_VERIFIED",
        "authority_kind": binding["authority_kind"],
        "authority_sha256": hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_root": str(source_root),
        "source_python_file_count": len(python_files),
        "source_tree_sha256": source_tree_sha256,
        "deployment_receipt_path": str(deployment_path),
        "deployment_receipt_sha256": sha256_file(deployment_path),
        "formal_plan_path": str(plan_path),
        "formal_plan_sha256": sha256_file(plan_path),
        "formal_plan_receipt_path": str(plan_receipt_path),
        "formal_plan_receipt_sha256": sha256_file(plan_receipt_path),
        "required_modules": sorted(required_modules),
        "anchor_files": sorted(anchors),
        "origin_verified": True,
    }


def _attribution_a2_payload_tree(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _attribution_a2_source_binding(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if (
        binding.get("authority_kind") != "sha256_verified_lightweight_zip"
        or binding.get("git_head_role") != "context_only_untracked"
        or binding.get("sha256") != ATTRIBUTION_A2_PACKAGE_SHA256
        or binding.get("bytes") != 140588387
        or binding.get("manifest_member") != ATTRIBUTION_A2_MANIFEST_MEMBER
        or binding.get("manifest_sha256") != ATTRIBUTION_A2_MANIFEST_SHA256
        or binding.get("manifest_member_count") != 2606
        or binding.get("archive_member_count") != 2607
        or binding.get("payload_tree_sha256") != ATTRIBUTION_A2_PAYLOAD_TREE_SHA256
        or binding.get("archive_source_prefix") != ATTRIBUTION_A2_SOURCE_PREFIX
        or binding.get("namespace") != ATTRIBUTION_A2_NAMESPACE
        or binding.get("namespace_member_count") != 19
        or binding.get("materialized_member_count") != 19
        or binding.get("archive_inventory_verified") is not True
        or binding.get("origin_verified") is not True
        or set(binding.get("required_modules", ())) != ATTRIBUTION_A2_REQUIRED_MODULES
    ):
        raise FinalizationError("Attribution A2 sealed-source authority contract changed")
    package_value = binding.get("path")
    if not isinstance(package_value, str) or not Path(package_value).is_absolute():
        raise FinalizationError("Attribution A2 sealed-package path is not absolute")
    package = _require_file(Path(package_value), "Attribution A2 sealed package").resolve()
    if package.stat().st_size != 140588387 or sha256_file(package) != ATTRIBUTION_A2_PACKAGE_SHA256:
        raise FinalizationError("Attribution A2 sealed-package bytes changed")

    try:
        with zipfile.ZipFile(package) as archive:
            actual_names = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(actual_names) != 2607 or len(actual_names) != len(set(actual_names)):
                raise FinalizationError("Attribution A2 ZIP inventory is not 2607 unique files")
            try:
                manifest_bytes = archive.read(ATTRIBUTION_A2_MANIFEST_MEMBER)
                manifest = json.loads(manifest_bytes)
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FinalizationError("Attribution A2 package manifest is invalid") from error
            records = manifest.get("members") if isinstance(manifest, dict) else None
            if (
                not isinstance(manifest, dict)
                or set(manifest)
                != {"schema_version", "package_kind", "payload_tree_sha256", "members"}
                or manifest.get("schema_version") != 1
                or manifest.get("package_kind") != "lightweight"
                or manifest.get("payload_tree_sha256") != ATTRIBUTION_A2_PAYLOAD_TREE_SHA256
                or hashlib.sha256(manifest_bytes).hexdigest()
                != ATTRIBUTION_A2_MANIFEST_SHA256
                or not isinstance(records, list)
                or len(records) != 2606
            ):
                raise FinalizationError("Attribution A2 package-manifest contract changed")
            expected_names: list[str] = []
            namespace_prefix = (
                f"{ATTRIBUTION_A2_SOURCE_PREFIX}/"
                f"{ATTRIBUTION_A2_NAMESPACE.replace('.', '/')}/"
            )
            namespace_members: dict[str, dict[str, Any]] = {}
            for index, record in enumerate(records):
                if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
                    raise FinalizationError(f"Attribution A2 member[{index}] is malformed")
                member = record.get("path")
                expected_bytes = record.get("bytes")
                expected_sha256 = record.get("sha256")
                if (
                    not isinstance(member, str)
                    or not member
                    or member.startswith("/")
                    or ".." in PurePosixPath(member).parts
                    or isinstance(expected_bytes, bool)
                    or not isinstance(expected_bytes, int)
                    or expected_bytes < 0
                    or not isinstance(expected_sha256, str)
                    or len(expected_sha256) != 64
                ):
                    raise FinalizationError(f"Attribution A2 member[{index}] identity is invalid")
                try:
                    with archive.open(member) as stream:
                        digest = hashlib.sha256()
                        observed_bytes = 0
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            observed_bytes += len(chunk)
                            digest.update(chunk)
                except KeyError as error:
                    raise FinalizationError(
                        f"Attribution A2 sealed member is missing: {member}"
                    ) from error
                if observed_bytes != expected_bytes or digest.hexdigest() != expected_sha256:
                    raise FinalizationError(f"Attribution A2 sealed member changed: {member}")
                expected_names.append(member)
                if member.startswith(namespace_prefix):
                    if not member.endswith(".py"):
                        raise FinalizationError("Attribution A2 namespace has a non-Python member")
                    namespace_members[member] = {
                        "archive_member": member,
                        "bytes": expected_bytes,
                        "sha256": expected_sha256,
                    }
            if expected_names != sorted(expected_names) or len(expected_names) != len(
                set(expected_names)
            ):
                raise FinalizationError("Attribution A2 manifest inventory is not canonical")
            if set(actual_names) != {ATTRIBUTION_A2_MANIFEST_MEMBER, *expected_names}:
                raise FinalizationError("Attribution A2 ZIP and manifest inventories differ")
            payload_tree_sha256 = _attribution_a2_payload_tree(records)
            if payload_tree_sha256 != ATTRIBUTION_A2_PAYLOAD_TREE_SHA256:
                raise FinalizationError("Attribution A2 payload-tree digest changed")
    except (OSError, zipfile.BadZipFile) as error:
        raise FinalizationError(f"Attribution A2 sealed package is unreadable: {error}") from error

    if len(namespace_members) != 19 or binding.get("namespace_members") != namespace_members:
        raise FinalizationError("Attribution A2 19-member namespace binding changed")
    required_members = {
        f"{ATTRIBUTION_A2_SOURCE_PREFIX}/{ATTRIBUTION_A2_NAMESPACE.replace('.', '/')}/"
        f"{name}.py"
        for name in ATTRIBUTION_A2_REQUIRED_MODULES
    }
    if not required_members.issubset(namespace_members):
        raise FinalizationError("Attribution A2 namespace lacks required modules")

    expected_import_root = (root / "provenance/historical_sources/attribution_idsds").resolve()
    import_value = binding.get("import_root")
    namespace_value = binding.get("materialized_namespace")
    if (
        not isinstance(import_value, str)
        or Path(import_value).resolve() != expected_import_root
        or not isinstance(namespace_value, str)
    ):
        raise FinalizationError("Attribution A2 materialized import root changed")
    materialized_namespace = Path(namespace_value).resolve()
    expected_namespace = expected_import_root / "cmr/decaf_idsds_funnybirds_v1"
    if materialized_namespace != expected_namespace or not expected_namespace.is_dir():
        raise FinalizationError("Attribution A2 materialized namespace changed")
    observed_paths = list(expected_import_root.rglob("*"))
    if any(path.is_symlink() for path in (expected_import_root, *observed_paths)):
        raise FinalizationError("Attribution A2 materialized source contains a symlink")
    expected_materialized: set[Path] = set()
    for member, record in namespace_members.items():
        relative = PurePosixPath(member).relative_to(ATTRIBUTION_A2_SOURCE_PREFIX)
        output = expected_import_root.joinpath(*relative.parts).resolve()
        expected_materialized.add(output)
        _require_file(output, f"Attribution A2 materialized member {member}")
        if output.stat().st_size != record["bytes"] or sha256_file(output) != record["sha256"]:
            raise FinalizationError(f"Attribution A2 materialized bytes changed: {member}")

    shim = binding.get("parent_package_shim")
    shim_path = expected_import_root / "cmr/__init__.py"
    if (
        not isinstance(shim, dict)
        or shim.get("path") != str(shim_path)
        or shim.get("bytes") != 76
        or shim.get("sha256") != ATTRIBUTION_A2_PARENT_SHIM_SHA256
        or shim.get("role") != "verification_only_import_isolation"
        or shim.get("historical_source") is not False
        or binding.get("parent_package_origin") != str(shim_path)
    ):
        raise FinalizationError("Attribution A2 verification-only parent shim changed")
    _require_file(shim_path, "Attribution A2 parent-package shim")
    if (
        shim_path.stat().st_size != 76
        or sha256_file(shim_path) != ATTRIBUTION_A2_PARENT_SHIM_SHA256
    ):
        raise FinalizationError("Attribution A2 parent-package shim bytes changed")
    actual_materialized = {path.resolve() for path in observed_paths if path.is_file()}
    if actual_materialized != expected_materialized | {shim_path.resolve()}:
        raise FinalizationError("Attribution A2 materialized inventory is stale or mixed")

    expected_origins: dict[str, str | None] = {ATTRIBUTION_A2_NAMESPACE: None}
    for name in ATTRIBUTION_A2_REQUIRED_MODULES:
        expected_origins[f"{ATTRIBUTION_A2_NAMESPACE}.{name}"] = str(
            (expected_namespace / f"{name}.py").resolve()
        )
    if binding.get("loaded_module_origins") != dict(sorted(expected_origins.items())):
        raise FinalizationError("Attribution A2 loaded-module origins changed")

    return {
        "status": "SEALED_ZIP_MATERIALIZED_SOURCE_AND_ORIGINS_SHA256_VERIFIED",
        "authority_kind": binding["authority_kind"],
        "authority_sha256": hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "package_path": str(package),
        "package_sha256": sha256_file(package),
        "manifest_member": ATTRIBUTION_A2_MANIFEST_MEMBER,
        "manifest_sha256": ATTRIBUTION_A2_MANIFEST_SHA256,
        "payload_tree_sha256": payload_tree_sha256,
        "manifest_member_count": 2606,
        "archive_member_count": 2607,
        "namespace_member_count": 19,
        "materialized_member_count": 19,
        "parent_package_shim_sha256": ATTRIBUTION_A2_PARENT_SHIM_SHA256,
        "required_modules": sorted(ATTRIBUTION_A2_REQUIRED_MODULES),
        "origin_verified": True,
    }


def validate_attribution_historical_source_bindings(root: Path) -> dict[str, Any]:
    """Require 3/3 identical A0 and 3/3 identical A2 source authorities."""

    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = {"a0": [], "a2": []}
    selection_manifests: dict[str, dict[str, str]] = {}
    for key, (dataset, _model_id) in ATTRIBUTION_BRIDGES.items():
        path = root / f"manifests/attribution__{key}_selection.json"
        manifest = _read_json(path, f"Attribution source-bound selection {key}")
        digest = sha256_file(path)
        if digest != ATTRIBUTION_SELECTION_SHA256[key]:
            raise FinalizationError(f"Attribution selection manifest SHA-256 changed: {key}")
        binding = manifest.get("historical_source_binding")
        if not isinstance(binding, dict):
            raise FinalizationError(f"Attribution selection lacks source authority: {key}")
        group = "a0" if dataset == "funnybirds" else "a2"
        grouped[group].append((key, binding))
        selection_manifests[key] = {"path": str(path.resolve()), "sha256": digest}
    for group, records in grouped.items():
        if len(records) != 3 or any(binding != records[0][1] for _, binding in records[1:]):
            raise FinalizationError(
                f"Attribution {group.upper()} selections do not share one exact 3/3 authority"
            )
    a0 = _attribution_a0_source_binding(grouped["a0"][0][1])
    a2 = _attribution_a2_source_binding(root, grouped["a2"][0][1])
    return {
        "status": "SIX_SELECTIONS_TWO_AUTHORITIES_FULLY_SHA256_VERIFIED",
        "selection_manifests": selection_manifests,
        "a0_funnybirds": {**a0, "selection_count": 3, "authority_equal_3_of_3": True},
        "a2_imagenet1k_idsds": {**a2, "selection_count": 3, "authority_equal_3_of_3": True},
    }


def validate_attribution_artifact_chain(root: Path) -> dict[str, Any]:
    """Validate all six historical/current attribution inputs and their aggregate."""

    aggregate_path = root / "manifests/attribution_aggregate.json"
    aggregate = _read_json(aggregate_path, "Attribution aggregate manifest")
    if aggregate.get("schema_version") != 1 or aggregate.get("experiment_family") != "attribution":
        raise FinalizationError("Attribution aggregate manifest has an unexpected schema/family")
    aggregate_output = root / "trajectories/attribution.parquet"
    aggregate_output_sha256 = _validate_artifact_binding(
        aggregate,
        aggregate_output,
        path_key="output",
        sha_key="output_sha256",
        label="Attribution aggregate output",
    )

    source_records = aggregate.get("source_records")
    if not isinstance(source_records, list) or len(source_records) != len(ATTRIBUTION_BRIDGES):
        raise FinalizationError("Attribution aggregate must bind exactly six source records")
    indexed_sources: dict[Path, Mapping[str, Any]] = {}
    for index, record in enumerate(source_records):
        if not isinstance(record, dict):
            raise FinalizationError(f"Attribution aggregate source_records[{index}] is invalid")
        path_value = record.get("path")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise FinalizationError(
                f"Attribution aggregate source_records[{index}] path is not absolute"
            )
        source_path = Path(path_value).resolve()
        if source_path in indexed_sources:
            raise FinalizationError("Attribution aggregate repeats a source-record path")
        indexed_sources[source_path] = record

    expected_sources = {
        key: (root / f"trajectories/attribution__{key}.parquet").resolve()
        for key in ATTRIBUTION_BRIDGES
    }
    if set(indexed_sources) != set(expected_sources.values()):
        raise FinalizationError("Attribution aggregate source-record inventory is stale or mixed")

    legacy_sources: dict[str, dict[str, Any]] = {}
    current_outputs: dict[str, dict[str, Any]] = {}
    for key, (dataset, model_id) in ATTRIBUTION_BRIDGES.items():
        historical_path = expected_sources[key]
        source_sha256 = _validate_artifact_binding(
            indexed_sources[historical_path],
            historical_path,
            path_key="path",
            sha_key="sha256",
            label=f"Attribution aggregate source {key}",
        )
        selection_path = root / f"manifests/attribution__{key}_selection.json"
        selection = _read_json(selection_path, f"Attribution legacy selection {key}")
        if (
            selection.get("schema_version") != 1
            or selection.get("experiment_family") != "attribution"
            or selection.get("dataset") != dataset
            or selection.get("model_id") != model_id
        ):
            raise FinalizationError(f"Attribution legacy selection {key} has mixed identity")
        selection_trajectory_sha256 = _validate_artifact_binding(
            selection,
            historical_path,
            path_key="trajectory_record",
            sha_key="trajectory_record_sha256",
            label=f"Attribution legacy selection trajectory {key}",
        )
        if selection_trajectory_sha256 != source_sha256:
            raise FinalizationError(
                f"Attribution aggregate/legacy selection hashes differ for {key}"
            )

        receipt_path = root / f"provenance/attribution_current__{key}.json"
        receipt = _read_json(receipt_path, f"Attribution current provenance receipt {key}")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("experiment_family") != "attribution"
            or receipt.get("dataset") != dataset
            or receipt.get("model_id") != model_id
        ):
            raise FinalizationError(f"Attribution current receipt {key} has mixed identity")
        current_path = root / f"trajectories/attribution_current__{key}.parquet"
        current_sha256 = _validate_artifact_binding(
            receipt,
            current_path,
            path_key="output",
            sha_key="output_sha256",
            label=f"Attribution current receipt output {key}",
        )
        legacy_sources[key] = {
            "trajectory_path": str(historical_path),
            "trajectory_sha256": source_sha256,
            "selection_manifest_path": str(selection_path.resolve()),
            "selection_manifest_sha256": sha256_file(selection_path),
        }
        current_outputs[key] = {
            "trajectory_path": str(current_path.resolve()),
            "trajectory_sha256": current_sha256,
            "provenance_receipt_path": str(receipt_path.resolve()),
            "provenance_receipt_sha256": sha256_file(receipt_path),
        }

    return {
        "status": "COMPLETE_SIX_BRIDGE_SHA256_CHAIN",
        "aggregate_manifest_path": str(aggregate_path.resolve()),
        "aggregate_manifest_sha256": sha256_file(aggregate_path),
        "aggregate_output_path": str(aggregate_output.resolve()),
        "aggregate_output_sha256": aggregate_output_sha256,
        "source_count": len(legacy_sources),
        "legacy_sources": legacy_sources,
        "current_outputs": current_outputs,
        "validation": (
            "aggregate_output_and_six_sources_plus_legacy_selection_and_current_receipt_sha256"
        ),
    }


def _require_selection_manifests(root: Path) -> None:
    for phase in ("c0", "c1", "c2"):
        _require_file(
            root / f"manifests/controlled_{phase}_selection.json",
            f"Controlled {phase.upper()} selection manifest",
        )
    _require_file(root / "manifests/imagenet9_selection.csv", "ImageNet-9 selection manifest")
    _require_file(
        root / "manifests/imagenet9_historical_patch_orders.json",
        "ImageNet-9 historical patch-order manifest",
    )
    _require_file(root / "manifests/covertype_selection.json", "Covertype selection manifest")
    attribution = sorted(root.glob("manifests/attribution*selection.json"))
    if len(attribution) < 6 or any(path.is_symlink() for path in attribution):
        raise FinalizationError(
            "Attribution requires selection manifests for all six dataset/model bridges"
        )


def _require_recorded_bool(
    frame: pd.DataFrame,
    column: str,
    expected: np.ndarray | pd.Series,
    *,
    label: str,
) -> pd.Series:
    recorded = _bool_series(frame, column, label)
    wanted = np.asarray(expected, dtype=bool)
    if len(recorded) != len(wanted) or not np.array_equal(recorded.to_numpy(dtype=bool), wanted):
        raise FinalizationError(f"{label}.{column} differs from independent row recomputation")
    return recorded


def _require_recorded_numeric(
    frame: pd.DataFrame,
    column: str,
    expected: np.ndarray | pd.Series,
    *,
    label: str,
) -> np.ndarray:
    recorded = _finite_numeric(frame, column, label)
    wanted = np.asarray(expected, dtype=np.float64)
    if len(recorded) != len(wanted) or not np.allclose(
        recorded,
        wanted,
        atol=1.0e-14,
        rtol=1.0e-12,
    ):
        raise FinalizationError(f"{label}.{column} differs from independent row recomputation")
    return recorded


def _dominant_values(
    e_values: np.ndarray, c_values: np.ndarray, f_values: np.ndarray
) -> np.ndarray:
    result: list[str] = []
    for e_value, c_value, f_value in zip(e_values, c_values, f_values, strict=True):
        values = {"E": float(e_value), "C": float(c_value), "F": float(f_value)}
        maximum = max(values.values())
        result.append("|".join(name for name in ("E", "C", "F") if values[name] == maximum))
    return np.asarray(result, dtype=object)


def _require_recorded_text(
    frame: pd.DataFrame,
    column: str,
    expected: np.ndarray | pd.Series,
    *,
    label: str,
) -> None:
    if column not in frame:
        raise FinalizationError(f"{label} lacks {column!r}")
    recorded = frame[column].astype(str).to_numpy(dtype=str)
    wanted = np.asarray(expected, dtype=str)
    if len(recorded) != len(wanted) or not np.array_equal(recorded, wanted):
        raise FinalizationError(f"{label}.{column} differs from independent row recomputation")


def _tier_and_hard_recomputation(
    frame: pd.DataFrame,
    *,
    metric_names: Sequence[str],
    current: Mapping[str, np.ndarray],
    historical: Mapping[str, np.ndarray],
    boundary: np.ndarray,
    gate_match: np.ndarray,
    orientation_match: np.ndarray,
    dominant_match: np.ndarray,
    identity_match: np.ndarray,
    label: str,
) -> None:
    errors: list[np.ndarray] = []
    close: list[np.ndarray] = []
    for name in metric_names:
        signed = current[name] - historical[name]
        absolute = np.abs(signed)
        _require_recorded_numeric(frame, f"signed_error_{name}", signed, label=label)
        _require_recorded_numeric(frame, f"abs_error_{name}", absolute, label=label)
        errors.append(absolute)
        close.append(
            np.isclose(
                current[name],
                historical[name],
                atol=core_comparison.TIER_A_ATOL,
                rtol=core_comparison.TIER_A_RTOL,
            )
        )
    maximum_error = np.max(np.column_stack(errors), axis=1)
    tier_a = np.all(np.column_stack(close), axis=1)
    tier_b = (
        (maximum_error <= core_comparison.TIER_B_ABS)
        & (boundary | (gate_match & orientation_match))
        & dominant_match
    )
    tier = np.where(tier_a, "A", np.where(tier_b, "B", "FAIL"))
    hard = (
        (maximum_error > core_comparison.HARD_MISMATCH_ABS)
        | (~boundary & (~gate_match | ~orientation_match))
        | ~dominant_match
        | ~identity_match
    )
    _require_recorded_bool(frame, "tier_a_pass", tier_a, label=label)
    _require_recorded_bool(frame, "tier_b_pass", tier_b, label=label)
    _require_recorded_text(frame, "tier", tier, label=label)
    _require_recorded_bool(frame, "hard_mismatch", hard, label=label)


def _imagenet9_identity_contract(root: Path) -> tuple[set[tuple[str, str, str]], dict[str, int]]:
    selection = _read_csv(root / "manifests/imagenet9_selection.csv", "ImageNet-9 selection")
    required_selection = {"pair_id", "pair_type", "class_id"}
    if (
        not required_selection.issubset(selection.columns)
        or len(selection) != 16
        or selection["pair_id"].astype(str).duplicated().any()
    ):
        raise FinalizationError("ImageNet-9 selection identity inventory changed")
    pairs = selection["pair_id"].astype(str)
    pair_types = selection["pair_type"].astype(str)
    derived_types = pairs.str.rsplit("__", n=1).str[-1]
    classes = pd.to_numeric(selection["class_id"], errors="coerce").to_numpy(dtype=float)
    if (
        not pair_types.isin(IMAGENET9_PAIR_TYPES).all()
        or not np.array_equal(pair_types.to_numpy(dtype=str), derived_types.to_numpy(dtype=str))
        or not np.isfinite(classes).all()
        or not np.equal(classes, np.floor(classes)).all()
    ):
        raise FinalizationError("ImageNet-9 selection pair identity changed")
    class_by_pair = {
        pair_id: int(class_id) for pair_id, class_id in zip(pairs, classes, strict=True)
    }

    bridge = _read_json(root / "provenance/imagenet9_bridge.json", "ImageNet-9 bridge")
    receipt = _read_json(
        root / "provenance/imagenet9_current_e2e.json", "ImageNet-9 current E2E receipt"
    )
    bridge_models = bridge.get("models")
    receipt_models = receipt.get("models")
    if not isinstance(bridge_models, list) or not isinstance(receipt_models, list):
        raise FinalizationError("ImageNet-9 model identity receipts are incomplete")
    bridge_by_model = {
        str(record.get("current_model_id")): record
        for record in bridge_models
        if isinstance(record, dict)
    }
    receipt_by_model = {
        str(record.get("model_id")): record for record in receipt_models if isinstance(record, dict)
    }
    if set(bridge_by_model) != set(IMAGENET9_E2E_MODEL_BINDINGS) or set(
        receipt_by_model
    ) != set(IMAGENET9_E2E_MODEL_BINDINGS):
        raise FinalizationError("ImageNet-9 model identity inventory changed")
    for model_id, expected in IMAGENET9_E2E_MODEL_BINDINGS.items():
        bridge_record = bridge_by_model[model_id]
        receipt_record = receipt_by_model[model_id]
        if (
            bridge_record.get("historical_model_id") != expected["historical_model_id"]
            or bridge_record.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or bridge_record.get("sealed_sample_sha256") != expected["sealed_sample_sha256"]
            or receipt_record.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        ):
            raise FinalizationError(f"ImageNet-9 model/checkpoint identity changed: {model_id}")
    expected_units = {
        (model_id, pair_id, reveal_path)
        for model_id in IMAGENET9_E2E_MODEL_BINDINGS
        for pair_id in class_by_pair
        for reveal_path in IMAGENET9_REVEAL_PATHS
    }
    return expected_units, class_by_pair


def _read_parquet(path: Path, label: str) -> pd.DataFrame:
    _require_file(path, label)
    try:
        frame = pd.read_parquet(path)
    except Exception as error:
        raise FinalizationError(f"invalid {label}: {path}: {error}") from error
    if frame.empty:
        raise FinalizationError(f"{label} is empty: {path}")
    return frame


def _validate_imagenet9_sealed_summaries(
    root: Path,
    comparison: pd.DataFrame,
    class_by_pair: Mapping[str, int],
) -> list[dict[str, Any]]:
    bridge = _read_json(root / "provenance/imagenet9_bridge.json", "ImageNet-9 bridge")
    model_records = bridge.get("models")
    if not isinstance(model_records, list):
        raise FinalizationError("ImageNet-9 bridge lacks sealed model summaries")
    records_by_model = {
        str(record.get("current_model_id")): record
        for record in model_records
        if isinstance(record, dict)
    }
    source_pairs = {pair_id.rsplit("__", 1)[0] for pair_id in class_by_pair}
    sealed_frames: list[pd.DataFrame] = []
    bindings: list[dict[str, Any]] = []
    for model_id, expected in IMAGENET9_E2E_MODEL_BINDINGS.items():
        record = records_by_model.get(model_id)
        path_value = record.get("sealed_sample") if isinstance(record, dict) else None
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise FinalizationError(f"ImageNet-9 {model_id} sealed sample path changed")
        path = _require_file(Path(path_value), f"ImageNet-9 {model_id} sealed sample").resolve()
        if (
            record.get("sealed_sample_sha256") != expected["sealed_sample_sha256"]
            or sha256_file(path) != expected["sealed_sample_sha256"]
        ):
            raise FinalizationError(f"ImageNet-9 {model_id} sealed sample SHA-256 changed")
        sealed = _read_parquet(path, f"ImageNet-9 {model_id} sealed sample")
        required = {
            "model_id",
            "pair_id",
            "true_in9_class",
            "architecture_family",
            "training_regime",
            "pair_type",
            "path",
            "epsilon",
            "M",
            "E",
            "C",
            "F",
            "Abs",
            "endpoint_delta",
            "endpoint_active",
        }
        if not required.issubset(sealed.columns):
            raise FinalizationError(f"ImageNet-9 {model_id} sealed sample columns changed")
        epsilon = pd.to_numeric(sealed["epsilon"], errors="coerce").to_numpy(dtype=float)
        selected = sealed.loc[
            sealed["pair_id"].astype(str).isin(source_pairs)
            & np.isclose(epsilon, 0.02, atol=0.0, rtol=0.0)
        ].copy()
        keys = ["pair_id", "pair_type", "path"]
        if (
            len(selected) != 48
            or selected.duplicated(keys).any()
            or set(selected["pair_type"].astype(str)) != IMAGENET9_PAIR_TYPES
            or set(selected["path"].astype(str)) != IMAGENET9_REVEAL_PATHS
            or set(selected["model_id"].astype(str)) != {expected["historical_model_id"]}
        ):
            raise FinalizationError(f"ImageNet-9 {model_id} sealed 48-row selection changed")
        selected["model_id"] = model_id
        selected["pair_id"] = (
            selected["pair_id"].astype(str) + "__" + selected["pair_type"].astype(str)
        )
        selected = selected.rename(
            columns={
                "path": "reveal_path",
                "endpoint_delta": "historical_endpoint_d_sealed",
                "endpoint_active": "historical_gate_sealed",
                **{name: f"historical_{name}_sealed" for name in SUMMARY_NAMES},
            }
        )
        sealed_frames.append(selected)
        bindings.append(
            {
                "model_id": model_id,
                "path": str(path),
                "sha256": expected["sealed_sample_sha256"],
                "selected_rows": 48,
            }
        )
    sealed_selected = pd.concat(sealed_frames, ignore_index=True)
    unit_keys = ["model_id", "pair_id", "reveal_path"]
    bound = comparison.merge(
        sealed_selected.loc[
            :,
            [
                *unit_keys,
                "historical_endpoint_d_sealed",
                "historical_gate_sealed",
                *(f"historical_{name}_sealed" for name in SUMMARY_NAMES),
            ],
        ],
        on=unit_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(bound) != 144:
        raise FinalizationError("ImageNet-9 E2E rows do not bind all sealed summaries")
    for column in ("endpoint_d", *SUMMARY_NAMES):
        if not np.allclose(
            _finite_numeric(bound, f"historical_{column}", "ImageNet-9 E2E CSV"),
            _finite_numeric(
                bound,
                f"historical_{column}_sealed",
                "ImageNet-9 sealed summaries",
            ),
            atol=1.0e-14,
            rtol=1.0e-12,
        ):
            raise FinalizationError(
                f"ImageNet-9 E2E historical_{column} differs from sealed summary bytes"
            )
    _require_recorded_bool(
        bound,
        "historical_gate",
        _bool_series(bound, "historical_gate_sealed", "ImageNet-9 sealed summaries"),
        label="ImageNet-9 E2E CSV",
    )
    return bindings


def _validate_imagenet9_stage_evidence(
    root: Path,
    comparison: pd.DataFrame,
    expected_units: set[tuple[str, str, str]],
    class_by_pair: Mapping[str, int],
) -> dict[str, Any]:
    sealed_summary_bindings = _validate_imagenet9_sealed_summaries(
        root, comparison, class_by_pair
    )
    current_path = root / "trajectories/imagenet9_current_e2e_scans.parquet"
    legacy_path = root / "trajectories/imagenet9_legacy_stage_scores.parquet"
    stage_csv_path = root / "comparisons/imagenet9_stage_responses.csv"
    current = _read_parquet(current_path, "ImageNet-9 current stage evidence")
    legacy = _read_parquet(legacy_path, "ImageNet-9 historical stage evidence")
    stage_csv = _read_csv(stage_csv_path, "ImageNet-9 stage-response comparison")
    current_columns = {
        "pair_id",
        "pair_type",
        "model_id",
        "reveal_path",
        "stage_index",
        "alpha",
        "response",
        "checkpoint_sha256",
    }
    legacy_columns = {
        "historical_model_id",
        "model_id",
        "checkpoint_sha256",
        "source_pair_id",
        "pair_id",
        "pair_type",
        "class_id",
        "reveal_path",
        "stage_index",
        "alpha",
        "score_plus",
        "score_minus",
        "response",
    }
    if set(current.columns) != current_columns or set(legacy.columns) != legacy_columns:
        raise FinalizationError("ImageNet-9 stage evidence columns changed")
    stage_keys = ["model_id", "pair_id", "reveal_path", "stage_index"]
    expected_stage_keys = {
        (*unit, stage_index) for unit in expected_units for stage_index in range(9)
    }
    for name, frame in (("current", current), ("historical", legacy)):
        indices = _finite_numeric(frame, "stage_index", f"ImageNet-9 {name} stages")
        observed_stage_keys = set(
            zip(
                frame["model_id"].astype(str),
                frame["pair_id"].astype(str),
                frame["reveal_path"].astype(str),
                indices.astype(int),
                strict=True,
            )
        )
        if (
            len(frame) != 1296
            or not np.equal(indices, np.floor(indices)).all()
            or len(observed_stage_keys) != 1296
            or observed_stage_keys != expected_stage_keys
        ):
            raise FinalizationError(f"ImageNet-9 {name} stage cartesian inventory changed")
        alpha = _finite_numeric(frame, "alpha", f"ImageNet-9 {name} stages")
        expected_alpha = np.linspace(0.0, 1.0, 9)[indices.astype(int)]
        if not np.array_equal(alpha, expected_alpha):
            raise FinalizationError(f"ImageNet-9 {name} stage alpha grid changed")
        expected_pair_types = frame["pair_id"].astype(str).str.rsplit("__", n=1).str[-1]
        if not np.array_equal(
            frame["pair_type"].astype(str).to_numpy(dtype=str),
            expected_pair_types.to_numpy(dtype=str),
        ):
            raise FinalizationError(f"ImageNet-9 {name} stage pair identity changed")
        expected_checkpoints = frame["model_id"].astype(str).map(
            lambda value: IMAGENET9_E2E_MODEL_BINDINGS[value]["checkpoint_sha256"]
        )
        if not np.array_equal(
            frame["checkpoint_sha256"].astype(str).to_numpy(dtype=str),
            expected_checkpoints.to_numpy(dtype=str),
        ):
            raise FinalizationError(f"ImageNet-9 {name} stage checkpoint identity changed")
        _finite_numeric(frame, "response", f"ImageNet-9 {name} stages")
    expected_historical_models = legacy["model_id"].astype(str).map(
        lambda value: IMAGENET9_E2E_MODEL_BINDINGS[value]["historical_model_id"]
    )
    expected_sources = legacy["pair_id"].astype(str).str.rsplit("__", n=1).str[0]
    expected_classes = legacy["pair_id"].astype(str).map(class_by_pair).to_numpy(dtype=int)
    if (
        not np.array_equal(
            legacy["historical_model_id"].astype(str).to_numpy(dtype=str),
            expected_historical_models.to_numpy(dtype=str),
        )
        or not np.array_equal(
            legacy["source_pair_id"].astype(str).to_numpy(dtype=str),
            expected_sources.to_numpy(dtype=str),
        )
        or not np.array_equal(
            _finite_numeric(legacy, "class_id", "ImageNet-9 historical stages"),
            expected_classes,
        )
    ):
        raise FinalizationError("ImageNet-9 historical stage identity columns changed")
    score_plus = _finite_numeric(legacy, "score_plus", "ImageNet-9 historical stages")
    score_minus = _finite_numeric(legacy, "score_minus", "ImageNet-9 historical stages")
    legacy_response = _finite_numeric(legacy, "response", "ImageNet-9 historical stages")
    if not np.allclose(
        legacy_response,
        score_plus - score_minus,
        atol=1.0e-14,
        rtol=1.0e-12,
    ):
        raise FinalizationError(
            "ImageNet-9 historical stage response is not score_plus-score_minus"
        )

    neutral_path = root / "trajectories/imagenet9.parquet"
    try:
        neutral = read_trajectory_record(
            _require_file(neutral_path, "ImageNet-9 neutral trajectory")
        )
    except Exception as error:
        raise FinalizationError(f"ImageNet-9 neutral trajectory is invalid: {error}") from error
    if len(neutral) != 1296:
        raise FinalizationError("ImageNet-9 neutral trajectory must contain exactly 1296 stages")
    expected_neutral_unit_ids = {
        f"imagenet9::{model_id}::{pair_id}::{reveal_path}"
        for model_id, pair_id, reveal_path in expected_units
    }
    if set(neutral["unit_id"].astype(str)) != expected_neutral_unit_ids:
        raise FinalizationError("ImageNet-9 neutral trajectory unit identity changed")
    neutral_stages = neutral.rename(
        columns={
            "sample_or_pair_id": "pair_id",
            "protocol": "reveal_path",
            "stage_t": "alpha_neutral",
            "stage_r": "response_neutral",
            "checkpoint_sha256": "checkpoint_sha256_neutral",
        }
    )
    historical_crosscheck = legacy.merge(
        neutral_stages.loc[
            :,
            [
                *stage_keys,
                "unit_id",
                "alpha_neutral",
                "response_neutral",
                "checkpoint_sha256_neutral",
                "endpoint_d",
                *(f"historical_{name}" for name in SUMMARY_NAMES),
            ],
        ],
        on=stage_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(historical_crosscheck) != 1296:
        raise FinalizationError("ImageNet-9 historical stages do not bind the neutral trajectory")
    for left, right in (
        ("alpha", "alpha_neutral"),
        ("response", "response_neutral"),
    ):
        if not np.allclose(
            _finite_numeric(historical_crosscheck, left, "ImageNet-9 historical stages"),
            _finite_numeric(historical_crosscheck, right, "ImageNet-9 neutral trajectory"),
            atol=1.0e-14,
            rtol=1.0e-12,
        ):
            raise FinalizationError(
                f"ImageNet-9 historical stage {left} differs from the neutral trajectory"
            )
    if not np.array_equal(
        historical_crosscheck["checkpoint_sha256"].astype(str).to_numpy(dtype=str),
        historical_crosscheck["checkpoint_sha256_neutral"].astype(str).to_numpy(dtype=str),
    ):
        raise FinalizationError("ImageNet-9 historical/neutral checkpoint identities differ")
    historical_units = historical_crosscheck.loc[
        historical_crosscheck["stage_index"].astype(int).eq(0),
        ["unit_id", "endpoint_d", *(f"historical_{name}" for name in SUMMARY_NAMES)],
    ].rename(
        columns={
            "endpoint_d": "historical_endpoint_d_neutral",
            **{
                f"historical_{name}": f"historical_{name}_neutral"
                for name in SUMMARY_NAMES
            },
        }
    )
    expected_comparison_unit_ids = (
        "imagenet9::"
        + comparison["model_id"].astype(str)
        + "::"
        + comparison["pair_id"].astype(str)
        + "::"
        + comparison["reveal_path"].astype(str)
    )
    comparison_historical = comparison.copy()
    comparison_historical["unit_id"] = expected_comparison_unit_ids
    historical_bound = comparison_historical.merge(
        historical_units,
        on="unit_id",
        how="inner",
        validate="one_to_one",
    )
    if len(historical_bound) != 144:
        raise FinalizationError("ImageNet-9 E2E historical summaries lack neutral bindings")
    for column in SUMMARY_NAMES:
        if not np.allclose(
            _finite_numeric(
                historical_bound,
                f"historical_{column}",
                "ImageNet-9 E2E historical summary",
            ),
            _finite_numeric(
                historical_bound,
                f"historical_{column}_neutral",
                "ImageNet-9 neutral historical summary",
            ),
            atol=1.0e-14,
            rtol=1.0e-12,
        ):
            raise FinalizationError(
                f"ImageNet-9 E2E historical_{column} differs from neutral evidence"
            )

    merged = current.merge(
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
    if len(merged) != 1296 or not np.array_equal(
        merged["alpha_current"].to_numpy(dtype=float),
        merged["alpha_legacy"].to_numpy(dtype=float),
    ):
        raise FinalizationError("ImageNet-9 current/historical stage identities differ")
    merged["response_abs_error"] = np.abs(
        merged["response_current"].to_numpy(dtype=float)
        - merged["response_legacy"].to_numpy(dtype=float)
    )
    expected_stage_csv = merged.loc[
        :,
        [
            "pair_id",
            "pair_type",
            "model_id",
            "reveal_path",
            "stage_index",
            "alpha_current",
            "response_current",
            "checkpoint_sha256",
            "alpha_legacy",
            "score_plus",
            "score_minus",
            "response_legacy",
            "source_pair_id",
            "historical_model_id",
            "response_abs_error",
        ],
    ].sort_values(stage_keys, kind="stable").reset_index(drop=True)
    if set(stage_csv.columns) != set(expected_stage_csv.columns) or len(stage_csv) != 1296:
        raise FinalizationError("ImageNet-9 stage-response CSV inventory changed")
    observed_stage_csv = stage_csv.sort_values(stage_keys, kind="stable").reset_index(drop=True)
    for column in expected_stage_csv:
        wanted = expected_stage_csv[column]
        if pd.api.types.is_numeric_dtype(wanted.dtype):
            observed = pd.to_numeric(observed_stage_csv[column], errors="coerce").to_numpy(float)
            if not np.isfinite(observed).all() or not np.allclose(
                observed,
                wanted.to_numpy(float),
                atol=1.0e-14,
                rtol=1.0e-12,
            ):
                raise FinalizationError(
                    f"ImageNet-9 stage-response CSV.{column} differs from stage parquets"
                )
        elif observed_stage_csv[column].astype(str).tolist() != wanted.astype(str).tolist():
            raise FinalizationError(
                f"ImageNet-9 stage-response CSV.{column} differs from stage parquets"
            )

    unit_keys = stage_keys[:-1]
    aggregates = (
        merged.groupby(unit_keys, sort=True)["response_abs_error"]
        .agg(
            stage_response_median_abs_error="median",
            stage_response_max_abs_error="max",
        )
        .reset_index()
    )
    p95 = (
        merged.groupby(unit_keys, sort=True)["response_abs_error"]
        .apply(lambda values: float(np.percentile(values, 95)))
        .rename("stage_response_p95_abs_error")
        .reset_index()
    )
    aggregates = aggregates.merge(p95, on=unit_keys, validate="one_to_one")
    endpoints = merged.loc[merged["stage_index"].astype(int).eq(8)].copy()
    endpoints["current_legacy_endpoint_abs_error"] = np.abs(
        endpoints["response_current"] - endpoints["response_legacy"]
    )
    aggregates = aggregates.merge(
        endpoints.loc[
            :,
            [*unit_keys, "response_legacy", "current_legacy_endpoint_abs_error"],
        ].rename(columns={"response_legacy": "legacy_fresh_endpoint_d"}),
        on=unit_keys,
        validate="one_to_one",
    )
    aligned = comparison.merge(
        aggregates,
        on=unit_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_recomputed"),
    )
    if len(aligned) != 144:
        raise FinalizationError("ImageNet-9 E2E rows do not bind the 144 stage trajectories")
    for column in (
        "stage_response_median_abs_error",
        "stage_response_p95_abs_error",
        "stage_response_max_abs_error",
        "legacy_fresh_endpoint_d",
        "current_legacy_endpoint_abs_error",
    ):
        if not np.allclose(
            _finite_numeric(aligned, column, "ImageNet-9 E2E CSV"),
            _finite_numeric(aligned, f"{column}_recomputed", "ImageNet-9 stage recomputation"),
            atol=1.0e-14,
            rtol=1.0e-12,
        ):
            raise FinalizationError(f"ImageNet-9 E2E {column} differs from stage parquets")
    scores = evaluate_response_frame(current, epsilon=0.02)
    scores = scores.rename(
        columns={
            **{name: f"current_{name}_recomputed" for name in SUMMARY_NAMES},
            "endpoint_delta": "current_endpoint_d_recomputed",
            "endpoint_active": "current_gate_recomputed",
        }
    )
    aligned_scores = comparison.merge(scores, on=unit_keys, validate="one_to_one")
    for column in (*SUMMARY_NAMES, "endpoint_d"):
        if not np.allclose(
            _finite_numeric(aligned_scores, f"current_{column}", "ImageNet-9 E2E CSV"),
            _finite_numeric(
                aligned_scores,
                f"current_{column}_recomputed",
                "ImageNet-9 current-stage recomputation",
            ),
            atol=1.0e-14,
            rtol=1.0e-12,
        ):
            raise FinalizationError(
                f"ImageNet-9 E2E current_{column} differs from current stage evidence"
            )
    _require_recorded_bool(
        aligned_scores,
        "current_gate",
        _bool_series(
            aligned_scores,
            "current_gate_recomputed",
            "ImageNet-9 current-stage recomputation",
        ),
        label="ImageNet-9 E2E CSV",
    )
    return {
        "current_stage_path": str(current_path.resolve()),
        "current_stage_sha256": sha256_file(current_path),
        "historical_stage_path": str(legacy_path.resolve()),
        "historical_stage_sha256": sha256_file(legacy_path),
        "neutral_trajectory_path": str(neutral_path.resolve()),
        "neutral_trajectory_sha256": sha256_file(neutral_path),
        "stage_comparison_path": str(stage_csv_path.resolve()),
        "stage_comparison_sha256": sha256_file(stage_csv_path),
        "stage_count": 1296,
        "unit_count": 144,
        "sealed_summary_bindings": sealed_summary_bindings,
        "validation": (
            "two_parquet_exact_cartesian_identity_checkpoint_alpha_response_"
            "plus_stage_csv_and_current_summary_recomputation"
        ),
    }


def _validate_imagenet9_e2e_rows(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    label = "ImageNet-9 E2E CSV"
    required = {
        "model_id",
        "historical_model_id",
        "pair_id",
        "reveal_path",
        "pair_type",
        "pair_type_current_label",
        "pair_type_historical_label",
        "true_in9_class",
        "epsilon",
        "current_endpoint_d",
        "historical_endpoint_d",
        "current_gate",
        "historical_gate",
        "current_orientation",
        "historical_orientation",
        "current_dominant",
        "historical_dominant",
        "checkpoint_identity_match",
        "sample_identity_match",
        "identity_match",
        *(f"current_{name}" for name in SUMMARY_NAMES),
        *(f"historical_{name}" for name in SUMMARY_NAMES),
        *(f"abs_error_{name}" for name in SUMMARY_NAMES),
        *(f"signed_error_{name}" for name in SUMMARY_NAMES),
        "boundary",
        "gate_match",
        "orientation_match",
        "dominant_match",
        "tier_a_pass",
        "tier_b_pass",
        "tier",
        "hard_mismatch",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FinalizationError(f"{label} lacks columns: {', '.join(missing)}")
    expected_units, class_by_pair = _imagenet9_identity_contract(root)
    stage_binding = _validate_imagenet9_stage_evidence(
        root, frame, expected_units, class_by_pair
    )
    observed_units = list(
        zip(
            frame["model_id"].astype(str),
            frame["pair_id"].astype(str),
            frame["reveal_path"].astype(str),
            strict=True,
        )
    )
    if (
        len(observed_units) != 144
        or len(set(observed_units)) != 144
        or set(observed_units) != expected_units
    ):
        raise FinalizationError("ImageNet-9 E2E unit identity/cartesian inventory changed")

    model_ids = frame["model_id"].astype(str)
    pair_ids = frame["pair_id"].astype(str)
    reveal_paths = frame["reveal_path"].astype(str)
    expected_historical_models = model_ids.map(
        lambda value: IMAGENET9_E2E_MODEL_BINDINGS[value]["historical_model_id"]
    )
    expected_pair_types = pair_ids.str.rsplit("__", n=1).str[-1]
    expected_classes = pair_ids.map(class_by_pair).to_numpy(dtype=int)
    _require_recorded_text(
        frame, "historical_model_id", expected_historical_models, label=label
    )
    for column in ("pair_type", "pair_type_current_label", "pair_type_historical_label"):
        _require_recorded_text(frame, column, expected_pair_types, label=label)
    _require_recorded_numeric(frame, "true_in9_class", expected_classes, label=label)
    checkpoint_identity = np.ones(len(frame), dtype=bool)
    sample_identity = pair_ids.isin(class_by_pair).to_numpy() & reveal_paths.isin(
        IMAGENET9_REVEAL_PATHS
    ).to_numpy()
    identity = checkpoint_identity & sample_identity
    _require_recorded_bool(
        frame, "checkpoint_identity_match", checkpoint_identity, label=label
    )
    _require_recorded_bool(frame, "sample_identity_match", sample_identity, label=label)
    _require_recorded_bool(frame, "identity_match", identity, label=label)

    epsilon = _finite_numeric(frame, "epsilon", label)
    if not np.array_equal(epsilon, np.full(len(frame), 0.02, dtype=float)):
        raise FinalizationError("ImageNet-9 E2E epsilon changed")
    endpoints = {
        side: _finite_numeric(frame, f"{side}_endpoint_d", label)
        for side in ("current", "historical")
    }
    current = {
        name: _finite_numeric(frame, f"current_{name}", label) for name in SUMMARY_NAMES
    }
    historical = {
        name: _finite_numeric(frame, f"historical_{name}", label) for name in SUMMARY_NAMES
    }
    for side, values in (("current", current), ("historical", historical)):
        if not np.allclose(
            values["M"], np.abs(endpoints[side]), atol=1.0e-14, rtol=1.0e-12
        ):
            raise FinalizationError(f"ImageNet-9 E2E {side} M/endpoint identity changed")
    current_gate = np.abs(endpoints["current"]) >= epsilon
    historical_gate = np.abs(endpoints["historical"]) >= epsilon
    current_orientation = np.where(current_gate, np.sign(endpoints["current"]).astype(int), 0)
    historical_orientation = np.where(
        historical_gate, np.sign(endpoints["historical"]).astype(int), 0
    )
    boundary = np.abs(np.abs(endpoints["historical"]) - epsilon) <= core_comparison.BOUNDARY_ABS
    gate_match = current_gate == historical_gate
    orientation_match = current_orientation == historical_orientation
    current_dominant = _dominant_values(current["E"], current["C"], current["F"])
    historical_dominant = _dominant_values(
        historical["E"], historical["C"], historical["F"]
    )
    dominant_match = current_dominant == historical_dominant
    for column, expected in (
        ("current_gate", current_gate),
        ("historical_gate", historical_gate),
        ("boundary", boundary),
        ("gate_match", gate_match),
        ("orientation_match", orientation_match),
        ("dominant_match", dominant_match),
    ):
        _require_recorded_bool(frame, column, expected, label=label)
    _require_recorded_numeric(
        frame, "current_orientation", current_orientation, label=label
    )
    _require_recorded_numeric(
        frame, "historical_orientation", historical_orientation, label=label
    )
    _require_recorded_text(frame, "current_dominant", current_dominant, label=label)
    _require_recorded_text(frame, "historical_dominant", historical_dominant, label=label)
    _tier_and_hard_recomputation(
        frame,
        metric_names=SUMMARY_NAMES,
        current=current,
        historical=historical,
        boundary=boundary,
        gate_match=gate_match,
        orientation_match=orientation_match,
        dominant_match=dominant_match,
        identity_match=identity,
        label=label,
    )
    patched = frame.copy()
    patched["unit_id"] = [
        f"imagenet9::{model_id}::{pair_id}::{reveal_path}"
        for model_id, pair_id, reveal_path in observed_units
    ]
    patched.attrs["stage_binding"] = stage_binding
    return patched


def _validate_attribution_e2e_rows(frame: pd.DataFrame) -> pd.DataFrame:
    label = "Attribution E2E CSV"
    required = {
        "unit_id",
        "dataset",
        "model",
        "method",
        "image_id",
        "factor_or_part_id",
        "historical_checkpoint_sha256",
        "current_checkpoint_sha256",
        "historical_target",
        "current_target",
        "historical_counterfactual_map",
        "current_counterfactual_map",
        "historical_reference",
        "current_reference",
        "historical_intervention_operator",
        "current_intervention_operator",
        "current_endpoint_d",
        "historical_endpoint_d",
        "current_gate",
        "historical_gate",
        "current_orientation",
        "historical_orientation",
        "current_dominant",
        "historical_dominant",
        "checkpoint_match",
        "target_match",
        "counterfactual_map_match",
        "reference_match",
        "intervention_operator_match",
        "identity_match",
        "current_input_domain",
        "current_preprocess_inside_forward",
        *(f"current_{name}" for name in ATTRIBUTION_VECTOR_METRICS),
        *(f"historical_{name}" for name in ATTRIBUTION_VECTOR_METRICS),
        *(f"abs_error_{name}" for name in ATTRIBUTION_VECTOR_METRICS),
        *(f"signed_error_{name}" for name in ATTRIBUTION_VECTOR_METRICS),
        "boundary",
        "gate_match",
        "orientation_match",
        "dominant_match",
        "tier_a_pass",
        "tier_b_pass",
        "tier",
        "hard_mismatch",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FinalizationError(f"{label} lacks columns: {', '.join(missing)}")
    if len(frame) != 1476 or frame["unit_id"].astype(str).duplicated().any():
        raise FinalizationError("Attribution E2E must contain 1476 unique vector units")
    dataset_models = set(
        zip(frame["dataset"].astype(str), frame["model"].astype(str), strict=True)
    )
    allowed_dataset_models = set(ATTRIBUTION_BRIDGES.values())
    if dataset_models != allowed_dataset_models or set(frame["method"].astype(str)) != {
        "decaf_3",
        "decaf_5",
        "decaf_9",
    }:
        raise FinalizationError("Attribution E2E dataset/model/method identity changed")
    expected_unit_ids = (
        "attribution::"
        + frame["dataset"].astype(str)
        + "::"
        + frame["model"].astype(str)
        + "::"
        + frame["method"].astype(str)
        + "::"
        + frame["image_id"].astype(str)
        + "::"
        + frame["factor_or_part_id"].astype(str)
    )
    if not np.array_equal(
        frame["unit_id"].astype(str).to_numpy(dtype=str),
        expected_unit_ids.to_numpy(dtype=str),
    ):
        raise FinalizationError("Attribution E2E unit IDs differ from their identity columns")

    checkpoint_match = (
        frame["current_checkpoint_sha256"].astype(str)
        == frame["historical_checkpoint_sha256"].astype(str)
    ).to_numpy()
    current_target = _finite_numeric(frame, "current_target", label)
    historical_target = _finite_numeric(frame, "historical_target", label)
    if not np.equal(current_target, np.floor(current_target)).all() or not np.equal(
        historical_target, np.floor(historical_target)
    ).all():
        raise FinalizationError("Attribution E2E target identities are non-integral")
    target_match = current_target == historical_target
    string_identity_columns = {
        "counterfactual_map": (
            frame["current_counterfactual_map"].astype(str)
            == frame["historical_counterfactual_map"].astype(str)
        ).to_numpy(),
        "reference": (
            frame["current_reference"].astype(str)
            == frame["historical_reference"].astype(str)
        ).to_numpy(),
        "intervention_operator": (
            frame["current_intervention_operator"].astype(str)
            == frame["historical_intervention_operator"].astype(str)
        ).to_numpy(),
    }
    identity = checkpoint_match & target_match
    for name, expected in string_identity_columns.items():
        identity &= expected
        _require_recorded_bool(frame, f"{name}_match", expected, label=label)
    _require_recorded_bool(frame, "checkpoint_match", checkpoint_match, label=label)
    _require_recorded_bool(frame, "target_match", target_match, label=label)
    _require_recorded_bool(frame, "identity_match", identity, label=label)
    expected_domain = frame["dataset"].astype(str).map(
        {"funnybirds": "raw_rgb_float_0_1", "imagenet1k_idsds": "fixed_shape_model_input"}
    )
    expected_preprocess = frame["dataset"].astype(str).eq("funnybirds").to_numpy()
    _require_recorded_text(frame, "current_input_domain", expected_domain, label=label)
    _require_recorded_bool(
        frame, "current_preprocess_inside_forward", expected_preprocess, label=label
    )

    endpoints = {
        side: _finite_numeric(frame, f"{side}_endpoint_d", label)
        for side in ("current", "historical")
    }
    current = {
        name: _finite_numeric(frame, f"current_{name}", label)
        for name in ATTRIBUTION_VECTOR_METRICS
    }
    historical = {
        name: _finite_numeric(frame, f"historical_{name}", label)
        for name in ATTRIBUTION_VECTOR_METRICS
    }
    current_gate = current["M"] >= ATTRIBUTION_ENDPOINT_EPSILON
    historical_gate = historical["M"] >= ATTRIBUTION_ENDPOINT_EPSILON
    current_orientation = np.where(current_gate, np.sign(endpoints["current"]).astype(int), 0)
    historical_orientation = np.where(
        historical_gate, np.sign(endpoints["historical"]).astype(int), 0
    )
    boundary = (
        np.abs(historical["M"] - ATTRIBUTION_ENDPOINT_EPSILON)
        <= core_comparison.BOUNDARY_ABS
    )
    gate_match = current_gate == historical_gate
    orientation_match = current_orientation == historical_orientation
    current_dominant = _dominant_values(current["E"], current["C"], current["F"])
    historical_dominant = _dominant_values(
        historical["E"], historical["C"], historical["F"]
    )
    dominant_match = current_dominant == historical_dominant
    for column, expected in (
        ("current_gate", current_gate),
        ("historical_gate", historical_gate),
        ("boundary", boundary),
        ("gate_match", gate_match),
        ("orientation_match", orientation_match),
        ("dominant_match", dominant_match),
    ):
        _require_recorded_bool(frame, column, expected, label=label)
    _require_recorded_numeric(
        frame, "current_orientation", current_orientation, label=label
    )
    _require_recorded_numeric(
        frame, "historical_orientation", historical_orientation, label=label
    )
    _require_recorded_text(frame, "current_dominant", current_dominant, label=label)
    _require_recorded_text(frame, "historical_dominant", historical_dominant, label=label)
    _tier_and_hard_recomputation(
        frame,
        metric_names=ATTRIBUTION_VECTOR_METRICS,
        current=current,
        historical=historical,
        boundary=boundary,
        gate_match=gate_match,
        orientation_match=orientation_match,
        dominant_match=dominant_match,
        identity_match=identity,
        label=label,
    )
    return frame.copy()


def _status_from_e2e_csv(root: Path, frame: pd.DataFrame, family: str) -> dict[str, Any]:
    if family == "imagenet9":
        validated = _validate_imagenet9_e2e_rows(root, frame)
        label = "imagenet9 E2E"
    elif family == "attribution":
        validated = _validate_attribution_e2e_rows(frame)
        label = "attribution E2E"
    else:
        raise FinalizationError(f"unsupported row-level E2E family: {family}")
    result = aggregate_unit_comparisons([validated], label=label)
    result["row_level_validation"] = (
        "independent_numeric_identity_semantic_tier_and_hard_mismatch_recomputation"
    )
    if family == "imagenet9":
        result["stage_evidence"] = validated.attrs["stage_binding"]
    return result


def _validate_e2e_summary(
    summary: Mapping[str, Any], stats: Mapping[str, Any], *, label: str
) -> None:
    if int(summary.get("unit_count", -1)) != int(stats["unit_count"]):
        raise FinalizationError(f"{label} unit count differs from its comparison CSV")
    for key in (
        "tier_a_fraction",
        "tier_b_fraction",
        "tier_a_or_b_fraction",
        "hard_mismatch_fraction",
        "gate_agreement",
        "orientation_agreement",
        "dominant_mechanism_agreement",
        "identity_agreement",
    ):
        if key not in summary:
            raise FinalizationError(f"{label} lacks {key}")
        if not np.isclose(float(summary[key]), float(stats[key]), atol=1.0e-12, rtol=0.0):
            raise FinalizationError(f"{label}.{key} differs from its comparison CSV")


def _attribution_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).all():
        raise FinalizationError(f"Attribution {name} is not a finite vector of length >= 2")
    return vector


def _require_attribution_vector_binding(
    recorded: Any, authoritative: Any, *, label: str
) -> None:
    observed = _attribution_vector(recorded, name=f"{label} recorded")
    expected = _attribution_vector(authoritative, name=f"{label} authoritative")
    if observed.shape != expected.shape or not np.allclose(
        observed, expected, atol=1.0e-14, rtol=1.0e-12
    ):
        raise FinalizationError(f"Attribution {label} differs from bound raw output vector")


def _current_member_spearman(row: Any, *, funnybirds: bool) -> float:
    """Recompute one production Spearman value from persisted raw vectors."""

    patch = _attribution_vector(row.patch_scores, name="patch_scores")
    if funnybirds:
        first = _attribution_vector(
            row.heldout_background_texture_effects,
            name="heldout_background_texture_effects",
        )
        second = _attribution_vector(
            row.heldout_telea_dilate3_effects,
            name="heldout_telea_dilate3_effects",
        )
        if first.shape != patch.shape or second.shape != patch.shape:
            raise FinalizationError("FunnyBirds held-out vectors differ from patch-score shape")
        first_score = float(row_spearman(patch, first)[0])
        second_score = float(row_spearman(patch, second)[0])
        expected = float(np.mean((first_score, second_score)))
        for name, observed in (
            ("spearman_background_texture", row.spearman_background_texture),
            ("spearman_telea_dilate3", row.spearman_telea_dilate3),
            ("spearman", row.spearman),
        ):
            target = {
                "spearman_background_texture": first_score,
                "spearman_telea_dilate3": second_score,
                "spearman": expected,
            }[name]
            if not np.isclose(float(observed), target, atol=1.0e-12, rtol=0.0):
                raise FinalizationError(
                    f"current FunnyBirds {name} differs from held-out raw-vector recomputation"
                )
        # Older sealed B200 output used this label for the same equal-operator mean.
        if str(row.quality_aggregation) not in {
            "equal_mean_within_image",
            "equal_mean_of_operator_spearman",
        }:
            raise FinalizationError("current FunnyBirds quality aggregation changed")
        return expected
    endpoint = _attribution_vector(row.endpoint_effects, name="endpoint_effects")
    if endpoint.shape != patch.shape:
        raise FinalizationError("IDSDS endpoint vector differs from patch-score shape")
    expected = float(row_spearman(patch, endpoint)[0])
    if not np.isclose(float(row.spearman), expected, atol=1.0e-12, rtol=0.0):
        raise FinalizationError("current IDSDS Spearman differs from raw-vector recomputation")
    return expected


def _attribution_reference_receipt(root: Path) -> tuple[Path, dict[str, Any]]:
    replay_root = (root / "verification/replay").resolve()
    family_replay_root = replay_root / "family_replays"
    family_receipt_path = _require_file(
        family_replay_root / "family_replay_receipt.json",
        "current family-replay receipt",
    )
    analysis = _read_json(root / "verification/analysis_replay.json", "analysis replay")
    family_receipt = _read_json(family_receipt_path, "current family-replay receipt")
    if analysis.get("family_replay_receipt_sha256") != sha256_file(family_receipt_path):
        raise FinalizationError("analysis replay does not bind the current family-replay receipt")

    families = family_receipt.get("families")
    if not isinstance(families, list):
        raise FinalizationError("current family-replay receipt lacks a family list")
    matches = [
        record
        for record in families
        if isinstance(record, dict) and record.get("family") == "attribution"
    ]
    if len(matches) != 1:
        raise FinalizationError("current family-replay receipt must bind one attribution replay")
    attribution = matches[0]
    contract = attribution.get("contract")
    analysis_record = attribution.get("analysis")
    path_value = attribution.get("path")
    relative = PurePosixPath(path_value) if isinstance(path_value, str) else None
    if (
        attribution.get("status") != "completed"
        or not isinstance(contract, dict)
        or contract.get("status") != "passed"
        or not isinstance(analysis_record, dict)
        or analysis_record.get("source_mode") != "sealed_reference_replay"
        or relative is None
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 3
        or relative.parts[0] != "family_replays"
        or not relative.parts[1].startswith("invocation-")
        or relative.parts[2] != "attribution"
    ):
        raise FinalizationError("current attribution family-replay binding is invalid")

    attribution_root = replay_root.joinpath(*relative.parts)
    if (
        not attribution_root.is_dir()
        or attribution_root.is_symlink()
        or attribution_root.resolve().parent.parent != family_replay_root.resolve()
    ):
        raise FinalizationError("current attribution family-replay path is stale or mixed")
    receipt_path = _require_file(
        attribution_root / "receipts/attribution_reference_inputs.json",
        "Attribution reference-input receipt",
    ).resolve()
    return receipt_path, _read_json(receipt_path, "Attribution reference-input receipt")


def _reconstruct_attribution_spearman(
    root: Path,
    comparison: pd.DataFrame,
    attribution_chain: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = list(ATTRIBUTION_SPEARMAN_IDENTITY_COLUMNS)
    vector_keys = [*keys, "factor_or_part_id", "vector_index"]
    required = {
        *vector_keys,
        "historical_score",
        "historical_endpoint_d",
        "current_score",
        "current_endpoint_d",
        "historical_spearman",
        "current_spearman",
    }
    if len(comparison) != 1476 or not required.issubset(comparison.columns):
        raise FinalizationError("Attribution E2E lacks the exact 1,476 raw vector elements")
    for column in (
        "historical_score",
        "historical_endpoint_d",
        "current_score",
        "current_endpoint_d",
    ):
        _finite_numeric(comparison, column, "Attribution E2E raw vectors")
    for side in ("historical", "current"):
        for name in SUMMARY_NAMES:
            _finite_numeric(
                comparison, f"{side}_{name}", "Attribution E2E DECAF vectors"
            )
    index = _finite_numeric(comparison, "vector_index", "Attribution E2E raw vectors")
    if not np.equal(index, np.floor(index)).all() or comparison.duplicated(vector_keys).any():
        raise FinalizationError("Attribution E2E vector identity/index inventory changed")

    neutral_path = Path(str(attribution_chain.get("aggregate_output_path", ""))).resolve()
    if (
        neutral_path != (root / "trajectories/attribution.parquet").resolve()
        or sha256_file(neutral_path) != attribution_chain.get("aggregate_output_sha256")
    ):
        raise FinalizationError("Attribution neutral trajectory chain changed")
    try:
        neutral = read_trajectory_record(neutral_path)
    except Exception as error:
        raise FinalizationError(f"Attribution neutral trajectory is invalid: {error}") from error
    neutral_unit_keys = [
        "unit_id",
        "sample_or_pair_id",
        "factor_or_part_id",
        "protocol",
        "endpoint_d",
        "historical_M",
        "historical_E",
        "historical_C",
        "historical_F",
        "historical_Abs",
        "metadata_json",
    ]
    neutral_units = neutral.loc[neutral["stage_index"].astype(int).eq(0), neutral_unit_keys]
    if len(neutral_units) != 1476 or neutral_units["unit_id"].astype(str).duplicated().any():
        raise FinalizationError("Attribution neutral trajectory unit inventory changed")
    historical_bound = comparison.merge(
        neutral_units,
        on="unit_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_neutral"),
    )
    if len(historical_bound) != 1476:
        raise FinalizationError("Attribution comparison does not bind all neutral units")
    for left, right in (
        ("image_id", "sample_or_pair_id"),
        ("factor_or_part_id", "factor_or_part_id_neutral"),
        ("method", "protocol"),
    ):
        if not np.array_equal(
            historical_bound[left].astype(str).to_numpy(dtype=str),
            historical_bound[right].astype(str).to_numpy(dtype=str),
        ):
            raise FinalizationError(f"Attribution neutral identity differs: {left}")
    for left, right in (
        ("historical_score", "historical_E_neutral"),
        *((f"historical_{name}", f"historical_{name}_neutral") for name in SUMMARY_NAMES),
    ):
        if not np.allclose(
            _finite_numeric(historical_bound, left, "Attribution comparison historical"),
            _finite_numeric(historical_bound, right, "Attribution neutral trajectory"),
            atol=1.0e-14,
            rtol=1.0e-12,
        ):
            raise FinalizationError(
                f"Attribution comparison {left} differs from SHA-bound neutral evidence"
            )
    metadata_historical_endpoint: list[float] = []
    metadata_regenerated_endpoint: list[float] = []
    for raw in historical_bound["metadata_json"]:
        try:
            payload = json.loads(str(raw))
            metadata_historical_endpoint.append(float(payload["historical_endpoint_d"]))
            metadata_regenerated_endpoint.append(float(payload["regenerated_endpoint_d"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise FinalizationError(
                f"Attribution neutral endpoint metadata is invalid: {error}"
            ) from error
    if not np.allclose(
        _finite_numeric(
            historical_bound, "historical_endpoint_d", "Attribution comparison historical"
        ),
        np.asarray(metadata_historical_endpoint, dtype=np.float64),
        atol=1.0e-14,
        rtol=1.0e-12,
    ):
        raise FinalizationError("Attribution historical endpoint differs from neutral metadata")
    if not np.allclose(
        _finite_numeric(historical_bound, "endpoint_d", "Attribution neutral trajectory"),
        np.asarray(metadata_regenerated_endpoint, dtype=np.float64),
        atol=1.0e-14,
        rtol=1.0e-12,
    ):
        raise FinalizationError("Attribution regenerated endpoint differs from neutral metadata")

    current_outputs = attribution_chain.get("current_outputs")
    if not isinstance(current_outputs, Mapping) or set(current_outputs) != set(
        ATTRIBUTION_BRIDGES
    ):
        raise FinalizationError("Attribution current-output chain is incomplete")
    computed_current: dict[tuple[str, ...], float] = {}
    current_bindings: list[dict[str, Any]] = []
    for bridge, (dataset, model) in ATTRIBUTION_BRIDGES.items():
        binding = current_outputs[bridge]
        if not isinstance(binding, Mapping):
            raise FinalizationError(f"Attribution current binding {bridge} is invalid")
        output = Path(str(binding.get("trajectory_path", ""))).resolve()
        receipt_path = Path(str(binding.get("provenance_receipt_path", ""))).resolve()
        if (
            output != (root / f"trajectories/attribution_current__{bridge}.parquet").resolve()
            or receipt_path
            != (root / f"provenance/attribution_current__{bridge}.json").resolve()
            or sha256_file(output) != binding.get("trajectory_sha256")
            or sha256_file(receipt_path) != binding.get("provenance_receipt_sha256")
        ):
            raise FinalizationError(f"Attribution current output/receipt binding changed: {bridge}")
        receipt = _read_json(receipt_path, f"Attribution current receipt {bridge}")
        selected = comparison.loc[
            comparison["dataset"].astype(str).eq(dataset)
            & comparison["model"].astype(str).eq(model)
        ].copy()
        image_ids = sorted(selected["image_id"].astype(str).unique())
        manifest_value = receipt.get("fixed_sample_manifest")
        runtime_value = receipt.get("runtime_run")
        if not isinstance(manifest_value, str) or not Path(manifest_value).is_absolute():
            raise FinalizationError(f"Attribution current receipt lacks fixed manifest: {bridge}")
        if not isinstance(runtime_value, str) or not Path(runtime_value).is_absolute():
            raise FinalizationError(f"Attribution current receipt lacks runtime root: {bridge}")
        manifest_path = _require_file(
            Path(manifest_value), f"Attribution fixed sample manifest {bridge}"
        ).resolve()
        runtime_root = Path(runtime_value).resolve()
        if (
            not runtime_root.is_dir()
            or not manifest_path.is_relative_to(runtime_root / "manifests/fixed_samples")
            or sha256_file(manifest_path) != receipt.get("fixed_sample_manifest_sha256")
            or receipt.get("executor")
            != "decaf.experiments.attribution.gpu_runtime.evaluate_member"
            or receipt.get("repository_commit") != DINO_REPOSITORY_COMMIT
            or receipt.get("methods") != ["decaf_3", "decaf_5", "decaf_9"]
            or sorted(map(str, receipt.get("image_ids", []))) != image_ids
            or receipt.get("rows") != 24
            or receipt.get("sample_count") != 8
            or receipt.get("runtime_cuda_matmul_allow_tf32") is not False
            or receipt.get("runtime_cudnn_allow_tf32") is not False
        ):
            raise FinalizationError(f"Attribution current receipt contract changed: {bridge}")
        fixed = _read_json(manifest_path, f"Attribution fixed sample manifest {bridge}")
        if (
            fixed.get("schema_version") != 1
            or fixed.get("dataset") != dataset
            or fixed.get("model_id") != model
            or sorted(map(str, fixed.get("image_ids", []))) != image_ids
            or fixed.get("targets") != receipt.get("targets")
        ):
            raise FinalizationError(f"Attribution fixed sample identity changed: {bridge}")
        member = _read_parquet(output, f"Attribution current raw member {bridge}")
        member_keys = ["dataset", "model", "method", "image_id"]
        required_member = {
            *member_keys,
            "patch_scores",
            "endpoint_effects",
            "part_names",
            "spearman",
            "finite_complete",
            "numeric_audit_passed",
            *(f"decaf_{name}" for name in SUMMARY_NAMES),
        }
        funnybirds = dataset == "funnybirds"
        if funnybirds:
            required_member |= {
                "heldout_background_texture_effects",
                "heldout_telea_dilate3_effects",
                "spearman_background_texture",
                "spearman_telea_dilate3",
                "quality_aggregation",
            }
        if (
            len(member) != 24
            or not required_member.issubset(member.columns)
            or member.duplicated(member_keys).any()
            or set(member["dataset"].astype(str)) != {dataset}
            or set(member["model"].astype(str)) != {model}
            or not _bool_series(member, "finite_complete", "Attribution current member").all()
            or not _bool_series(
                member, "numeric_audit_passed", "Attribution current member"
            ).all()
        ):
            raise FinalizationError(f"Attribution current raw-member inventory changed: {bridge}")
        for row in member.itertuples(index=False):
            identity = tuple(str(getattr(row, column)) for column in member_keys)
            group = selected.loc[
                selected["method"].astype(str).eq(identity[2])
                & selected["image_id"].astype(str).eq(identity[3])
            ].sort_values("vector_index", kind="stable")
            expected_index = np.arange(len(group), dtype=np.int64)
            if not np.array_equal(
                _finite_numeric(group, "vector_index", "Attribution E2E vector group").astype(
                    np.int64
                ),
                expected_index,
            ):
                raise FinalizationError("Attribution vector indexes are not contiguous from zero")
            patch = _attribution_vector(row.patch_scores, name="patch_scores")
            endpoint = _attribution_vector(row.endpoint_effects, name="endpoint_effects")
            part_names = np.asarray(row.part_names).astype(str)
            if (
                len(group) != len(patch)
                or endpoint.shape != patch.shape
                or not np.array_equal(
                    group["factor_or_part_id"].astype(str).to_numpy(dtype=str), part_names
                )
                or not np.allclose(
                    _finite_numeric(group, "current_score", "Attribution E2E vectors"),
                    patch,
                    atol=1.0e-14,
                    rtol=1.0e-12,
                )
                or not np.allclose(
                    _finite_numeric(group, "current_endpoint_d", "Attribution E2E vectors"),
                    endpoint,
                    atol=1.0e-14,
                    rtol=1.0e-12,
                )
            ):
                raise FinalizationError("Attribution current raw vectors differ from comparison")
            for name in SUMMARY_NAMES:
                raw_decaf = _attribution_vector(
                    getattr(row, f"decaf_{name}"), name=f"decaf_{name}"
                )
                if raw_decaf.shape != patch.shape:
                    raise FinalizationError(f"Attribution raw decaf_{name} shape changed")
                _require_attribution_vector_binding(
                    _finite_numeric(
                        group, f"current_{name}", "Attribution E2E current DECAF vectors"
                    ),
                    raw_decaf,
                    label=f"current_{name}",
                )
            computed_current[identity] = _current_member_spearman(
                row, funnybirds=funnybirds
            )
        current_bindings.append(
            {
                "bridge": bridge,
                "output_path": str(output),
                "output_sha256": binding["trajectory_sha256"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": binding["provenance_receipt_sha256"],
                "fixed_sample_manifest_path": str(manifest_path),
                "fixed_sample_manifest_sha256": receipt["fixed_sample_manifest_sha256"],
                "runtime_run": str(runtime_root),
                "raw_rows": 24,
            }
        )
    if len(computed_current) != 144:
        raise FinalizationError("Attribution current raw outputs do not yield 144 vectors")

    # FunnyBirds historical quality is recomputed from the A0 sealed formal two-operator
    # rows.  The A2 reused-quality member is the aggregate's direct source and must be an
    # exact value-level cross-check, but is not represented as an independent raw recompute.
    receipt_path, reference_receipt = _attribution_reference_receipt(root)
    archives = reference_receipt.get("archives")
    inputs = reference_receipt.get("inputs")
    a0_archives = [
        record
        for record in archives if isinstance(record, Mapping) and record.get("run_id") == "A0"
    ] if isinstance(archives, list) else []
    a0_inputs = [
        record
        for record in inputs
        if isinstance(record, Mapping)
        and record.get("run_id") == "A0"
        and record.get("resolved_member") == ATTRIBUTION_A0_HELDOUT_QUALITY_MEMBER
    ] if isinstance(inputs, list) else []
    if len(a0_archives) != 1 or len(a0_inputs) != 1:
        raise FinalizationError("A0 sealed held-out quality receipt binding is incomplete")
    a0_archive = a0_archives[0]
    a0_input = a0_inputs[0]
    a0_package = Path(str(a0_archive.get("resolved_path", ""))).resolve()
    if (
        not a0_package.is_file()
        or a0_archive.get("sha256") != ATTRIBUTION_A0_PACKAGE_SHA256
        or sha256_file(a0_package) != ATTRIBUTION_A0_PACKAGE_SHA256
        or a0_input.get("sha256") != ATTRIBUTION_A0_HELDOUT_QUALITY_SHA256
        or a0_input.get("relative_path")
        != f"A0/{ATTRIBUTION_A0_HELDOUT_QUALITY_MEMBER}"
    ):
        raise FinalizationError("A0 sealed held-out quality receipt/package changed")
    try:
        with zipfile.ZipFile(a0_package) as archive:
            a0_bytes = archive.read(ATTRIBUTION_A0_HELDOUT_QUALITY_MEMBER)
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise FinalizationError(f"A0 sealed held-out quality is unreadable: {error}") from error
    if hashlib.sha256(a0_bytes).hexdigest() != ATTRIBUTION_A0_HELDOUT_QUALITY_SHA256:
        raise FinalizationError("A0 sealed held-out quality member SHA-256 changed")
    try:
        a0_quality = pd.read_parquet(io.BytesIO(a0_bytes))
    except Exception as error:
        raise FinalizationError(
            f"A0 sealed held-out quality parquet is invalid: {error}"
        ) from error

    funny_keys = {
        tuple(values)
        for values in comparison.loc[
            comparison["dataset"].astype(str).eq("funnybirds"), keys
        ].astype(str).drop_duplicates().itertuples(index=False, name=None)
    }
    a0_selected = a0_quality.loc[
        a0_quality["dataset"].astype(str).eq("funnybirds")
        & a0_quality["method"].astype(str).isin({"decaf_3", "decaf_5", "decaf_9"})
        & a0_quality["track"].astype(str).eq("main")
        & a0_quality["reference"].astype(str).eq("gaussian_blur_k31_sigma12")
        & a0_quality["metric"].astype(str).eq("spearman")
    ].copy()
    a0_selected = a0_selected.loc[
        [tuple(values) in funny_keys for values in a0_selected[keys].astype(str).itertuples(
            index=False, name=None
        )]
    ]
    a0_operator_keys = [*keys, "operator"]
    if (
        len(a0_selected) != 144
        or a0_selected.duplicated(a0_operator_keys).any()
        or set(a0_selected["operator"].astype(str))
        != {"background_texture", "telea_dilate3"}
        or not np.isfinite(
            pd.to_numeric(a0_selected["value"], errors="coerce").to_numpy(dtype=float)
        ).all()
    ):
        raise FinalizationError("A0 historical FunnyBirds two-operator inventory changed")
    historical_funny = (
        a0_selected.groupby(keys, sort=True)["value"].mean().astype(float).to_dict()
    )
    if set(historical_funny) != funny_keys:
        raise FinalizationError("A0 historical FunnyBirds identities are incomplete")

    historical_sources = attribution_chain.get("historical_source_bindings")
    a2_binding = (
        historical_sources.get("a2_imagenet1k_idsds")
        if isinstance(historical_sources, Mapping)
        else None
    )
    gate_path = root / "manifests/attribution_funnybirds_a2_gate.json"
    gate = _read_json(gate_path, "Attribution FunnyBirds A2 reused-quality gate")
    if not isinstance(a2_binding, Mapping):
        raise FinalizationError("A2 sealed package binding is unavailable")
    a2_package = Path(str(a2_binding.get("package_path", ""))).resolve()
    if (
        a2_binding.get("package_sha256") != ATTRIBUTION_A2_PACKAGE_SHA256
        or sha256_file(a2_package) != ATTRIBUTION_A2_PACKAGE_SHA256
        or gate.get("reused_quality_sha256") != ATTRIBUTION_A2_FUNNY_QUALITY_SHA256
        or gate.get("rows") != 72
        or gate.get("methods") != ["decaf_3", "decaf_5", "decaf_9"]
    ):
        raise FinalizationError("A2 FunnyBirds reused-quality binding changed")
    try:
        with zipfile.ZipFile(a2_package) as archive:
            a2_bytes = archive.read(ATTRIBUTION_A2_FUNNY_QUALITY_MEMBER)
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise FinalizationError(f"A2 reused FunnyBirds quality is unreadable: {error}") from error
    external_a2 = Path(str(gate.get("reused_quality", ""))).resolve()
    if (
        hashlib.sha256(a2_bytes).hexdigest() != ATTRIBUTION_A2_FUNNY_QUALITY_SHA256
        or not external_a2.is_file()
        or sha256_file(external_a2) != ATTRIBUTION_A2_FUNNY_QUALITY_SHA256
        or external_a2.read_bytes() != a2_bytes
    ):
        raise FinalizationError("A2 reused-quality extracted/package bytes differ")
    a2_quality = pd.read_parquet(io.BytesIO(a2_bytes))
    a2_selected = a2_quality.loc[
        [tuple(values) in funny_keys for values in a2_quality[keys].astype(str).itertuples(
            index=False, name=None
        )]
    ].copy()
    if (
        len(a2_selected) != 72
        or a2_selected.duplicated(keys).any()
        or not _bool_series(a2_selected, "finite_complete", "A2 reused quality").all()
        or not _bool_series(a2_selected, "correctly_classified", "A2 reused quality").all()
        or set(a2_selected["source"].astype(str)) != {"reused_locked_formal_result"}
    ):
        raise FinalizationError("A2 reused FunnyBirds selected inventory changed")
    a2_values = a2_selected.set_index(keys)["spearman"].astype(float).to_dict()
    if set(a2_values) != set(historical_funny) or any(
        not np.isclose(a2_values[key], value, atol=1.0e-15, rtol=0.0)
        for key, value in historical_funny.items()
    ):
        raise FinalizationError("A2 reused FunnyBirds values differ from A0 formal authority")

    rows: list[dict[str, Any]] = []
    for identity, group in comparison.groupby(keys, sort=True):
        identity = tuple(map(str, identity))
        group = group.sort_values("vector_index", kind="stable")
        current = computed_current[identity]
        if identity[0] == "funnybirds":
            historical = historical_funny[identity]
        else:
            historical = float(
                row_spearman(
                    _finite_numeric(group, "historical_score", "Attribution raw vectors"),
                    _finite_numeric(
                        group, "historical_endpoint_d", "Attribution raw vectors"
                    ),
                )[0]
            )
            current_from_comparison = float(
                row_spearman(
                    _finite_numeric(group, "current_score", "Attribution raw vectors"),
                    _finite_numeric(group, "current_endpoint_d", "Attribution raw vectors"),
                )[0]
            )
            if not np.isclose(current, current_from_comparison, atol=1.0e-12, rtol=0.0):
                raise FinalizationError("IDSDS current output/comparison Spearman differs")
        for column, expected in (
            ("historical_spearman", historical),
            ("current_spearman", current),
        ):
            recorded = _finite_numeric(group, column, "Attribution derived Spearman")
            if not np.allclose(recorded, expected, atol=1.0e-12, rtol=0.0):
                raise FinalizationError(
                    f"Attribution {column} differs from authoritative reconstruction"
                )
        rows.append(dict(zip(keys, identity, strict=True)) | {
            "historical_spearman": historical,
            "current_spearman": current,
        })
    reconstructed = pd.DataFrame(rows)
    if len(reconstructed) != 144 or reconstructed.duplicated(keys).any():
        raise FinalizationError("Attribution raw reconstruction did not yield 144 vectors")
    return reconstructed, {
        "status": "RAW_RECOMPUTED_WITH_SEALED_HISTORICAL_AUTHORITY",
        "current": {
            "status": "SIX_RECEIPT_SHA256_BOUND_RAW_VECTOR_OUTPUTS",
            "outputs": current_bindings,
            "funnybirds_formula": (
                "mean(row_spearman(patch_scores, background_texture), "
                "row_spearman(patch_scores, telea_dilate3))"
            ),
            "idsds_formula": "row_spearman(patch_scores, endpoint_effects)",
        },
        "historical_funnybirds": {
            "status": "A0_SEALED_FORMAL_TWO_OPERATOR_MEAN_RECOMPUTED",
            "reference_receipt_path": str(receipt_path),
            "reference_receipt_sha256": sha256_file(receipt_path),
            "package_path": str(a0_package),
            "package_sha256": ATTRIBUTION_A0_PACKAGE_SHA256,
            "member": ATTRIBUTION_A0_HELDOUT_QUALITY_MEMBER,
            "member_sha256": ATTRIBUTION_A0_HELDOUT_QUALITY_SHA256,
            "selected_operator_rows": 144,
            "reconstructed_vectors": 72,
            "a2_reused_quality_crosscheck": {
                "gate_path": str(gate_path.resolve()),
                "gate_sha256": sha256_file(gate_path),
                "package_path": str(a2_package),
                "package_sha256": ATTRIBUTION_A2_PACKAGE_SHA256,
                "member": ATTRIBUTION_A2_FUNNY_QUALITY_MEMBER,
                "member_sha256": ATTRIBUTION_A2_FUNNY_QUALITY_SHA256,
                "selected_rows": 72,
                "value_exact": True,
                "role": "aggregate_direct_source_crosscheck_not_raw_recompute",
            },
            "executor_source_authority": "A0 deployed reference-locked tree",
        },
        "historical_idsds": {
            "status": "RAW_SCORE_ENDPOINT_VECTORS_RECOMPUTED",
            "reconstructed_vectors": 72,
            "neutral_trajectory_path": str(neutral_path),
            "neutral_trajectory_sha256": attribution_chain["aggregate_output_sha256"],
            "neutral_unit_rows_bound": 1476,
        },
    }


def _validate_attribution_spearman(
    root: Path,
    attribution_e2e: pd.DataFrame,
    summary: Mapping[str, Any],
    attribution_chain: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate 144 per-image comparisons from raw or sealed-authority evidence."""

    label = "Attribution Spearman CSV"
    path = root / "comparisons/attribution_spearman.csv"
    digest = _validate_artifact_binding(
        summary,
        path,
        path_key="spearman_comparison",
        sha_key="spearman_comparison_sha256",
        label="Attribution Spearman comparison",
    )
    frame = _read_csv(path, label)
    if set(frame.columns) != ATTRIBUTION_SPEARMAN_COLUMNS:
        raise FinalizationError("Attribution Spearman CSV columns changed")
    keys = list(ATTRIBUTION_SPEARMAN_IDENTITY_COLUMNS)
    if len(frame) != 144 or frame.duplicated(keys).any():
        raise FinalizationError("Attribution Spearman CSV must contain 144 unique image keys")
    source, reconstruction = _reconstruct_attribution_spearman(
        root, attribution_e2e, attribution_chain
    )
    if len(source) != 144 or source.duplicated(keys).any():
        raise FinalizationError("Attribution E2E does not yield exactly 144 feature vectors")
    source = source.sort_values(keys, kind="stable").reset_index(drop=True)
    observed = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    for column in keys:
        if not np.array_equal(
            observed[column].astype(str).to_numpy(dtype=str),
            source[column].astype(str).to_numpy(dtype=str),
        ):
            raise FinalizationError(
                f"Attribution Spearman identity column differs from E2E rows: {column}"
            )
    historical = _finite_numeric(source, "historical_spearman", label)
    current = _finite_numeric(source, "current_spearman", label)
    _require_recorded_numeric(observed, "historical_spearman", historical, label=label)
    _require_recorded_numeric(observed, "current_spearman", current, label=label)
    signed = current - historical
    absolute = np.abs(signed)
    tier_a = np.isclose(
        current,
        historical,
        atol=core_comparison.TIER_A_ATOL,
        rtol=core_comparison.TIER_A_RTOL,
    )
    tier_b = absolute <= core_comparison.TIER_B_ABS
    tier = np.where(tier_a, "A", np.where(tier_b, "B", "FAIL"))
    hard = absolute > core_comparison.HARD_MISMATCH_ABS
    _require_recorded_numeric(observed, "signed_error", signed, label=label)
    _require_recorded_numeric(observed, "absolute_error", absolute, label=label)
    _require_recorded_bool(observed, "tier_a_pass", tier_a, label=label)
    _require_recorded_bool(observed, "tier_b_pass", tier_b, label=label)
    _require_recorded_text(observed, "tier", tier, label=label)
    _require_recorded_bool(observed, "hard_mismatch", hard, label=label)
    tier_a_fraction = float(tier_a.mean())
    tier_b_fraction = float((~tier_a & tier_b).mean())
    tier_a_or_b_fraction = float((tier_a | tier_b).mean())
    hard_fraction = float(hard.mean())
    result = {
        **_error_distribution(absolute, agreement=tier_a_or_b_fraction),
        "unit_count": 144,
        "tier_a_count": int(tier_a.sum()),
        "tier_b_count": int((~tier_a & tier_b).sum()),
        "fail_count": int((~tier_a & ~tier_b).sum()),
        "tier_a_fraction": tier_a_fraction,
        "tier_b_fraction": tier_b_fraction,
        "tier_a_or_b_fraction": tier_a_or_b_fraction,
        "hard_mismatch_count": int(hard.sum()),
        "hard_mismatch_fraction": hard_fraction,
        "mean_signed_error": float(signed.mean()),
        "evidence_layer": "current_e2e_feature_vectors",
        "identity_columns": keys,
        "reconstruction": reconstruction,
        "artifact_binding": {
            "path": str(path.resolve()),
            "sha256": digest,
            "row_count": 144,
            "validation": (
                "exact_columns_unique_image_identity_values_errors_tiers_and_hard_"
                "mismatches_recomputed_against_1476_vector_elements"
            ),
        },
    }
    recorded = summary.get("spearman")
    expected_summary = {
        "median_absolute_error": result["median_absolute_error"],
        "p95_absolute_error": result["p95_absolute_error"],
        "maximum_absolute_error": result["maximum_absolute_error"],
        "mean_signed_error": result["mean_signed_error"],
        "tier_a_fraction": tier_a_fraction,
        "tier_b_fraction": tier_b_fraction,
        "tier_a_or_b_fraction": tier_a_or_b_fraction,
        "hard_mismatch_fraction": hard_fraction,
    }
    if not isinstance(recorded, Mapping) or summary.get("image_method_count") != 144:
        raise FinalizationError("Attribution summary lacks the 144-vector Spearman summary")
    for key, expected in expected_summary.items():
        value = recorded.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isclose(
            float(value), float(expected), atol=1.0e-12, rtol=0.0
        ):
            raise FinalizationError(
                f"Attribution summary.spearman.{key} differs from its 144-row CSV"
            )
    return result


def validate_covertype_historical_source_binding(root: Path) -> dict[str, Any]:
    """Independently validate the sealed and materialized Covertype source snapshot."""

    selection_path = root / "manifests/covertype_selection.json"
    selection = _read_json(selection_path, "Covertype selection manifest")
    binding = selection.get("historical_source_binding")
    if (
        selection.get("schema_version") != 1
        or selection.get("family") != "covertype"
        or selection.get("historical_repository_head_role") != "context_only_untracked"
        or not isinstance(binding, dict)
    ):
        raise FinalizationError("Covertype selection/source binding contract changed")

    digest_record = dict(selection)
    recorded_selection_sha256 = digest_record.pop("selection_sha256", None)
    selection_sha256 = hashlib.sha256(
        json.dumps(
            digest_record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        selection.get("selection_key") != "sha256(namespace|sealed_test_source_index)"
        or selection.get("selection_uses_model_outputs") is not False
        or recorded_selection_sha256 != selection_sha256
    ):
        raise FinalizationError("Covertype fixed selection digest/contract changed")

    package_value = binding.get("path")
    if not isinstance(package_value, str) or not Path(package_value).is_absolute():
        raise FinalizationError("Covertype sealed package path is not absolute")
    package = _require_file(Path(package_value), "Covertype sealed historical package")
    package_sha256 = sha256_file(package)
    if (
        binding.get("authority_kind") != "sha256_verified_lightweight_zip"
        or binding.get("git_head_role") != "context_only_untracked"
        or binding.get("sha256") != COVERTYPE_HISTORICAL_PACKAGE_SHA256
        or package_sha256 != COVERTYPE_HISTORICAL_PACKAGE_SHA256
        or binding.get("manifest_member") != COVERTYPE_MANIFEST_MEMBER
        or binding.get("manifest_sha256") != COVERTYPE_HISTORICAL_MANIFEST_SHA256
        or binding.get("archive_source_prefix") != COVERTYPE_ARCHIVE_SOURCE_PREFIX
        or binding.get("archive_inventory_verified") is not True
    ):
        raise FinalizationError("Covertype sealed-package authority changed")

    try:
        with zipfile.ZipFile(package) as archive:
            archive_files = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(archive_files) != 111 or len(archive_files) != len(set(archive_files)):
                raise FinalizationError("Covertype sealed ZIP must contain 111 unique files")
            try:
                manifest_bytes = archive.read(COVERTYPE_MANIFEST_MEMBER)
                manifest = json.loads(manifest_bytes)
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FinalizationError("Covertype sealed package manifest is invalid") from error
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            files = manifest.get("files") if isinstance(manifest, dict) else None
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != 1
                or manifest.get("namespace") != "decaf_covertype_v1"
                or manifest.get("lightweight") is not True
                or manifest_sha256 != COVERTYPE_HISTORICAL_MANIFEST_SHA256
                or not isinstance(files, list)
                or len(files) != 110
            ):
                raise FinalizationError("Covertype sealed package manifest contract changed")

            manifest_members: dict[str, dict[str, Any]] = {}
            namespace_members: dict[str, dict[str, Any]] = {}
            expected_archive_files = {COVERTYPE_MANIFEST_MEMBER}
            for index, record in enumerate(files):
                if not isinstance(record, dict):
                    raise FinalizationError(f"Covertype package member[{index}] is invalid")
                member = record.get("path")
                expected_bytes = record.get("bytes")
                expected_sha256 = record.get("sha256")
                member_path = PurePosixPath(member) if isinstance(member, str) else None
                if (
                    not isinstance(member, str)
                    or not member
                    or member.startswith("/")
                    or member_path is None
                    or ".." in member_path.parts
                    or member in manifest_members
                    or isinstance(expected_bytes, bool)
                    or not isinstance(expected_bytes, int)
                    or expected_bytes < 0
                    or not isinstance(expected_sha256, str)
                    or len(expected_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in expected_sha256)
                ):
                    raise FinalizationError(f"Covertype package member[{index}] is malformed")
                try:
                    payload = archive.read(member)
                except KeyError as error:
                    raise FinalizationError(
                        f"Covertype sealed package member is missing: {member}"
                    ) from error
                if (
                    len(payload) != expected_bytes
                    or hashlib.sha256(payload).hexdigest() != expected_sha256
                ):
                    raise FinalizationError(
                        f"Covertype sealed package member identity changed: {member}"
                    )
                compact = {
                    "archive_member": member,
                    "bytes": expected_bytes,
                    "sha256": expected_sha256,
                }
                manifest_members[member] = compact
                expected_archive_files.add(member)
                if member.startswith(COVERTYPE_NAMESPACE_PREFIX) and member.endswith(".py"):
                    namespace_members[member] = compact
            if set(archive_files) != expected_archive_files:
                raise FinalizationError("Covertype sealed ZIP inventory differs from its manifest")
    except (OSError, zipfile.BadZipFile) as error:
        raise FinalizationError(f"Covertype sealed package is unreadable: {error}") from error

    recorded_namespace = binding.get("namespace_members")
    if (
        len(namespace_members) != 26
        or binding.get("namespace_member_count") != 26
        or recorded_namespace != namespace_members
        or set(binding.get("required_modules", ())) != COVERTYPE_REQUIRED_MODULES
    ):
        raise FinalizationError("Covertype sealed namespace/member binding changed")
    required_archive_members = {
        f"{COVERTYPE_NAMESPACE_PREFIX}{name}.py" for name in COVERTYPE_REQUIRED_MODULES
    }
    if not required_archive_members.issubset(namespace_members):
        raise FinalizationError("Covertype sealed namespace lacks required runtime modules")

    import_root_value = binding.get("import_root")
    namespace_value = binding.get("materialized_namespace")
    expected_import_root = (root / "provenance/historical_sources/covertype").resolve()
    if (
        not isinstance(import_root_value, str)
        or not Path(import_root_value).is_absolute()
        or Path(import_root_value).resolve() != expected_import_root
        or not isinstance(namespace_value, str)
        or not Path(namespace_value).is_absolute()
    ):
        raise FinalizationError("Covertype materialized import root is stale or mixed")
    import_root = Path(import_root_value).resolve()
    namespace = Path(namespace_value).resolve()
    if namespace != import_root / "cmr/decaf_covertype_v1" or not namespace.is_dir():
        raise FinalizationError("Covertype materialized namespace path changed")
    observed_paths = list(import_root.rglob("*"))
    if any(path.is_symlink() for path in (import_root, namespace, *observed_paths)):
        raise FinalizationError("Covertype materialized source contains a symlink")

    expected_materialized: dict[Path, dict[str, Any]] = {}
    for archive_member, record in namespace_members.items():
        relative = PurePosixPath(archive_member).relative_to(COVERTYPE_ARCHIVE_SOURCE_PREFIX)
        output = import_root.joinpath(*relative.parts).resolve()
        expected_materialized[output] = record
        _require_file(output, f"materialized Covertype source {archive_member}")
        if output.stat().st_size != record["bytes"] or sha256_file(output) != record["sha256"]:
            raise FinalizationError(
                f"materialized Covertype source differs from sealed bytes: {archive_member}"
            )

    shim = binding.get("parent_package_shim")
    shim_path = import_root / "cmr/__init__.py"
    if (
        not isinstance(shim, dict)
        or shim.get("path") != str(shim_path)
        or shim.get("bytes") != 74
        or shim.get("sha256") != COVERTYPE_PARENT_SHIM_SHA256
        or shim.get("role") != "verification_only_import_isolation"
        or shim.get("historical_source") is not False
        or binding.get("parent_package_origin") != str(shim_path)
    ):
        raise FinalizationError("Covertype verification-only parent-package shim changed")
    _require_file(shim_path, "Covertype parent-package isolation shim")
    if shim_path.stat().st_size != 74 or sha256_file(shim_path) != COVERTYPE_PARENT_SHIM_SHA256:
        raise FinalizationError("Covertype parent-package shim bytes changed")
    actual_materialized = {path.resolve() for path in observed_paths if path.is_file()}
    if (
        binding.get("materialized_member_count") != 26
        or actual_materialized != set(expected_materialized) | {shim_path.resolve()}
    ):
        raise FinalizationError("Covertype materialized source inventory changed")

    loaded_origins = binding.get("loaded_module_origins")
    expected_origins: dict[str, str] = {}
    for name in COVERTYPE_LOADED_MODULES:
        module = (
            "cmr.decaf_covertype_v1"
            if name == "__init__"
            else f"cmr.decaf_covertype_v1.{name}"
        )
        filename = "__init__.py" if name == "__init__" else f"{name}.py"
        expected_origins[module] = str(namespace / filename)
    if (
        binding.get("origin_verified") is not True
        or not isinstance(loaded_origins, dict)
        or loaded_origins != dict(sorted(expected_origins.items()))
    ):
        raise FinalizationError("Covertype loaded-module origins are not the exact snapshot")

    return {
        "status": "SHA256_VERIFIED_SEALED_ZIP_AND_MATERIALIZED_ORIGINS",
        "selection_manifest_path": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_sha256": selection_sha256,
        "package_path": str(package.resolve()),
        "package_sha256": package_sha256,
        "manifest_member": COVERTYPE_MANIFEST_MEMBER,
        "manifest_sha256": manifest_sha256,
        "manifest_file_count": len(manifest_members),
        "archive_file_count": len(archive_files),
        "namespace_member_count": len(namespace_members),
        "materialized_member_count": len(expected_materialized),
        "parent_package_shim": dict(shim),
        "loaded_module_origins": dict(sorted(expected_origins.items())),
        "required_modules": sorted(COVERTYPE_REQUIRED_MODULES),
        "archive_inventory_verified": True,
        "materialized_inventory_verified": True,
        "origin_verified": True,
    }


def _materialize_provenance_snapshot(
    destination: Path,
    payload: bytes,
    *,
    expected_sha256: str,
    label: str,
    write_outputs: bool,
) -> dict[str, Any]:
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise FinalizationError(f"{label} source bytes differ from the fixed SHA-256")
    if destination.exists():
        _require_file(destination, f"existing {label} provenance snapshot")
        if (
            destination.stat().st_size != len(payload)
            or sha256_file(destination) != expected_sha256
        ):
            raise FinalizationError(f"existing {label} provenance snapshot is stale or mixed")
        if destination.read_bytes() != payload:
            raise FinalizationError(f"existing {label} provenance snapshot bytes changed")
    if write_outputs:
        _atomic_bytes(destination, payload)
        _require_file(destination, f"written {label} provenance snapshot")
        if (
            destination.stat().st_size != len(payload)
            or sha256_file(destination) != expected_sha256
            or destination.read_bytes() != payload
        ):
            raise FinalizationError(f"written {label} provenance snapshot failed verification")
    return {
        "path": str(destination.resolve()),
        "sha256": expected_sha256,
        "bytes": len(payload),
        "materialized": destination.is_file() and not destination.is_symlink(),
        "copy_semantics": "original_bytes_exact",
    }


def _snapshot_package_binding(root: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(snapshot["path"])).resolve()
    relative = path.relative_to(root.resolve())
    package_member = (
        PurePosixPath("decaf_cross_generation_equivalence_v2")
        / PurePosixPath(relative.as_posix())
    )
    materialized = bool(snapshot["materialized"])
    selected = False
    if materialized:
        selected_members = {
            source.resolve(): member for source, member in _package_members(root)
        }
        selected = selected_members.get(path) == package_member
        if not selected:
            raise FinalizationError(
                f"provenance snapshot is not selected under its recorded package member: {path}"
            )
    return {
        **snapshot,
        "package_member": package_member.as_posix(),
        "package_member_verified": selected,
    }


def _snapshot_bound_file(
    root: Path,
    record: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    expected_sha256: str,
    destination_key: str,
    label: str,
    write_outputs: bool,
) -> dict[str, Any]:
    source_value = record.get(path_key)
    if (
        not isinstance(source_value, str)
        or not Path(source_value).is_absolute()
        or record.get(sha_key) != expected_sha256
    ):
        raise FinalizationError(f"{label} source binding is stale or mixed")
    source = _require_file(Path(source_value), f"{label} source").resolve()
    if sha256_file(source) != expected_sha256:
        raise FinalizationError(f"{label} source SHA-256 changed before snapshotting")
    snapshot = _materialize_provenance_snapshot(
        root / HISTORICAL_SNAPSHOT_RELATIVES[destination_key],
        source.read_bytes(),
        expected_sha256=expected_sha256,
        label=label,
        write_outputs=write_outputs,
    )
    return _snapshot_package_binding(root, {**snapshot, "source_path": str(source)})


def _snapshot_zip_manifest(
    root: Path,
    record: Mapping[str, Any],
    *,
    expected_package_sha256: str,
    expected_manifest_member: str,
    expected_manifest_sha256: str,
    destination_key: str,
    label: str,
    write_outputs: bool,
) -> dict[str, Any]:
    package_value = record.get("package_path")
    if (
        not isinstance(package_value, str)
        or not Path(package_value).is_absolute()
        or record.get("package_sha256") != expected_package_sha256
        or record.get("manifest_member") != expected_manifest_member
        or record.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise FinalizationError(f"{label} manifest source binding is stale or mixed")
    package = _require_file(Path(package_value), f"{label} sealed package").resolve()
    if sha256_file(package) != expected_package_sha256:
        raise FinalizationError(f"{label} sealed package changed before snapshotting")
    try:
        with zipfile.ZipFile(package) as archive:
            matches = [
                info
                for info in archive.infolist()
                if not info.is_dir() and info.filename == expected_manifest_member
            ]
            if len(matches) != 1:
                raise FinalizationError(f"{label} manifest member is missing or repeated")
            payload = archive.read(matches[0])
    except (OSError, zipfile.BadZipFile) as error:
        raise FinalizationError(f"{label} sealed package is unreadable: {error}") from error
    snapshot = _materialize_provenance_snapshot(
        root / HISTORICAL_SNAPSHOT_RELATIVES[destination_key],
        payload,
        expected_sha256=expected_manifest_sha256,
        label=f"{label} raw manifest",
        write_outputs=write_outputs,
    )
    return _snapshot_package_binding(root, {
        **snapshot,
        "source_path": str(package),
        "source_archive_member": expected_manifest_member,
        "source_package_sha256": expected_package_sha256,
    })


def snapshot_historical_source_provenance(
    root: Path,
    attribution_sources: Mapping[str, Any],
    covertype_source: Mapping[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Snapshot small source-authority receipts/manifests without copying sealed ZIPs."""

    a0 = attribution_sources.get("a0_funnybirds")
    a2 = attribution_sources.get("a2_imagenet1k_idsds")
    if not isinstance(a0, Mapping) or not isinstance(a2, Mapping):
        raise FinalizationError("Attribution source authorities are incomplete for snapshotting")
    snapshots = {
        "status": "FIXED_SHA256_ORIGINAL_BYTES_PROVENANCE_SNAPSHOTS",
        "attribution_a0": {
            "deployment_receipt": _snapshot_bound_file(
                root,
                a0,
                path_key="deployment_receipt_path",
                sha_key="deployment_receipt_sha256",
                expected_sha256=ATTRIBUTION_A0_DEPLOYMENT_SHA256,
                destination_key="attribution_a0_deployment_receipt",
                label="Attribution A0 deployment receipt",
                write_outputs=write_outputs,
            ),
            "formal_plan": _snapshot_bound_file(
                root,
                a0,
                path_key="formal_plan_path",
                sha_key="formal_plan_sha256",
                expected_sha256=ATTRIBUTION_A0_PLAN_SHA256,
                destination_key="attribution_a0_formal_plan",
                label="Attribution A0 formal plan",
                write_outputs=write_outputs,
            ),
            "formal_plan_receipt": _snapshot_bound_file(
                root,
                a0,
                path_key="formal_plan_receipt_path",
                sha_key="formal_plan_receipt_sha256",
                expected_sha256=ATTRIBUTION_A0_PLAN_RECEIPT_SHA256,
                destination_key="attribution_a0_formal_plan_receipt",
                label="Attribution A0 formal-plan receipt",
                write_outputs=write_outputs,
            ),
        },
        "attribution_a2": {
            "package_manifest": _snapshot_zip_manifest(
                root,
                a2,
                expected_package_sha256=ATTRIBUTION_A2_PACKAGE_SHA256,
                expected_manifest_member=ATTRIBUTION_A2_MANIFEST_MEMBER,
                expected_manifest_sha256=ATTRIBUTION_A2_MANIFEST_SHA256,
                destination_key="attribution_a2_package_manifest",
                label="Attribution A2",
                write_outputs=write_outputs,
            )
        },
        "covertype": {
            "package_manifest": _snapshot_zip_manifest(
                root,
                covertype_source,
                expected_package_sha256=COVERTYPE_HISTORICAL_PACKAGE_SHA256,
                expected_manifest_member=COVERTYPE_MANIFEST_MEMBER,
                expected_manifest_sha256=COVERTYPE_HISTORICAL_MANIFEST_SHA256,
                destination_key="covertype_package_manifest",
                label="Covertype",
                write_outputs=write_outputs,
            )
        },
    }
    snapshots["all_materialized"] = all(
        bool(record["materialized"])
        for group in (
            snapshots["attribution_a0"],
            snapshots["attribution_a2"],
            snapshots["covertype"],
        )
        for record in group.values()
    )
    return snapshots


def _covertype_e2e(root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    comparison = _read_csv(root / "comparisons/covertype.csv", "Covertype E2E CSV")
    required = {
        "test_units",
        "tier_a_fraction",
        "tier_b_fraction",
        "tier_a_or_b_fraction",
        "hard_mismatch_fraction",
        "gate_agreement",
        "orientation_agreement",
        "dominant_agreement",
        "identity_exact",
        "summary_median_abs_error",
        "summary_p95_abs_error",
        "summary_max_abs_error",
    }
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise FinalizationError(f"Covertype E2E CSV lacks columns: {', '.join(missing)}")
    weights = _finite_numeric(comparison, "test_units", "Covertype E2E CSV")
    if np.any(weights <= 0) or not np.equal(weights, np.floor(weights)).all():
        raise FinalizationError("Covertype E2E test_units must be positive integers")
    units = int(weights.sum())

    def weighted(column: str) -> float:
        values = _finite_numeric(comparison, column, "Covertype E2E CSV")
        return float(np.average(values, weights=weights))

    identity = _bool_series(comparison, "identity_exact", "Covertype E2E CSV")
    if not identity.all() or summary.get("identity_exact") is not True:
        raise FinalizationError("Covertype E2E identities are not exact")
    exact_contract = {
        "training_performed": False,
        "model_fit_calls": 0,
        "historical_decomposition_called": False,
        "legal_support_only": True,
    }
    for key, expected in exact_contract.items():
        if summary.get(key) != expected:
            raise FinalizationError(f"Covertype exact-estimator contract failed: {key}")
    if int(summary.get("model_count", 0)) < 10:
        raise FinalizationError("Covertype exact-estimator bridge covers fewer than ten models")
    if int(summary.get("current_e2e_units", -1)) != units:
        raise FinalizationError("Covertype E2E summary unit count differs from CSV weights")
    result = {
        "unit_count": units,
        "tier_a_fraction": weighted("tier_a_fraction"),
        "tier_b_fraction": weighted("tier_b_fraction"),
        "tier_a_or_b_fraction": weighted("tier_a_or_b_fraction"),
        "hard_mismatch_fraction": weighted("hard_mismatch_fraction"),
        "gate_agreement": weighted("gate_agreement"),
        "orientation_agreement": weighted("orientation_agreement"),
        "dominant_mechanism_agreement": weighted("dominant_agreement"),
        "identity_agreement": 1.0,
        "median_absolute_error": weighted("summary_median_abs_error"),
        "p95_absolute_error": weighted("summary_p95_abs_error"),
        "maximum_absolute_error": float(
            _finite_numeric(comparison, "summary_max_abs_error", "Covertype E2E CSV").max()
        ),
        "aggregation": "exact_test_unit_weighted_model_statistics",
    }
    output = summary.get("outputs")
    expected_comparison = (root / "comparisons/covertype.csv").resolve()
    if (
        not isinstance(output, dict)
        or not isinstance(output.get("comparison"), str)
        or not Path(str(output["comparison"])).is_absolute()
        or Path(str(output["comparison"])).resolve() != expected_comparison
    ):
        raise FinalizationError("Covertype E2E summary points to a stale/mixed comparison")
    summary_keys = {
        "tier_a_fraction": "tier_a_fraction",
        "tier_b_fraction": "tier_b_fraction",
        "tier_a_or_b_fraction": "tier_a_or_b_fraction",
        "hard_mismatch_fraction": "hard_mismatch_fraction",
        "gate_agreement": "gate_agreement",
        "orientation_agreement": "orientation_agreement",
        "dominant_agreement": "dominant_mechanism_agreement",
        "median_absolute_error": "median_absolute_error",
        "p95_absolute_error": "p95_absolute_error",
        "maximum_absolute_error": "maximum_absolute_error",
    }
    for summary_key, result_key in summary_keys.items():
        if summary_key not in summary or not np.isclose(
            float(summary[summary_key]),
            float(result[result_key]),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise FinalizationError(
                f"Covertype E2E summary.{summary_key} differs from its comparison CSV"
            )
    result["artifact_binding"] = {
        "comparison_path": str(expected_comparison),
        "comparison_sha256": sha256_file(expected_comparison),
        "validation": "actual_sha256_plus_exact_unit_weighted_summary_recomputation",
    }
    return result


def _combine_e2e(families: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not families:
        raise FinalizationError("no E2E family statistics supplied")
    unit_count = sum(int(stats["unit_count"]) for stats in families.values())

    def weighted(key: str) -> float:
        return float(
            sum(int(stats["unit_count"]) * float(stats[key]) for stats in families.values())
            / unit_count
        )

    return {
        "unit_count": unit_count,
        "tier_a_fraction": weighted("tier_a_fraction"),
        "tier_b_fraction": weighted("tier_b_fraction"),
        "tier_a_or_b_fraction": weighted("tier_a_or_b_fraction"),
        "hard_mismatch_fraction": weighted("hard_mismatch_fraction"),
        "gate_agreement": weighted("gate_agreement"),
        "orientation_agreement": weighted("orientation_agreement"),
        "dominant_mechanism_agreement": weighted("dominant_mechanism_agreement"),
        "identity_agreement": weighted("identity_agreement"),
        "aggregation": "exact_unit_weighted_across_current_e2e_families",
    }


def _error_distribution(errors: np.ndarray, *, agreement: float) -> dict[str, Any]:
    if errors.size == 0 or not np.isfinite(errors).all():
        raise FinalizationError("variable error distribution is empty or non-finite")
    return {
        "median_absolute_error": float(np.median(errors)),
        "p95_absolute_error": float(np.percentile(errors, 95)),
        "maximum_absolute_error": float(np.max(errors)),
        "agreement": float(agreement),
    }


def _endpoint_d_statistics(frames: Sequence[pd.DataFrame]) -> dict[str, Any]:
    errors: list[np.ndarray] = []
    agreements: list[np.ndarray] = []
    pairs = (
        ("current_endpoint_d", "historical_endpoint_d"),
        ("current_d", "historical_d"),
        ("endpoint_d_current", "endpoint_d_historical"),
    )
    for index, frame in enumerate(frames):
        selected = next(
            ((left, right) for left, right in pairs if left in frame and right in frame), None
        )
        if selected is None:
            if "abs_error_d" not in frame:
                raise FinalizationError(f"E2E frame {index} lacks an endpoint-d comparison")
            error = _finite_numeric(frame, "abs_error_d", f"E2E frame {index}")
            errors.append(error)
            agreements.append(error <= 0.002)
            continue
        current = _finite_numeric(frame, selected[0], f"E2E frame {index}")
        historical = _finite_numeric(frame, selected[1], f"E2E frame {index}")
        errors.append(np.abs(current - historical))
        agreements.append(np.isclose(current, historical, atol=5.0e-4, rtol=5.0e-3))
    all_errors = np.concatenate(errors)
    all_agreements = np.concatenate(agreements)
    return _error_distribution(all_errors, agreement=float(all_agreements.mean()))


def _variable_statistics(
    core: Mapping[str, Any],
    imagenet9_e2e_frame: pd.DataFrame,
    attribution_e2e_frame: pd.DataFrame,
    attribution_spearman: Mapping[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "d": _endpoint_d_statistics([imagenet9_e2e_frame, attribution_e2e_frame])
    }
    values["d"]["evidence_layer"] = "current_e2e"
    for name in SUMMARY_NAMES:
        summary = dict(core["metric_summaries"][name])
        summary["evidence_layer"] = "current_core"
        values[name] = summary
    values.update(
        {
            "gate": {
                "agreement": float(core["gate_agreement"]),
                "evidence_layer": "current_core",
            },
            "orientation": {
                "agreement": float(core["orientation_agreement"]),
                "evidence_layer": "current_core",
            },
            "dominant mechanism": {
                "agreement": float(core["dominant_mechanism_agreement"]),
                "evidence_layer": "current_core",
            },
        }
    )
    values["feature-vector agreement"] = dict(attribution_spearman)
    return values


def _hard_mismatch_rows(frame: pd.DataFrame, *, family: str) -> list[dict[str, Any]]:
    hard = _bool_series(frame, "hard_mismatch", f"{family} hard-mismatch disclosure")
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(hard.to_numpy(dtype=bool)):
        row = frame.iloc[int(index)]
        errors: dict[str, float] = {}
        for column in frame.columns:
            if not column.startswith("abs_error_"):
                continue
            name = column.removeprefix("abs_error_")
            current_column = f"current_{name}"
            historical_column = f"historical_{name}"
            value = (
                abs(float(row[current_column]) - float(row[historical_column]))
                if current_column in frame and historical_column in frame
                else float(row[column])
            )
            if np.isfinite(value):
                errors[name] = value
        maximum = max(errors.values())
        maximum_variables = sorted(
            name
            for name, value in errors.items()
            if np.isclose(value, maximum, atol=1.0e-14, rtol=1.0e-12)
        )
        flags = {
            name: bool(_bool_series(frame.iloc[[int(index)]], name, family).iloc[0])
            for name in (
                "boundary",
                "identity_match",
                "gate_match",
                "orientation_match",
                "dominant_match",
            )
        }
        triggers: list[str] = []
        if maximum > core_comparison.HARD_MISMATCH_ABS:
            triggers.append("maximum_absolute_error_gt_0.01")
        if not flags["boundary"] and (not flags["gate_match"] or not flags["orientation_match"]):
            triggers.append("non_boundary_gate_or_orientation_mismatch")
        if not flags["dominant_match"]:
            triggers.append("dominant_mechanism_mismatch")
        if not flags["identity_match"]:
            triggers.append("identity_mismatch")
        if not triggers:
            raise FinalizationError(f"{family} hard mismatch has no independently defined trigger")
        unit_id = str(row.get("unit_id", ""))
        if not unit_id:
            unit_id = "::".join(
                (
                    family,
                    str(row.get("model_id", "")),
                    str(row.get("pair_id", "")),
                    str(row.get("reveal_path", "")),
                )
            )
        rows.append(
            {
                "family": family,
                "unit_id": unit_id,
                "dataset": str(row.get("dataset", family)),
                "model": str(row.get("model", row.get("model_id", ""))),
                "method": str(row.get("method", "")),
                "image_id": str(row.get("image_id", row.get("pair_id", ""))),
                "factor_or_part_id": str(
                    row.get("factor_or_part_id", row.get("reveal_path", ""))
                ),
                "tier": str(row["tier"]),
                **flags,
                "maximum_absolute_error": maximum,
                "maximum_error_variables": maximum_variables,
                "absolute_errors": dict(sorted(errors.items())),
                "triggers": triggers,
            }
        )
    return sorted(rows, key=lambda record: record["unit_id"])


def _scientific_mismatch_disclosure(
    *,
    c0_qualification: Mapping[str, Any],
    core_frames: Mapping[str, pd.DataFrame],
    imagenet9_e2e_frame: pd.DataFrame,
    attribution_e2e_frame: pd.DataFrame,
    covertype_e2e: Mapping[str, Any],
) -> dict[str, Any]:
    core_frame = pd.concat(list(core_frames.values()), ignore_index=True)
    core_rows = _hard_mismatch_rows(core_frame, family="current_core")
    imagenet9_rows = _hard_mismatch_rows(imagenet9_e2e_frame, family="imagenet9")
    attribution_rows = _hard_mismatch_rows(attribution_e2e_frame, family="attribution")
    e2e_rows = sorted(
        [*imagenet9_rows, *attribution_rows], key=lambda record: record["unit_id"]
    )
    covertype_units = int(covertype_e2e["unit_count"])
    covertype_hard = float(covertype_e2e["hard_mismatch_fraction"]) * covertype_units
    if not np.isclose(covertype_hard, round(covertype_hard), atol=1.0e-9, rtol=0.0):
        raise FinalizationError("Covertype weighted hard-mismatch count is non-integral")
    covertype_hard_count = int(round(covertype_hard))
    if covertype_hard_count:
        raise FinalizationError(
            "Covertype reports hard mismatches but exposes no row-level disclosure table"
        )
    core_ids = {record["unit_id"] for record in core_rows}
    e2e_ids = {record["unit_id"] for record in e2e_rows}
    unique_ids = sorted(core_ids | e2e_ids)
    core_by_id = {record["unit_id"]: record for record in core_rows}
    e2e_by_id = {record["unit_id"]: record for record in e2e_rows}
    disclosed_rows: list[dict[str, Any]] = []
    for unit_id in unique_ids:
        layers = [
            layer
            for layer, records in (
                ("current_core", core_by_id),
                ("current_e2e", e2e_by_id),
            )
            if unit_id in records
        ]
        source = e2e_by_id.get(unit_id, core_by_id[unit_id])
        disclosed_rows.append({**source, "layers": layers})
    shared_ids = sorted(core_ids & e2e_ids)
    core_only_ids = sorted(core_ids - e2e_ids)
    e2e_only_ids = sorted(e2e_ids - core_ids)
    return {
        "status": "ALL_HARD_MISMATCHES_EXPLICITLY_DISCLOSED",
        "unique_hard_mismatch_count": len(unique_ids),
        "unique_hard_mismatch_unit_ids": unique_ids,
        "current_core": {
            "hard_mismatch_count": len(core_rows),
            "unit_ids": sorted(core_ids),
        },
        "current_e2e": {
            "hard_mismatch_count": len(e2e_rows),
            "row_level_families": {
                "imagenet9": len(imagenet9_rows),
                "attribution": len(attribution_rows),
            },
            "covertype_exact_estimator_test_units": {
                "unit_count": covertype_units,
                "hard_mismatch_count": covertype_hard_count,
                "evidence_boundary": (
                    "exact test-unit-weighted estimator summaries; no synthetic row expansion"
                ),
            },
        },
        "same_units_in_core_and_row_level_e2e": core_ids == e2e_ids,
        "shared_unit_ids": shared_ids,
        "current_core_only_unit_ids": core_only_ids,
        "current_e2e_only_unit_ids": e2e_only_ids,
        "hard_rows": disclosed_rows,
        "c0_qualification_exclusions": list(c0_qualification["excluded"]),
        "c0_qualification_exclusion_count": int(c0_qualification["excluded_count"]),
    }


def determine_overall_verdict(
    families: Mapping[str, Mapping[str, Any]], paper: Mapping[str, Any]
) -> str:
    """Apply the runbook's overall-verdict definitions."""

    mandatory_core = all(
        families.get(family, {}).get("status") in PASSING_CORE_STATUSES
        for family in MANDATORY_FAMILIES
    )
    required_e2e = all(
        families.get(family, {}).get("status") == "PASS_CORE_AND_E2E"
        for family in ("imagenet9", "attribution")
    )
    covertype_exact = families.get("covertype", {}).get("status") == "PASS_CORE_AND_E2E"
    replay = (
        paper.get("status") == "PASS"
        and int(paper.get("assertions_passed", -1)) == 27
        and int(paper.get("assertions_total", -1)) == 27
    )
    if mandatory_core and required_e2e and covertype_exact and replay:
        return "PASS_FOR_PAPER_REPRODUCTION"
    if mandatory_core and required_e2e and replay:
        return "PASS_WITH_SCOPED_EXECUTOR_GAPS"
    if any(
        families.get(family, {}).get("status") == "FAIL_NUMERICAL" for family in MANDATORY_FAMILIES
    ):
        return "FAIL"
    return "PARTIAL"


def _run_git(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalizationError(f"git {' '.join(arguments)} failed: {error}") from error
    return result.stdout.rstrip("\n")


def _run_git_bytes(repository: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalizationError(f"git {' '.join(arguments)} failed: {error}") from error
    return result.stdout


def _git_index_identity(repository: Path, head_tree: str) -> dict[str, Any]:
    """Bind the index without invoking an object-writing Git command."""

    stage_payload = _run_git_bytes(repository, "ls-files", "--stage", "-z")
    records = [record for record in stage_payload.split(b"\0") if record]
    compared = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--no-ext-diff", "HEAD", "--"],
        cwd=repository,
        check=False,
        capture_output=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if compared.returncode not in {0, 1}:
        raise FinalizationError("git index-to-HEAD comparison failed")
    matches_head = compared.returncode == 0
    return {
        "index_tree": head_tree if matches_head else None,
        "index_matches_head": matches_head,
        "index_identity_kind": "git_ls_files_stage_z_sha256",
        "index_snapshot_sha256": hashlib.sha256(stage_payload).hexdigest(),
        "index_entry_count": len(records),
        "index_tree_derivation": (
            "verified_equal_to_HEAD_tree_via_git_diff_cached_quiet"
            if matches_head
            else "not_materialized; byte_safe_ls_files_stage_snapshot_is_authority"
        ),
    }


def _untracked_patch(repository: Path, relative: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "--no-index", "--", "/dev/null", relative],
        cwd=repository,
        check=False,
        capture_output=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if result.returncode not in {0, 1}:
        raise FinalizationError(f"could not capture untracked file in diff: {relative}")
    return result.stdout.decode("utf-8", errors="replace")


def _working_tree_manifest(repository: Path) -> tuple[list[dict[str, Any]], str]:
    listed = _run_git(repository, "ls-files", "-co", "--exclude-standard", "-z")
    paths = sorted(set(value for value in listed.split("\0") if value))
    records: list[dict[str, Any]] = []
    for relative in paths:
        path = repository / relative
        if path.is_symlink():
            target = os.readlink(path)
            records.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": target,
                    "sha256": hashlib.sha256(target.encode()).hexdigest(),
                }
            )
        elif path.is_file():
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": f"{path.stat().st_mode & 0o777:04o}",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            records.append({"path": relative, "kind": "missing"})
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return records, hashlib.sha256(encoded).hexdigest()


def capture_repository_provenance(
    root: Path, repository: Path, *, write_outputs: bool = True
) -> dict[str, Any]:
    """Capture repository identity, working diff, and source hashes."""

    repository = repository.resolve()
    commit = _run_git(repository, "rev-parse", "HEAD")
    head_tree = _run_git(repository, "rev-parse", "HEAD^{tree}")
    index_identity = _git_index_identity(repository, head_tree)
    branch = _run_git(repository, "branch", "--show-current")
    status_text = _run_git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    diff_text = _run_git(repository, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked = sorted(
        value
        for value in _run_git(repository, "ls-files", "--others", "--exclude-standard", "-z").split(
            "\0"
        )
        if value
    )
    if untracked:
        diff_text += "".join(_untracked_patch(repository, relative) for relative in untracked)
    diff_payload = diff_text + ("\n" if diff_text else "")
    status_payload = status_text + ("\n" if status_text else "")
    source_paths = sorted(
        [
            repository / "tools/crossgen",
            repository / "tests/unit/crossgen",
            repository / "src/decaf/core",
            repository / "src/decaf/experiments/attribution",
            repository / "src/decaf/experiments/imagenet9",
        ]
    )
    source_hashes: dict[str, str] = {}
    for source_root in source_paths:
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".sh"}
            ):
                relative = path.relative_to(repository).as_posix()
                source_hashes[relative] = sha256_file(path)
    working_manifest, working_digest = _working_tree_manifest(repository)
    payload = {
        "schema_version": 1,
        "repository": str(repository),
        "branch": branch,
        "commit": commit,
        "head_tree": head_tree,
        **index_identity,
        "working_tree_snapshot_sha256": working_digest,
        "working_tree_entry_count": len(working_manifest),
        "tracked_and_untracked_status_clean": not bool(status_text),
        "untracked_paths": untracked,
        "status_sha256": hashlib.sha256(status_payload.encode()).hexdigest(),
        "diff_sha256": hashlib.sha256(diff_payload.encode()).hexdigest(),
        "source_hashes": source_hashes,
    }
    if not write_outputs:
        return payload

    diff_path = root / "provenance/repository_diff.patch"
    _atomic_text(diff_path, diff_payload)
    status_path = root / "provenance/repository_status.txt"
    _atomic_text(status_path, status_payload)
    working_manifest_path = root / "provenance/repository_working_tree_manifest.json"
    _atomic_json(
        working_manifest_path,
        {
            "schema_version": 1,
            "entries": working_manifest,
            "snapshot_sha256": working_digest,
        },
    )
    payload.update(
        {
            "working_tree_manifest": str(working_manifest_path.resolve()),
            "working_tree_manifest_sha256": sha256_file(working_manifest_path),
            "status_path": str(status_path.resolve()),
            "diff_path": str(diff_path.resolve()),
        }
    )
    identity_path = root / "provenance/repository_identity.json"
    _atomic_json(identity_path, payload)
    return payload


def _paper_evidence(replay_root: Path) -> dict[str, Any]:
    analysis = _read_json(replay_root / "analysis_replay.json", "analysis replay")
    headline = _read_json(replay_root / "headline_assertions.json", "headline assertions")
    wrapper = _read_json(replay_root / "cpu_verification.json", "analysis replay wrapper")
    analysis_status = str(analysis.get("status", "")).lower()
    assertion_status = str(headline.get("status", "")).lower()
    count = int(headline.get("assertion_count", -1))
    verified = int(headline.get("verified_count", -1))
    if (
        analysis_status != "passed"
        or str(analysis.get("headline_assertions_status", "")).lower() != "passed"
        or assertion_status != "passed"
        or count != 27
        or verified != count
        or str(wrapper.get("status", "")).lower() != "passed"
        or wrapper.get("mode") != "analysis-replay"
        or wrapper.get("steps", {}).get("analysis_replay") != analysis
    ):
        raise FinalizationError("paper replay/headline evidence did not pass 27/27")
    return {
        "status": "PASS",
        "assertions_passed": verified,
        "assertions_total": count,
        "analysis_replay": str((replay_root / "analysis_replay.json").resolve()),
        "analysis_replay_sha256": sha256_file(replay_root / "analysis_replay.json"),
        "headline_assertions": str((replay_root / "headline_assertions.json").resolve()),
        "headline_assertions_sha256": sha256_file(replay_root / "headline_assertions.json"),
        "wrapper": str((replay_root / "cpu_verification.json").resolve()),
        "wrapper_sha256": sha256_file(replay_root / "cpu_verification.json"),
        "repository_commit_recorded": analysis.get("repository_commit"),
        "repository_tree_recorded": analysis.get("repository_tree"),
    }


def _dino_evidence(b200_root: Path) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    status_path = b200_root / "B200_VERIFICATION_STATUS.json"
    run_root = b200_root / "runs/dinov2_g"
    run_path = run_root / "run.json"
    status = _read_json(status_path, "prior B200 verification status")
    run = _read_json(run_path, "prior B200 DINO run receipt")
    repository = status.get("repository")
    machine = status.get("machine")
    gpu = machine.get("gpu") if isinstance(machine, dict) else None
    gates = status.get("acceptance_gates")
    checkpoint_fingerprints = status.get("checkpoint_fingerprints")
    final_audits = status.get("final_audits")
    shard_group = status.get("representative_shards")
    shard = shard_group.get("dinov2_g") if isinstance(shard_group, dict) else None
    if (
        status.get("schema_version") != 1
        or str(status.get("status", "")).lower() != "passed"
        or not isinstance(repository, dict)
        or repository.get("commit") != DINO_REPOSITORY_COMMIT
        or repository.get("tree") != DINO_REPOSITORY_TREE
        or repository.get("tracked_worktree_clean") is not True
        or not isinstance(gpu, dict)
        or gpu.get("name") != "NVIDIA B200"
        or not isinstance(gates, dict)
        or any(gates.get(gate) is not True for gate in DINO_REQUIRED_GATES)
        or not isinstance(shard, dict)
        or shard.get("status") != "PASS"
        or shard.get("scope") != "real_cuda_single_b200_shard"
        or shard.get("member_count") != 16
        or not isinstance(checkpoint_fingerprints, dict)
        or checkpoint_fingerprints.get("status") != "PASS"
        or not isinstance(final_audits, dict)
        or final_audits.get("repository_audit") != "PASS"
        or run.get("schema_version") != 1
        or str(run.get("status", "")).lower() != "completed"
        or run.get("run_id") != "dinov2_g"
        or run.get("experiment") != "attribution"
        or run.get("profile") != "large-model-smoke"
        or run.get("completed_stages") != ["prepare", "compute", "analyze", "paper"]
    ):
        raise FinalizationError("prior real-B200 DINO status/run evidence failed closed")

    plan_path = run_root / "manifests/plan.json"
    checkpoint_path = run_root / "manifests/checkpoints.json"
    data_path = run_root / "manifests/data.json"
    compute_path = run_root / "receipts/compute.json"
    compute_members_path = run_root / "receipts/compute_members.json"
    plan = _read_json(plan_path, "prior B200 DINO plan")
    checkpoints = _read_json(checkpoint_path, "prior B200 DINO checkpoint binding")
    data = _read_json(data_path, "prior B200 DINO data binding")
    compute = _read_json(compute_path, "prior B200 DINO compute receipt")
    compute_members = _read_json(
        compute_members_path, "prior B200 DINO compute-member receipt"
    )
    plan_contract_sha256 = plan.get("plan_contract_sha256")
    config_sha256 = plan.get("config_sha256")
    plan_members = plan.get("members")
    plan_audit = plan.get("audit")
    if (
        plan.get("schema_version") != 1
        or plan.get("experiment") != "attribution"
        or plan.get("profile") != "large-model-smoke"
        or tuple(plan.get("scope_names", ())) != DINO_SCOPES
        or plan.get("member_count") != 16
        or plan.get("expected_member_count") != 16
        or plan.get("endpoint_m_stage") != "analyze"
        or not isinstance(plan_contract_sha256, str)
        or len(plan_contract_sha256) != 64
        or not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or not isinstance(plan_audit, dict)
        or plan_audit.get("passed") is not True
        or plan_audit.get("checked_members") != 16
        or plan_audit.get("errors") != []
        or not isinstance(plan_members, list)
        or len(plan_members) != 16
    ):
        raise FinalizationError("prior B200 DINO plan contract changed")
    jobs: dict[str, Mapping[str, Any]] = {}
    output_paths: set[str] = set()
    receipt_paths: set[str] = set()
    for index, job in enumerate(plan_members):
        if not isinstance(job, dict):
            raise FinalizationError(f"prior B200 DINO plan member[{index}] is invalid")
        member_id = job.get("member_id")
        output_path = job.get("output_path")
        receipt_path = job.get("receipt_path")
        output_relative = PurePosixPath(output_path) if isinstance(output_path, str) else None
        receipt_relative = PurePosixPath(receipt_path) if isinstance(receipt_path, str) else None
        if (
            not isinstance(member_id, str)
            or member_id in jobs
            or job.get("scope") not in DINO_SCOPES
            or job.get("model_id") != "dinov2_vit_g_14"
            or not isinstance(output_path, str)
            or output_relative is None
            or output_relative.is_absolute()
            or ".." in output_relative.parts
            or output_path in output_paths
            or not isinstance(receipt_path, str)
            or receipt_relative is None
            or receipt_relative.is_absolute()
            or ".." in receipt_relative.parts
            or receipt_path in receipt_paths
            or job.get("config_sha256") != config_sha256
            or job.get("plan_contract_sha256") != plan_contract_sha256
            or not isinstance(job.get("job_sha256"), str)
            or len(job["job_sha256"]) != 64
        ):
            raise FinalizationError(f"prior B200 DINO plan member[{index}] is stale or mixed")
        jobs[member_id] = job
        output_paths.add(output_path)
        receipt_paths.add(receipt_path)

    if (
        checkpoints.get("schema_version") != 1
        or checkpoints.get("resolved") is not True
        or checkpoints.get("execution_claimed") is not False
        or checkpoints.get("config_sha256") != config_sha256
        or checkpoints.get("plan_contract_sha256") != plan_contract_sha256
        or not isinstance(checkpoints.get("items"), list)
        or len(checkpoints["items"]) != 1
        or not isinstance(checkpoints["items"][0], dict)
        or checkpoints["items"][0].get("model_id") != "dinov2_vit_g_14"
        or not isinstance(checkpoints["items"][0].get("checkpoints"), list)
        or len(checkpoints["items"][0]["checkpoints"]) != 2
    ):
        raise FinalizationError("prior B200 DINO checkpoint binding changed")
    for checkpoint in checkpoints["items"][0]["checkpoints"]:
        if not isinstance(checkpoint, dict):
            raise FinalizationError("prior B200 DINO checkpoint record is malformed")
        checkpoint_value = checkpoint.get("resolved_path")
        if (
            not isinstance(checkpoint_value, str)
            or not Path(checkpoint_value).is_absolute()
            or not isinstance(checkpoint.get("bytes_sha256"), str)
            or len(checkpoint["bytes_sha256"]) != 64
        ):
            raise FinalizationError("prior B200 DINO checkpoint record is malformed")
        checkpoint_file = _require_file(Path(checkpoint_value), "prior B200 DINO checkpoint")
        if sha256_file(checkpoint_file) != checkpoint["bytes_sha256"]:
            raise FinalizationError("prior B200 DINO checkpoint bytes changed")

    data_items = data.get("items")
    if (
        data.get("schema_version") != 1
        or data.get("resolved") is not True
        or data.get("execution_claimed") is not False
        or data.get("config_sha256") != config_sha256
        or data.get("plan_contract_sha256") != plan_contract_sha256
        or not isinstance(data_items, list)
        or len(data_items) != 2
        or {item.get("scope") for item in data_items if isinstance(item, dict)}
        != set(DINO_SCOPES)
    ):
        raise FinalizationError("prior B200 DINO data binding changed")
    for item in data_items:
        if not isinstance(item, dict):
            raise FinalizationError("prior B200 DINO data record is malformed")
        data_value = item.get("resolved_path")
        if (
            item.get("dataset") != "imagenet1k_idsds"
            or item.get("images") != 8
            or not isinstance(data_value, str)
            or not Path(data_value).is_absolute()
            or item.get("bytes_sha256") != item.get("expected_sha256")
        ):
            raise FinalizationError("prior B200 DINO data record is malformed")
        data_file = _require_file(Path(data_value), "prior B200 DINO data manifest")
        if sha256_file(data_file) != item["bytes_sha256"]:
            raise FinalizationError("prior B200 DINO data-manifest bytes changed")

    details = compute.get("details")
    global_details = compute_members.get("details")
    member_statuses = compute_members.get("members")
    if (
        compute.get("schema_version") != 1
        or compute.get("stage") != "compute"
        or compute.get("status") != "completed"
        or not isinstance(details, dict)
        or details.get("backend") != "gpu"
        or details.get("scheduler") != "single_gpu_dynamic_queue"
        or details.get("device") != 0
        or details.get("member_count") != 16
        or details.get("completed_members") != 16
        or details.get("resumed_members") != 0
        or details.get("failed_members") != 0
        or compute_members.get("schema_version") != 1
        or compute_members.get("kind") != "global"
        or compute_members.get("all_processes_exited") is not True
        or compute_members.get("member_count") != 16
        or not isinstance(global_details, dict)
        or global_details.get("backend") != "gpu"
        or global_details.get("scheduler") != "single_gpu_dynamic_queue"
        or global_details.get("visible_device") != "cuda:0"
        or global_details.get("exclusive_member_concurrency") != 1
        or global_details.get("duplicate_execution") is not False
        or global_details.get("failures") != {}
        or global_details.get("member_count") != 16
        or global_details.get("plan_contract_sha256") != plan_contract_sha256
        or global_details.get("config_sha256") != config_sha256
        or global_details.get("checkpoint_binding_manifest_sha256")
        != sha256_file(checkpoint_path)
        or global_details.get("data_binding_manifest_sha256") != sha256_file(data_path)
        or not isinstance(member_statuses, dict)
        or set(member_statuses) != set(jobs)
        or any(
            not isinstance(value, dict)
            or value.get("status") != "completed"
            or value.get("optional") is not False
            for value in member_statuses.values()
        )
    ):
        raise FinalizationError("prior B200 DINO compute receipt chain changed")

    persisted_members: dict[str, dict[str, str]] = {}
    for member_id, job in jobs.items():
        receipt_path = _require_file(
            run_root / str(job["receipt_path"]), f"prior B200 DINO member receipt {member_id}"
        )
        receipt = _read_json(receipt_path, f"prior B200 DINO member receipt {member_id}")
        member_details = receipt.get("details")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("kind") != "member"
            or receipt.get("status") != "completed"
            or receipt.get("error") is not None
            or receipt.get("optional") is not False
            or receipt.get("member_id") != member_id
            or not isinstance(member_details, dict)
            or member_details.get("output_path") != job["output_path"]
            or member_details.get("scope") != job.get("scope")
            or member_details.get("model_id") != "dinov2_vit_g_14"
            or member_details.get("job_sha256") != job.get("job_sha256")
            or member_details.get("config_sha256") != config_sha256
            or member_details.get("plan_contract_sha256") != plan_contract_sha256
            or member_details.get("checkpoint_binding_manifest_sha256")
            != global_details["checkpoint_binding_manifest_sha256"]
            or member_details.get("data_binding_manifest_sha256")
            != global_details["data_binding_manifest_sha256"]
        ):
            raise FinalizationError(f"prior B200 DINO member lineage changed: {member_id}")
        output = _require_file(
            run_root / str(job["output_path"]), f"prior B200 DINO member output {member_id}"
        )
        if sha256_file(output) != member_details.get("output_sha256"):
            raise FinalizationError(f"prior B200 DINO member output changed: {member_id}")
        persisted_members[member_id] = {
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": sha256_file(receipt_path),
            "output_path": str(output.resolve()),
            "output_sha256": sha256_file(output),
        }

    bound_artifacts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in (
            ("verification_status", status_path),
            ("run", run_path),
            ("plan", plan_path),
            ("checkpoints", checkpoint_path),
            ("data", data_path),
            ("compute", compute_path),
            ("compute_members", compute_members_path),
        )
    }
    return {
        "status": "PASS_CORE_WITH_METADATA_LIMIT",
        "current_real_compute_path": "PASS",
        "historical_paper_output_replay": "PASS",
        "exact_same_unit_bridge": "not_available",
        "scoped_gap": (
            "Historical formal vectors use PartImageNet while the released executable "
            "smoke path uses ImageNet-1k IDSDS."
        ),
        "member_count": int(shard.get("member_count", 0)),
        "repository_commit": repository["commit"],
        "repository_tree": repository["tree"],
        "tracked_worktree_clean": True,
        "gpu_name": gpu["name"],
        "mandatory_gates": {gate: True for gate in sorted(DINO_REQUIRED_GATES)},
        "scope": shard["scope"],
        "bound_artifacts": bound_artifacts,
        "member_artifacts": persisted_members,
        "verification_status": str(status_path.resolve()),
        "verification_status_sha256": sha256_file(status_path),
        "run_receipt": str(run_path.resolve()),
        "run_receipt_sha256": sha256_file(run_path),
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_report(status: Mapping[str, Any]) -> str:
    families = status["families"]
    attribution_sources = families["attribution"]["historical_source_bindings"]
    attribution_snapshots = attribution_sources["provenance_snapshots"]
    attribution_a0 = attribution_sources["a0_funnybirds"]
    attribution_a2 = attribution_sources["a2_imagenet1k_idsds"]
    covertype_source = families["covertype"]["historical_source_binding"]
    mismatches = status["scientific_mismatches"]
    c0 = families["controlled"]["c0_candidate_qualification"]
    c0_attribution = families["controlled"]["c0_runtime_attribution"]
    rows = []
    for family in ("controlled", "imagenet9", "attribution", "dinov2-g", "covertype"):
        value = families[family]
        core = value.get("current_core")
        e2e = value.get("current_e2e")
        core_units = (core or {}).get("unit_count")
        e2e_units = (e2e or {}).get("unit_count")
        if core_units is not None and e2e_units is not None and core_units != e2e_units:
            units = f"{core_units} core / {e2e_units} E2E"
        else:
            units = core_units if core_units is not None else (e2e_units or "—")
        rows.append(
            "| {family} | {units} | {core_compared} | {e2e_compared} | {tier_a} | "
            "{tier_b} | {hard} | {family_status} |".format(
                family=family,
                units=units,
                core_compared="yes" if core else "no",
                e2e_compared="yes" if e2e else "no",
                tier_a=_format_percent(float(core["tier_a_fraction"])) if core else "—",
                tier_b=_format_percent(float(core["tier_b_fraction"])) if core else "—",
                hard=(_format_percent(float(core["hard_mismatch_fraction"])) if core else "—"),
                family_status=value["status"],
            )
        )
    core = status["current_core_agreement"]
    e2e = status["current_e2e_agreement"]
    variables = status["variable_statistics"]
    core_variable_rows = []
    for name in (*SUMMARY_NAMES, "gate", "orientation", "dominant mechanism"):
        metrics = variables[name]
        core_variable_rows.append(
            "| {name} | {median} | {p95} | {maximum} | {agreement} |".format(
                name=name,
                median=(
                    f"{metrics['median_absolute_error']:.6g}"
                    if "median_absolute_error" in metrics
                    else "—"
                ),
                p95=(
                    f"{metrics['p95_absolute_error']:.6g}"
                    if "p95_absolute_error" in metrics
                    else "—"
                ),
                maximum=(
                    f"{metrics['maximum_absolute_error']:.6g}"
                    if "maximum_absolute_error" in metrics
                    else "—"
                ),
                agreement=_format_percent(float(metrics["agreement"])),
            )
        )
    endpoint = variables["d"]
    vector = variables["feature-vector agreement"]
    e2e_variable_rows = [
        "| d | {median:.6g} | {p95:.6g} | {maximum:.6g} | {agreement} |".format(
            median=endpoint["median_absolute_error"],
            p95=endpoint["p95_absolute_error"],
            maximum=endpoint["maximum_absolute_error"],
            agreement=_format_percent(float(endpoint["agreement"])),
        ),
        "| feature vector | {median:.6g} | {p95:.6g} | {maximum:.6g} | "
        "{agreement} |".format(
            median=vector["median_absolute_error"],
            p95=vector["p95_absolute_error"],
            maximum=vector["maximum_absolute_error"],
            agreement=_format_percent(float(vector["agreement"])),
        ),
    ]
    gaps = "\n".join(f"- {gap}" for gap in status["remaining_scoped_gaps"])
    c0_exclusions = "\n".join(
        "- `{model_id}` / base `{base_id}`: maximum absolute error "
        "`{maximum_absolute_error!r}`; reason `{reason_code}`; excluded without "
        "replacement.".format(**record)
        for record in c0["excluded"]
    )
    hard_mismatch_rows = "\n".join(
        "| {layers} | {dataset} | {model} | {method} | `{image_id}` | `{factor_or_part_id}` | "
        "{tier} | {variables} | `{maximum_absolute_error!r}` | {identity} | {semantics} |".format(
            layers=" + ".join(record["layers"]),
            dataset=record["dataset"],
            model=record["model"],
            method=record["method"],
            image_id=record["image_id"],
            factor_or_part_id=record["factor_or_part_id"],
            tier=record["tier"],
            variables=", ".join(record["maximum_error_variables"]),
            maximum_absolute_error=record["maximum_absolute_error"],
            identity="yes" if record["identity_match"] else "no",
            semantics=(
                "yes"
                if record["gate_match"]
                and record["orientation_match"]
                and record["dominant_match"]
                else "no"
            ),
        )
        for record in mismatches["hard_rows"]
    )
    covertype_disclosed_units = mismatches["current_e2e"][
        "covertype_exact_estimator_test_units"
    ]["unit_count"]
    mismatch_table_header = (
        "| Evidence layer(s) | Dataset | Model | Method | Image | Part | Tier | "
        "Maximum-error variable | "
        "Maximum absolute error | Identity exact | Gate/orientation/dominant exact |"
    )
    if mismatches["same_units_in_core_and_row_level_e2e"]:
        mismatch_layer_summary = (
            f"The same {len(mismatches['shared_unit_ids'])} units appear in both the "
            "current-core and row-level current-E2E layers."
        )
    else:
        mismatch_layer_summary = (
            f"Layer membership differs: {len(mismatches['shared_unit_ids'])} shared, "
            f"{len(mismatches['current_core_only_unit_ids'])} current-core-only, and "
            f"{len(mismatches['current_e2e_only_unit_ids'])} current-E2E-only units."
        )
    all_funny_vgg_decaf3 = bool(mismatches["hard_rows"]) and all(
        record["dataset"] == "funnybirds"
        and record["model"] == "funnybirds_vgg16"
        and record["method"] == "decaf_3"
        for record in mismatches["hard_rows"]
    )
    mismatch_science_summary = (
        "Every disclosed row is a FunnyBirds VGG16 DECAF-3 unit."
        if all_funny_vgg_decaf3
        else "The table reports the complete dynamic union of hard mismatches across layers."
    )
    index_description = (
        f"`{status['repository']['index_tree']}` (verified equal to HEAD without writing "
        "a Git object)"
        if status["repository"]["index_matches_head"]
        else (
            "not materialized; byte-safe `git ls-files --stage -z` snapshot "
            f"`{status['repository']['index_snapshot_sha256']}`"
        )
    )
    return f"""# DECAF Cross-Generation Equivalence Report V2

Overall verdict: **{status["overall_verdict"]}**

This verification connects representative exact historical endpoint/stage responses to
the current `decaf.core`, current executors to exact historical units where released
runtimes exist, and sealed full historical outputs to the current paper analysis.

## Family results

| Family | Units | Core | E2E | Core Tier A | Core Tier B | Core hard mismatch | Status |
|---|---:|:---:|:---:|---:|---:|---:|---|
{chr(10).join(rows)}

The three family-level numerical columns above are **current-core** statistics only;
they are not end-to-end values. Counts shown as core counts are neutral trajectory
units. E2E statistics are aggregated separately by actual comparison units; Covertype
contributes its exact 480,000 estimator/test units, not ten unweighted model summaries.

## Variable-level agreement by evidence layer

### Current-core variables and semantics

| Variable | Median absolute error | p95 absolute error | Maximum absolute error | Agreement |
|---|---:|---:|---:|---:|
{chr(10).join(core_variable_rows)}

`M`, `E`, `C`, `F`, and `Abs`, plus gate/orientation/dominant-mechanism agreement,
are derived exclusively from the current-core comparison layer.

### Current end-to-end variables

| Variable | Median absolute error | p95 absolute error | Maximum absolute error | Agreement |
|---|---:|---:|---:|---:|
{chr(10).join(e2e_variable_rows)}

Endpoint `d` and feature-vector agreement are derived exclusively from current
end-to-end executor comparisons; they are not current-core summary variables. Feature-vector
agreement is the independently recomputed 144-row per-image/method Spearman evidence,
not the 1,476 vector-element summary table.

Across all mandatory current-core units: Tier A {_format_percent(core["tier_a_fraction"])},
Tier B {_format_percent(core["tier_b_fraction"])}, hard mismatch
{_format_percent(core["hard_mismatch_fraction"])}. These are exact unit-weighted values.
Across current E2E comparisons: {e2e["unit_count"]} units, Tier A/B
{_format_percent(e2e["tier_a_or_b_fraction"])}, hard mismatch
{_format_percent(e2e["hard_mismatch_fraction"])}.

## Evidence layers and distinctions

- **Current-core equivalence:** historical factual/counterfactual endpoint and stage
  responses were read from neutral records and decomposed only by the current core.
- **Current end-to-end equivalence:** ImageNet-9 and attribution current executors were
  compared on exact historical identities; Covertype loaded historical estimators and
  exercised exact formal test/support inputs without retraining.
- **Verification-only historical trajectory exporter:** historical code was used only to
  recover factual/counterfactual scores and path identity where raw stages were not
  sealed. Historical decomposition was not used as the new result.
- **Paper bridge:** sealed full formal outputs were replayed through current analysis;
  {status["paper_replay"]["assertions_passed"]}/{status["paper_replay"]["assertions_total"]}
  headline assertions passed. The replay commit and HEAD tree exactly match repository
  provenance. Replay does not claim uncommitted working-tree state; the packaged binary
  diff, exact working-tree manifest, and source hashes bind that state separately.
- **ImageNet-9 historical source:** the verification bridge imports the required
  `__init__`, `data`, `decaf`, `models`, `reveal`, and `run` modules from the SHA-verified
  sealed lightweight ZIP (`{families["imagenet9"]["historical_source_binding"]["package_sha256"]}`),
  not from historical Git HEAD. The patch-order manifest and current E2E receipt bind
  the same package-derived order artifact.
- **Attribution A0 historical source:** all three FunnyBirds selections share the exact
  deployed-tree authority; the complete {attribution_a0["source_python_file_count"]}-file
  Python tree (`{attribution_a0["source_tree_sha256"]}`), required modules, anchors,
  actual import origins, deployment receipt, formal plan, and formal-plan receipt are
  SHA-verified. Original receipt/plan bytes are packaged at
  `{attribution_snapshots["attribution_a0"]["deployment_receipt"]["package_member"]}`,
  `{attribution_snapshots["attribution_a0"]["formal_plan"]["package_member"]}`, and
  `{attribution_snapshots["attribution_a0"]["formal_plan_receipt"]["package_member"]}` with the
  fixed status-recorded SHA-256 values.
- **Attribution A2 historical source:** all three ImageNet-1k IDSDS selections share the
  exact sealed authority (`{attribution_a2["package_sha256"]}`); the complete
  {attribution_a2["manifest_member_count"]}-payload/{attribution_a2["archive_member_count"]}-file
  ZIP inventory, {attribution_a2["namespace_member_count"]}-member materialized namespace,
  shim, and import origins are independently verified. The original raw ZIP-manifest
  bytes are packaged at
  `{attribution_snapshots["attribution_a2"]["package_manifest"]["package_member"]}`
  (`{attribution_snapshots["attribution_a2"]["package_manifest"]["sha256"]}`). Historical
  Git HEAD remains context only.
- **Attribution feature-vector quality:** all 144 current values are recomputed with the
  production `row_spearman` from six receipt/SHA-bound raw current outputs. FunnyBirds
  historical values are independently reconstructed from 144 two-operator rows in the
  A0 sealed `formal/heldout_quality.parquet` member
  (`{vector["reconstruction"]["historical_funnybirds"]["member_sha256"]}`) into 72 equal
  operator means. The A2 sealed `reused_quality.parquet` member is the aggregate's direct
  source and is retained as an exact 72-value cross-check; the A0 deployed tree is the
  historical executor source authority. IDSDS historical values are recomputed from the
  bound score/endpoint vectors.
- **Covertype historical source:** the verification bridge independently validates the
  sealed lightweight ZIP, its complete 110-member manifest/111-file archive inventory,
  all 26 namespace members and materialized bytes, the sole verification-only `cmr`
  parent shim, and the exact nine loaded-module origins. Historical Git HEAD is context
  only and is not the source authority. Its original raw ZIP-manifest bytes are packaged
  at `{covertype_source["provenance_snapshot"]["package_member"]}`
  (`{covertype_source["provenance_snapshot"]["sha256"]}`).

## Explicit scientific mismatch disclosure

Exactly {mismatches["unique_hard_mismatch_count"]} unique hard-mismatch units exist.
{mismatch_layer_summary}
ImageNet-9 has zero and Covertype's exact {covertype_disclosed_units:,}
test-unit-weighted estimator evidence has zero. Covertype is reported at its genuine
exact-estimator aggregation boundary and is not expanded into synthetic rows.

{mismatch_table_header}
|---|---|---|---|---|---|---:|---|---:|:---:|:---:|
{hard_mismatch_rows}

{mismatch_science_summary} Each has exact identity and exact
gate/orientation/dominant semantics where those flags are shown; each row records its
independently evaluated hard-mismatch trigger. Tier and hard mismatch are independent,
so a row may remain Tier A under the registered relative tolerance while still being a
hard mismatch. The C0 qualification exclusions below are additional scientific disclosures,
not hidden replacements and not members of this hard-mismatch table.

## Controlled aggregate

Controlled combines exactly 28 formal units at the unit level: C0 contributes 6, C1
contributes 10, and C2 contributes 12. C0 evaluated all eight fixed candidates, but it
is **6/8 strict aggregate-qualified**, not an 8/8 pass. The six selected candidates
retain the registered architecture, active/null, mixed-E/C, and counterfactual-map
coverage; the two failed candidates remain in the manifest and diagnostic and are not
hidden or replaced:

{c0_exclusions}

The C0 selection manifest (`{c0["manifest_sha256"]}`), final candidate diagnostic
(`{c0["diagnostic_sha256"]}`), sealed aggregate-audit CSV
(`{c0["sealed_aggregate_audit_sha256"]}`), and runtime attribution
(`{c0_attribution["sha256"]}`) are SHA-bound and packaged. These two exclusions set
`scientific_mismatches_found` to `{str(status["scientific_mismatches_found"]).lower()}`,
while the six-unit formal C0 subset remains eligible for the mandatory family gates.

Historical MIG execution versus the current full-B200 run is a plausible contributor
only after the exclusions above. It does **not** prove causality: the historical driver,
cuDNN/kernel selection, numeric flags, parent environment, and exact kernel/runtime
metadata were not locked, and no controlled rerun under the original complete runtime
is available.

Controlled's public paper-scale scheduler remains an external accelerator-bundle
interface; that executor gap does not alter the core equivalence test.

## DINOv2-g scope

The prior real single-B200 DINOv2-g compute path passed and its historical timing/quality
outputs pass current paper analysis. Exact same-unit PartImageNet bridging is not
available and is explicitly optional under the V2 protocol.

## Remaining scoped gaps

{gaps}

## Provenance and package

- Repository commit: `{status["repository"]["commit"]}`
- Repository HEAD tree: `{status["repository"]["head_tree"]}`
- Repository index identity: {index_description}
- Working-tree content snapshot: `{status["repository"]["working_tree_snapshot_sha256"]}`
- Machine: 1 × NVIDIA B200
- Final ZIP: `{status["package"]["path"]}`
- ZIP SHA256: `{status["package"]["sha256"]}`

No mismatches are suppressed: unit CSVs, metric error distributions, semantic agreement,
selection manifests, source hashes, repository status, and diff are retained in the
package. The package includes only the exact verified Attribution A2 and Covertype
materialized Python subtrees plus the small source-authority snapshots; external sealed
ZIPs, raw datasets, checkpoints, and large outputs are excluded.
"""


def _terminal_block(status: Mapping[str, Any]) -> str:
    families = status["families"]
    core = status["current_core_agreement"]
    e2e = status["current_e2e_agreement"]
    gaps = "; ".join(status["remaining_scoped_gaps"])
    return f"""DECAF CROSS-GENERATION EQUIVALENCE V2 COMPLETE

tmux session:
decaf-crossgen

Repository commit:
{status["repository"]["commit"]}

Machine:
1 × NVIDIA B200

Per-family status:
- controlled: {families["controlled"]["status"]}
- imagenet9: {families["imagenet9"]["status"]}
- attribution: {families["attribution"]["status"]}
- dinov2-g: {families["dinov2-g"]["status"]}
- covertype: {families["covertype"]["status"]}

Current-core agreement:
- units: {core["unit_count"]}
- Tier A: {_format_percent(core["tier_a_fraction"])}
- Tier B: {_format_percent(core["tier_b_fraction"])}
- hard mismatch: {_format_percent(core["hard_mismatch_fraction"])}

Current end-to-end agreement:
- units: {e2e["unit_count"]}
- Tier A/B: {_format_percent(e2e["tier_a_or_b_fraction"])}
- hard mismatch: {_format_percent(e2e["hard_mismatch_fraction"])}

Semantic agreement:
- gate: {_format_percent(core["gate_agreement"])}
- orientation: {_format_percent(core["orientation_agreement"])}
- dominant mechanism: {_format_percent(core["dominant_mechanism_agreement"])}

Historical full-output -> paper replay:
PASS

Paper headline assertions:
{status["paper_replay"]["assertions_passed"]}/{status["paper_replay"]["assertions_total"]}

Overall verdict:
{status["overall_verdict"]}

Remaining scoped gaps:
{gaps}

Final ZIP:
{status["package"]["path"]}

SHA256:
{status["package"]["sha256"]}"""


def _package_members(root: Path) -> list[tuple[Path, PurePosixPath]]:
    members: list[tuple[Path, PurePosixPath]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in FORBIDDEN_PACKAGE_PARTS for part in relative.parts):
            continue
        if (
            relative.parts[0] not in PACKAGE_DIRECTORIES
            and relative.as_posix() not in PACKAGE_ROOT_FILES
            and relative.as_posix() not in PACKAGE_EXACT_FILES
        ):
            continue
        relative_posix = PurePosixPath(relative.as_posix())
        allowed_materialized_python = relative.suffix.lower() == ".py" and any(
            relative_posix.is_relative_to(prefix) for prefix in PACKAGE_PYTHON_SUBTREES
        )
        if relative.suffix.lower() not in PACKAGE_SUFFIXES and not allowed_materialized_python:
            continue
        if relative.name.endswith(".zip") or path.stat().st_size > MAX_PACKAGE_MEMBER_BYTES:
            continue
        members.append((path, PurePosixPath("decaf_cross_generation_equivalence_v2") / relative))
    if not members:
        raise FinalizationError("package allowlist selected no files")
    return members


def write_deterministic_zip(
    root: Path, destination: Path, *, excluded: Iterable[Path] = ()
) -> dict[str, Any]:
    """Write a stable-order, stable-metadata ZIP and SHA-256 sidecar."""

    excluded_resolved = {path.resolve() for path in excluded}
    members = [
        pair for pair in _package_members(root) if pair[0].resolve() not in excluded_resolved
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for source, member in members:
                info = zipfile.ZipInfo(member.as_posix(), date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    digest = sha256_file(destination)
    sidecar = destination.with_name(f"{destination.name}.sha256")
    _atomic_text(sidecar, f"{digest}  {destination.name}\n")
    return {
        "path": str(destination.resolve()),
        "sha256": digest,
        "sha256_sidecar": str(sidecar.resolve()),
        "member_count": len(members),
        "members": [member.as_posix() for _, member in members],
    }


def _write_controlled_aggregate(
    root: Path, frames: Sequence[pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    combined = pd.concat(frames, ignore_index=True)
    if "controlled_phase" not in combined:
        raise FinalizationError("internal controlled aggregate lacks controlled_phase")
    phase_counts = {
        phase: int((combined["controlled_phase"] == phase).sum()) for phase in C0_PHASE_UNITS
    }
    if phase_counts != C0_PHASE_UNITS:
        raise FinalizationError(
            f"internal controlled aggregate has wrong phase coverage: {phase_counts!r}"
        )
    destination = root / "comparisons/controlled.csv"
    descriptor, temporary = tempfile.mkstemp(prefix=".controlled.csv.", dir=destination.parent)
    os.close(descriptor)
    try:
        combined.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    stats = aggregate_unit_comparisons([combined], label="controlled aggregate")
    summary = {
        "schema_version": 1,
        "family": "controlled",
        "status": "PASS_CORE",
        "phases": phase_counts,
        "comparison": str(destination.resolve()),
        "comparison_sha256": sha256_file(destination),
        **stats,
    }
    _atomic_json(root / "comparisons/controlled_summary.json", summary)
    return combined, summary


def _snapshot_prior_dino_evidence(root: Path, status: dict[str, Any]) -> None:
    dino = status["families"]["dinov2-g"]
    artifacts = dino.get("bound_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "verification_status",
        "run",
        "plan",
        "checkpoints",
        "data",
        "compute",
        "compute_members",
    }:
        raise FinalizationError("prior DINO evidence lacks the complete bound artifact set")
    packaged: list[dict[str, Any]] = []
    for name, record in sorted(artifacts.items()):
        if not isinstance(record, dict):
            raise FinalizationError(f"prior DINO {name} binding is invalid")
        source = Path(str(record.get("path", "")))
        expected_sha256 = record.get("sha256")
        destination = root / f"provenance/prior_b200_dinov2_g_{name}.json"
        _require_file(source, "prior DINO evidence")
        if sha256_file(source) != expected_sha256:
            raise FinalizationError(f"prior DINO evidence changed before packaging: {source}")
        _atomic_bytes(destination, source.read_bytes())
        packaged.append(
            {
                "kind": name,
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
            }
        )
    dino["packaged_evidence"] = packaged


def _write_input_inventory(root: Path, status: dict[str, Any]) -> None:
    inventory_path = root / "readiness/finalization_input_inventory.json"
    members = [
        (source, member)
        for source, member in _package_members(root)
        if source.resolve() != inventory_path.resolve() and source.name not in PACKAGE_ROOT_FILES
    ]
    entries = [
        {
            "path": member.relative_to("decaf_cross_generation_equivalence_v2").as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for source, member in members
    ]
    payload = {
        "schema_version": 1,
        "entry_count": len(entries),
        "entries": entries,
        "mandatory_family_statuses": {
            family: status["families"][family]["status"]
            for family in ("controlled", "imagenet9", "attribution", "covertype")
        },
        "paper_replay": status["paper_replay"]["status"],
        "headline_assertions": (
            f"{status['paper_replay']['assertions_passed']}/"
            f"{status['paper_replay']['assertions_total']}"
        ),
    }
    _atomic_json(inventory_path, payload)
    status["finalization_input_inventory"] = {
        "path": str(inventory_path.resolve()),
        "sha256": sha256_file(inventory_path),
        "entry_count": len(entries),
    }


def collect_evidence(
    root: Path,
    repository: Path,
    replay_root: Path,
    b200_root: Path,
    *,
    write_provenance: bool = True,
) -> dict[str, Any]:
    """Load, validate, and aggregate all mandatory V2 evidence."""

    controlled_frames: list[pd.DataFrame] = []
    controlled_phase_stats: dict[str, dict[str, Any]] = {}
    artifact_bindings: dict[str, dict[str, Any]] = {"current_core": {}, "current_e2e": {}}
    for phase in ("c0", "c1", "c2"):
        comparison_path = root / f"comparisons/controlled_{phase}.csv"
        frame = _read_csv(comparison_path, f"Controlled {phase.upper()} core CSV")
        frame.insert(0, "controlled_phase", phase)
        controlled_frames.append(frame)
        phase_summary = _read_json(
            root / f"comparisons/controlled_{phase}_summary.json",
            f"Controlled {phase.upper()} core summary",
        )
        phase_manifest = _read_json(
            root / f"manifests/controlled_{phase}_selection.json",
            f"Controlled {phase.upper()} selection manifest",
        )
        artifact_bindings["current_core"][f"controlled_{phase}"] = _validate_core_artifact_chain(
            trajectory_path=root / f"trajectories/controlled_{phase}.parquet",
            comparison_path=comparison_path,
            comparison=frame.drop(columns="controlled_phase"),
            summary=phase_summary,
            selection_manifest=phase_manifest,
            label=f"Controlled {phase.upper()}",
        )
        phase_stats = aggregate_unit_comparisons([frame], label=f"controlled {phase.upper()} core")
        _validate_core_summary(
            phase_summary, phase_stats, label=f"Controlled {phase.upper()} core summary"
        )
        expected_units = C0_PHASE_UNITS[phase]
        if int(phase_stats["unit_count"]) != expected_units:
            raise FinalizationError(
                f"Controlled {phase.upper()} must contain exactly {expected_units} formal units"
            )
        controlled_phase_stats[phase] = phase_stats

    c0_qualification = validate_c0_candidate_qualification(root, controlled_frames[0])
    c0_runtime_attribution = validate_c0_runtime_attribution(root, c0_qualification)
    controlled_combined = pd.concat(controlled_frames, ignore_index=True)
    controlled_stats = aggregate_unit_comparisons(
        [controlled_combined], label="controlled aggregate"
    )
    if int(controlled_stats["unit_count"]) != sum(C0_PHASE_UNITS.values()):
        raise FinalizationError("Controlled aggregate must contain exactly 28 formal units")
    controlled_coverage = {
        phase: int(stats["unit_count"]) for phase, stats in controlled_phase_stats.items()
    }

    core_frames: dict[str, pd.DataFrame] = {"controlled": controlled_combined}
    family_summaries: dict[str, dict[str, Any]] = {}
    core_summaries: dict[str, dict[str, Any]] = {}
    for family in ("imagenet9", "attribution", "covertype"):
        comparison_path = root / f"comparisons/{family}_core.csv"
        core_frames[family] = _read_csv(comparison_path, f"{family} core CSV")
        core_summaries[family] = _read_json(
            root / f"comparisons/{family}_core_summary.json", f"{family} core summary"
        )
        artifact_bindings["current_core"][family] = _validate_core_artifact_chain(
            trajectory_path=root / f"trajectories/{family}.parquet",
            comparison_path=comparison_path,
            comparison=core_frames[family],
            summary=core_summaries[family],
            label=family,
        )
        summary = _read_json(root / f"comparisons/{family}_summary.json", f"{family} E2E summary")
        status = _summary_status(summary, f"{family} E2E summary")
        if status != "PASS_CORE_AND_E2E":
            raise FinalizationError(f"mandatory {family} E2E status is {status}")
        family_summaries[family] = summary

    attribution_chain = validate_attribution_artifact_chain(root)
    attribution_source_bindings = validate_attribution_historical_source_bindings(root)
    attribution_chain["historical_source_bindings"] = attribution_source_bindings
    if (
        attribution_chain["aggregate_output_sha256"]
        != artifact_bindings["current_core"]["attribution"]["trajectory_sha256"]
    ):
        raise FinalizationError("Attribution aggregate/core summary bind different trajectories")
    artifact_bindings["attribution_chain"] = attribution_chain

    core_stats = {
        family: aggregate_unit_comparisons([frame], label=f"{family} core")
        for family, frame in core_frames.items()
    }
    core_stats["controlled"] = controlled_stats
    for family in ("imagenet9", "attribution", "covertype"):
        _validate_core_summary(
            core_summaries[family], core_stats[family], label=f"{family} core summary"
        )
    for family, stats in core_stats.items():
        _validate_core_family(family, stats)
    combined_core = aggregate_unit_comparisons(
        list(core_frames.values()), label="mandatory current core"
    )

    imagenet9_e2e_frame = _read_csv(root / "comparisons/imagenet9.csv", "ImageNet-9 E2E CSV")
    imagenet9_e2e = _status_from_e2e_csv(root, imagenet9_e2e_frame, "imagenet9")
    attribution_e2e_frame = _read_csv(root / "comparisons/attribution.csv", "Attribution E2E CSV")
    attribution_e2e = _status_from_e2e_csv(root, attribution_e2e_frame, "attribution")
    for family, stats in (
        ("imagenet9", imagenet9_e2e),
        ("attribution", attribution_e2e),
    ):
        summary = family_summaries[family]
        _validate_e2e_summary(summary, stats, label=f"{family} E2E summary")
        comparison_path = root / f"comparisons/{family}.csv"
        comparison_sha256 = _validate_artifact_binding(
            summary,
            comparison_path,
            path_key="comparison",
            sha_key="comparison_sha256",
            label=f"{family} E2E comparison",
        )
        artifact_bindings["current_e2e"][family] = {
            "comparison_path": str(comparison_path.resolve()),
            "comparison_sha256": comparison_sha256,
            "validation": (
                "summary_sha256_plus_independent_row_numeric_identity_semantic_"
                "tier_hard_and_exact_unit_summary_recomputation"
            ),
        }
        if family == "imagenet9":
            artifact_bindings["current_e2e"][family]["stage_evidence"] = imagenet9_e2e[
                "stage_evidence"
            ]
    attribution_trajectory_sha256 = _validate_artifact_binding(
        family_summaries["attribution"],
        root / "trajectories/attribution.parquet",
        path_key="neutral_trajectory",
        sha_key="neutral_trajectory_sha256",
        label="Attribution E2E neutral trajectory",
    )
    if (
        attribution_trajectory_sha256
        != artifact_bindings["current_core"]["attribution"]["trajectory_sha256"]
    ):
        raise FinalizationError("Attribution core/E2E summaries bind different trajectories")
    for summary, expected_path, path_key, sha_key, label in (
        (
            family_summaries["imagenet9"],
            root / "comparisons/imagenet9_selected_ratios.csv",
            "ratio_comparison",
            "ratio_comparison_sha256",
            "ImageNet-9 selected-ratio comparison",
        ),
        (
            family_summaries["imagenet9"],
            root / "comparisons/imagenet9_stage_responses.csv",
            "stage_comparison",
            "stage_comparison_sha256",
            "ImageNet-9 stage-response comparison",
        ),
        (
            family_summaries["attribution"],
            root / "comparisons/attribution_spearman.csv",
            "spearman_comparison",
            "spearman_comparison_sha256",
            "Attribution Spearman comparison",
        ),
    ):
        _validate_artifact_binding(
            summary,
            expected_path,
            path_key=path_key,
            sha_key=sha_key,
            label=label,
        )
    attribution_spearman = _validate_attribution_spearman(
        root,
        attribution_e2e_frame,
        family_summaries["attribution"],
        attribution_chain,
    )
    artifact_bindings["current_e2e"]["attribution"]["feature_vector_evidence"] = (
        attribution_spearman["artifact_binding"]
    )
    covertype_e2e = _covertype_e2e(root, family_summaries["covertype"])
    artifact_bindings["current_e2e"]["covertype"] = covertype_e2e["artifact_binding"]
    e2e_stats = {
        "imagenet9": imagenet9_e2e,
        "attribution": attribution_e2e,
        "covertype": covertype_e2e,
    }
    for family, stats in e2e_stats.items():
        if (
            float(stats["tier_a_or_b_fraction"]) < 0.95
            or float(stats["hard_mismatch_fraction"]) > 0.05
            or float(stats["identity_agreement"]) != 1.0
            or float(stats["gate_agreement"]) < 0.99
            or float(stats["orientation_agreement"]) < 0.99
            or float(stats["dominant_mechanism_agreement"]) < 0.95
        ):
            raise FinalizationError(f"mandatory {family} E2E acceptance failed")
        if (
            "metric_summaries" in stats
            and max(
                abs(float(stats["metric_summaries"][name]["mean_signed_error"]))
                for name in SUMMARY_NAMES
            )
            > 0.002
        ):
            raise FinalizationError(f"mandatory {family} E2E has systematic signed bias")

    _require_selection_manifests(root)
    imagenet9_source_binding = validate_imagenet9_historical_source_binding(root)
    covertype_source_binding = validate_covertype_historical_source_binding(root)
    historical_source_snapshots = snapshot_historical_source_provenance(
        root,
        attribution_source_bindings,
        covertype_source_binding,
        write_outputs=write_provenance,
    )
    attribution_source_bindings["provenance_snapshots"] = {
        "attribution_a0": historical_source_snapshots["attribution_a0"],
        "attribution_a2": historical_source_snapshots["attribution_a2"],
    }
    covertype_source_binding["provenance_snapshot"] = historical_source_snapshots[
        "covertype"
    ]["package_manifest"]
    artifact_bindings["historical_source_snapshots"] = historical_source_snapshots
    artifact_bindings["historical_source"] = {
        "imagenet9": imagenet9_source_binding,
        "attribution": attribution_source_bindings,
        "covertype": covertype_source_binding,
    }
    paper = _paper_evidence(replay_root)
    dino = _dino_evidence(b200_root)
    repository_identity = capture_repository_provenance(
        root, repository, write_outputs=write_provenance
    )
    if (
        paper.get("repository_commit_recorded") != repository_identity["commit"]
        or paper.get("repository_tree_recorded") != repository_identity["head_tree"]
    ):
        raise FinalizationError(
            "paper replay repository commit/HEAD tree differs from final repository identity"
        )
    paper["repository_binding"] = {
        "scope": "HEAD commit/tree",
        "commit_exact": True,
        "head_tree_exact": True,
        "working_tree_scope": (
            "uncommitted changes are not claimed by replay; the final package binds them "
            "through repository diff, exact working-tree manifest, and source hashes"
        ),
        "working_tree_snapshot_sha256": repository_identity["working_tree_snapshot_sha256"],
    }
    families = {
        "controlled": {
            "status": "PASS_CORE",
            "current_core": core_stats["controlled"],
            "current_e2e": None,
            "coverage": controlled_coverage,
            "c0_candidate_qualification": c0_qualification,
            "c0_runtime_attribution": c0_runtime_attribution,
        },
        "imagenet9": {
            "status": "PASS_CORE_AND_E2E",
            "current_core": core_stats["imagenet9"],
            "current_e2e": imagenet9_e2e,
            "historical_source_binding": imagenet9_source_binding,
        },
        "attribution": {
            "status": "PASS_CORE_AND_E2E",
            "current_core": core_stats["attribution"],
            "current_e2e": attribution_e2e,
            "historical_source_bindings": attribution_source_bindings,
        },
        "dinov2-g": {**dino, "current_core": None, "current_e2e": None},
        "covertype": {
            "status": "PASS_CORE_AND_E2E",
            "current_core": core_stats["covertype"],
            "current_e2e": covertype_e2e,
            "historical_source_binding": covertype_source_binding,
        },
    }
    if any(families[name]["status"] not in PASSING_CORE_STATUSES for name in MANDATORY_FAMILIES):
        raise FinalizationError("one or more mandatory families lack a passing core status")
    verdict = determine_overall_verdict(families, paper)
    if verdict != "PASS_FOR_PAPER_REPRODUCTION":
        raise FinalizationError(f"mandatory evidence yielded unexpected verdict: {verdict}")
    combined_e2e = _combine_e2e(e2e_stats)
    scientific_mismatches = _scientific_mismatch_disclosure(
        c0_qualification=c0_qualification,
        core_frames=core_frames,
        imagenet9_e2e_frame=imagenet9_e2e_frame,
        attribution_e2e_frame=attribution_e2e_frame,
        covertype_e2e=covertype_e2e,
    )
    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC_TIMEZONE).isoformat().replace("+00:00", "Z"),
        "tmux_session": "decaf-crossgen",
        "machine": {"gpu_count": 1, "gpu_name": "NVIDIA B200"},
        "overall_verdict": verdict,
        "families": families,
        "current_core_agreement": combined_core,
        "current_e2e_agreement": combined_e2e,
        "variable_statistics": _variable_statistics(
            combined_core,
            imagenet9_e2e_frame,
            attribution_e2e_frame,
            attribution_spearman,
        ),
        "paper_replay": paper,
        "repository": repository_identity,
        "artifact_bindings": artifact_bindings,
        "remaining_scoped_gaps": [
            (
                "Controlled full paper-scale public scheduler remains an external "
                "accelerator-bundle interface."
            ),
            (
                "Controlled C0 has 6/8 strict aggregate-qualified candidates; two "
                "fixed context_gate endpoint-null candidates are retained as explicit "
                "UNRESOLVED_HISTORICAL_RUNTIME_METADATA exclusions. MIG versus full-B200 "
                "is only a post-exclusion hypothesis: causality is not proven and the "
                "historical kernel/runtime metadata was not locked."
            ),
            (
                "DINOv2-g exact PartImageNet same-unit bridge is unavailable; prior "
                "real-B200 compute and paper replay pass."
            ),
        ],
        "scientific_mismatches": scientific_mismatches,
        "scientific_mismatches_found": bool(
            int(c0_qualification["excluded_count"])
            or int(scientific_mismatches["unique_hard_mismatch_count"])
        ),
    }


def finalize(
    *,
    root: Path = DEFAULT_ROOT,
    repository: Path = DEFAULT_REPOSITORY,
    package_directory: Path = DEFAULT_PACKAGE_DIRECTORY,
    replay_root: Path = DEFAULT_REPLAY_ROOT,
    b200_root: Path = DEFAULT_B200_ROOT,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Validate evidence, emit report/status, package it, and print §20 output."""

    root = root.resolve()
    if not root.is_dir():
        raise FinalizationError(f"V2 root does not exist: {root}")
    status = collect_evidence(root, repository, replay_root, b200_root)
    controlled_frames = []
    for phase in ("c0", "c1", "c2"):
        frame = _read_csv(
            root / f"comparisons/controlled_{phase}.csv",
            f"Controlled {phase.upper()} core CSV",
        )
        frame.insert(0, "controlled_phase", phase)
        controlled_frames.append(frame)
    _write_controlled_aggregate(root, controlled_frames)
    _snapshot_prior_dino_evidence(root, status)
    _write_input_inventory(root, status)
    stamp = timestamp or datetime.now(UTC_TIMEZONE).strftime("%Y%m%dT%H%M%SZ")
    if not stamp or any(character not in "0123456789TZ_-" for character in stamp):
        raise FinalizationError(f"invalid package timestamp: {stamp!r}")
    destination = package_directory / f"decaf_cross_generation_equivalence_v2_b200_{stamp}.zip"
    report_path = root / "CROSS_GENERATION_EQUIVALENCE_REPORT_V2.md"
    status_path = root / "CROSS_GENERATION_EQUIVALENCE_STATUS_V2.json"

    status["package"] = {
        "path": str(destination.resolve()),
        "sha256": "PENDING_EXTERNAL_SHA256_SIDECAR",
        "sha256_sidecar": str(destination.with_name(f"{destination.name}.sha256").resolve()),
        "report_and_status_inside_zip": True,
        "packaged_copy_kind": "pre_digest_snapshot",
        "snapshot_note": (
            "The packaged report/status snapshot points to the external SHA256 sidecar; "
            "the paired authoritative copies are atomically updated with the final digest."
        ),
    }
    _atomic_json(status_path, status)
    _atomic_text(report_path, _render_report(status))
    package = write_deterministic_zip(root, destination)
    required_members = {
        "decaf_cross_generation_equivalence_v2/CROSS_GENERATION_EQUIVALENCE_REPORT_V2.md",
        "decaf_cross_generation_equivalence_v2/CROSS_GENERATION_EQUIVALENCE_STATUS_V2.json",
    }
    if not required_members.issubset(package["members"]):
        raise FinalizationError("final ZIP omits the required V2 report/status snapshot")
    status["package"] = {
        **package,
        "report_and_status_inside_zip": True,
        "packaged_copy_kind": "pre_digest_snapshot",
        "snapshot_digest_marker": "PENDING_EXTERNAL_SHA256_SIDECAR",
        "snapshot_note": (
            "The packaged report/status snapshot is non-recursive. The paired external "
            "authoritative copies and SHA256 sidecar record this final ZIP digest."
        ),
    }
    _atomic_json(status_path, status)
    _atomic_text(report_path, _render_report(status))
    print(_terminal_block(status), flush=True)
    return status


def check_only(
    *,
    root: Path = DEFAULT_ROOT,
    repository: Path = DEFAULT_REPOSITORY,
    replay_root: Path = DEFAULT_REPLAY_ROOT,
    b200_root: Path = DEFAULT_B200_ROOT,
) -> dict[str, Any]:
    """Validate every input without creating aggregate, report, or package outputs."""

    root = root.resolve()
    if not root.is_dir():
        raise FinalizationError(f"V2 root does not exist: {root}")
    return collect_evidence(
        root,
        repository,
        replay_root,
        b200_root,
        write_provenance=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--package-directory", type=Path, default=DEFAULT_PACKAGE_DIRECTORY)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--b200-root", type=Path, default=DEFAULT_B200_ROOT)
    parser.add_argument("--timestamp")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="fail-closed validation without writing report, status, provenance, or ZIP",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.check_only:
        status = check_only(
            root=arguments.root,
            repository=arguments.repository,
            replay_root=arguments.replay_root,
            b200_root=arguments.b200_root,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "check-only",
                    "overall_verdict": status["overall_verdict"],
                    "current_core_units": status["current_core_agreement"]["unit_count"],
                    "current_e2e_units": status["current_e2e_agreement"]["unit_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finalize(
        root=arguments.root,
        repository=arguments.repository,
        package_directory=arguments.package_directory,
        replay_root=arguments.replay_root,
        b200_root=arguments.b200_root,
        timestamp=arguments.timestamp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
