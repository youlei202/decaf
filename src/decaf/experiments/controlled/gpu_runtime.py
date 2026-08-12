"""Controlled single-B200 verification over real historical assets.

This module is dormant unless ``DECAF_B200_VERIFY=1``.  It deliberately keeps
all external paths in environment-selected, untracked manifests while binding
their byte identities into the run-local plan, prepared manifests, member
receipts, and resume validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from decaf.core.manifests import atomic_write_json, read_json, sha256_file
from decaf.experiments.common import RunContext
from decaf.experiments.controlled.evaluate import (
    PlanMember,
    configuration_sha256,
    member_contract_sha256,
    prepared_run_bindings,
    receipt_reusable,
)
from decaf.experiments.controlled.gpu_models import (
    ARCHITECTURES,
    load_historical_model,
    normalize_architecture,
    require_single_cuda,
)
from decaf.experiments.controlled.protocols import decompose_score_trajectory

B200_GATE_ENV = "DECAF_B200_VERIFY"
IMAGES_ENV = "DECAF_CONTROLLED_IMAGES_32_UINT8"
FACTORS_ENV = "DECAF_CONTROLLED_FACTOR_INDICES"
CHECKPOINT_MANIFEST_ENV = "DECAF_CONTROLLED_B200_CHECKPOINT_MANIFEST"
SAMPLES_ENV = "DECAF_CONTROLLED_B200_SAMPLES"
BATCH_SIZE_ENV = "DECAF_CONTROLLED_B200_BATCH_SIZE"

FACTOR_NAMES = (
    "floor_color",
    "wall_color",
    "object_color",
    "object_size",
    "object_shape",
    "orientation",
)
FACTOR_CARDINALITIES = (10, 10, 10, 8, 4, 15)
FACTOR_STRIDES = (48_000, 4_800, 480, 60, 15, 1)
NUM_IMAGES = 480_000
REQUIRED_FAMILIES = frozenset({"base", "evidence", "fragility", "context_swap"})
REQUIRED_BEHAVIORS = frozenset({"active", "null", "aligned", "opposed"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SAFE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def b200_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Return whether the explicit real-GPU gate is enabled."""

    env = os.environ if environment is None else environment
    return env.get(B200_GATE_ENV) == "1"


def _safe_token(value: Any, *, label: str) -> str:
    token = str(value).strip()
    if not token or not SAFE_TOKEN_PATTERN.fullmatch(token) or token in {".", ".."}:
        raise ValueError(f"unsafe {label}: {value!r}")
    return token


def _required_path(environment: Mapping[str, str], variable: str) -> Path:
    raw = environment.get(variable)
    if not raw:
        raise RuntimeError(f"{variable} is required when {B200_GATE_ENV}=1")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{variable} must be an absolute path")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{variable} is missing or not a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _relative_logical_path(value: Any, *, fallback: str) -> str:
    logical = Path(str(fallback if value is None or value == "" else value))
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        raise ValueError(f"checkpoint logical_path must be a contained relative path: {logical}")
    return logical.as_posix()


def _integer_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of integer row IDs")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError(f"{label} cannot contain booleans")
        try:
            selected = int(item)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{label} must contain integer row IDs") from error
        if selected < 0 or selected >= NUM_IMAGES:
            raise ValueError(f"{label} row ID outside [0, {NUM_IMAGES}): {selected}")
        result.append(selected)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate row IDs")
    return tuple(result)


def _alpha_grid(value: Any, *, context_swap: bool) -> tuple[float, ...]:
    raw = (0.0, 0.5, 1.0) if context_swap else (0.0, 0.25, 0.5, 0.75, 1.0)
    if value is not None and value != "":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError("alpha_grid must be a sequence")
        raw = tuple(float(item) for item in value)
    grid = np.asarray(raw, dtype=np.float64)
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.isfinite(grid).all()
        or grid[0] != 0.0
        or grid[-1] != 1.0
        or not np.all(np.diff(grid) > 0.0)
    ):
        raise ValueError("alpha_grid must be finite, increasing, and span [0, 1]")
    if context_swap and grid.size != 3:
        raise ValueError("context_swap verification requires exactly [0, midpoint, 1]")
    return tuple(map(float, grid))


@dataclass(frozen=True, slots=True)
class B200Case:
    """One historical checkpoint and its real-shard behavior contract."""

    case_id: str
    family: str
    expected_behavior: str
    architecture: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_bytes: int
    checkpoint_logical_path: str
    model_id: str
    task: str
    factor: str
    seed: int
    target: int
    epsilon: float
    alpha_grid: tuple[float, ...]
    sample_ids: tuple[int, ...]
    counterfactual_ids: tuple[int, ...]
    fingerprint: bool
    fingerprint_sample_ids: tuple[int, ...]
    model_config: Mapping[str, Any]
    state_dict_key: str | None
    strip_prefix: str | None

    def member_metadata(self, *, source_manifest_sha256: str) -> dict[str, Any]:
        return {
            "family": self.family,
            "verification_case": self.case_id,
            "expected_behavior": self.expected_behavior,
            "architecture": self.architecture,
            "model_id": self.model_id,
            "task": self.task,
            "factor": self.factor,
            "target": self.target,
            "endpoint_epsilon": self.epsilon,
            "alpha_grid": list(self.alpha_grid),
            "checkpoint_logical_path": self.checkpoint_logical_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
            "checkpoint_manifest_sha256": source_manifest_sha256,
            "sample_ids": list(self.sample_ids),
            "counterfactual_ids": list(self.counterfactual_ids),
            "model_config": dict(self.model_config),
            "state_dict_key": self.state_dict_key,
            "strip_prefix": self.strip_prefix,
            "precision": "float32",
            "execution_backend": "real_cuda",
        }

    def checkpoint_record(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "case_id": self.case_id,
            "family": self.family,
            "architecture": self.architecture,
            "logical_path": self.checkpoint_logical_path,
            "bytes": self.checkpoint_bytes,
            "sha256": self.checkpoint_sha256,
            "strict_load": True,
        }


@dataclass(frozen=True, slots=True)
class B200Inventory:
    """Validated external manifest and all behavior cases."""

    manifest_path: Path
    manifest_sha256: str
    cases: tuple[B200Case, ...]


@dataclass(frozen=True, slots=True)
class B200Assets:
    """Hash-bound Shapes3D arrays loaded read-only via NumPy mmap."""

    images_path: Path
    factors_path: Path
    images_sha256: str
    factors_sha256: str
    images_bytes: int
    factors_bytes: int
    images: np.ndarray
    factors: np.ndarray

    def data_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "scope": "controlled_real_cuda_shard",
            "gpu_real_shard_verification": "ready",
            "items": [
                {
                    "id": "images_32_uint8",
                    "kind": "npy",
                    "source_environment": IMAGES_ENV,
                    "shape": [NUM_IMAGES, 32, 32, 3],
                    "dtype": "uint8",
                    "bytes": self.images_bytes,
                    "sha256": self.images_sha256,
                },
                {
                    "id": "factor_indices",
                    "kind": "npy",
                    "source_environment": FACTORS_ENV,
                    "shape": [NUM_IMAGES, len(FACTOR_NAMES)],
                    "dtype": "uint8",
                    "bytes": self.factors_bytes,
                    "sha256": self.factors_sha256,
                    "factor_names": list(FACTOR_NAMES),
                    "factor_cardinalities": list(FACTOR_CARDINALITIES),
                    "row_major_order": True,
                },
            ],
        }


def _load_document(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("controlled B200 checkpoint manifest must be an object")
    return payload


def _parse_case(raw: Mapping[str, Any], manifest: Path, *, verify_files: bool) -> B200Case:
    case_id = _safe_token(raw.get("case_id"), label="case_id")
    family = str(raw.get("family", "")).strip().lower().replace("-", "_")
    behavior = str(raw.get("expected_behavior", raw.get("behavior", ""))).strip().lower()
    if family not in REQUIRED_FAMILIES:
        raise ValueError(f"case {case_id} has unsupported family {family!r}")
    if behavior not in REQUIRED_BEHAVIORS:
        raise ValueError(f"case {case_id} has unsupported expected_behavior {behavior!r}")
    architecture = normalize_architecture(raw.get("architecture"))
    raw_path = raw.get("checkpoint_path", raw.get("path"))
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"case {case_id} has no checkpoint_path")
    checkpoint_source = Path(raw_path).expanduser()
    if not checkpoint_source.is_absolute():
        checkpoint_source = manifest.parent / checkpoint_source
    checkpoint = checkpoint_source.resolve()
    digest = str(raw.get("checkpoint_sha256", raw.get("sha256", ""))).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"case {case_id} has no valid checkpoint SHA256")
    try:
        recorded_bytes = int(raw.get("checkpoint_bytes", raw.get("bytes", -1)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"case {case_id} has invalid checkpoint bytes") from error
    if verify_files:
        if not checkpoint_source.is_file() or checkpoint_source.is_symlink():
            raise FileNotFoundError(
                f"case {case_id} checkpoint is missing or not a regular non-symlink file: "
                f"{checkpoint}"
            )
        actual_bytes = checkpoint.stat().st_size
        if recorded_bytes not in {-1, actual_bytes}:
            raise ValueError(f"case {case_id} checkpoint byte count differs")
        if sha256_file(checkpoint) != digest:
            raise ValueError(f"case {case_id} checkpoint SHA256 differs")
        recorded_bytes = actual_bytes
    elif recorded_bytes < 1:
        raise ValueError(f"case {case_id} must record checkpoint_bytes")
    logical = _relative_logical_path(
        raw.get("logical_path"),
        fallback=f"historical/{case_id}/{checkpoint.name}",
    )
    model_id = _safe_token(raw.get("model_id", case_id), label="model_id")
    task = _safe_token(raw.get("task", "object_shape"), label="task")
    factor = str(
        raw.get(
            "factor",
            "object_color"
            if family == "context_swap"
            else ("wall_color" if family == "fragility" else "object_shape"),
        )
    )
    if factor not in FACTOR_NAMES:
        raise ValueError(f"case {case_id} has unknown factor {factor!r}")
    seed = int(raw.get("seed", 0))
    target = int(raw.get("target", 1))
    if target < 0:
        raise ValueError(f"case {case_id} target must be non-negative")
    epsilon = float(raw.get("endpoint_epsilon", raw.get("epsilon", 0.02)))
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"case {case_id} endpoint epsilon must be positive and finite")
    samples = _integer_tuple(raw.get("sample_ids"), label=f"{case_id}.sample_ids")
    counterfactuals = _integer_tuple(
        raw.get("counterfactual_ids"),
        label=f"{case_id}.counterfactual_ids",
    )
    if counterfactuals and len(counterfactuals) != len(samples):
        raise ValueError(f"case {case_id} sample/counterfactual lengths differ")
    fingerprint_ids = _integer_tuple(
        raw.get("fingerprint_sample_ids"),
        label=f"{case_id}.fingerprint_sample_ids",
    )
    model_config = raw.get("model_config", {})
    if not isinstance(model_config, Mapping):
        raise TypeError(f"case {case_id} model_config must be an object")
    state_dict_key = raw.get("state_dict_key")
    strip_prefix = raw.get("strip_prefix")
    if state_dict_key is not None and not isinstance(state_dict_key, str):
        raise TypeError(f"case {case_id} state_dict_key must be a string")
    if strip_prefix is not None and not isinstance(strip_prefix, str):
        raise TypeError(f"case {case_id} strip_prefix must be a string")
    return B200Case(
        case_id=case_id,
        family=family,
        expected_behavior=behavior,
        architecture=architecture,
        checkpoint_path=checkpoint,
        checkpoint_sha256=digest,
        checkpoint_bytes=recorded_bytes,
        checkpoint_logical_path=logical,
        model_id=model_id,
        task=task,
        factor=factor,
        seed=seed,
        target=target,
        epsilon=epsilon,
        alpha_grid=_alpha_grid(raw.get("alpha_grid"), context_swap=family == "context_swap"),
        sample_ids=samples,
        counterfactual_ids=counterfactuals,
        fingerprint=bool(raw.get("fingerprint", False)),
        fingerprint_sample_ids=fingerprint_ids,
        model_config=dict(model_config),
        state_dict_key=state_dict_key,
        strip_prefix=strip_prefix,
    )


def load_b200_inventory(
    environment: Mapping[str, str] | None = None,
    *,
    verify_files: bool = True,
) -> B200Inventory:
    """Load the external case manifest and fail closed on missing coverage."""

    env = os.environ if environment is None else environment
    manifest = _required_path(env, CHECKPOINT_MANIFEST_ENV)
    payload = _load_document(manifest)
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("controlled B200 checkpoint manifest cases must be a non-empty list")
    cases = tuple(
        sorted(
            (
                _parse_case(raw, manifest, verify_files=verify_files)
                for raw in raw_cases
                if isinstance(raw, Mapping)
            ),
            key=lambda case: case.case_id,
        )
    )
    if len(cases) != len(raw_cases):
        raise TypeError("every controlled B200 manifest case must be an object")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("controlled B200 manifest contains duplicate case IDs")
    logical_paths = [case.checkpoint_logical_path for case in cases]
    if len(logical_paths) != len(set(logical_paths)):
        raise ValueError("controlled B200 manifest contains duplicate logical checkpoint paths")
    families = {case.family for case in cases}
    behaviors = {case.expected_behavior for case in cases}
    architectures = {case.architecture for case in cases}
    if not REQUIRED_FAMILIES.issubset(families):
        raise ValueError(
            f"controlled B200 family coverage incomplete: {sorted(REQUIRED_FAMILIES - families)}"
        )
    if not REQUIRED_BEHAVIORS.issubset(behaviors):
        raise ValueError(
            "controlled B200 behavior coverage incomplete: "
            f"{sorted(REQUIRED_BEHAVIORS - behaviors)}"
        )
    if not set(ARCHITECTURES).issubset(architectures):
        raise ValueError(
            f"controlled B200 architecture coverage incomplete: "
            f"{sorted(set(ARCHITECTURES) - architectures)}"
        )
    return B200Inventory(manifest, sha256_file(manifest), cases)


def load_b200_cases(
    environment: Mapping[str, str] | None = None,
    *,
    verify_files: bool = True,
) -> tuple[B200Case, ...]:
    """Convenience accessor for validated external cases."""

    return load_b200_inventory(environment, verify_files=verify_files).cases


def _audit_factor_index(factors: np.ndarray) -> None:
    cardinalities = np.asarray(FACTOR_CARDINALITIES, dtype=np.int64)
    for start in range(0, NUM_IMAGES, 16_384):
        stop = min(NUM_IMAGES, start + 16_384)
        ids = np.arange(start, stop, dtype=np.int64)
        expected = _unravel(ids)
        observed = np.asarray(factors[start:stop], dtype=np.int64)
        if np.any(observed < 0) or np.any(observed >= cardinalities):
            raise ValueError(f"factor_indices contains an out-of-support value near row {start}")
        if not np.array_equal(observed, expected):
            mismatch = int(np.flatnonzero(np.any(observed != expected, axis=1))[0]) + start
            raise ValueError(f"factor_indices row-major identity differs at row {mismatch}")


def load_b200_assets(environment: Mapping[str, str] | None = None) -> B200Assets:
    """Load and fully validate the environment-selected Shapes3D NPY assets."""

    env = os.environ if environment is None else environment
    images_path = _required_path(env, IMAGES_ENV)
    factors_path = _required_path(env, FACTORS_ENV)
    images = np.load(images_path, mmap_mode="r", allow_pickle=False)
    factors = np.load(factors_path, mmap_mode="r", allow_pickle=False)
    if images.shape != (NUM_IMAGES, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError(
            f"{IMAGES_ENV} must be uint8[{NUM_IMAGES},32,32,3]; got {images.dtype}{images.shape}"
        )
    if factors.shape != (NUM_IMAGES, len(FACTOR_NAMES)) or factors.dtype != np.uint8:
        raise ValueError(
            f"{FACTORS_ENV} must be uint8[{NUM_IMAGES},6]; got {factors.dtype}{factors.shape}"
        )
    _audit_factor_index(factors)
    return B200Assets(
        images_path=images_path,
        factors_path=factors_path,
        images_sha256=sha256_file(images_path),
        factors_sha256=sha256_file(factors_path),
        images_bytes=images_path.stat().st_size,
        factors_bytes=factors_path.stat().st_size,
        images=images,
        factors=factors,
    )


def build_b200_members(
    config: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
) -> tuple[PlanMember, ...]:
    """Build the real-GPU member universe without changing smoke.yaml."""

    if str(config.get("profile", "")) != "smoke":
        raise ValueError("DECAF_B200_VERIFY=1 is supported only for the controlled smoke profile")
    inventory = load_b200_inventory(environment)
    members = [
        PlanMember(
            member_id=f"b200__{case.case_id}",
            phase=f"b200_{case.family}",
            resource="cuda:0",
            seed=case.seed,
            output=f"raw/b200/{case.family}/{case.case_id}.json",
            metadata=case.member_metadata(source_manifest_sha256=inventory.manifest_sha256),
        )
        for case in inventory.cases
    ]
    outputs = [member.output for member in members]
    if len(outputs) != len(set(outputs)):
        raise AssertionError("controlled B200 member outputs are not unique")
    return tuple(members)


def build_b200_plan(
    config: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the static real-CUDA shard plan selected by the explicit gate."""

    inventory = load_b200_inventory(environment)
    members = build_b200_members(config, environment)
    counts = {
        "scheduled_members": len(members),
        "base_cases": sum(case.family == "base" for case in inventory.cases),
        "evidence_cases": sum(case.family == "evidence" for case in inventory.cases),
        "fragility_cases": sum(case.family == "fragility" for case in inventory.cases),
        "context_swap_cases": sum(case.family == "context_swap" for case in inventory.cases),
        "architectures": len({case.architecture for case in inventory.cases}),
        "behaviors": len({case.expected_behavior for case in inventory.cases}),
    }
    assertions = {
        "families": {
            "expected": sorted(REQUIRED_FAMILIES),
            "actual": sorted({case.family for case in inventory.cases}),
            "passed": True,
        },
        "architectures": {
            "expected": list(ARCHITECTURES),
            "actual": sorted({case.architecture for case in inventory.cases}),
            "passed": True,
        },
        "behaviors": {
            "expected": sorted(REQUIRED_BEHAVIORS),
            "actual": sorted({case.expected_behavior for case in inventory.cases}),
            "passed": True,
        },
    }
    return {
        "schema_version": 2,
        "experiment": "controlled",
        "profile": "smoke",
        "verification_mode": "single_b200_real_cuda",
        "configuration_sha256": configuration_sha256(config),
        "member_contract_sha256": member_contract_sha256(members),
        "checkpoint_case_manifest_sha256": inventory.manifest_sha256,
        "scientific_counts": counts,
        "assertions": assertions,
        "contracts": {
            "gpu_execution_performed_by_this_cli": True,
            "single_visible_b200_required": True,
            "historical_repository_imported": False,
            "historical_checkpoints_strict_load": True,
            "score_precision": "float32",
            "decomposition_precision": "float64",
            "prepared_input_lineage_bound": True,
            "atomic_member_receipts": True,
            "receipt_driven_resume": True,
            "pointwise_conservation": True,
            "integrated_conservation": True,
            "tiny_endpoint_swap": True,
        },
        "members": [member.as_dict() for member in members],
    }


def b200_prepared_manifests(
    environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate external bytes and return portable prepared manifests."""

    inventory = load_b200_inventory(environment)
    assets = load_b200_assets(environment)
    checkpoint_manifest = {
        "schema_version": 2,
        "scope": "controlled_real_cuda_shard",
        "source_environment": CHECKPOINT_MANIFEST_ENV,
        "source_manifest_sha256": inventory.manifest_sha256,
        "strict_local_byte_identity": True,
        "gpu_real_shard_verification": "ready",
        "items": [
            {
                "id": "b200_historical_checkpoints",
                "count": len(inventory.cases),
                "checkpoints": [case.checkpoint_record() for case in inventory.cases],
            }
        ],
    }
    return assets.data_manifest(), checkpoint_manifest


def _unravel(ids: Any) -> np.ndarray:
    values = np.asarray(ids, dtype=np.int64).reshape(-1)
    if np.any(values < 0) or np.any(values >= NUM_IMAGES):
        raise ValueError("Shapes3D row ID is out of bounds")
    remainder = values.copy()
    result = np.empty((values.size, len(FACTOR_NAMES)), dtype=np.int64)
    for column, stride in enumerate(FACTOR_STRIDES):
        result[:, column] = remainder // stride
        remainder %= stride
    return result


def _ravel(factors: Any) -> np.ndarray:
    values = np.asarray(factors, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(FACTOR_NAMES):
        raise ValueError("factor tuples must have shape [N,6]")
    cardinalities = np.asarray(FACTOR_CARDINALITIES, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= cardinalities):
        raise ValueError("factor tuple leaves official Shapes3D support")
    return values @ np.asarray(FACTOR_STRIDES, dtype=np.int64)


def _changed_ids(ids: np.ndarray, factor: str) -> np.ndarray:
    factual = _unravel(ids)
    changed = factual.copy()
    column = FACTOR_NAMES.index(factor)
    values = changed[:, column]
    if factor in {"floor_color", "wall_color", "object_color"}:
        changed[:, column] = np.where(values < 5, values + 5, values - 5)
    elif factor == "object_shape":
        if not np.isin(values, (0, 1)).all():
            raise ValueError("object_shape verification pairs require levels {0,1}")
        changed[:, column] = 1 - values
    elif factor == "object_size":
        changed[:, column] = (values + 4) % 8
    elif factor == "orientation":
        changed[:, column] = (values + 7) % 15
    differences = factual != changed
    if not np.all(differences.sum(axis=1) == 1) or not np.all(differences[:, column]):
        raise AssertionError("counterfactual pair changed the wrong factor")
    return np.ascontiguousarray(_ravel(changed), dtype=np.int64)


def _deterministic_candidates(case: B200Case, count: int) -> np.ndarray:
    requested = max(1, count)
    start = int(case.seed) % NUM_IMAGES
    step = 7_919 + 2 * (abs(int(case.seed)) % 1_000)
    while math.gcd(step, NUM_IMAGES) != 1:
        step += 2
    ids = (start + np.arange(requested * 64, dtype=np.int64) * step) % NUM_IMAGES
    factors = _unravel(ids)
    eligible = np.ones(ids.size, dtype=bool)
    shape = factors[:, FACTOR_NAMES.index("object_shape")]
    if case.task == "object_shape" or case.family in {"evidence", "fragility", "context_swap"}:
        eligible &= np.isin(shape, (0, 1))
    if case.factor == "object_shape":
        eligible &= shape == 0
    if case.family == "context_swap":
        eligible &= factors[:, FACTOR_NAMES.index("wall_color")] >= 5
        eligible &= factors[:, FACTOR_NAMES.index("object_color")] < 5
    selected = ids[eligible][:requested]
    if selected.size < count:
        raise RuntimeError(f"could not construct enough deterministic pairs for {case.case_id}")
    return np.ascontiguousarray(selected, dtype=np.int64)


def _generic_pairs(case: B200Case, count: int) -> tuple[np.ndarray, np.ndarray]:
    factual = (
        np.asarray(case.sample_ids, dtype=np.int64)
        if case.sample_ids
        else _deterministic_candidates(case, count)
    )
    counterfactual = (
        np.asarray(case.counterfactual_ids, dtype=np.int64)
        if case.counterfactual_ids
        else _changed_ids(factual, case.factor)
    )
    factual_factors = _unravel(factual)
    counterfactual_factors = _unravel(counterfactual)
    differences = factual_factors != counterfactual_factors
    column = FACTOR_NAMES.index(case.factor)
    if (
        factual.shape != counterfactual.shape
        or not np.all(differences.sum(axis=1) == 1)
        or not np.all(differences[:, column])
    ):
        raise ValueError(f"case {case.case_id} pairs do not change exactly {case.factor}")
    return factual, counterfactual


def _context_groups(case: B200Case, count: int) -> dict[str, np.ndarray]:
    base = (
        np.asarray(case.sample_ids, dtype=np.int64)
        if case.sample_ids
        else _deterministic_candidates(case, count)
    )
    factors = _unravel(base)
    wall = FACTOR_NAMES.index("wall_color")
    object_color = FACTOR_NAMES.index("object_color")
    shape = FACTOR_NAMES.index("object_shape")
    if (
        not np.all(factors[:, wall] >= 5)
        or not np.all(factors[:, object_color] < 5)
        or not np.all(np.isin(factors[:, shape], (0, 1)))
    ):
        raise ValueError(f"context case {case.case_id} bases must have G=1, A=0, H in {{0,1}}")
    endpoint_cf = factors.copy()
    endpoint_cf[:, object_color] += 5
    swap_fact = factors.copy()
    swap_fact[:, wall] -= 5
    swap_cf = endpoint_cf.copy()
    swap_cf[:, wall] -= 5
    return {
        "endpoint_fact": np.ascontiguousarray(base, dtype=np.int64),
        "endpoint_cf": np.ascontiguousarray(_ravel(endpoint_cf), dtype=np.int64),
        "swap_fact": np.ascontiguousarray(_ravel(swap_fact), dtype=np.int64),
        "swap_cf": np.ascontiguousarray(_ravel(swap_cf), dtype=np.int64),
    }


def preprocess_uint8(images: Any) -> np.ndarray:
    """Apply the registered uint8 NHWC -> float32 NCHW [-1,1] transform."""

    source = np.asarray(images)
    if source.ndim != 4 or source.shape[1:] != (32, 32, 3) or source.dtype != np.uint8:
        raise ValueError(
            f"controlled preprocessing expected uint8 NHWC 32x32; got {source.dtype}{source.shape}"
        )
    result = np.ascontiguousarray(np.transpose(source, (0, 3, 1, 2)), dtype=np.float32)
    result /= np.float32(127.5)
    result -= np.float32(1.0)
    if not np.isfinite(result).all():
        raise FloatingPointError("controlled preprocessing produced a non-finite tensor")
    return result


def canonical_tensor_identity(value: Any) -> dict[str, Any]:
    """Hash canonical little-endian, C-contiguous float32 tensor bytes."""

    array = np.ascontiguousarray(np.asarray(value), dtype=np.dtype("<f4"))
    payload = memoryview(array).cast("B")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dtype": "float32",
        "shape": list(array.shape),
        "bytes": int(array.nbytes),
        "byte_order": "little-endian",
        "layout": "C-contiguous",
    }


def _positive_integer(environment: Mapping[str, str], variable: str, default: int) -> int:
    try:
        value = int(environment.get(variable, str(default)))
    except ValueError as error:
        raise ValueError(f"{variable} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{variable} must be a positive integer")
    return value


def _forward_logits(
    model: Any,
    inputs: np.ndarray,
    *,
    target: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch, _nn, _functional, _device_record = require_single_cuda(device)
    logits_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(inputs[start : start + batch_size]))
            batch = batch.to(device=torch.device(device), dtype=torch.float32, non_blocking=False)
            logits = model(batch)
            if not torch.is_tensor(logits) or logits.ndim != 2:
                raise RuntimeError("historical controlled model must return a rank-2 logits tensor")
            logits = logits.float()
            if target >= logits.shape[1]:
                raise ValueError(f"target {target} is outside {logits.shape[1]} model classes")
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("historical controlled model produced non-finite logits")
            logits_parts.append(logits.cpu().numpy())
    torch.cuda.synchronize(torch.device(device))
    all_logits = np.ascontiguousarray(np.concatenate(logits_parts), dtype=np.float32)
    maximum = np.max(all_logits, axis=1, keepdims=True)
    exponent = np.exp(np.asarray(all_logits - maximum, dtype=np.float64))
    all_probabilities = exponent / np.sum(exponent, axis=1, keepdims=True)
    if not np.isfinite(all_probabilities).all():
        raise FloatingPointError("softmax produced non-finite probabilities")
    return all_logits, np.ascontiguousarray(all_probabilities, dtype=np.float64)


def tiny_endpoint_swap_audit(
    alpha: Sequence[float],
    response: Any,
    endpoint: Any,
    *,
    epsilon: float,
    limit: int = 2,
) -> dict[str, Any]:
    """Audit sign-swapped endpoint invariance using the authoritative core."""

    values = np.asarray(response, dtype=np.float64)
    anchor = np.asarray(endpoint, dtype=np.float64)
    selected = min(limit, values.shape[0]) if values.ndim > 1 else 1
    forward_values = values[:selected] if values.ndim > 1 else values
    forward_anchor = anchor[:selected] if anchor.ndim > 0 else anchor
    forward = decompose_score_trajectory(
        alpha,
        forward_values,
        endpoint=forward_anchor,
        epsilon=epsilon,
    )
    swapped = decompose_score_trajectory(
        alpha,
        -forward_values,
        endpoint=-forward_anchor,
        epsilon=epsilon,
    )
    maximum_error = 0.0
    for name in ("E", "C", "F", "Abs", "Net"):
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(np.asarray(forward[name]) - np.asarray(swapped[name])))),
            float(
                np.max(
                    np.abs(
                        np.asarray(forward["pointwise_components"][name])
                        - np.asarray(swapped["pointwise_components"][name])
                    )
                )
            ),
        )
    passed = bool(maximum_error <= 1.0e-12)
    if not passed:
        raise AssertionError(f"tiny endpoint-swap invariance failed: {maximum_error}")
    return {"passed": passed, "pairs": selected, "max_abs_error": maximum_error}


def audit_score_trajectory(
    alpha: Sequence[float],
    response: Any,
    endpoint: Any,
    *,
    epsilon: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all finite/nonnegative/conservation/swap checks through DECAF core."""

    values = np.asarray(response, dtype=np.float64)
    anchor = np.asarray(endpoint, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(anchor).all():
        raise FloatingPointError("controlled score trajectory contains a non-finite value")
    scores = decompose_score_trajectory(alpha, values, endpoint=anchor, epsilon=epsilon)
    pointwise = scores["pointwise_components"]
    for name in ("E", "C", "F"):
        if np.any(np.asarray(pointwise[name]) < -1.0e-12):
            raise AssertionError(f"pointwise {name} contains a negative value")
        if np.any(np.asarray(scores[name]) < -1.0e-12):
            raise AssertionError(f"integrated {name} contains a negative value")
    numeric = scores["numeric_audit"]
    if not numeric["pointwise"]["passed"] or not numeric["integrated"]["passed"]:
        raise AssertionError("authoritative DECAF conservation audit failed")
    swap = tiny_endpoint_swap_audit(
        alpha,
        values,
        anchor,
        epsilon=epsilon,
    )
    audit = {
        "passed": True,
        "finite_model_scores": True,
        "nonnegative_ecf": True,
        "pointwise_conservation": numeric["pointwise"],
        "integrated_conservation": numeric["integrated"],
        "tiny_endpoint_swap": swap,
    }
    return scores, audit


def _behavior_mask(scores: Mapping[str, Any], behavior: str) -> np.ndarray:
    active = np.asarray(scores["endpoint_active"], dtype=bool).reshape(-1)
    evidence = np.asarray(scores["E"], dtype=np.float64).reshape(-1)
    contradiction = np.asarray(scores["C"], dtype=np.float64).reshape(-1)
    if behavior == "active":
        return active
    if behavior == "null":
        return ~active
    if behavior == "aligned":
        return active & (evidence > 1.0e-10)
    if behavior == "opposed":
        return active & (contradiction > 1.0e-10)
    raise KeyError(behavior)


def _range_record(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {"minimum": float(np.min(array)), "maximum": float(np.max(array))}


def _case_trajectory(
    case: B200Case,
    assets: B200Assets,
    model: Any,
    *,
    device: str,
    desired_samples: int,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    # Endpoint-null examples are intentionally sparse under the historical
    # fragile checkpoint.  Search a larger deterministic real-data pool so the
    # requested output still contains only observed null trajectories rather
    # than weakening the behavior assertion or padding with synthetic scores.
    candidate_multiplier = 256 if case.expected_behavior == "null" else 16
    candidate_count = max(desired_samples * candidate_multiplier, 64)
    if case.family == "context_swap":
        groups = _context_groups(case, candidate_count)
        joined = np.concatenate(
            [
                groups[name]
                for name in (
                    "endpoint_fact",
                    "endpoint_cf",
                    "swap_fact",
                    "swap_cf",
                )
            ]
        )
        inputs = preprocess_uint8(assets.images[joined])
        logits, probabilities = _forward_logits(
            model,
            inputs,
            target=case.target,
            device=device,
            batch_size=batch_size,
        )
        target_scores = probabilities[:, case.target]
        split = groups["endpoint_fact"].size
        q_endpoint_fact, q_endpoint_cf, q_swap_fact, q_swap_cf = np.split(
            target_scores, (split, 2 * split, 3 * split)
        )
        endpoint = q_endpoint_cf - q_endpoint_fact
        swapped = q_swap_cf - q_swap_fact
        response = np.stack((np.zeros_like(endpoint), swapped, endpoint), axis=1)
        scores, audit = audit_score_trajectory(
            case.alpha_grid,
            response,
            endpoint,
            epsilon=case.epsilon,
        )
        mask = _behavior_mask(scores, case.expected_behavior)
        selected = np.flatnonzero(mask)[:desired_samples]
        if selected.size < desired_samples:
            raise RuntimeError(
                f"case {case.case_id} found {selected.size}/{desired_samples} "
                f"{case.expected_behavior} context responses"
            )
        identifiers = {name: values[selected] for name, values in groups.items()}
        sample_record: dict[str, Any] = {
            "protocol": "four_image_context_swap",
            **{name + "_ids": values.tolist() for name, values in identifiers.items()},
        }
    else:
        factual, counterfactual = _generic_pairs(case, candidate_count)
        factual_inputs = preprocess_uint8(assets.images[factual])
        counterfactual_inputs = preprocess_uint8(assets.images[counterfactual])
        grid = np.asarray(case.alpha_grid, dtype=np.float32)
        states = (
            factual_inputs[:, None] * (np.float32(1.0) - grid[None, :, None, None, None])
            + counterfactual_inputs[:, None] * grid[None, :, None, None, None]
        )
        flat_states = np.ascontiguousarray(states.reshape((-1, 3, 32, 32)), dtype=np.float32)
        logits, probabilities = _forward_logits(
            model,
            flat_states,
            target=case.target,
            device=device,
            batch_size=batch_size,
        )
        target_scores = probabilities[:, case.target].reshape((factual.size, grid.size))
        response = target_scores - target_scores[:, :1]
        endpoint = response[:, -1]
        scores, audit = audit_score_trajectory(
            case.alpha_grid,
            response,
            endpoint,
            epsilon=case.epsilon,
        )
        mask = _behavior_mask(scores, case.expected_behavior)
        selected = np.flatnonzero(mask)[:desired_samples]
        if selected.size < desired_samples:
            raise RuntimeError(
                f"case {case.case_id} found {selected.size}/{desired_samples} "
                f"{case.expected_behavior} trajectories"
            )
        sample_record = {
            "protocol": "linear_pixel_counterfactual",
            "factual_ids": factual[selected].tolist(),
            "counterfactual_ids": counterfactual[selected].tolist(),
        }
    selected_scores = {
        name: np.asarray(scores[name]).reshape(-1)[selected]
        for name in ("M", "E", "C", "F", "Abs", "Net")
    }
    selected_active = np.asarray(scores["endpoint_active"], dtype=bool).reshape(-1)[selected]
    result = {
        "sample_ids": sample_record,
        "metrics": {name: value.tolist() for name, value in selected_scores.items()},
        "endpoint_active": selected_active.tolist(),
        "observed_behaviors": {
            "active": int(np.count_nonzero(selected_active)),
            "null": int(np.count_nonzero(~selected_active)),
            "aligned": int(np.count_nonzero(selected_scores["E"] > 1.0e-10)),
            "opposed": int(np.count_nonzero(selected_scores["C"] > 1.0e-10)),
        },
        "numeric_audit": audit,
        "model_score_range": _range_record(probabilities),
        "model_logit_range": _range_record(logits),
    }
    return result, logits, probabilities


def b200_member_executor(
    inventory: B200Inventory,
    assets: B200Assets,
    *,
    environment: Mapping[str, str] | None = None,
    device: str = "cuda:0",
):
    """Create the sequential real-CUDA member executor."""

    env = os.environ if environment is None else environment
    desired_samples = _positive_integer(env, SAMPLES_ENV, 8)
    batch_size = _positive_integer(env, BATCH_SIZE_ENV, 128)
    by_id = {f"b200__{case.case_id}": case for case in inventory.cases}

    def executor(context: RunContext, member: PlanMember) -> Sequence[Path]:
        try:
            case = by_id[member.member_id]
        except KeyError as error:
            raise KeyError(f"unexpected controlled B200 member: {member.member_id}") from error
        if (
            case.checkpoint_path.stat().st_size != case.checkpoint_bytes
            or sha256_file(case.checkpoint_path) != case.checkpoint_sha256
        ):
            raise ValueError(f"checkpoint byte identity changed before {case.case_id}")
        model, load_record, device_record = load_historical_model(
            case.checkpoint_path,
            case.architecture,
            device=device,
            model_config=case.model_config or None,
            state_dict_key=case.state_dict_key,
            strip_prefix=case.strip_prefix,
        )
        try:
            result, _logits, _probabilities = _case_trajectory(
                case,
                assets,
                model,
                device=device,
                desired_samples=desired_samples,
                batch_size=batch_size,
            )
            payload = {
                "schema_version": 2,
                "kind": "controlled_real_cuda_shard",
                "family": case.family,
                "case_id": case.case_id,
                "expected_behavior": case.expected_behavior,
                "architecture": case.architecture,
                "model_id": case.model_id,
                "task": case.task,
                "factor": case.factor,
                "target": case.target,
                "alpha_grid": list(case.alpha_grid),
                "endpoint_epsilon": case.epsilon,
                "checkpoint": case.checkpoint_record(),
                "checkpoint_load": load_record,
                "data_bindings": {
                    "images_sha256": assets.images_sha256,
                    "factor_indices_sha256": assets.factors_sha256,
                    "checkpoint_manifest_sha256": inventory.manifest_sha256,
                },
                "execution": {
                    "backend": "cuda",
                    "device": device_record,
                    "precision": "float32",
                    "decomposition_precision": "float64",
                    "batch_size": batch_size,
                    "gpu_verification": "passed",
                },
                **result,
            }
            output = context.path / member.output
            atomic_write_json(output, payload)
            return (output,)
        finally:
            torch, _nn, _functional, _record = require_single_cuda(device)
            del model
            torch.cuda.empty_cache()

    return executor


def _fingerprint_cases(inventory: B200Inventory) -> tuple[B200Case, B200Case]:
    marked = [case for case in inventory.cases if case.fingerprint]
    if marked:
        selected = marked
    else:
        selected = [case for case in inventory.cases if case.family == "base"]
    if len(selected) != 2 or {case.architecture for case in selected} != set(ARCHITECTURES):
        raise ValueError(
            "controlled checkpoint fingerprints require exactly two marked cases "
            "(or exactly two base cases): one resnet18 and one small_vit"
        )
    return tuple(sorted(selected, key=lambda case: case.architecture))  # type: ignore[return-value]


def validate_checkpoint_fingerprint_records(records: Any) -> list[dict[str, Any]]:
    """Validate the strict two-case contract consumed by checkpoint-fingerprint."""

    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("controlled fingerprint collector must return a list of exactly two cases")
    architectures: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for position, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"controlled fingerprint case {position} must be an object")
        record = dict(raw)
        if record.get("family") != "controlled":
            raise ValueError("controlled fingerprint family must equal 'controlled'")
        for key in ("case_id", "model_id", "precision", "device"):
            if key not in record or record[key] is None or record[key] == "":
                raise ValueError(f"controlled fingerprint case is missing {key}")
        architecture = normalize_architecture(record.get("architecture"))
        architectures.add(architecture)
        checkpoints = record.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 1:
            raise ValueError("controlled fingerprint case must bind exactly one checkpoint")
        checkpoint = checkpoints[0]
        if not isinstance(checkpoint, Mapping):
            raise TypeError("controlled fingerprint checkpoint must be an object")
        path = Path(str(checkpoint.get("path", "")))
        digest = str(checkpoint.get("sha256", "")).lower()
        try:
            checkpoint_bytes = int(checkpoint.get("bytes"))
        except (TypeError, ValueError) as error:
            raise ValueError("controlled fingerprint checkpoint bytes are invalid") from error
        if not path.is_absolute() or not SHA256_PATTERN.fullmatch(digest) or checkpoint_bytes < 1:
            raise ValueError("controlled fingerprint checkpoint identity is incomplete")
        sample_ids = record.get("sample_ids")
        if (
            not isinstance(sample_ids, list)
            or not sample_ids
            or any(isinstance(value, bool) or not isinstance(value, int) for value in sample_ids)
        ):
            raise ValueError("controlled fingerprint sample_ids must be non-empty integers")
        tensor = record.get("preprocessed_tensor")
        if not isinstance(tensor, Mapping):
            raise ValueError("controlled fingerprint preprocessed_tensor is missing")
        if (
            not SHA256_PATTERN.fullmatch(str(tensor.get("sha256", "")).lower())
            or tensor.get("dtype") != "float32"
            or tensor.get("byte_order") != "little-endian"
            or tensor.get("layout") != "C-contiguous"
            or tensor.get("shape") != [len(sample_ids), 3, 32, 32]
        ):
            raise ValueError("controlled fingerprint preprocessed tensor identity is invalid")
        target_values = record.get("target_class")
        if (
            not isinstance(target_values, list)
            or len(target_values) != len(sample_ids)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in target_values)
            or any(value < 0 for value in target_values)
            or len(set(target_values)) != 1
        ):
            raise ValueError("controlled fingerprint target_class must be one integer per sample")
        target = int(target_values[0])
        logits = np.asarray(record.get("logits"), dtype=np.float64)
        probabilities = np.asarray(record.get("probabilities"), dtype=np.float64)
        if (
            logits.ndim != 2
            or probabilities.shape != logits.shape
            or logits.shape[0] != len(sample_ids)
            or target >= logits.shape[1]
            or not np.isfinite(logits).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-7, rtol=0.0)
        ):
            raise ValueError("controlled fingerprint logits/probabilities are invalid")
        if record.get("precision") != "float32":
            raise ValueError("controlled fingerprint precision must be float32")
        normalized.append(record)
    if architectures != set(ARCHITECTURES):
        raise ValueError("controlled fingerprints must cover resnet18 and small_vit exactly")
    return normalized


def collect_checkpoint_fingerprints(device: str = "cuda:0") -> list[dict[str, Any]]:
    """Return exactly two real-CUDA controlled checkpoint fingerprints.

    The external manifest must select one ResNet-18 and one Small-ViT.  Missing
    arrays, checkpoint bytes, or an ambiguous selection fail closed.
    """

    device_name = str(device)
    inventory = load_b200_inventory()
    assets = load_b200_assets()
    cases = _fingerprint_cases(inventory)
    torch, _nn, _functional, device_record = require_single_cuda(device_name)
    output: list[dict[str, Any]] = []
    for case in cases:
        if case.fingerprint_sample_ids:
            sample_ids = np.asarray(case.fingerprint_sample_ids, dtype=np.int64)
        elif case.sample_ids:
            sample_ids = np.asarray(case.sample_ids[:2], dtype=np.int64)
        else:
            factual, counterfactual = _generic_pairs(case, 1)
            sample_ids = np.asarray([factual[0], counterfactual[0]], dtype=np.int64)
        if sample_ids.size < 1:
            raise ValueError(f"fingerprint case {case.case_id} has no sample IDs")
        preprocessed = preprocess_uint8(assets.images[sample_ids])
        model, load_record, case_device = load_historical_model(
            case.checkpoint_path,
            case.architecture,
            device=device_name,
            model_config=case.model_config or None,
            state_dict_key=case.state_dict_key,
            strip_prefix=case.strip_prefix,
        )
        try:
            logits, probabilities = _forward_logits(
                model,
                preprocessed,
                target=case.target,
                device=device_name,
                batch_size=max(1, sample_ids.size),
            )
            output.append(
                {
                    "schema_version": 1,
                    "experiment": "controlled",
                    "family": "controlled",
                    "case_id": f"controlled__{case.case_id}",
                    "architecture": case.architecture,
                    "model_id": case.model_id,
                    "checkpoints": [
                        {
                            "identity": case.model_id,
                            "path": str(case.checkpoint_path),
                            "logical_path": case.checkpoint_logical_path,
                            "sha256": case.checkpoint_sha256,
                            "bytes": case.checkpoint_bytes,
                            "manifest_sha256": inventory.manifest_sha256,
                        }
                    ],
                    "checkpoint_load": load_record,
                    "sample_ids": sample_ids.tolist(),
                    "preprocessed_tensor": {
                        **canonical_tensor_identity(preprocessed),
                        "definition": "uint8 NHWC -> float32 NCHW via x / 127.5 - 1",
                        "images_source_sha256": assets.images_sha256,
                        "factor_indices_source_sha256": assets.factors_sha256,
                    },
                    "target_class": [case.target] * int(sample_ids.size),
                    "logits": logits.tolist(),
                    "probabilities": probabilities.tolist(),
                    "precision": "float32",
                    "device": device_name,
                    "device_details": case_device,
                    "libraries": {
                        "torch": str(torch.__version__),
                        "cuda": str(torch.version.cuda),
                        "cudnn": int(torch.backends.cudnn.version() or 0),
                        "numpy": str(np.__version__),
                    },
                }
            )
        finally:
            del model
            torch.cuda.empty_cache()
    output = validate_checkpoint_fingerprint_records(output)
    if device_record != output[0]["device_details"] or device_record != output[1]["device_details"]:
        raise AssertionError("controlled fingerprint device identity changed between cases")
    return output


def validate_b200_prepare_resume(context: RunContext) -> dict[str, Any]:
    """Revalidate every prepared byte binding before skipping prepare."""

    inventory = load_b200_inventory()
    members = build_b200_members(context.config)
    expected_plan = build_b200_plan(context.config)
    expected_data, expected_checkpoints = b200_prepared_manifests()
    persisted_plan = read_json(context.path / "manifests" / "plan.json")
    persisted_data = read_json(context.path / "manifests" / "data.json")
    persisted_checkpoints = read_json(context.path / "manifests" / "checkpoints.json")
    if persisted_plan != expected_plan:
        raise ValueError("controlled B200 prepared plan changed before resume")
    if persisted_data != expected_data:
        raise ValueError("controlled B200 prepared data manifest changed before resume")
    if persisted_checkpoints != expected_checkpoints:
        raise ValueError("controlled B200 prepared checkpoint manifest changed before resume")
    jobs = [
        json.loads(line)
        for line in (context.path / "manifests" / "jobs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if jobs != [member.as_dict() for member in members]:
        raise ValueError("controlled B200 jobs manifest changed before resume")
    prepared_run_bindings(context, members)
    return {"validated": True, "members": len(members), "manifest": inventory.manifest_sha256}


def validate_b200_compute_resume(context: RunContext) -> dict[str, Any]:
    """Require every completed member artifact/hash before skipping compute."""

    validate_b200_prepare_resume(context)
    members = build_b200_members(context.config)
    bindings = prepared_run_bindings(context, members)
    invalid = [
        member.member_id
        for member in members
        if not receipt_reusable(context, member, run_bindings=bindings)
    ]
    if invalid:
        raise ValueError(f"controlled B200 member receipts are not reusable: {invalid}")
    global_receipt = read_json(context.path / "receipts" / "compute_members.json")
    expected_ids = {member.member_id for member in members}
    details = global_receipt.get("details", {})
    if (
        global_receipt.get("kind") != "global"
        or global_receipt.get("status") != "completed"
        or not global_receipt.get("all_processes_exited")
        or set(global_receipt.get("members", {})) != expected_ids
        or any(
            record.get("status") != "completed"
            for record in global_receipt.get("members", {}).values()
        )
        or details.get("member_contract_sha256") != member_contract_sha256(members)
        or details.get("run_bindings") != bindings
    ):
        raise ValueError("controlled B200 global compute receipt is not reusable")
    return {"validated": True, "members": len(members), "output_hashes": "unchanged"}


def validate_b200_analyze_resume(context: RunContext) -> dict[str, Any]:
    """Validate real-CUDA analysis products before skipping analyze."""

    validate_b200_compute_resume(context)
    summary = read_json(context.path / "metrics" / "controlled_smoke_summary.json")
    metrics = context.path / "metrics" / "controlled_smoke_metrics.csv"
    if (
        not metrics.is_file()
        or metrics.stat().st_size < 1
        or summary.get("scope") != "real_cuda_single_b200_shard"
        or summary.get("gpu_real_shard_verification") != "passed"
        or summary.get("metrics_sha256") != sha256_file(metrics)
    ):
        raise ValueError("controlled B200 analysis outputs are not reusable")
    return {"validated": True, "metrics_sha256": sha256_file(metrics)}


def validate_b200_paper_resume(context: RunContext) -> dict[str, Any]:
    """Validate paper-data artifact hashes before skipping the paper stage."""

    validate_b200_analyze_resume(context)
    root = context.path / "paper_data" / "controlled"
    receipt = read_json(root / "controlled_receipt.json")
    if (
        receipt.get("scope") != "real_cuda_single_b200_shard"
        or receipt.get("gpu_real_shard_verification") != "passed"
    ):
        raise ValueError("controlled B200 paper receipt is not reusable")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("controlled B200 paper receipt has no artifacts")
    for record in artifacts:
        path = root / str(record.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"controlled B200 paper artifact changed: {path.name}")
    return {"validated": True, "artifacts": len(artifacts)}


__all__ = [
    "B200Assets",
    "B200Case",
    "B200Inventory",
    "B200_GATE_ENV",
    "BATCH_SIZE_ENV",
    "CHECKPOINT_MANIFEST_ENV",
    "FACTORS_ENV",
    "IMAGES_ENV",
    "SAMPLES_ENV",
    "audit_score_trajectory",
    "b200_enabled",
    "b200_member_executor",
    "b200_prepared_manifests",
    "build_b200_members",
    "build_b200_plan",
    "canonical_tensor_identity",
    "collect_checkpoint_fingerprints",
    "load_b200_assets",
    "load_b200_cases",
    "load_b200_inventory",
    "preprocess_uint8",
    "tiny_endpoint_swap_audit",
    "validate_checkpoint_fingerprint_records",
    "validate_b200_analyze_resume",
    "validate_b200_compute_resume",
    "validate_b200_paper_resume",
    "validate_b200_prepare_resume",
]
