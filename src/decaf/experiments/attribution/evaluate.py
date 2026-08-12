"""Member planning, CPU score oracle, and resumable attribution evaluation."""

from __future__ import annotations

import importlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.core.manifests import sha256_file
from decaf.core.receipts import (
    finalize_global_receipt,
    load_member_receipt,
    utc_now,
    write_member_receipt,
)
from decaf.experiments.attribution.endpoint import row_spearman
from decaf.experiments.attribution.methods import decaf_trajectory
from decaf.experiments.attribution.models import checkpoint_coverage
from decaf.experiments.attribution.plan import (
    DELETION_TARGET_METHOD,
    FUNNYBIRDS_DELETION_TARGET_METHOD,
    FUNNYBIRDS_HELDOUT_METHODS,
    build_plan,
    canonical_sha256,
    validate_plan,
)
from decaf.experiments.attribution.timing import require_gpu_runtime
from decaf.experiments.common import (
    RunContext,
    TerminationRequested,
    atomic_json,
    atomic_text,
)

MemberEvaluator = Callable[[Mapping[str, Any], RunContext], pd.DataFrame]

_HELDOUT_QUALITY_AGGREGATION = "equal_mean_of_operator_spearman"
_HELDOUT_EFFECT_COLUMNS = (
    "heldout_background_texture_effects",
    "heldout_telea_dilate3_effects",
)
_HELDOUT_SPEARMAN_COLUMNS = (
    "heldout_background_texture_spearman",
    "heldout_telea_dilate3_spearman",
)


def _operatorwise_heldout_spearman(
    patch_scores: Sequence[np.ndarray],
    background_effects: Sequence[np.ndarray],
    telea_effects: Sequence[np.ndarray],
) -> tuple[list[float], list[float], list[float]]:
    """Score each held-out operator first, then take their frozen equal mean."""

    background_scores = [
        float(row_spearman(patch, effects)[0])
        for patch, effects in zip(patch_scores, background_effects, strict=True)
    ]
    telea_scores = [
        float(row_spearman(patch, effects)[0])
        for patch, effects in zip(patch_scores, telea_effects, strict=True)
    ]
    averaged = [
        0.5 * (first + second)
        for first, second in zip(background_scores, telea_scores, strict=True)
    ]
    return background_scores, telea_scores, averaged


def _contained(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ValueError("member paths must be relative to the run directory")
    destination = (root / value).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"member path escapes the run directory: {relative}") from error
    return destination


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write and verify a Parquet member."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("member evaluator must return a non-empty DataFrame")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        observed = pd.read_parquet(temporary)
        if len(observed) != len(frame) or tuple(observed.columns) != tuple(frame.columns):
            raise RuntimeError(f"Parquet member verification failed: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _plan(context: RunContext) -> dict[str, Any]:
    path = context.path / "manifests/plan.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("stored attribution plan is not an object")
        validate_plan(payload, raise_on_error=True)
        current = build_plan(context.config)
        if canonical_sha256(payload) != canonical_sha256(current):
            raise RuntimeError("stored attribution plan does not match the current config")
        return payload
    return build_plan(context.config)


def _execution_config(context: RunContext) -> Mapping[str, Any]:
    value = context.config.get("execution", {})
    if not isinstance(value, Mapping):
        raise TypeError("execution configuration must be a mapping")
    return value


def _configured_binding(
    context: RunContext,
    *,
    mapping_name: str,
    identities: tuple[str, ...],
    root_env_name: str,
) -> Path:
    execution = _execution_config(context)
    mapping = execution.get(mapping_name, {})
    if not isinstance(mapping, Mapping):
        raise TypeError(f"execution.{mapping_name} must be a mapping")
    configured = next(
        (mapping[identity] for identity in identities if identity in mapping),
        None,
    )
    if not isinstance(configured, str) or not configured:
        raise RuntimeError(
            f"formal compute requires execution.{mapping_name} binding for one of {identities}"
        )
    environment_match = re.fullmatch(r"\$\{([A-Z][A-Z0-9_]*)\}(?:/(.*))?", configured)
    if environment_match is not None:
        variable, suffix = environment_match.groups()
        root_value = os.environ.get(variable)
        if not root_value:
            raise RuntimeError(f"configured runtime binding requires ${variable}")
        path = Path(root_value).expanduser()
        if suffix:
            path /= suffix
    else:
        path = Path(configured).expanduser()
    if not path.is_absolute():
        root_value = os.environ.get(root_env_name)
        if not root_value:
            raise RuntimeError(
                f"relative execution.{mapping_name} binding requires ${root_env_name}"
            )
        path = Path(root_value).expanduser() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured runtime binding is absent: {path}")
    return path


def _runtime_from_manifests(
    context: RunContext,
    plan: Mapping[str, Any],
    data_payload: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        data_payload.get("schema_version") != 1
        or checkpoint_payload.get("schema_version") != 1
        or data_payload.get("resolved") is not True
        or checkpoint_payload.get("resolved") is not True
        or data_payload.get("plan_contract_sha256") != plan.get("plan_contract_sha256")
        or checkpoint_payload.get("plan_contract_sha256") != plan.get("plan_contract_sha256")
        or data_payload.get("config_sha256") != plan.get("config_sha256")
        or checkpoint_payload.get("config_sha256") != plan.get("config_sha256")
        or data_payload.get("execution_claimed") is not False
        or checkpoint_payload.get("execution_claimed") is not False
    ):
        raise RuntimeError("runtime binding manifests do not match the stored plan")
    data_rows = data_payload.get("items")
    checkpoint_rows = checkpoint_payload.get("items")
    if not isinstance(data_rows, list) or not isinstance(checkpoint_rows, list):
        raise RuntimeError("runtime binding manifests have invalid inventories")
    if not all(isinstance(row, Mapping) for row in (*data_rows, *checkpoint_rows)):
        raise RuntimeError("runtime binding manifests contain non-object rows")
    data_ids = [str(row.get("scope")) for row in data_rows]
    checkpoint_model_ids = [str(row.get("model_id")) for row in checkpoint_rows]
    if len(data_ids) != len(set(data_ids)) or len(checkpoint_model_ids) != len(
        set(checkpoint_model_ids)
    ):
        raise RuntimeError("runtime binding manifests contain duplicate identities")
    data_by_scope = {str(row["scope"]): row for row in data_rows}
    checkpoints_by_model = {str(row["model_id"]): row for row in checkpoint_rows}
    if set(data_by_scope) != set(plan.get("scope_names", ())) or set(checkpoints_by_model) != {
        str(job["model_id"]) for job in plan["members"]
    }:
        raise RuntimeError("runtime binding manifest inventory drifted")
    scopes_by_name = {str(scope["name"]): scope for scope in plan["scopes"]}
    for scope_name, row in data_by_scope.items():
        scope = scopes_by_name[scope_name]
        if (
            row.get("dataset") != scope.get("dataset")
            or int(row.get("images", -1)) != int(scope.get("images", -2))
            or row.get("expected_sha256") != scope.get("manifest_sha256")
        ):
            raise RuntimeError(f"dataset binding contract drifted: {scope_name}")
        path_value = row.get("resolved_path")
        if path_value is None:
            if row.get("dataset") != "oracle":
                raise RuntimeError("formal dataset binding has no resolved path")
        else:
            path = Path(str(path_value))
            if not path.is_file() or sha256_file(path) != row.get("bytes_sha256"):
                raise RuntimeError(f"dataset binding bytes drifted: {path}")
            if row.get("bytes_sha256") != scope.get("manifest_sha256"):
                raise RuntimeError(f"dataset manifest digest drifted: {path}")
    expected_coverage = checkpoint_coverage(tuple(sorted(checkpoints_by_model)))
    formal = str(plan.get("profile_key", plan.get("profile"))) != "smoke"
    for model_id, row in checkpoints_by_model.items():
        records = row.get("checkpoints")
        checkpoint_ids = row.get("checkpoint_ids")
        expected_ids = list(expected_coverage[model_id])
        if (
            not isinstance(checkpoint_ids, list)
            or checkpoint_ids != expected_ids
            or len(checkpoint_ids) != len(set(checkpoint_ids))
            or not isinstance(records, list)
            or not all(isinstance(record, Mapping) for record in records)
        ):
            raise RuntimeError("checkpoint binding row is invalid")
        record_ids = [str(record.get("checkpoint_id")) for record in records]
        resolved_paths = [str(record.get("resolved_path")) for record in records]
        if (
            len(record_ids) != len(set(record_ids))
            or (formal and record_ids != expected_ids)
            or (not formal and record_ids)
            or len(resolved_paths) != len(set(resolved_paths))
        ):
            raise RuntimeError(f"checkpoint binding inventory drifted: {model_id}")
        for record in records:
            path = Path(str(record.get("resolved_path", "")))
            if not path.is_file() or sha256_file(path) != record.get("bytes_sha256"):
                raise RuntimeError(f"checkpoint binding bytes drifted: {path}")
    job_bindings: dict[str, dict[str, str | None]] = {}
    for job in plan["members"]:
        data_row = data_by_scope[str(job["scope"])]
        checkpoint_row = checkpoints_by_model[str(job["model_id"])]
        checkpoint_bytes_sha256 = canonical_sha256(checkpoint_row.get("checkpoints", []))
        input_manifest_sha256 = canonical_sha256(
            {
                "dataset_manifest_bytes_sha256": data_row.get("bytes_sha256"),
                "checkpoint_bytes_sha256": checkpoint_bytes_sha256,
                "input_contract_sha256": job["input_contract_sha256"],
                "image_start": job["image_start"],
                "image_stop": job["image_stop"],
            }
        )
        job_bindings[str(job["member_id"])] = {
            "dataset_manifest_bytes_sha256": data_row.get("bytes_sha256"),
            "checkpoint_bytes_sha256": checkpoint_bytes_sha256,
            "input_manifest_sha256": input_manifest_sha256,
        }
    data_path = context.path / "manifests/data.json"
    checkpoint_path = context.path / "manifests/checkpoints.json"
    return {
        "jobs": job_bindings,
        "data_binding_manifest_sha256": sha256_file(data_path),
        "checkpoint_binding_manifest_sha256": sha256_file(checkpoint_path),
    }


def _resolve_runtime_bindings(context: RunContext, plan: Mapping[str, Any]) -> dict[str, Any]:
    execution = _execution_config(context)
    formal = str(plan.get("profile_key", plan.get("profile"))) != "smoke"
    dataset_root_env = str(execution.get("dataset_root_env", "DECAF_DATA_ROOT"))
    checkpoint_root_env = str(execution.get("checkpoint_root_env", "DECAF_CACHE_ROOT"))
    data_rows: list[dict[str, Any]] = []
    for scope in plan["scopes"]:
        expected = scope.get("manifest_sha256")
        if formal:
            path = _configured_binding(
                context,
                mapping_name="dataset_manifests",
                identities=(
                    str(scope["name"]),
                    str(expected),
                    str(scope["dataset"]),
                ),
                root_env_name=dataset_root_env,
            )
            observed = sha256_file(path)
            if observed != expected:
                raise RuntimeError(
                    f"dataset manifest bytes do not match {scope['name']}: {observed} != {expected}"
                )
            resolved_path: str | None = str(path)
        else:
            observed = None
            resolved_path = None
        data_rows.append(
            {
                "scope": scope["name"],
                "dataset": scope["dataset"],
                "images": scope["images"],
                "expected_sha256": expected,
                "resolved_path": resolved_path,
                "bytes_sha256": observed,
            }
        )
    model_ids = tuple(sorted({str(job["model_id"]) for job in plan["members"]}))
    checkpoint_rows: list[dict[str, Any]] = []
    for model_id, checkpoint_ids in checkpoint_coverage(model_ids).items():
        records: list[dict[str, str]] = []
        for checkpoint_id in checkpoint_ids:
            if not formal:
                continue
            path = _configured_binding(
                context,
                mapping_name="checkpoint_files",
                identities=(checkpoint_id,),
                root_env_name=checkpoint_root_env,
            )
            records.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "resolved_path": str(path),
                    "bytes_sha256": sha256_file(path),
                }
            )
        checkpoint_rows.append(
            {
                "model_id": model_id,
                "checkpoint_ids": list(checkpoint_ids),
                "checkpoints": records,
            }
        )
    common = {
        "schema_version": 1,
        "resolved": True,
        "execution_claimed": False,
        "plan_contract_sha256": plan["plan_contract_sha256"],
        "config_sha256": plan["config_sha256"],
    }
    data_payload = {**common, "items": data_rows}
    checkpoint_payload = {**common, "items": checkpoint_rows}
    atomic_json(context.path / "manifests/data.json", data_payload)
    atomic_json(context.path / "manifests/checkpoints.json", checkpoint_payload)
    return _runtime_from_manifests(context, plan, data_payload, checkpoint_payload)


def _load_runtime_bindings(context: RunContext, plan: Mapping[str, Any]) -> dict[str, Any]:
    data_path = context.path / "manifests/data.json"
    checkpoint_path = context.path / "manifests/checkpoints.json"
    try:
        data_payload = json.loads(data_path.read_text(encoding="utf-8"))
        checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("runtime binding manifests are unreadable") from error
    return _runtime_from_manifests(context, plan, data_payload, checkpoint_payload)


def prepare(context: RunContext) -> dict[str, Any]:
    """Persist the exact plan and portable data/checkpoint bindings."""

    plan = build_plan(context.config)
    atomic_json(context.path / "manifests/plan.json", plan)
    job_lines = "".join(
        json.dumps(member, sort_keys=True, separators=(",", ":")) + "\n"
        for member in plan["members"]
    )
    atomic_text(context.path / "manifests/jobs.jsonl", job_lines)
    scopes = plan["scopes"]
    common = {
        "schema_version": 1,
        "resolved": False,
        "execution_claimed": False,
        "plan_contract_sha256": plan["plan_contract_sha256"],
        "config_sha256": plan["config_sha256"],
    }
    data_items = [
        {
            "scope": scope["name"],
            "dataset": scope["dataset"],
            "images": scope["images"],
            "expected_sha256": scope["manifest_sha256"],
            "resolved_path": None,
            "bytes_sha256": None,
        }
        for scope in scopes
    ]
    atomic_json(
        context.path / "manifests/data.json",
        {**common, "items": data_items},
    )
    model_ids = tuple(sorted({member["model_id"] for member in plan["members"]}))
    coverage = checkpoint_coverage(model_ids)
    atomic_json(
        context.path / "manifests/checkpoints.json",
        {
            **common,
            "items": [
                {
                    "model_id": model_id,
                    "checkpoint_ids": list(checkpoint_ids),
                    "checkpoints": [],
                }
                for model_id, checkpoint_ids in coverage.items()
            ],
        },
    )
    return {
        "member_count": plan["member_count"],
        "scope_names": plan["scope_names"],
        "plan_audit_passed": plan["audit"]["passed"],
        "endpoint_m_stage": plan["endpoint_m_stage"],
    }


def oracle_member(job: Mapping[str, Any], _context: RunContext) -> pd.DataFrame:
    """Run a deterministic, real DECAF score-oracle member on CPU."""

    if job.get("scope") != "oracle" or job.get("method_id") != "decaf_5":
        raise ValueError("the CPU oracle only accepts the smoke member")
    grid = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    base = np.linspace(-1.35, 1.45, 16, dtype=np.float64)
    base += np.where(np.arange(16) % 2 == 0, -0.07, 0.09)
    rows: list[dict[str, Any]] = []
    for image_index in range(8):
        generator = np.random.default_rng(12_001 + image_index)
        endpoint = base + generator.normal(0.0, 0.045, size=16)
        profile = generator.normal(0.0, 0.18, size=16)
        response = (
            grid[:, None] * endpoint[None, :] + np.sin(np.pi * grid)[:, None] * profile[None, :]
        )
        scores = decaf_trajectory("decaf_5", grid, response, endpoint, axis=0)
        patch_scores = np.asarray(scores["signed_E"], dtype=np.float64)
        quality = float(row_spearman(patch_scores, endpoint)[0])
        rows.append(
            {
                "image_index": image_index,
                "dataset": "oracle",
                "scope": "oracle",
                "model": "oracle_linear",
                "method": "decaf_5",
                "image_id": f"oracle-{image_index:03d}",
                "spearman": quality,
                "patch_scores": patch_scores,
                "endpoint_effects": endpoint,
                "quality_target_effects": endpoint.copy(),
                "decaf_M": np.asarray(scores["M"], dtype=np.float64),
                "decaf_E": np.asarray(scores["E"], dtype=np.float64),
                "decaf_C": np.asarray(scores["C"], dtype=np.float64),
                "decaf_F": np.asarray(scores["F"], dtype=np.float64),
                "decaf_Abs": np.asarray(scores["Abs"], dtype=np.float64),
                "finite_complete": True,
                "numeric_audit_passed": bool(scores["numeric_audit"]["passed"]),
            }
        )
    result = pd.DataFrame(rows)
    if not result["numeric_audit_passed"].all():
        raise AssertionError("the attribution score oracle failed a core numeric audit")
    return result


def _load_adapter(config: Mapping[str, Any]) -> MemberEvaluator:
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise TypeError("execution configuration must be a mapping")
    adapter = execution.get("adapter")
    if not isinstance(adapter, str) or ":" not in adapter:
        raise RuntimeError(
            "formal GPU compute requires execution.adapter='module:function'; "
            "use --plan-only on a CPU-only machine"
        )
    require_gpu_runtime()
    module_name, function_name = adapter.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise RuntimeError(f"attribution adapter is not callable: {adapter}")
    return function


QUALITY_KINDS = frozenset(
    {
        "cpu_score_oracle",
        "quality",
        "quality_supplement",
        "scale_check",
        "large_model_quality",
        "boundary_quality",
    }
)
TIMING_KINDS = frozenset({"timing", "large_model_timing"})
TARGET_KINDS = frozenset(
    {
        "shared_deletion_targets",
        "shared_part_deletion_targets",
        "shared_heldout_targets",
    }
)


def _vector(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with at least two values")
    return result


def _constant_column(frame: pd.DataFrame, column: str, expected: Any) -> None:
    if column not in frame.columns:
        raise ValueError(f"member output is missing lineage column: {column}")
    if expected is None:
        if not frame[column].isna().all():
            raise ValueError(f"member lineage drifted: {column}")
    elif not frame[column].map(lambda value: value == expected).all():
        raise ValueError(f"member lineage drifted: {column}")


def _lineage_values(
    job: Mapping[str, Any],
    binding: Mapping[str, Any],
    dependency_records: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "member_id": job["member_id"],
        "job_sha256": job["job_sha256"],
        "config_sha256": job["config_sha256"],
        "plan_contract_sha256": job["plan_contract_sha256"],
        "dataset_manifest_contract_sha256": job["dataset_manifest_sha256"],
        "dataset_manifest_bytes_sha256": binding["dataset_manifest_bytes_sha256"],
        "checkpoint_contract_sha256": job["checkpoint_contract_sha256"],
        "checkpoint_bytes_sha256": binding["checkpoint_bytes_sha256"],
        "input_contract_sha256": job["input_contract_sha256"],
        "input_manifest_sha256": binding["input_manifest_sha256"],
        "output_schema": job["output_schema"],
        "dependency_outputs_json": json.dumps(
            dependency_records, sort_keys=True, separators=(",", ":")
        ),
        "dependency_outputs_sha256": canonical_sha256(dependency_records),
    }


def _attach_lineage(
    frame: pd.DataFrame,
    job: Mapping[str, Any],
    binding: Mapping[str, Any],
    dependency_records: list[dict[str, str]],
) -> pd.DataFrame:
    result = frame.copy()
    for column, value in _lineage_values(job, binding, dependency_records).items():
        result[column] = value
    return result


def _validate_member_frame(
    frame: pd.DataFrame,
    job: Mapping[str, Any],
    binding: Mapping[str, Any],
    dependency_records: list[dict[str, str]],
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("member output must be a non-empty DataFrame")
    kind = str(job["kind"])
    required_identity = {"scope", "dataset", "model", "method"}
    missing_identity = sorted(required_identity - set(frame.columns))
    if missing_identity:
        raise ValueError(f"member output is missing identity columns: {missing_identity}")
    for column, expected in (
        ("scope", job["scope"]),
        ("dataset", job["dataset"]),
        ("model", job["model_id"]),
        ("method", job["method_id"]),
    ):
        _constant_column(frame, column, expected)
    if kind in TIMING_KINDS:
        required = {
            "repeat",
            "wall_seconds_per_image",
            "peak_allocated_bytes",
            "forward_rows_per_image",
            "backward_calls_per_image",
        }
        missing = sorted(required - set(frame.columns))
        if missing or len(frame) != 1:
            raise ValueError(f"timing member schema drifted; missing={missing}")
        _constant_column(frame, "repeat", int(job["repeat"]))
        values = frame.loc[:, sorted(required - {"repeat"})].apply(pd.to_numeric, errors="coerce")
        array = values.to_numpy(dtype=np.float64)
        if not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError("timing member values must be finite and non-negative")
    else:
        required = {"image_index", "image_id"}
        missing = sorted(required - set(frame.columns))
        if missing or len(frame) != int(job["image_count"]):
            raise ValueError(f"image member schema/count drifted; missing={missing}")
        indices = pd.to_numeric(frame["image_index"], errors="coerce").to_numpy()
        expected_indices = np.arange(
            int(job["image_start"]), int(job["image_stop"]), dtype=np.int64
        )
        if not np.array_equal(indices, expected_indices):
            raise ValueError("member image_index inventory is not the exact job range")
        image_ids = frame["image_id"].astype(str)
        if image_ids.eq("").any() or image_ids.duplicated().any():
            raise ValueError("member image_id inventory is invalid")
        if kind in TARGET_KINDS:
            if "target_effects" not in frame.columns:
                raise ValueError("target member is missing target_effects")
            for value in frame["target_effects"]:
                _vector(value, name="target_effects")
        elif kind in QUALITY_KINDS:
            required_quality = {
                "spearman",
                "patch_scores",
                "decaf_M",
                "endpoint_effects",
                "quality_target_effects",
                "finite_complete",
            }
            missing = sorted(required_quality - set(frame.columns))
            if missing:
                raise ValueError(f"quality member schema drifted; missing={missing}")
            if not frame["finite_complete"].map(bool).all():
                raise ValueError("quality member contains incomplete rows")
            if (
                "numeric_audit_passed" in frame.columns
                and not frame["numeric_audit_passed"].map(bool).all()
            ):
                raise ValueError("quality member contains failed numeric audits")
            observed = pd.to_numeric(frame["spearman"], errors="coerce").to_numpy(dtype=np.float64)
            heldout_contract = "quality_aggregation" in frame.columns
            if heldout_contract:
                required_heldout = {
                    *_HELDOUT_EFFECT_COLUMNS,
                    *_HELDOUT_SPEARMAN_COLUMNS,
                }
                missing_heldout = sorted(required_heldout - set(frame.columns))
                if missing_heldout:
                    raise ValueError(
                        f"held-out quality provenance is incomplete: {missing_heldout}"
                    )
                _constant_column(
                    frame,
                    "quality_aggregation",
                    _HELDOUT_QUALITY_AGGREGATION,
                )
            expected_quality: list[float] = []
            for row in frame.itertuples(index=False):
                patch = _vector(row.patch_scores, name="patch_scores")
                endpoint_m = _vector(row.decaf_M, name="decaf_M")
                endpoint = _vector(row.endpoint_effects, name="endpoint_effects")
                target = _vector(
                    row.quality_target_effects,
                    name="quality_target_effects",
                )
                if endpoint_m.shape != endpoint.shape or not np.allclose(
                    endpoint_m, np.abs(endpoint), atol=1.0e-12, rtol=0.0
                ):
                    raise ValueError("decaf_M is not the endpoint magnitude")
                if patch.shape != target.shape:
                    raise ValueError("patch scores and quality target shapes differ")
                if heldout_contract:
                    operator_effects = [
                        _vector(getattr(row, column), name=column)
                        for column in _HELDOUT_EFFECT_COLUMNS
                    ]
                    if any(value.shape != patch.shape for value in operator_effects):
                        raise ValueError("held-out operator and patch score shapes differ")
                    operator_scores = [
                        float(row_spearman(patch, value)[0]) for value in operator_effects
                    ]
                    recorded_scores = [
                        float(getattr(row, column)) for column in _HELDOUT_SPEARMAN_COLUMNS
                    ]
                    if not np.allclose(
                        recorded_scores,
                        operator_scores,
                        atol=1.0e-12,
                        rtol=0.0,
                    ):
                        raise ValueError("held-out operator Spearman provenance drifted")
                    expected_quality.append(float(np.mean(operator_scores)))
                else:
                    expected_quality.append(float(row_spearman(patch, target)[0]))
            expected_array = np.asarray(expected_quality, dtype=np.float64)
            if (
                not np.isfinite(observed).all()
                or np.any(np.abs(observed) > 1.0 + 1.0e-12)
                or not np.allclose(observed, expected_array, atol=1.0e-12, rtol=0.0)
            ):
                raise ValueError("persisted Spearman quality does not match its target")
        else:
            raise ValueError(f"unsupported attribution member kind: {kind}")
    for column, expected in _lineage_values(job, binding, dependency_records).items():
        _constant_column(frame, column, expected)


def _load_completed_dependency(
    context: RunContext,
    dependency: Mapping[str, Any],
    jobs_by_id: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, str]]:
    member_id = str(dependency["member_id"])
    target_job = jobs_by_id.get(member_id)
    if target_job is None or target_job["job_sha256"] != dependency["job_sha256"]:
        raise RuntimeError(f"dependency job binding drifted: {member_id}")
    output_path = _contained(context.path, str(dependency["output_path"]))
    receipt_path = _contained(context.path, str(dependency["receipt_path"]))
    if not output_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"dependency output is incomplete: {member_id}")
    receipt = load_member_receipt(receipt_path)
    details = receipt.get("details")
    output_sha256 = sha256_file(output_path)
    if (
        receipt.get("status") != "completed"
        or receipt.get("member_id") != member_id
        or not isinstance(details, Mapping)
        or details.get("job_sha256") != target_job["job_sha256"]
        or details.get("output_sha256") != output_sha256
    ):
        raise RuntimeError(f"dependency receipt drifted: {member_id}")
    frame = pd.read_parquet(output_path)
    _validate_member_frame(frame, target_job, runtime["jobs"][member_id], [])
    return frame, {
        "member_id": member_id,
        "job_sha256": str(target_job["job_sha256"]),
        "output_sha256": output_sha256,
        "relationship": str(dependency["relationship"]),
    }


def _bind_dependency_targets(
    context: RunContext,
    job: Mapping[str, Any],
    frame: pd.DataFrame,
    jobs_by_id: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    dependencies = job.get("depends_on", [])
    if not isinstance(dependencies, list):
        raise RuntimeError("member dependency list is invalid")
    if not dependencies:
        return frame.copy(), []
    result = frame.copy()
    endpoint_values: list[np.ndarray] | None = None
    heldout_values: dict[str, list[np.ndarray]] = {}
    records: list[dict[str, str]] = []
    for dependency in dependencies:
        target, record = _load_completed_dependency(context, dependency, jobs_by_id, runtime)
        if not np.array_equal(
            result["image_index"].to_numpy(), target["image_index"].to_numpy()
        ) or not np.array_equal(
            result["image_id"].astype(str).to_numpy(),
            target["image_id"].astype(str).to_numpy(),
        ):
            raise RuntimeError(f"dependency image inventory drifted: {record['member_id']}")
        values = [_vector(value, name="target_effects") for value in target["target_effects"]]
        method_id = str(dependency["method_id"])
        if method_id in {DELETION_TARGET_METHOD, FUNNYBIRDS_DELETION_TARGET_METHOD}:
            if endpoint_values is not None:
                raise RuntimeError("member has multiple endpoint target dependencies")
            endpoint_values = values
        elif method_id in FUNNYBIRDS_HELDOUT_METHODS:
            if method_id in heldout_values:
                raise RuntimeError("member has duplicate held-out target dependencies")
            heldout_values[method_id] = values
        else:
            raise RuntimeError(f"unknown dependency target role: {method_id}")
        records.append(record)
    if endpoint_values is None:
        raise RuntimeError("quality member has no endpoint target dependency")
    if heldout_values:
        if set(heldout_values) != set(FUNNYBIRDS_HELDOUT_METHODS):
            raise RuntimeError("held-out quality requires exactly two target operators")
        background_values = heldout_values["__heldout_background_texture__"]
        telea_values = heldout_values["__heldout_telea_dilate3__"]
        quality_values: list[np.ndarray] = []
        for first, second in zip(background_values, telea_values, strict=True):
            if first.shape != second.shape:
                raise RuntimeError("held-out target operator shapes differ")
            # Retain the mean vector as a diagnostic/backward-compatible
            # target column, but never use it as the metric basis: Spearman is
            # nonlinear and the frozen study averages the two correlations.
            quality_values.append((first + second) / 2.0)
    else:
        quality_values = [value.copy() for value in endpoint_values]
    result["endpoint_effects"] = endpoint_values
    result["quality_target_effects"] = quality_values
    result["decaf_M"] = [np.abs(value) for value in endpoint_values]
    if "patch_scores" in result.columns:
        patches = [_vector(value, name="patch_scores") for value in result["patch_scores"]]
        if heldout_values:
            background_scores, telea_scores, averaged_scores = _operatorwise_heldout_spearman(
                patches, background_values, telea_values
            )
            result[_HELDOUT_EFFECT_COLUMNS[0]] = background_values
            result[_HELDOUT_EFFECT_COLUMNS[1]] = telea_values
            result[_HELDOUT_SPEARMAN_COLUMNS[0]] = background_scores
            result[_HELDOUT_SPEARMAN_COLUMNS[1]] = telea_scores
            result["quality_aggregation"] = _HELDOUT_QUALITY_AGGREGATION
            result["quality_target_effects_role"] = (
                "diagnostic_operator_effect_mean_not_metric_basis"
            )
            result["spearman"] = averaged_scores
        else:
            result["spearman"] = [
                float(row_spearman(patch, target)[0])
                for patch, target in zip(patches, quality_values, strict=True)
            ]
    return result, records


def _receipt_details(
    frame: pd.DataFrame,
    output_path: Path,
    job: Mapping[str, Any],
    binding: Mapping[str, Any],
    runtime: Mapping[str, Any],
    dependency_records: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "output_path": str(job["output_path"]),
        "output_sha256": sha256_file(output_path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "scope": job["scope"],
        "dataset": job["dataset"],
        "model_id": job["model_id"],
        "method_id": job["method_id"],
        "job_sha256": job["job_sha256"],
        "config_sha256": job["config_sha256"],
        "plan_contract_sha256": job["plan_contract_sha256"],
        "checkpoint_contract_sha256": job["checkpoint_contract_sha256"],
        "input_contract_sha256": job["input_contract_sha256"],
        "output_schema": job["output_schema"],
        **dict(binding),
        "data_binding_manifest_sha256": runtime["data_binding_manifest_sha256"],
        "checkpoint_binding_manifest_sha256": runtime["checkpoint_binding_manifest_sha256"],
        "dependency_outputs": dependency_records,
        "dependency_outputs_sha256": canonical_sha256(dependency_records),
    }


def _validate_completed_member(
    context: RunContext,
    job: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = _contained(context.path, str(job["receipt_path"]))
    output_path = _contained(context.path, str(job["output_path"]))
    if not receipt_path.is_file() or not output_path.is_file():
        raise FileNotFoundError(f"member artifacts are incomplete: {job['member_id']}")
    receipt = load_member_receipt(receipt_path)
    details = receipt.get("details")
    if (
        receipt.get("status") != "completed"
        or receipt.get("member_id") != job["member_id"]
        or not isinstance(details, Mapping)
    ):
        raise RuntimeError(f"member receipt is not completed: {job['member_id']}")
    frame = pd.read_parquet(output_path)
    dependency_records = details.get("dependency_outputs")
    if not isinstance(dependency_records, list) or not all(
        isinstance(record, Mapping) for record in dependency_records
    ):
        raise RuntimeError(f"member dependency receipt drifted: {job['member_id']}")
    normalized_records = [dict(record) for record in dependency_records]
    expected_dependency_records: list[dict[str, str]] = []
    for dependency in job.get("depends_on", []):
        dependency_output = _contained(context.path, str(dependency["output_path"]))
        dependency_receipt_path = _contained(context.path, str(dependency["receipt_path"]))
        if not dependency_output.is_file() or not dependency_receipt_path.is_file():
            raise FileNotFoundError(f"member dependency is incomplete: {dependency['member_id']}")
        dependency_receipt = load_member_receipt(dependency_receipt_path)
        dependency_details = dependency_receipt.get("details")
        dependency_hash = sha256_file(dependency_output)
        if (
            dependency_receipt.get("status") != "completed"
            or dependency_receipt.get("member_id") != dependency["member_id"]
            or not isinstance(dependency_details, Mapping)
            or dependency_details.get("job_sha256") != dependency["job_sha256"]
            or dependency_details.get("output_sha256") != dependency_hash
        ):
            raise RuntimeError(f"member dependency receipt drifted: {dependency['member_id']}")
        expected_dependency_records.append(
            {
                "member_id": str(dependency["member_id"]),
                "job_sha256": str(dependency["job_sha256"]),
                "output_sha256": dependency_hash,
                "relationship": str(dependency["relationship"]),
            }
        )
    if normalized_records != expected_dependency_records:
        raise RuntimeError(f"member dependency lineage drifted: {job['member_id']}")
    _validate_member_frame(frame, job, runtime["jobs"][str(job["member_id"])], normalized_records)
    expected = _receipt_details(
        frame,
        output_path,
        job,
        runtime["jobs"][str(job["member_id"])],
        runtime,
        normalized_records,
    )
    for key, value in expected.items():
        if details.get(key) != value:
            raise RuntimeError(f"member receipt field drifted: {job['member_id']}:{key}")
    return receipt


def _completed_member(
    context: RunContext,
    job: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> bool:
    receipt_path = _contained(context.path, str(job["receipt_path"]))
    output_path = _contained(context.path, str(job["output_path"]))
    if not receipt_path.exists() and not output_path.exists():
        return False
    if receipt_path.is_file():
        receipt = load_member_receipt(receipt_path)
        if receipt.get("status") in {"running", "failed", "skipped"}:
            return False
    elif output_path.exists():
        return False
    _validate_completed_member(context, job, runtime)
    return True


def run_member(
    context: RunContext,
    job: Mapping[str, Any],
    evaluator: MemberEvaluator,
    *,
    jobs_by_id: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Run one member with atomic output and terminal receipt semantics."""

    member_id = str(job["member_id"])
    receipt_path = _contained(context.path, str(job["receipt_path"]))
    output_path = _contained(context.path, str(job["output_path"]))
    if context.resume and _completed_member(context, job, runtime):
        return "skipped", load_member_receipt(receipt_path)
    started_at = utc_now()
    write_member_receipt(receipt_path, member_id, "running", started_at=started_at)
    try:
        frame = evaluator(job, context)
        frame, dependency_records = _bind_dependency_targets(
            context, job, frame, jobs_by_id, runtime
        )
        frame = _attach_lineage(
            frame,
            job,
            runtime["jobs"][member_id],
            dependency_records,
        )
        _validate_member_frame(
            frame,
            job,
            runtime["jobs"][member_id],
            dependency_records,
        )
        atomic_parquet(frame, output_path)
        persisted = pd.read_parquet(output_path)
        _validate_member_frame(
            persisted,
            job,
            runtime["jobs"][member_id],
            dependency_records,
        )
        details = _receipt_details(
            persisted,
            output_path,
            job,
            runtime["jobs"][member_id],
            runtime,
            dependency_records,
        )
        write_member_receipt(
            receipt_path,
            member_id,
            "completed",
            started_at=started_at,
            details=details,
        )
        return "completed", load_member_receipt(receipt_path)
    except Exception as error:
        write_member_receipt(
            receipt_path,
            member_id,
            "failed",
            started_at=started_at,
            error=f"{type(error).__name__}: {error}",
        )
        raise


_GPU_METHOD_ORDER = {
    DELETION_TARGET_METHOD: 0,
    FUNNYBIRDS_DELETION_TARGET_METHOD: 0,
    "__heldout_background_texture__": 1,
    "__heldout_telea_dilate3__": 2,
    "decaf_3": 10,
    "decaf_5": 11,
    "decaf_9": 12,
    "ig_16": 20,
    "ig_32": 21,
    "ig_u_32": 22,
    "gradient_shap": 23,
    "smoothgrad_16": 24,
    "deep_lift": 25,
    "part_occlusion": 30,
    "exact_part_shapley": 31,
    "rise_512": 40,
    "kernel_shap_512": 41,
}


def _gpu_queue_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    """Keep one model resident while still preferring short receipts first."""

    kind = str(job["kind"])
    kind_order = 0 if kind in TARGET_KINDS else (2 if kind in TIMING_KINDS else 1)
    return (
        str(job["dataset"]),
        str(job["model_id"]),
        kind_order,
        _GPU_METHOD_ORDER.get(str(job["method_id"]), 100),
        int(job["repeat"]),
        int(job["shard"]),
        str(job["member_id"]),
    )


def _gpu_global_details(
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    backend: str,
    queue_events: Sequence[Mapping[str, Any]],
    failures: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "backend": backend,
        "endpoint_m_stage": "analyze",
        "member_count": len(plan["members"]),
        "plan_contract_sha256": plan["plan_contract_sha256"],
        "config_sha256": plan["config_sha256"],
        "data_binding_manifest_sha256": runtime["data_binding_manifest_sha256"],
        "checkpoint_binding_manifest_sha256": runtime["checkpoint_binding_manifest_sha256"],
        "scheduler": "single_gpu_dynamic_queue",
        "visible_device": "cuda:0",
        "exclusive_member_concurrency": 1,
        "dynamic_refill": True,
        "duplicate_execution": False,
        "queue_events": list(queue_events),
        "failures": dict(sorted(failures.items())),
        "multi_gpu_real_execution": "NOT_TESTED_SINGLE_GPU_NODE",
    }


def _compute_single_gpu_queue(
    context: RunContext,
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    evaluator: MemberEvaluator,
    *,
    backend: str,
) -> dict[str, Any]:
    """Run a dependency-aware, failure-isolating queue at concurrency one."""

    jobs = [dict(job) for job in plan["members"]]
    jobs_by_id = {str(job["member_id"]): job for job in jobs}
    expected = [str(job["member_id"]) for job in jobs]
    receipts: dict[str, dict[str, Any]] = {}
    successful: set[str] = set()
    statuses = Counter()
    failures: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}

    for job in jobs:
        member_id = str(job["member_id"])
        if context.resume and _completed_member(context, job, runtime):
            receipt_path = _contained(context.path, str(job["receipt_path"]))
            receipts[member_id] = load_member_receipt(receipt_path)
            successful.add(member_id)
            statuses["skipped"] += 1
            events.append({"member_id": member_id, "event": "resume_skip", "device": 0})
        else:
            pending[member_id] = job

    interrupted: BaseException | None = None
    while pending and interrupted is None:
        ready: list[dict[str, Any]] = []
        blocked_by_failure: list[tuple[dict[str, Any], list[str]]] = []
        for job in pending.values():
            dependencies = [str(value["member_id"]) for value in job.get("depends_on", [])]
            failed_dependencies = [value for value in dependencies if value in failures]
            if failed_dependencies:
                blocked_by_failure.append((job, failed_dependencies))
            elif all(value in successful for value in dependencies):
                ready.append(job)

        for job, dependencies in blocked_by_failure:
            member_id = str(job["member_id"])
            error = f"dependency members failed: {dependencies}"
            receipt_path = _contained(context.path, str(job["receipt_path"]))
            write_member_receipt(
                receipt_path,
                member_id,
                "failed",
                error=f"RuntimeError: {error}",
            )
            receipts[member_id] = load_member_receipt(receipt_path)
            failures[member_id] = error
            statuses["failed"] += 1
            events.append(
                {
                    "member_id": member_id,
                    "event": "dependency_failure_isolated",
                    "device": 0,
                }
            )
            pending.pop(member_id)

        if not ready:
            if pending and not blocked_by_failure:
                unresolved = sorted(pending)
                interrupted = RuntimeError(
                    f"single-GPU queue has an unresolved dependency cycle: {unresolved}"
                )
            continue

        job = min(ready, key=_gpu_queue_key)
        member_id = str(job["member_id"])
        events.append({"member_id": member_id, "event": "start", "device": 0})
        try:
            status, receipt = run_member(
                context,
                job,
                evaluator,
                jobs_by_id=jobs_by_id,
                runtime=runtime,
            )
        except TerminationRequested as error:
            receipt_path = _contained(context.path, str(job["receipt_path"]))
            if receipt_path.is_file():
                receipts[member_id] = load_member_receipt(receipt_path)
            failures[member_id] = str(error)
            statuses["failed"] += 1
            events.append({"member_id": member_id, "event": "terminated", "device": 0})
            pending.pop(member_id)
            interrupted = error
        except Exception as error:
            receipt_path = _contained(context.path, str(job["receipt_path"]))
            if receipt_path.is_file():
                receipts[member_id] = load_member_receipt(receipt_path)
            message = f"{type(error).__name__}: {error}"
            failures[member_id] = message
            statuses["failed"] += 1
            events.append({"member_id": member_id, "event": "failure_isolated", "device": 0})
            pending.pop(member_id)
        else:
            receipts[member_id] = receipt
            successful.add(member_id)
            statuses[status] += 1
            events.append({"member_id": member_id, "event": status, "device": 0})
            pending.pop(member_id)

    global_path = context.path / "receipts/compute_members.json"
    finalize_global_receipt(
        global_path,
        context.path.name,
        receipts,
        expected_members=expected,
        details=_gpu_global_details(
            plan,
            runtime,
            backend=backend,
            queue_events=events,
            failures=failures,
        ),
    )
    if interrupted is not None:
        raise interrupted
    if failures:
        raise RuntimeError(
            f"{len(failures)} attribution members failed in isolation; see {global_path}"
        )
    return {
        "backend": backend,
        "completed_members": statuses["completed"],
        "resumed_members": statuses["skipped"],
        "failed_members": statuses["failed"],
        "member_count": len(expected),
        "scheduler": "single_gpu_dynamic_queue",
        "device": 0,
    }


def _finalize_gpu_queue_after_termination(
    context: RunContext,
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    backend: str,
    error: TerminationRequested,
) -> None:
    """Close queue receipts if SIGTERM lands outside an active member call."""

    global_path = context.path / "receipts/compute_members.json"
    if global_path.is_file():
        try:
            current = json.loads(global_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
        if current.get("all_processes_exited") is True and current.get("status") != "running":
            return

    receipts: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for job in plan["members"]:
        member_id = str(job["member_id"])
        receipt_path = _contained(context.path, str(job["receipt_path"]))
        if not receipt_path.is_file():
            continue
        receipt = load_member_receipt(receipt_path)
        if receipt.get("status") == "running":
            message = f"{type(error).__name__}: {error}"
            write_member_receipt(
                receipt_path,
                member_id,
                "failed",
                started_at=str(receipt["started_at"]),
                error=message,
            )
            receipt = load_member_receipt(receipt_path)
            failures[member_id] = message
        elif receipt.get("status") == "failed":
            failures[member_id] = str(receipt.get("error") or "member failed")
        receipts[member_id] = receipt

    finalize_global_receipt(
        global_path,
        context.path.name,
        receipts,
        expected_members=[str(job["member_id"]) for job in plan["members"]],
        details=_gpu_global_details(
            plan,
            runtime,
            backend=backend,
            queue_events=[
                {
                    "event": "sigterm_terminalize",
                    "device": 0,
                    "error": f"{type(error).__name__}: {error}",
                }
            ],
            failures=failures,
        ),
    )


def compute(context: RunContext) -> dict[str, Any]:
    """Execute resumable members through the oracle or a lazily loaded GPU adapter."""

    if not (context.path / "manifests/plan.json").is_file():
        if context.profile == "smoke-resume" and context.stage == "compute":
            prepare(context)
        else:
            raise RuntimeError("attribution compute requires a persisted prepare-stage plan")
    plan = _plan(context)
    runtime = _resolve_runtime_bindings(context, plan)
    execution = context.config.get("execution", {})
    backend = execution.get("backend", "gpu") if isinstance(execution, Mapping) else "gpu"
    evaluator = oracle_member if backend == "oracle" else _load_adapter(context.config)
    scheduler = execution.get("scheduler") if isinstance(execution, Mapping) else None
    if backend != "oracle" and scheduler == "single_gpu_dynamic_queue":
        try:
            return _compute_single_gpu_queue(
                context,
                plan,
                runtime,
                evaluator,
                backend=str(backend),
            )
        except TerminationRequested as error:
            _finalize_gpu_queue_after_termination(
                context,
                plan,
                runtime,
                backend=str(backend),
                error=error,
            )
            raise
    statuses = Counter()
    receipts: dict[str, dict[str, Any]] = {}
    jobs_by_id = {str(job["member_id"]): job for job in plan["members"]}
    for scope_name in plan["scope_names"]:
        scope_jobs = [job for job in plan["members"] if job["scope"] == scope_name]
        with ThreadPoolExecutor(max_workers=context.workers) as executor:
            futures = {
                executor.submit(
                    run_member,
                    context,
                    job,
                    evaluator,
                    jobs_by_id=jobs_by_id,
                    runtime=runtime,
                ): job
                for job in scope_jobs
            }
            for future in as_completed(futures):
                status, receipt = future.result()
                statuses[status] += 1
                receipts[str(futures[future]["member_id"])] = receipt
    expected = [str(job["member_id"]) for job in plan["members"]]
    finalize_global_receipt(
        context.path / "receipts/compute_members.json",
        context.path.name,
        receipts,
        expected_members=expected,
        details={
            "backend": backend,
            "endpoint_m_stage": "analyze",
            "member_count": len(expected),
            "plan_contract_sha256": plan["plan_contract_sha256"],
            "config_sha256": plan["config_sha256"],
            "data_binding_manifest_sha256": runtime["data_binding_manifest_sha256"],
            "checkpoint_binding_manifest_sha256": runtime["checkpoint_binding_manifest_sha256"],
        },
    )
    return {
        "backend": backend,
        "completed_members": statuses["completed"],
        "resumed_members": statuses["skipped"],
        "member_count": len(expected),
    }


def validate_compute_members(context: RunContext) -> dict[str, Any]:
    """Fail closed on the exact plan, runtime bindings, and member inventory."""

    plan = _plan(context)
    runtime = _load_runtime_bindings(context, plan)
    expected_outputs = {str(job["output_path"]) for job in plan["members"]}
    expected_receipts = {str(job["receipt_path"]) for job in plan["members"]}
    actual_outputs = {
        path.relative_to(context.path).as_posix()
        for path in (context.path / "raw/members").rglob("*")
        if path.is_file()
    }
    actual_receipts = {
        path.relative_to(context.path).as_posix()
        for path in (context.path / "receipts/members").rglob("*")
        if path.is_file()
    }
    if actual_outputs != expected_outputs or actual_receipts != expected_receipts:
        raise RuntimeError("attribution compute artifact inventory drifted")
    receipts = {
        str(job["member_id"]): _validate_completed_member(context, job, runtime)
        for job in plan["members"]
    }
    global_path = context.path / "receipts/compute_members.json"
    if not global_path.is_file():
        raise FileNotFoundError("global attribution member receipt is absent")
    global_receipt = json.loads(global_path.read_text(encoding="utf-8"))
    details = global_receipt.get("details")
    if (
        global_receipt.get("status") != "completed"
        or global_receipt.get("all_processes_exited") is not True
        or global_receipt.get("member_count") != len(plan["members"])
        or set(global_receipt.get("members", {})) != set(receipts)
        or any(
            row.get("status") != "completed" for row in global_receipt.get("members", {}).values()
        )
        or not isinstance(details, Mapping)
        or details.get("plan_contract_sha256") != plan["plan_contract_sha256"]
        or details.get("config_sha256") != plan["config_sha256"]
        or details.get("data_binding_manifest_sha256") != runtime["data_binding_manifest_sha256"]
        or details.get("checkpoint_binding_manifest_sha256")
        != runtime["checkpoint_binding_manifest_sha256"]
    ):
        raise RuntimeError("global attribution member receipt drifted")
    return {
        "member_count": len(receipts),
        "plan_contract_sha256": plan["plan_contract_sha256"],
        "config_sha256": plan["config_sha256"],
    }


def load_quality_members(context: RunContext) -> pd.DataFrame:
    """Load only per-image quality members, excluding timing and target caches."""

    validate_compute_members(context)
    root = context.path
    plan = _plan(context)
    paths = [
        _contained(root, str(job["output_path"]))
        for job in plan["members"]
        if job.get("kind") in QUALITY_KINDS
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"attribution quality member is absent: {path}")
        frame = pd.read_parquet(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no attribution quality members are available for analysis")
    result = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "scope",
        "dataset",
        "model",
        "method",
        "image_id",
        "spearman",
        "endpoint_effects",
        "quality_target_effects",
        "decaf_M",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"attribution quality members are missing columns: {missing}")
    keys = ["scope", "dataset", "model", "method", "image_id"]
    if result.duplicated(keys).any():
        raise ValueError("attribution quality members contain duplicate keys")
    result.loc[result["scope"].astype(str).eq("funnybirds_supplement"), "scope"] = (
        "funnybirds_primary"
    )
    if result.duplicated(keys).any():
        raise ValueError("normalized attribution quality members contain duplicate keys")
    return result


__all__ = [
    "MemberEvaluator",
    "atomic_parquet",
    "compute",
    "load_quality_members",
    "oracle_member",
    "prepare",
    "run_member",
    "validate_compute_members",
]
