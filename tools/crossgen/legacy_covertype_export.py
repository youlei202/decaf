"""Export exact historical Covertype trajectories for current-core verification.

The adapter is intentionally verification-only.  It loads immutable historical
data, mechanism caches, estimators, receipts, and sealed outputs from read-only
locations.  It never fits an estimator.  Historical code is used only to
recover exact identities and query the four legal binary support points; all
fresh DECAF routing and integration use the current ``decaf.core`` package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# The historical repository is a read-only evidence source. Suppress import-cache
# writes before importing any of its modules dynamically.
sys.dont_write_bytecode = True

CURRENT_REPOSITORY = Path(__file__).resolve().parents[2]
CURRENT_SOURCE = CURRENT_REPOSITORY / "src"
if str(CURRENT_SOURCE) not in sys.path:
    sys.path.insert(0, str(CURRENT_SOURCE))

from decaf.core.decomposition import decompose, endpoint_orientation  # noqa: E402
from decaf.core.quadrature import integrate_components  # noqa: E402
from tools.crossgen.schema import (  # noqa: E402
    NEUTRAL_COLUMNS,
    sha256_file,
    trapezoid_weights,
    write_trajectory_record,
)

MODEL_FAMILIES = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "mlp",
)
SUMMARY_NAMES = ("M", "E", "C", "F", "Abs")
MECHANISM_NAMES = ("E", "C", "F")
REFERENCE_RUN = "decaf_covertype_v1_formal"
SELECTION_NAMESPACE = "decaf-crossgen-v2|covertype|sealed-test-source-index"
HISTORICAL_PACKAGE = Path(
    "/work/Users/leiyo/decaf_covertype_v1_results/packages/"
    "decaf_covertype_v1_20260811T090808Z_lightweight.zip"
)
HISTORICAL_PACKAGE_SHA256 = (
    "e9acaf30491dcdf654fdfb691df915e19d75e9c19d0ffe2546312d0d34f87927"
)
HISTORICAL_PACKAGE_MANIFEST_MEMBER = "PACKAGE_CONTENTS.json"
HISTORICAL_PACKAGE_CODE_PREFIX = "code/src"
HISTORICAL_MODULE_NAMESPACE = "cmr.decaf_covertype_v1"
PARENT_PACKAGE_SHIM = (
    b'"""Verification-only parent package for the sealed Covertype snapshot."""\n'
)
TIER_A_ATOL = 5.0e-4
TIER_A_RTOL = 5.0e-3
TIER_B_ATOL = 2.0e-3
HARD_ATOL = 1.0e-2
BOUNDARY_ATOL = 5.0e-4


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _clean_json(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_clean_json(dict(value)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_receipt(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = _read_json(path)
    if receipt.get("status") != "completed":
        raise AssertionError(f"receipt is not completed: {path}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AssertionError(f"receipt contains no artifacts: {path}")
    verified: list[dict[str, Any]] = []
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise TypeError(f"invalid receipt artifact: {path}")
        artifact = Path(str(raw.get("path", ""))).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        observed_bytes = artifact.stat().st_size
        observed_sha256 = sha256_file(artifact)
        if observed_bytes != int(raw.get("bytes", -1)):
            raise AssertionError(f"artifact byte-size mismatch: {artifact}")
        if observed_sha256 != raw.get("sha256"):
            raise AssertionError(f"artifact SHA-256 mismatch: {artifact}")
        verified.append(
            {
                "path": str(artifact),
                "bytes": observed_bytes,
                "sha256": observed_sha256,
            }
        )
    return receipt, verified


def _stable_positions(source_indices: Any, count: int) -> np.ndarray:
    """Select samples without consulting any model score or DECAF output."""

    values = np.asarray(source_indices, dtype=np.int64)
    selected_count = int(count)
    if values.ndim != 1 or not 0 < selected_count <= len(values):
        raise ValueError("count must lie within the one-dimensional source-index vector")
    if len(np.unique(values)) != len(values):
        raise ValueError("source indices must be unique")
    order = sorted(
        range(len(values)),
        key=lambda position: (
            hashlib.sha256(f"{SELECTION_NAMESPACE}|{int(values[position])}".encode()).digest(),
            int(values[position]),
        ),
    )
    return np.asarray(order[:selected_count], dtype=np.int64)


def _dominant(values: Mapping[str, Any]) -> np.ndarray:
    matrix = np.column_stack(
        [np.asarray(values[name], dtype=np.float64) for name in MECHANISM_NAMES]
    )
    maximum = np.max(matrix, axis=1, keepdims=True)
    names = np.asarray(MECHANISM_NAMES, dtype=object)
    return np.asarray(
        ["|".join(names[row == maximum[index, 0]].tolist()) for index, row in enumerate(matrix)],
        dtype=object,
    )


def _numeric_stats(first: Any, second: Any) -> dict[str, float]:
    current = np.asarray(first, dtype=np.float64)
    historical = np.asarray(second, dtype=np.float64)
    if current.shape != historical.shape or current.size == 0:
        raise ValueError("comparison arrays must be non-empty and aligned")
    error = np.abs(current - historical)
    return {
        "median": float(np.median(error)),
        "p95": float(np.quantile(error, 0.95)),
        "max": float(np.max(error)),
    }


@lru_cache(maxsize=1)
def _historical_source_binding() -> dict[str, Any]:
    """Validate the complete sealed Covertype namespace and return its binding."""

    package = HISTORICAL_PACKAGE.resolve()
    if package.is_symlink() or not package.is_file():
        raise FileNotFoundError(f"sealed Covertype package is missing or unsafe: {package}")
    package_sha256 = sha256_file(package)
    if package_sha256 != HISTORICAL_PACKAGE_SHA256:
        raise ValueError(
            "sealed Covertype package SHA-256 changed: "
            f"{package_sha256} != {HISTORICAL_PACKAGE_SHA256}"
        )
    with zipfile.ZipFile(package) as archive:
        try:
            manifest_bytes = archive.read(HISTORICAL_PACKAGE_MANIFEST_MEMBER)
            manifest = json.loads(manifest_bytes)
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("sealed Covertype package manifest is invalid") from error
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("namespace") != "decaf_covertype_v1"
            or manifest.get("lightweight") is not True
            or not isinstance(files, list)
        ):
            raise ValueError("sealed Covertype package manifest contract changed")
        actual_files = {info.filename for info in archive.infolist() if not info.is_dir()}
        expected_files = {HISTORICAL_PACKAGE_MANIFEST_MEMBER}
        namespace_prefix = "code/src/cmr/decaf_covertype_v1/"
        namespace_members: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(files):
            if not isinstance(record, dict):
                raise TypeError(f"sealed Covertype member[{index}] is not an object")
            relative = record.get("path")
            expected_sha256 = record.get("sha256")
            expected_bytes = record.get("bytes")
            if (
                not isinstance(relative, str)
                or not relative
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0
            ):
                raise ValueError(f"sealed Covertype member[{index}] has invalid identity")
            try:
                payload = archive.read(relative)
            except KeyError as error:
                raise ValueError(f"sealed Covertype member is missing: {relative}") from error
            if (
                len(payload) != expected_bytes
                or hashlib.sha256(payload).hexdigest() != expected_sha256
            ):
                raise ValueError(f"sealed Covertype member identity changed: {relative}")
            expected_files.add(relative)
            if relative.startswith(namespace_prefix) and relative.endswith(".py"):
                if relative in namespace_members:
                    raise ValueError(f"duplicate Covertype source member: {relative}")
                namespace_members[relative] = {
                    "archive_member": relative,
                    "bytes": expected_bytes,
                    "sha256": expected_sha256,
                }
        if actual_files != expected_files:
            raise ValueError("sealed Covertype package inventory differs from its manifest")
    required_names = {
        "__init__",
        "behaviors",
        "compatibility",
        "config",
        "data",
        "decaf",
        "mechanisms",
        "models",
    }
    required_members = {
        f"code/src/cmr/decaf_covertype_v1/{name}.py" for name in required_names
    }
    if not required_members.issubset(namespace_members):
        raise ValueError("sealed Covertype package lacks required runtime modules")
    return {
        "authority_kind": "sha256_verified_lightweight_zip",
        "path": str(package),
        "sha256": package_sha256,
        "manifest_member": HISTORICAL_PACKAGE_MANIFEST_MEMBER,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "archive_source_prefix": HISTORICAL_PACKAGE_CODE_PREFIX,
        "namespace_member_count": len(namespace_members),
        "namespace_members": namespace_members,
        "required_modules": sorted(required_names),
        "archive_inventory_verified": True,
        "origin_verified": False,
        "git_head_role": "context_only_untracked",
    }


def _materialize_historical_source(destination: Path) -> dict[str, Any]:
    """Materialize only the verified historical namespace for filesystem import."""

    binding = _historical_source_binding()
    target = destination.resolve()
    namespace = target / "cmr/decaf_covertype_v1"
    namespace.mkdir(parents=True, exist_ok=True)
    parent_shim = target / "cmr/__init__.py"
    temporary_shim = parent_shim.with_name(f".{parent_shim.name}.part")
    temporary_shim.write_bytes(PARENT_PACKAGE_SHIM)
    temporary_shim.replace(parent_shim)
    member_prefix = f"{HISTORICAL_PACKAGE_CODE_PREFIX}/cmr/decaf_covertype_v1/"
    expected_relative: set[str] = set()
    with zipfile.ZipFile(binding["path"]) as archive:
        for archive_member, record in binding["namespace_members"].items():
            relative = archive_member.removeprefix(member_prefix)
            if relative == archive_member or not relative:
                raise ValueError(f"invalid Covertype namespace member: {archive_member}")
            expected_relative.add(relative)
            output = namespace / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(archive_member)
            if (
                len(payload) != int(record["bytes"])
                or hashlib.sha256(payload).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"Covertype source changed during extraction: {archive_member}")
            temporary = output.with_name(f".{output.name}.part")
            temporary.write_bytes(payload)
            temporary.replace(output)
    observed_relative = {
        path.relative_to(namespace).as_posix()
        for path in namespace.rglob("*")
        if path.is_file()
    }
    if observed_relative != expected_relative:
        raise ValueError("materialized Covertype source contains an unexpected file")
    materialized = {
        **binding,
        "import_root": str(target),
        "materialized_namespace": str(namespace),
        "materialized_member_count": len(observed_relative),
        "parent_package_shim": {
            "path": str(parent_shim),
            "bytes": len(PARENT_PACKAGE_SHIM),
            "sha256": hashlib.sha256(PARENT_PACKAGE_SHIM).hexdigest(),
            "role": "verification_only_import_isolation",
            "historical_source": False,
        },
    }
    return materialized


def _load_legacy(
    repository: Path,
    *,
    source_binding: dict[str, Any],
) -> dict[str, Callable[..., Any]]:
    del repository
    source = str(source_binding["import_root"])
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    parent = sys.modules.get("cmr")
    expected_parent_origin = f"{source}/cmr/__init__.py"
    if parent is not None and getattr(parent, "__file__", None) != expected_parent_origin:
        raise RuntimeError("the cmr parent package was already loaded from another source")
    try:
        from cmr.decaf_covertype_v1.behaviors import (
            contextual_direction_behavior,
            endpoint_null_fragility_behavior,
        )
        from cmr.decaf_covertype_v1.compatibility import config_sha256_compatible
        from cmr.decaf_covertype_v1.config import config_sha256, load_config
        from cmr.decaf_covertype_v1.data import data_fingerprint, load_data_bundle
        from cmr.decaf_covertype_v1.decaf import query_responses
        from cmr.decaf_covertype_v1.mechanisms import (
            load_module_c_bundle,
            load_module_f_bundle,
            mechanism_fingerprint,
        )
        from cmr.decaf_covertype_v1.models import predict_positive
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "sealed historical runtime dependencies are unavailable"
        ) from error
    expected_origin = f"{source}/cmr/decaf_covertype_v1/"
    loaded = {
        name: getattr(module, "__file__", None)
        for name, module in sys.modules.items()
        if name == HISTORICAL_MODULE_NAMESPACE
        or name.startswith(f"{HISTORICAL_MODULE_NAMESPACE}.")
    }
    if not loaded or any(
        not isinstance(origin, str) or not origin.startswith(expected_origin)
        for origin in loaded.values()
    ):
        raise RuntimeError(
            "Covertype historical modules were not loaded exclusively from the "
            "SHA-verified package snapshot"
        )
    parent_origin = getattr(sys.modules.get("cmr"), "__file__", None)
    if parent_origin != expected_parent_origin:
        raise RuntimeError(
            "the Covertype parent package shim did not isolate the sealed snapshot"
        )
    required_loaded = {
        f"{HISTORICAL_MODULE_NAMESPACE}.{name}"
        for name in source_binding["required_modules"]
        if name != "__init__"
    }
    if not required_loaded.issubset(loaded):
        raise RuntimeError("not all required sealed Covertype modules were loaded")
    source_binding["origin_verified"] = True
    source_binding["parent_package_origin"] = parent_origin
    source_binding["loaded_module_origins"] = dict(sorted(loaded.items()))
    return {
        "config_sha256": config_sha256,
        "config_sha256_compatible": config_sha256_compatible,
        "contextual_direction_behavior": contextual_direction_behavior,
        "data_fingerprint": data_fingerprint,
        "endpoint_null_fragility_behavior": endpoint_null_fragility_behavior,
        "load_config": load_config,
        "load_data_bundle": load_data_bundle,
        "load_module_c_bundle": load_module_c_bundle,
        "load_module_f_bundle": load_module_f_bundle,
        "mechanism_fingerprint": mechanism_fingerprint,
        "predict_positive": predict_positive,
        "query_responses": query_responses,
    }


def _model_specs(families: Sequence[str], seed: int, strength: float) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    token = f"{float(strength):.2f}"
    for family in families:
        specs.append(
            {
                "module": "C",
                "regime": "invert",
                "strength": float(strength),
                "model_family": family,
                "seed": int(seed),
                "model_id": f"c_invert_p{token}_{family}_s{int(seed)}",
            }
        )
        specs.append(
            {
                "module": "F",
                "regime": "fragile",
                "strength": None,
                "model_family": family,
                "seed": int(seed),
                "model_id": f"f_fragile_{family}_s{int(seed)}",
            }
        )
    return specs


def _identity_matches(
    sealed: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    source_indices: np.ndarray,
    context: np.ndarray,
    factor: np.ndarray,
) -> bool:
    return bool(
        len(sealed) == len(source_indices)
        and np.array_equal(
            sealed["sample_position"].to_numpy(),
            np.arange(len(source_indices), dtype=np.int64),
        )
        and np.array_equal(sealed["source_index"].to_numpy(), source_indices)
        and np.array_equal(sealed["factual_context"].to_numpy(), context)
        and np.array_equal(sealed["factual_factor"].to_numpy(), factor)
        and (sealed["model_id"].astype(str) == str(spec["model_id"])).all()
        and (sealed["module"].astype(str) == str(spec["module"])).all()
        and (sealed["regime"].astype(str) == str(spec["regime"])).all()
        and (sealed["model_family"].astype(str) == str(spec["model_family"])).all()
        and (sealed["seed"].astype(int) == int(spec["seed"])).all()
    )


def _trajectory_rows(
    *,
    spec: Mapping[str, Any],
    positions: np.ndarray,
    source_indices: np.ndarray,
    sealed: pd.DataFrame,
    scores: np.ndarray,
    response: np.ndarray,
    stages: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
    checkpoint_sha256: str,
    metadata_base: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    factor_name = "Z" if spec["module"] == "C" else "U"
    context_name = "G" if spec["module"] == "C" else "H"
    for raw_position in positions:
        position = int(raw_position)
        sample = sealed.iloc[position]
        source_index = int(source_indices[position])
        historical_values = {name: float(sample[name]) for name in SUMMARY_NAMES}
        metadata = {
            **metadata_base,
            "sample_position": position,
            "source_index": source_index,
            "y": int(sample["y"]),
            "y01": int(sample["y01"]),
            "factual_context": int(sample["factual_context"]),
            "factual_factor": int(sample["factual_factor"]),
            "alternate_score_plus": float(scores[position, 2]),
            "alternate_score_minus": float(scores[position, 3]),
            "alternate_response": float(sample["alternate_response"]),
            "historical_gate": bool(sample["endpoint_active"]),
            "historical_orientation": (
                int(np.sign(float(sample["endpoint_response"])))
                if bool(sample["endpoint_active"])
                else 0
            ),
            "historical_dominant": str(_dominant(historical_values)[0]),
            "identity_match": True,
            "current_model_id": str(spec["model_id"]),
            "current_checkpoint_sha256": checkpoint_sha256,
            "current_sample_or_pair_id": str(source_index),
            "current_factor_or_part_id": factor_name,
            "current_counterfactual_map": f"{context_name}:+1->-1",
            "current_protocol": "linear_mixture_of_legal_binary_context_responses",
        }
        metadata_json = _json_text(metadata)
        unit_id = f"{spec['model_id']}::source_index={source_index}"
        for stage_index, stage_t in enumerate(stages):
            rows.append(
                {
                    "experiment_family": "covertype",
                    "reference_run": REFERENCE_RUN,
                    "unit_id": unit_id,
                    "model_id": str(spec["model_id"]),
                    "checkpoint_sha256": checkpoint_sha256,
                    "sample_or_pair_id": str(source_index),
                    "factor_or_part_id": factor_name,
                    "counterfactual_map": f"{context_name}:+1->-1",
                    "protocol": "linear_mixture_of_legal_binary_context_responses",
                    "protocol_seed": int(spec["seed"]),
                    "stage_index": int(stage_index),
                    "stage_t": float(stage_t),
                    "quadrature_weight": float(weights[stage_index]),
                    "endpoint_epsilon": float(epsilon),
                    "endpoint_score_plus": float(scores[position, 0]),
                    "endpoint_score_minus": float(scores[position, 1]),
                    "endpoint_d": float(sample["endpoint_response"]),
                    "stage_score_plus": np.nan,
                    "stage_score_minus": np.nan,
                    "stage_r": float(response[position, stage_index]),
                    "historical_M": float(sample["M"]),
                    "historical_E": float(sample["E"]),
                    "historical_C": float(sample["C"]),
                    "historical_F": float(sample["F"]),
                    "historical_Abs": float(sample["Abs"]),
                    "metadata_json": metadata_json,
                }
            )
    return rows


def _run_model(
    *,
    spec: Mapping[str, Any],
    results: Path,
    inventory: pd.DataFrame,
    data: Any,
    source_indices: np.ndarray,
    config: Mapping[str, Any],
    runtime_config_sha256: str,
    natural_fingerprint: str,
    legacy: Mapping[str, Callable[..., Any]],
    positions: np.ndarray,
    stages: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], int]:
    model_id = str(spec["model_id"])
    inventory_rows = inventory.loc[inventory["model_id"].astype(str) == model_id]
    if len(inventory_rows) != 1 or str(inventory_rows.iloc[0]["status"]) != "completed":
        raise AssertionError(f"inventory identity is not unique/completed: {model_id}")
    checkpoint = (results / "checkpoints" / f"{model_id}.joblib").resolve()
    if checkpoint != Path(str(inventory_rows.iloc[0]["checkpoint_path"])).resolve():
        raise AssertionError(f"checkpoint path differs from inventory: {model_id}")
    checkpoint_before = {
        "sha256": sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "mtime_ns": checkpoint.stat().st_mtime_ns,
    }
    training_receipt_path = results / "jobs/training" / f"{model_id}.receipt.json"
    behavior_receipt_path = results / "jobs/decaf_behavior" / f"{model_id}.receipt.json"
    training_receipt, training_artifacts = _verify_receipt(training_receipt_path)
    behavior_receipt, behavior_artifacts = _verify_receipt(behavior_receipt_path)
    recorded_config_sha256 = str(behavior_receipt["config_sha256"])
    if (
        str(training_receipt["config_sha256"]) != recorded_config_sha256
        or str(training_receipt["task"]["model_id"]) != model_id
        or str(behavior_receipt["task"]["model_id"]) != model_id
        or str(behavior_receipt["task"]["input_checkpoint_sha256"]) != checkpoint_before["sha256"]
    ):
        raise AssertionError(f"receipt identity mismatch: {model_id}")
    if not legacy["config_sha256_compatible"](
        recorded_config_sha256, config, scope="decaf_behavior"
    ):
        raise AssertionError(f"runtime config is not a compatible successor: {model_id}")

    if spec["module"] == "C":
        strength_token = f"{float(spec['strength']):.2f}".replace(".", "p")
        mechanism_path = (
            results
            / "data_cache/mechanisms"
            / f"module_c__p_{strength_token}__seed_{spec['seed']}.npz"
        )
        bundle = legacy["load_module_c_bundle"](mechanism_path, data)
        context = np.asarray(bundle.test.G)
    else:
        mechanism_path = results / "data_cache/mechanisms" / f"module_f__seed_{spec['seed']}.npz"
        bundle = legacy["load_module_f_bundle"](mechanism_path, data)
        context = np.asarray(bundle.test.H)
    mechanism_sha256 = sha256_file(mechanism_path)
    mechanism_fingerprint = legacy["mechanism_fingerprint"](bundle)
    factor = np.asarray(bundle.test.factor(str(spec["regime"])))
    features = bundle.augmented(data, "test", str(spec["regime"]))
    model = joblib.load(checkpoint)
    queried = legacy["query_responses"](
        lambda value: legacy["predict_positive"](model, value),
        features,
    )
    if not bool(queried["all_queries_support_valid"]):
        raise AssertionError(f"legal support invariant failed: {model_id}")
    endpoint = np.asarray(queried["endpoint_response"], dtype=np.float64)
    alternate = np.asarray(queried["alternate_response"], dtype=np.float64)
    response = np.asarray(queried["response"], dtype=np.float64)
    scores = np.asarray(queried["scores"], dtype=np.float64)
    expected_rows = len(source_indices)
    if (
        endpoint.shape != (expected_rows,)
        or alternate.shape != (expected_rows,)
        or response.shape != (expected_rows, len(stages))
        or scores.shape != (expected_rows, 4)
    ):
        raise AssertionError(f"legal-query output shape mismatch: {model_id}")

    section = config["decaf"]["module_c" if spec["module"] == "C" else "module_f"]
    epsilon = float(section["primary_epsilon"])
    components = decompose(response, endpoint, epsilon, axis=1)
    integrated = integrate_components(stages, components, axis=1)
    current = {
        "M": np.abs(endpoint),
        **{name: np.asarray(integrated[name]) for name in ("E", "C", "F", "Abs")},
    }
    sealed_path = results / "jobs/decaf_behavior" / f"{model_id}.samples.parquet"
    model_table_path = results / "jobs/decaf_behavior" / f"{model_id}.model.parquet"
    sealed = pd.read_parquet(sealed_path)
    model_table = pd.read_parquet(model_table_path)
    if len(model_table) != 1:
        raise AssertionError(f"sealed model summary cardinality mismatch: {model_id}")
    historical_model = model_table.iloc[0]
    identity_exact = bool(
        _identity_matches(
            sealed,
            spec=spec,
            source_indices=source_indices,
            context=context,
            factor=factor,
        )
        and str(historical_model["config_sha256"]) == recorded_config_sha256
        and str(historical_model["natural_data_fingerprint"]) == natural_fingerprint
        and str(historical_model["mechanism_fingerprint"]) == mechanism_fingerprint
    )
    if not identity_exact:
        raise AssertionError(f"sealed sample identity mismatch: {model_id}")

    historical = {name: sealed[name].to_numpy(dtype=np.float64) for name in SUMMARY_NAMES}
    current_matrix = np.column_stack([current[name] for name in SUMMARY_NAMES])
    historical_matrix = np.column_stack([historical[name] for name in SUMMARY_NAMES])
    error_matrix = np.abs(current_matrix - historical_matrix)
    tier_a = np.all(
        np.isclose(
            current_matrix,
            historical_matrix,
            atol=TIER_A_ATOL,
            rtol=TIER_A_RTOL,
        ),
        axis=1,
    )
    current_gate, current_orientation = endpoint_orientation(endpoint, epsilon)
    historical_gate = sealed["endpoint_active"].to_numpy(dtype=bool)
    historical_orientation = endpoint_orientation(
        sealed["endpoint_response"].to_numpy(dtype=np.float64), epsilon
    )[1]
    current_dominant = _dominant(current)
    historical_dominant = _dominant(historical)
    gate_agree = current_gate == historical_gate
    orientation_agree = current_orientation == historical_orientation
    dominant_agree = current_dominant == historical_dominant
    boundary = np.abs(np.abs(endpoint) - epsilon) <= BOUNDARY_ATOL
    tier_b_eligible = (
        (np.max(error_matrix, axis=1) <= TIER_B_ATOL)
        & (gate_agree | boundary)
        & (orientation_agree | boundary)
        & dominant_agree
    )
    tier_b = (~tier_a) & tier_b_eligible
    hard = (
        (np.max(error_matrix, axis=1) > HARD_ATOL)
        | ((~gate_agree | ~orientation_agree) & ~boundary)
        | ~dominant_agree
    )
    suffix = "Z" if spec["module"] == "C" else "U"
    model_summary_errors = {
        name: abs(float(np.mean(current[name])) - float(historical_model[f"{name}_{suffix}"]))
        for name in SUMMARY_NAMES
    }
    if spec["module"] == "C":
        behavior = legacy["contextual_direction_behavior"](
            endpoint,
            alternate,
            delta=float(config["behaviors"]["module_c"]["delta"]),
        )["model"]
        behavior_names = ("preserve_rate", "collapse_rate", "invert_rate")
    else:
        alternate_mask = context == -1
        behavior = legacy["endpoint_null_fragility_behavior"](
            scores[alternate_mask],
            endpoint[alternate_mask],
            data.test.y[alternate_mask],
            factor[alternate_mask],
            epsilon=epsilon,
        )["model"]
        behavior_names = (
            "pairwise_prediction_change_rate",
            "null_context_prediction_change_rate",
            "null_context_probability_excursion",
        )
    behavior_errors = {
        name: abs(float(behavior[name]) - float(historical_model[name]))
        for name in behavior_names
        if np.isfinite(float(behavior[name])) and np.isfinite(float(historical_model[name]))
    }
    variable_stats = {
        name: _numeric_stats(current[name], historical[name]) for name in SUMMARY_NAMES
    }
    comparison: dict[str, Any] = {
        **dict(spec),
        "checkpoint_sha256": checkpoint_before["sha256"],
        "test_units": expected_rows,
        "identity_exact": identity_exact,
        "endpoint_max_abs_error": _numeric_stats(endpoint, sealed["endpoint_response"])["max"],
        "alternate_max_abs_error": _numeric_stats(alternate, sealed["alternate_response"])["max"],
        "summary_median_abs_error": float(np.median(error_matrix)),
        "summary_p95_abs_error": float(np.quantile(error_matrix, 0.95)),
        "summary_max_abs_error": float(np.max(error_matrix)),
        "tier_a_fraction": float(np.mean(tier_a)),
        "tier_b_fraction": float(np.mean(tier_b)),
        "tier_b_eligible_fraction": float(np.mean(tier_b_eligible)),
        "tier_a_or_b_fraction": float(np.mean(tier_a | tier_b_eligible)),
        "hard_mismatch_fraction": float(np.mean(hard)),
        "gate_agreement": (float(np.mean(gate_agree[~boundary])) if np.any(~boundary) else 1.0),
        "orientation_agreement": (
            float(np.mean(orientation_agree[~boundary])) if np.any(~boundary) else 1.0
        ),
        "dominant_agreement": float(np.mean(dominant_agree)),
        "endpoint_active_rate_current": float(np.mean(current_gate)),
        "endpoint_active_rate_historical": float(
            historical_model[f"endpoint_active_rate_{suffix}"]
        ),
        "endpoint_active_rate_abs_error": abs(
            float(np.mean(current_gate)) - float(historical_model[f"endpoint_active_rate_{suffix}"])
        ),
        "model_summary_max_abs_error": max(model_summary_errors.values()),
        "behavior_max_abs_error": max(behavior_errors.values(), default=0.0),
        "legal_support_queries": True,
        "historical_decomposition_called": False,
        "current_core_used": True,
        "model_fit_calls": 0,
    }
    for name, stats in variable_stats.items():
        for statistic, value in stats.items():
            comparison[f"{name}_{statistic}_abs_error"] = value
    for name, value in model_summary_errors.items():
        comparison[f"model_{name}_abs_error"] = value
    for name, value in behavior_errors.items():
        comparison[f"{name}_abs_error"] = value

    metadata_base = {
        "bridge": "verification_only_legacy_covertype_export",
        **dict(spec),
        "recorded_config_sha256": recorded_config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "runtime_config_compatible": True,
        "natural_data_fingerprint": natural_fingerprint,
        "mechanism_fingerprint": mechanism_fingerprint,
        "mechanism_sha256": mechanism_sha256,
        "sealed_samples_sha256": sha256_file(sealed_path),
        "all_queries_support_valid": True,
        "historical_decomposition_called": False,
        "model_fit_calls": 0,
    }
    trajectories = _trajectory_rows(
        spec=spec,
        positions=positions,
        source_indices=source_indices,
        sealed=sealed,
        scores=scores,
        response=response,
        stages=stages,
        weights=weights,
        epsilon=epsilon,
        checkpoint_sha256=str(checkpoint_before["sha256"]),
        metadata_base=metadata_base,
    )
    checkpoint_after = {
        "sha256": sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "mtime_ns": checkpoint.stat().st_mtime_ns,
    }
    if checkpoint_after != checkpoint_before:
        raise AssertionError(f"checkpoint changed during verification: {model_id}")
    provenance = {
        **dict(spec),
        "checkpoint_path": str(checkpoint),
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "checkpoint_unchanged": True,
        "mechanism_path": str(mechanism_path),
        "mechanism_sha256": mechanism_sha256,
        "mechanism_fingerprint": mechanism_fingerprint,
        "recorded_config_sha256": recorded_config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "runtime_config_compatible": True,
        "training_receipt_path": str(training_receipt_path),
        "training_receipt_sha256": sha256_file(training_receipt_path),
        "behavior_receipt_path": str(behavior_receipt_path),
        "behavior_receipt_sha256": sha256_file(behavior_receipt_path),
        "receipt_artifacts_verified": len(training_artifacts) + len(behavior_artifacts),
        "training_performed": False,
        "model_fit_calls": 0,
    }
    return (
        comparison,
        trajectories,
        provenance,
        len(training_artifacts) + len(behavior_artifacts),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(args.historical_repo).resolve()
    results = Path(args.historical_results).resolve()
    output_root = Path(args.selection_manifest).resolve().parent.parent
    source_binding = _materialize_historical_source(
        output_root / "provenance/historical_sources/covertype"
    )
    legacy = _load_legacy(repository, source_binding=source_binding)
    config_path = repository / "configs/decaf_covertype_v1/formal.yaml"
    config = legacy["load_config"](config_path)
    runtime_config_sha256 = str(legacy["config_sha256"](config))
    data_path = results / "data_cache/covertype_balanced_240000_split7601.npz"
    data = legacy["load_data_bundle"](data_path)
    natural_fingerprint = str(legacy["data_fingerprint"](data))
    split_manifest = _read_json(results / "data/split_manifest.json")
    if str(split_manifest.get("fingerprint")) != natural_fingerprint:
        raise AssertionError("split manifest does not match the cached exact split")
    source_indices = np.asarray(data.test.indices, dtype=np.int64)
    families = tuple(part.strip() for part in str(args.families).split(",") if part.strip())
    if not families or any(family not in MODEL_FAMILIES for family in families):
        raise ValueError(f"families must be selected from {MODEL_FAMILIES}")
    specs = _model_specs(families, int(args.seed), float(args.strength))
    positions = _stable_positions(source_indices, int(args.samples_per_model))
    stages = np.asarray(config["decaf"]["stages"], dtype=np.float64)
    weights = trapezoid_weights(stages)
    inventory = pd.read_csv(results / "inventory/model_manifest.csv")
    comparison_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    model_provenance: list[dict[str, Any]] = []
    receipt_artifacts_verified = 0
    for spec in specs:
        comparison, trajectories, provenance, verified_count = _run_model(
            spec=spec,
            results=results,
            inventory=inventory,
            data=data,
            source_indices=source_indices,
            config=config,
            runtime_config_sha256=runtime_config_sha256,
            natural_fingerprint=natural_fingerprint,
            legacy=legacy,
            positions=positions,
            stages=stages,
            weights=weights,
        )
        comparison_rows.append(comparison)
        trajectory_rows.extend(trajectories)
        model_provenance.append(provenance)
        receipt_artifacts_verified += verified_count

    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_target = Path(args.output_comparison)
    comparison_target.parent.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(comparison_target, index=False)
    trajectory_frame = pd.DataFrame(trajectory_rows, columns=NEUTRAL_COLUMNS)
    write_trajectory_record(trajectory_frame, args.output_record)
    expected_trajectory_units = len(specs) * len(positions)
    if trajectory_frame["unit_id"].nunique() != expected_trajectory_units or len(
        trajectory_frame
    ) != expected_trajectory_units * len(stages):
        raise AssertionError("trajectory selection cardinality mismatch")
    total_units = int(comparison_frame["test_units"].sum())

    def weighted(column: str) -> float:
        return float(
            np.average(
                comparison_frame[column],
                weights=comparison_frame["test_units"],
            )
        )

    status = (
        "PASS_CORE_AND_E2E"
        if (
            weighted("tier_a_or_b_fraction") >= 0.95
            and weighted("hard_mismatch_fraction") == 0.0
            and weighted("gate_agreement") >= 0.99
            and weighted("orientation_agreement") >= 0.99
            and weighted("dominant_agreement") >= 0.95
            and bool(comparison_frame["identity_exact"].all())
        )
        else "FAIL_NUMERICAL"
    )
    selection_manifest: dict[str, Any] = {
        "schema_version": 1,
        "family": "covertype",
        "reference_run": REFERENCE_RUN,
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_key": "sha256(namespace|sealed_test_source_index)",
        "selection_uses_model_outputs": False,
        "samples_per_model": len(positions),
        "test_positions": [int(value) for value in positions],
        "source_indices": [int(source_indices[value]) for value in positions],
        "test_split_size": len(source_indices),
        "full_e2e_units_per_model": len(source_indices),
        "full_e2e_units_total": total_units,
        "trajectory_units": expected_trajectory_units,
        "trajectory_rows": len(trajectory_frame),
        "model_count": len(specs),
        "models": model_provenance,
        "stages": [float(value) for value in stages],
        "quadrature_weights": [float(value) for value in weights],
        "data_cache_path": str(data_path),
        "data_cache_sha256": sha256_file(data_path),
        "data_cache_manifest_sha256": sha256_file(data_path.with_suffix(".manifest.json")),
        "split_manifest_sha256": sha256_file(results / "data/split_manifest.json"),
        "natural_data_fingerprint": natural_fingerprint,
        "historical_repository": str(repository),
        "historical_repository_head": str(args.historical_commit),
        "historical_repository_head_role": "context_only_untracked",
        "historical_source_binding": source_binding,
        "historical_results": str(results),
        "runtime_config_sha256": runtime_config_sha256,
        "training_performed": False,
        "model_fit_calls": 0,
        "all_checkpoints_unchanged": all(
            model["checkpoint_unchanged"] for model in model_provenance
        ),
    }
    selection_manifest["selection_sha256"] = hashlib.sha256(
        _json_text(selection_manifest).encode("utf-8")
    ).hexdigest()
    _write_json(args.selection_manifest, selection_manifest)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "family": "covertype",
        "status": status,
        "model_count": len(specs),
        "model_families": list(families),
        "invert_models": sum(spec["module"] == "C" for spec in specs),
        "fragile_models": sum(spec["module"] == "F" for spec in specs),
        "trajectory_units": expected_trajectory_units,
        "trajectory_rows": len(trajectory_frame),
        "current_core_units": total_units,
        "current_e2e_units": total_units,
        "full_test_rows_per_model": len(source_indices),
        "tier_a_fraction": weighted("tier_a_fraction"),
        "tier_b_fraction": weighted("tier_b_fraction"),
        "tier_b_eligible_fraction": weighted("tier_b_eligible_fraction"),
        "tier_a_or_b_fraction": weighted("tier_a_or_b_fraction"),
        "hard_mismatch_fraction": weighted("hard_mismatch_fraction"),
        "gate_agreement": weighted("gate_agreement"),
        "orientation_agreement": weighted("orientation_agreement"),
        "dominant_agreement": weighted("dominant_agreement"),
        "median_absolute_error": float(comparison_frame["summary_median_abs_error"].median()),
        "p95_absolute_error": float(comparison_frame["summary_p95_abs_error"].max()),
        "maximum_absolute_error": float(comparison_frame["summary_max_abs_error"].max()),
        "maximum_endpoint_error": float(comparison_frame["endpoint_max_abs_error"].max()),
        "maximum_behavior_error": float(comparison_frame["behavior_max_abs_error"].max()),
        "identity_exact": bool(comparison_frame["identity_exact"].all()),
        "legal_support_only": bool(comparison_frame["legal_support_queries"].all()),
        "historical_decomposition_called": False,
        "current_core_used": True,
        "receipt_artifacts_sha256_verified": receipt_artifacts_verified,
        "training_performed": False,
        "model_fit_calls": 0,
        "all_checkpoints_unchanged": bool(selection_manifest["all_checkpoints_unchanged"]),
        "natural_data_fingerprint": natural_fingerprint,
        "data_cache_sha256": sha256_file(data_path),
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_uses_model_outputs": False,
        "outputs": {
            "trajectory_record": str(Path(args.output_record).resolve()),
            "comparison": str(comparison_target.resolve()),
            "summary": str(Path(args.output_summary).resolve()),
            "selection_manifest": str(Path(args.selection_manifest).resolve()),
        },
    }
    _write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_pass and status != "PASS_CORE_AND_E2E":
        raise SystemExit(1)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-repo",
        default="/work/Users/leiyo/GitHub/covariance-matched-markov-revelation",
    )
    parser.add_argument(
        "--historical-results",
        default="/work/Users/leiyo/decaf_covertype_v1_results",
    )
    parser.add_argument(
        "--historical-commit",
        default="8555192d41e68423ac95be647fd9046dea0fb140",
    )
    parser.add_argument("--families", default=",".join(MODEL_FAMILIES))
    parser.add_argument("--seed", type=int, default=7701)
    parser.add_argument("--strength", type=float, default=0.95)
    parser.add_argument("--samples-per-model", type=int, default=64)
    parser.add_argument("--output-record", required=True)
    parser.add_argument("--output-comparison", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
