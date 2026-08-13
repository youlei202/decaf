"""Replay sealed legacy attribution paths into neutral trajectory records.

Legacy code generates only factual/counterfactual scores and identity metadata.
Historical M/E/C/F/Abs values always come from sealed result tables; the
current repository independently recomputes the decomposition from these rows.
One invocation owns one model and the fixed eight-sample selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Historical trees are read-only evidence.  Imports must not create cache files
# beside either the deployed A0 tree or the materialized A2 snapshot.
sys.dont_write_bytecode = True

from tools.crossgen.schema import (  # noqa: E402
    NEUTRAL_COLUMNS,
    sha256_file,
    trapezoid_weights,
    write_trajectory_record,
)

METHODS = ("decaf_3", "decaf_5", "decaf_9")
SCHEDULES = {
    "decaf_3": "DECAF-3",
    "decaf_5": "DECAF-5",
    "decaf_9": "DECAF-9",
}
MODELS = {
    "funnybirds": (
        "funnybirds_resnet50",
        "funnybirds_vgg16",
        "funnybirds_vit_b_16",
    ),
    "imagenet1k_idsds": ("resnet50", "vgg16", "vit_base_patch16_224"),
}
REFERENCE_ID = "gaussian_blur_k31_sigma12"
ENDPOINT_EPSILON = 0.02

A0_SOURCE_ROOT = Path("/work/Users/leiyo/decaf_reference_locked_v1_ready/code")
A0_READY_ROOT = Path("/work/Users/leiyo/decaf_reference_locked_v1_ready")
A0_DEPLOYMENT_RECEIPT = A0_READY_ROOT / "deployment_receipt.json"
A0_DEPLOYMENT_RECEIPT_SHA256 = (
    "f866a79876eaaff8ca81652394e2b9383665f869f9f8fd2061e49be1732d7703"
)
A0_FORMAL_PLAN_RECEIPT = A0_READY_ROOT / "plans/formal_jobs.jsonl.receipt.json"
A0_FORMAL_PLAN_RECEIPT_SHA256 = (
    "3789c430ebe7a1a05c973dfdc0f52b2850bff667e9fdfe45e1e56fbf039b6f05"
)
A0_FORMAL_SOURCE_SHA256 = (
    "46c53e874e685d95eb7bd06649ae747fd8d903b4805bebfb63a6b78abe33cfa9"
)
A0_FORMAL_PLAN_SHA256 = (
    "6d93f2ee4a0e52a338c74f755f0bec7b0ff1b2a8e1b909ab16a1a8e617e0531a"
)
A0_HISTORICAL_NAMESPACE = "cmr.decaf_reference_locked_v1"
A0_REQUIRED_MODULE_SHA256 = {
    "methods": "1d1621ab6c9b14c224e25bf93c0de9d6bc09a14b8e6a57e40abbb21703050a23",
    "models": "9c0427f993dfcd81116d57174630c9c81615c9acd61c85131c2ac3e8c2e16665",
    "worker": "6202a53bf76900577dd1599da0bb95a0b77c13ec95be9d9c2d30a77f570b1cdd",
}
A0_ANCHOR_FILE_SHA256 = {
    "cmr/decaf_reference_locked_v1/__init__.py": (
        "5f7fd80c768cfa51c55ef3e96a20e5ab2ff4c77853f0ee62061d2582961d6227"
    ),
    "cmr/decaf_imagenet9_v1/decaf.py": (
        "13c27814992ff86c4c6b59afc6bedb178ec73059f237b6d96c14d29dc35aeb41"
    ),
}
A0_CACHE_ROOT = Path("/work/Users/leiyo/decaf_reference_locked_v1_ready/cache")
A0_PLAN = Path(
    "/work/Users/leiyo/decaf_reference_locked_v1_ready/plans/formal_jobs.jsonl"
)
A0_COMPONENTS = Path(
    "/work/Users/leiyo/decaf_reference_locked_v1_results/formal/"
    "decaf_components.parquet"
)
A2_SOURCE_ROOT = Path(
    "/work/Users/leiyo/GitHub/covariance-matched-markov-revelation/src"
)
A2_HISTORICAL_PACKAGE = Path(
    "/work/Users/leiyo/decaf_idsds_funnybirds_v1_results/packages/"
    "decaf_idsds_funnybirds_v1_20260811T054516Z_lightweight.zip"
)
A2_HISTORICAL_PACKAGE_SHA256 = (
    "f68ed1fec48b39403fb677492283066f853722f466ce703edd5b468d59cc93a4"
)
A2_PACKAGE_MANIFEST_MEMBER = "PACKAGE_MANIFEST.json"
A2_PACKAGE_MANIFEST_SHA256 = (
    "6689282bef7fb97ca0a77174dff6c259ad3e348180a16d3de5cac50f15d40be5"
)
A2_PACKAGE_PAYLOAD_TREE_SHA256 = (
    "5a1f0bc9215b4c75f139400165c4450995e6189a92fc73196b5b91c007291598"
)
A2_PACKAGE_CODE_PREFIX = "code_snapshot/src"
A2_HISTORICAL_NAMESPACE = "cmr.decaf_idsds_funnybirds_v1"
A2_MATERIALIZED_SOURCE = Path(
    "/work/Users/leiyo/decaf_cross_generation_equivalence/v2/provenance/"
    "historical_sources/attribution_idsds"
)
A2_PARENT_PACKAGE_SHIM = (
    b'"""Verification-only parent package for the sealed Attribution snapshot."""\n'
)
A2_REQUIRED_MODULES = (
    "attribution",
    "data",
    "models",
    "contracts",
    "decomposition",
    "idsds",
)
A2_DATA_ROOT = Path("/work/Users/leiyo/decaf_idsds_funnybirds_v1_data")
A2_INPUT_MANIFEST = Path(
    "/work/Users/leiyo/decaf_idsds_funnybirds_v1_results/manifests/"
    "imagenet_idsds_10k.parquet"
)
A2_RESULTS = Path(
    "/work/Users/leiyo/decaf_idsds_funnybirds_v1_results/imagenet/"
    "per_image_idsds.parquet"
)


def _finite_vector(
    value: Any, *, name: str, length: int | None = None
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} has length {len(result)}, expected {length}")
    return result


def _dominant(e: float, c: float, f: float) -> str:
    names = ("E", "C", "F")
    values = (float(e), float(c), float(f))
    maximum = max(values)
    return "|".join(
        name for name, value in zip(names, values, strict=True) if value == maximum
    )


def _trajectory_rows(
    *,
    dataset: str,
    reference_run: str,
    model_id: str,
    checkpoint_sha256: str,
    image_id: str,
    target: int,
    method: str,
    part_names: Sequence[str],
    stage_t: Any,
    q_plus: Any,
    q_minus: Any,
    historical: Mapping[str, Any],
    counterfactual_map: str,
    historical_endpoint_d: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build neutral rows without importing either legacy tree."""

    if method not in METHODS:
        raise ValueError(f"unsupported attribution method: {method!r}")
    grid = _finite_vector(stage_t, name="stage_t")
    weights = trapezoid_weights(grid)
    plus = _finite_vector(q_plus, name="q_plus", length=len(grid))
    minus = np.asarray(q_minus, dtype=np.float64)
    names = tuple(str(name) for name in part_names)
    if (
        minus.ndim != 2
        or minus.shape != (len(names), len(grid))
        or not np.isfinite(minus).all()
    ):
        raise ValueError(
            "q_minus must be finite with shape "
            f"({len(names)}, {len(grid)}), got {minus.shape}"
        )
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("part_names must be non-empty and unique")

    summaries = {
        name: _finite_vector(
            historical[name], name=f"historical_{name}", length=len(names)
        )
        for name in ("M", "E", "C", "F", "Abs")
    }
    endpoint_history = (
        None
        if historical_endpoint_d is None
        else _finite_vector(
            historical_endpoint_d,
            name="historical_endpoint_d",
            length=len(names),
        )
    )
    if not np.allclose(
        summaries["Abs"],
        summaries["E"] + summaries["C"] + summaries["F"],
        atol=5.0e-4,
        rtol=5.0e-5,
    ):
        raise ValueError("sealed historical Abs differs from E + C + F")
    if endpoint_history is not None and not np.allclose(
        summaries["M"],
        np.abs(endpoint_history),
        atol=5.0e-4,
        rtol=5.0e-5,
    ):
        raise ValueError("sealed historical M differs from the signed endpoint effect")

    rows: list[dict[str, Any]] = []
    for part_index, part_name in enumerate(names):
        endpoint_d = float(plus[-1] - minus[part_index, -1])
        signed_history = (
            endpoint_d
            if endpoint_history is None
            else float(endpoint_history[part_index])
        )
        historical_gate = bool(summaries["M"][part_index] >= ENDPOINT_EPSILON)
        historical_orientation = int(np.sign(signed_history)) if historical_gate else 0
        payload: dict[str, Any] = {
            "target": int(target),
            "schedule": SCHEDULES[method],
            "historical_gate": historical_gate,
            "historical_orientation": historical_orientation,
            "historical_dominant": _dominant(
                summaries["E"][part_index],
                summaries["C"][part_index],
                summaries["F"][part_index],
            ),
            "historical_endpoint_d": signed_history,
            "regenerated_endpoint_d": endpoint_d,
            "raw_score_source": "historical_executable_replay",
            "historical_summary_source": "sealed_result_table",
            "current_model_id": model_id,
            "current_checkpoint_sha256": checkpoint_sha256,
            "current_sample_or_pair_id": image_id,
            "current_factor_or_part_id": part_name,
            "current_counterfactual_map": counterfactual_map,
            "current_protocol": method,
            "identity_match": True,
        }
        payload.update(dict(metadata or {}))
        metadata_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        unit_id = (
            f"attribution::{dataset}::{model_id}::{method}::{image_id}::{part_name}"
        )
        for stage_index, (stage, weight) in enumerate(
            zip(grid, weights, strict=True)
        ):
            stage_plus = float(plus[stage_index])
            stage_minus = float(minus[part_index, stage_index])
            rows.append(
                {
                    "experiment_family": "attribution",
                    "reference_run": reference_run,
                    "unit_id": unit_id,
                    "model_id": model_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "sample_or_pair_id": image_id,
                    "factor_or_part_id": part_name,
                    "counterfactual_map": counterfactual_map,
                    "protocol": method,
                    "protocol_seed": 0,
                    "stage_index": stage_index,
                    "stage_t": float(stage),
                    "quadrature_weight": float(weight),
                    "endpoint_epsilon": ENDPOINT_EPSILON,
                    "endpoint_score_plus": float(plus[-1]),
                    "endpoint_score_minus": float(minus[part_index, -1]),
                    "endpoint_d": endpoint_d,
                    "stage_score_plus": stage_plus,
                    "stage_score_minus": stage_minus,
                    "stage_r": stage_plus - stage_minus,
                    "historical_M": float(summaries["M"][part_index]),
                    "historical_E": float(summaries["E"][part_index]),
                    "historical_C": float(summaries["C"][part_index]),
                    "historical_F": float(summaries["F"][part_index]),
                    "historical_Abs": float(summaries["Abs"][part_index]),
                    "metadata_json": metadata_json,
                }
            )
    return rows


def _read_fixed_manifest(path: Path, *, dataset: str, model_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("fixed sample manifest must encode a JSON object")
    if payload.get("dataset") != dataset or payload.get("model_id") != model_id:
        raise ValueError("fixed sample manifest dataset/model identity mismatch")
    if payload.get("selection") != "first_eight_in_frozen_candidate_order":
        raise ValueError("fixed sample manifest selection contract drifted")
    image_ids = payload.get("image_ids")
    targets = payload.get("targets")
    if (
        not isinstance(image_ids, list)
        or not isinstance(targets, list)
        or len(image_ids) != 8
        or len(targets) != 8
        or len(set(map(str, image_ids))) != 8
    ):
        raise ValueError("fixed sample manifest must contain eight unique ids and targets")
    payload["image_ids"] = [str(value) for value in image_ids]
    payload["targets"] = [int(value) for value in targets]
    return payload


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not a readable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must encode a JSON object: {path}")
    return value


def _formal_python_tree_digest(root: Path) -> tuple[str, int]:
    """Reproduce the digest used by the sealed A0 formal-plan receipt."""

    source = root.resolve()
    files = sorted(source.rglob("*.py"))
    unsafe = [path for path in files if path.is_symlink()]
    if unsafe:
        raise ValueError(f"A0 deployed source tree contains a symlink: {unsafe[0]}")
    files = [
        path
        for path in files
        if path.is_file() and "__pycache__" not in path.parts
    ]
    if not files:
        raise ValueError(f"A0 deployed Python source tree is empty: {source}")
    material = [
        f"{path.relative_to(source).as_posix()}:{sha256_file(path)}" for path in files
    ]
    digest = hashlib.sha256("\n".join(material).encode("utf-8")).hexdigest()
    return digest, len(files)


@lru_cache(maxsize=1)
def _a0_historical_source_binding() -> dict[str, Any]:
    """Bind the deployed A0 tree to both authoritative formal receipts."""

    deployment_path = A0_DEPLOYMENT_RECEIPT.resolve()
    plan_receipt_path = A0_FORMAL_PLAN_RECEIPT.resolve()
    plan_path = A0_PLAN.resolve()
    source_root = A0_SOURCE_ROOT.resolve()
    for path, label in (
        (deployment_path, "A0 deployment receipt"),
        (plan_receipt_path, "A0 formal-plan receipt"),
        (plan_path, "A0 formal plan"),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    if not source_root.is_dir() or source_root.is_symlink():
        raise FileNotFoundError(f"A0 deployed source root is missing or unsafe: {source_root}")

    deployment_sha256 = sha256_file(deployment_path)
    if deployment_sha256 != A0_DEPLOYMENT_RECEIPT_SHA256:
        raise ValueError("A0 deployment receipt SHA-256 changed")
    plan_receipt_sha256 = sha256_file(plan_receipt_path)
    if plan_receipt_sha256 != A0_FORMAL_PLAN_RECEIPT_SHA256:
        raise ValueError("A0 formal-plan receipt SHA-256 changed")
    plan_sha256 = sha256_file(plan_path)
    if plan_sha256 != A0_FORMAL_PLAN_SHA256:
        raise ValueError("A0 formal plan SHA-256 changed")

    deployment = _read_json_object(deployment_path, label="A0 deployment receipt")
    plan_receipt = _read_json_object(
        plan_receipt_path, label="A0 formal-plan receipt"
    )
    deployment_plan = deployment.get("formal_job_plan")
    deployment_plan_receipt = deployment.get("formal_job_plan_receipt")
    deployment_rebind = deployment.get("formal_job_plan_rebind")
    expected_code_root = str(source_root)
    if (
        deployment.get("schema_version") != 1
        or deployment.get("code_root") != expected_code_root
        or deployment.get("entrypoint") != "cmr.decaf_reference_locked_v1.run"
        or deployment.get("code_sha256") != A0_FORMAL_SOURCE_SHA256
        or deployment.get("code_changes_required") is not False
        or deployment.get("downloads_required") is not False
        or deployment.get("installs_required") is not False
        or not isinstance(deployment_plan, dict)
        or deployment_plan.get("path") != "plans/formal_jobs.jsonl"
        or deployment_plan.get("sha256") != plan_sha256
        or deployment_plan.get("bytes") != plan_path.stat().st_size
        or not isinstance(deployment_plan_receipt, dict)
        or deployment_plan_receipt.get("path")
        != "plans/formal_jobs.jsonl.receipt.json"
        or deployment_plan_receipt.get("sha256") != plan_receipt_sha256
        or deployment_plan_receipt.get("bytes") != plan_receipt_path.stat().st_size
    ):
        raise ValueError("A0 deployment receipt contract changed")
    if (
        not isinstance(deployment_rebind, dict)
        or deployment_rebind.get("schema_version") != 1
        or deployment_rebind.get("kind")
        != "formal_job_plan_receipt_code_rebind"
        or deployment_rebind.get("code_sha256") != A0_FORMAL_SOURCE_SHA256
        or deployment_rebind.get("plan_sha256_unchanged") is not True
        or deployment_rebind.get("rebound_fields") != ["code_sha256"]
        or not isinstance(deployment_rebind.get("plan"), dict)
        or Path(str(deployment_rebind["plan"].get("path", ""))).resolve()
        != plan_path
        or deployment_rebind["plan"].get("bytes") != plan_path.stat().st_size
        or deployment_rebind["plan"].get("sha256") != plan_sha256
        or not isinstance(deployment_rebind.get("receipt"), dict)
        or Path(str(deployment_rebind["receipt"].get("path", ""))).resolve()
        != plan_receipt_path
        or deployment_rebind["receipt"].get("bytes")
        != plan_receipt_path.stat().st_size
        or deployment_rebind["receipt"].get("sha256") != plan_receipt_sha256
    ):
        raise ValueError("A0 formal-plan receipt rebind contract changed")
    if (
        plan_receipt.get("schema_version") != 1
        or plan_receipt.get("kind") != "decaf_reference_locked_formal_job_plan"
        or Path(str(plan_receipt.get("plan_path", ""))).resolve() != plan_path
        or plan_receipt.get("plan_sha256") != plan_sha256
        or plan_receipt.get("code_sha256") != A0_FORMAL_SOURCE_SHA256
        or plan_receipt.get("job_count") != 9184
    ):
        raise ValueError("A0 formal-plan receipt contract changed")

    source_tree_sha256, source_python_file_count = _formal_python_tree_digest(
        source_root / "cmr"
    )
    if source_tree_sha256 != A0_FORMAL_SOURCE_SHA256:
        raise ValueError("A0 deployed source-tree digest differs from its receipts")
    namespace_root = source_root / Path(*A0_HISTORICAL_NAMESPACE.split("."))
    modules: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in A0_REQUIRED_MODULE_SHA256.items():
        path = namespace_root / f"{name}.py"
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"required A0 module is missing or unsafe: {path}")
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"required A0 module SHA-256 changed: {name}")
        modules[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": observed_sha256,
        }
    anchor_files: dict[str, dict[str, Any]] = {}
    for relative, expected_sha256 in A0_ANCHOR_FILE_SHA256.items():
        path = source_root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"required A0 anchor is missing or unsafe: {path}")
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"required A0 anchor SHA-256 changed: {relative}")
        anchor_files[relative] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": observed_sha256,
        }
    return {
        "authority_kind": "deployed_tree_with_sha256_receipts",
        "source_root": str(source_root),
        "namespace": A0_HISTORICAL_NAMESPACE,
        "deployment_receipt": {
            "path": str(deployment_path),
            "bytes": deployment_path.stat().st_size,
            "sha256": deployment_sha256,
        },
        "formal_plan": {
            "path": str(plan_path),
            "bytes": plan_path.stat().st_size,
            "sha256": plan_sha256,
        },
        "formal_plan_receipt": {
            "path": str(plan_receipt_path),
            "bytes": plan_receipt_path.stat().st_size,
            "sha256": plan_receipt_sha256,
        },
        "source_tree_sha256": source_tree_sha256,
        "source_tree_digest_algorithm": "sha256_join_relative_py_path_colon_sha256_lf",
        "source_python_file_count": source_python_file_count,
        "required_modules": modules,
        "anchor_files": anchor_files,
        "origin_verified": False,
        "git_head_role": "context_only_untracked",
    }


def _a2_payload_tree_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _a2_historical_source_binding() -> dict[str, Any]:
    """Validate every member of the fixed A2 lightweight package."""

    package = A2_HISTORICAL_PACKAGE.resolve()
    if package.is_symlink() or not package.is_file():
        raise FileNotFoundError(f"sealed A2 package is missing or unsafe: {package}")
    package_sha256 = sha256_file(package)
    if package_sha256 != A2_HISTORICAL_PACKAGE_SHA256:
        raise ValueError("sealed A2 package SHA-256 changed")
    with zipfile.ZipFile(package) as archive:
        if archive.testzip() is not None:
            raise ValueError("sealed A2 package failed its ZIP CRC check")
        infos = [info for info in archive.infolist() if not info.is_dir()]
        actual_names = [info.filename for info in infos]
        if len(actual_names) != len(set(actual_names)):
            raise ValueError("sealed A2 package contains duplicate members")
        try:
            manifest_bytes = archive.read(A2_PACKAGE_MANIFEST_MEMBER)
            manifest = json.loads(manifest_bytes)
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("sealed A2 package manifest is invalid") from error
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {
                "schema_version",
                "package_kind",
                "payload_tree_sha256",
                "members",
            }
            or manifest.get("schema_version") != 1
            or manifest.get("package_kind") != "lightweight"
            or manifest.get("payload_tree_sha256")
            != A2_PACKAGE_PAYLOAD_TREE_SHA256
            or hashlib.sha256(manifest_bytes).hexdigest()
            != A2_PACKAGE_MANIFEST_SHA256
            or not isinstance(manifest.get("members"), list)
        ):
            raise ValueError("sealed A2 package manifest contract changed")
        records = manifest["members"]
        expected_names = [str(record.get("path", "")) for record in records]
        if expected_names != sorted(expected_names) or len(expected_names) != len(
            set(expected_names)
        ):
            raise ValueError("sealed A2 manifest member inventory is not canonical")
        if set(actual_names) != {A2_PACKAGE_MANIFEST_MEMBER, *expected_names}:
            raise ValueError("sealed A2 ZIP inventory differs from its manifest")

        namespace_prefix = f"{A2_PACKAGE_CODE_PREFIX}/{A2_HISTORICAL_NAMESPACE.replace('.', '/')}/"
        namespace_members: dict[str, dict[str, Any]] = {}
        verified_records: list[Mapping[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
                raise ValueError(f"sealed A2 manifest member[{index}] is invalid")
            relative = record["path"]
            expected_bytes = record["bytes"]
            expected_sha256 = record["sha256"]
            if (
                not isinstance(relative, str)
                or not relative
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
            ):
                raise ValueError(f"sealed A2 manifest member[{index}] identity is invalid")
            payload = archive.read(relative)
            if (
                len(payload) != expected_bytes
                or hashlib.sha256(payload).hexdigest() != expected_sha256
            ):
                raise ValueError(f"sealed A2 package member changed: {relative}")
            verified_records.append(record)
            if relative.startswith(namespace_prefix):
                if not relative.endswith(".py"):
                    raise ValueError("sealed A2 namespace contains a non-Python member")
                namespace_members[relative] = {
                    "archive_member": relative,
                    "bytes": expected_bytes,
                    "sha256": expected_sha256,
                }
        payload_tree_sha256 = _a2_payload_tree_sha256(verified_records)
        if payload_tree_sha256 != manifest["payload_tree_sha256"]:
            raise ValueError("sealed A2 package payload-tree digest changed")

    required_members = {
        f"{A2_PACKAGE_CODE_PREFIX}/{A2_HISTORICAL_NAMESPACE.replace('.', '/')}/{name}.py"
        for name in A2_REQUIRED_MODULES
    }
    if len(namespace_members) != 19 or not required_members.issubset(namespace_members):
        raise ValueError("sealed A2 package does not contain the exact runtime namespace")
    return {
        "authority_kind": "sha256_verified_lightweight_zip",
        "path": str(package),
        "bytes": package.stat().st_size,
        "sha256": package_sha256,
        "manifest_member": A2_PACKAGE_MANIFEST_MEMBER,
        "manifest_sha256": A2_PACKAGE_MANIFEST_SHA256,
        "archive_member_count": len(actual_names),
        "manifest_member_count": len(records),
        "payload_tree_sha256": payload_tree_sha256,
        "archive_source_prefix": A2_PACKAGE_CODE_PREFIX,
        "namespace": A2_HISTORICAL_NAMESPACE,
        "namespace_member_count": len(namespace_members),
        "namespace_members": namespace_members,
        "required_modules": list(A2_REQUIRED_MODULES),
        "archive_inventory_verified": True,
        "origin_verified": False,
        "git_head_role": "context_only_untracked",
    }


def _materialize_a2_historical_source(destination: Path) -> dict[str, Any]:
    """Materialize only the 19 verified A2 modules plus a parent-package shim."""

    binding = _a2_historical_source_binding()
    target = destination.resolve()
    namespace = target / Path(*A2_HISTORICAL_NAMESPACE.split("."))
    namespace.mkdir(parents=True, exist_ok=True)
    parent_shim = target / "cmr/__init__.py"
    parent_shim.parent.mkdir(parents=True, exist_ok=True)
    temporary_shim = parent_shim.with_name(f".{parent_shim.name}.part")
    temporary_shim.write_bytes(A2_PARENT_PACKAGE_SHIM)
    temporary_shim.replace(parent_shim)
    member_prefix = (
        f"{A2_PACKAGE_CODE_PREFIX}/{A2_HISTORICAL_NAMESPACE.replace('.', '/')}/"
    )
    expected_relative: set[str] = set()
    with zipfile.ZipFile(binding["path"]) as archive:
        for archive_member, record in binding["namespace_members"].items():
            relative = archive_member.removeprefix(member_prefix)
            if relative == archive_member or not relative or ".." in Path(relative).parts:
                raise ValueError(f"invalid A2 namespace member: {archive_member}")
            expected_relative.add(relative)
            output = namespace / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(archive_member)
            if (
                len(payload) != int(record["bytes"])
                or hashlib.sha256(payload).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"A2 source changed during materialization: {archive_member}")
            temporary = output.with_name(f".{output.name}.part")
            temporary.write_bytes(payload)
            temporary.replace(output)
    observed_relative = {
        path.relative_to(namespace).as_posix()
        for path in namespace.rglob("*")
        if path.is_file()
    }
    if observed_relative != expected_relative:
        raise ValueError("materialized A2 namespace contains stale or missing files")
    source_files = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    }
    expected_source_files = {
        "cmr/__init__.py",
        *(f"cmr/decaf_idsds_funnybirds_v1/{name}" for name in expected_relative),
    }
    if source_files != expected_source_files:
        raise ValueError("materialized A2 import root contains a stale extra file")
    return {
        **binding,
        "import_root": str(target),
        "materialized_namespace": str(namespace),
        "materialized_member_count": len(observed_relative),
        "parent_package_shim": {
            "path": str(parent_shim),
            "bytes": len(A2_PARENT_PACKAGE_SHIM),
            "sha256": hashlib.sha256(A2_PARENT_PACKAGE_SHIM).hexdigest(),
            "role": "verification_only_import_isolation",
            "historical_source": False,
        },
    }


def _bind_legacy_source(root: Path, package: str) -> None:
    root = root.resolve()
    loaded = sys.modules.get(package)
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file is not None and not Path(loaded_file).resolve().is_relative_to(root):
        raise RuntimeError(
            f"{package} is already imported from another tree: {loaded_file}"
        )
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)


def _verify_loaded_namespace(
    binding: dict[str, Any], *, required_modules: Sequence[str]
) -> None:
    namespace = str(binding["namespace"])
    raw_source_root = binding.get("import_root")
    if raw_source_root is None:
        raw_source_root = binding["source_root"]
    source_root = Path(str(raw_source_root)).resolve()
    namespace_root = source_root / Path(*namespace.split("."))
    loaded = {
        name: module
        for name, module in sys.modules.items()
        if name == namespace or name.startswith(f"{namespace}.")
    }
    if not loaded:
        raise RuntimeError(f"historical namespace was not imported: {namespace}")
    origins: dict[str, str | None] = {}
    for name, module in loaded.items():
        origin = getattr(module, "__file__", None)
        if name == namespace and origin is None:
            paths = [Path(value).resolve() for value in getattr(module, "__path__", ())]
            if paths != [namespace_root]:
                raise RuntimeError(f"historical namespace path is not isolated: {paths}")
            origins[name] = None
            continue
        if not isinstance(origin, str):
            raise RuntimeError(f"historical module has no filesystem origin: {name}")
        resolved = Path(origin).resolve()
        if not resolved.is_relative_to(namespace_root):
            raise RuntimeError(f"historical module escaped the bound namespace: {name}")
        origins[name] = str(resolved)
        if "namespace_members" in binding:
            relative = resolved.relative_to(namespace_root).as_posix()
            archive_member = (
                f"{binding['archive_source_prefix']}/{namespace.replace('.', '/')}/{relative}"
            )
            record = binding["namespace_members"].get(archive_member)
            if record is None or sha256_file(resolved) != record["sha256"]:
                raise RuntimeError(f"loaded historical module is not package-bound: {name}")
    required = {f"{namespace}.{name}" for name in required_modules}
    if not required.issubset(loaded):
        raise RuntimeError("not all required historical modules were imported")
    parent = sys.modules.get("cmr")
    parent_origin = getattr(parent, "__file__", None)
    expected_parent = source_root / "cmr/__init__.py"
    if not isinstance(parent_origin, str) or Path(parent_origin).resolve() != expected_parent:
        raise RuntimeError("historical parent package did not originate from the bound tree")
    anchor_origins: dict[str, str] = {}
    for relative, record in binding.get("anchor_files", {}).items():
        module_parts = Path(relative).with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module_name = ".".join(module_parts)
        module = sys.modules.get(module_name)
        origin = getattr(module, "__file__", None)
        expected_origin = Path(str(record["path"])).resolve()
        if (
            not isinstance(origin, str)
            or Path(origin).resolve() != expected_origin
            or sha256_file(expected_origin) != record["sha256"]
        ):
            raise RuntimeError(f"A0 anchor module origin is not receipt-bound: {module_name}")
        anchor_origins[module_name] = str(expected_origin)
    binding["origin_verified"] = True
    binding["parent_package_origin"] = str(expected_parent)
    binding["loaded_module_origins"] = dict(sorted(origins.items()))
    if anchor_origins:
        binding["loaded_anchor_origins"] = dict(sorted(anchor_origins.items()))


@contextmanager
def _strict_fp32(torch: Any) -> Iterator[dict[str, bool]]:
    old_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    old_cudnn = bool(torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _tensor_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64)


def _emit_progress(**payload: Any) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _historical_vectors(
    frame: pd.DataFrame, part_names: Sequence[str]
) -> dict[str, np.ndarray]:
    indexed = frame.set_index("part_group", verify_integrity=True)
    missing = sorted(set(part_names) - set(indexed.index.astype(str)))
    if missing:
        raise ValueError(f"sealed component rows are missing parts: {missing}")
    selected = indexed.loc[list(part_names)]
    return {
        name: selected[name].to_numpy(dtype=np.float64)
        for name in ("M", "E", "C", "F", "Abs")
    }


def _export_funnybirds(
    *,
    model_id: str,
    selection: Mapping[str, Any],
    device: str,
    methods: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_binding = _a0_historical_source_binding()
    _bind_legacy_source(A0_SOURCE_ROOT, "cmr")
    import torch
    from cmr.decaf_reference_locked_v1 import methods as legacy_methods
    from cmr.decaf_reference_locked_v1 import models as legacy_models
    from cmr.decaf_reference_locked_v1 import worker as legacy_worker

    _verify_loaded_namespace(
        source_binding, required_modules=tuple(A0_REQUIRED_MODULE_SHA256)
    )

    jobs = [json.loads(line) for line in A0_PLAN.read_text().splitlines() if line]
    candidates = sorted(
        (
            job
            for job in jobs
            if job.get("dataset") == "funnybirds"
            and job.get("model_id") == model_id
            and job.get("method_id") == "decaf_3"
            and job.get("mode") == "science"
            and job.get("track") == "main"
            and job.get("reference_id") == REFERENCE_ID
        ),
        key=lambda job: str(job["shard_id"]),
    )
    requested = list(selection["image_ids"])
    sample_by_id: dict[str, Any] = {}
    source_jobs: dict[str, Mapping[str, Any]] = {}
    for job in candidates:
        for sample in legacy_worker.load_frozen_samples(job):
            if sample.image_id in requested:
                sample_by_id[sample.image_id] = sample
                source_jobs[sample.image_id] = job
        if len(sample_by_id) == len(requested):
            break
    if set(sample_by_id) != set(requested):
        raise ValueError("not all fixed FunnyBirds samples were found in frozen shards")
    samples = [sample_by_id[image_id] for image_id in requested]
    for sample, target in zip(samples, selection["targets"], strict=True):
        if int(sample.target_class) != int(target):
            raise ValueError(f"target mismatch for {sample.image_id}")

    components = pd.read_parquet(A0_COMPONENTS)
    components = components[
        components["dataset"].eq("funnybirds")
        & components["model"].eq(model_id)
        & components["method"].isin(methods)
        & components["track"].eq("main")
        & components["reference"].eq(REFERENCE_ID)
        & components["image_id"].isin(requested)
    ].copy()
    layout = legacy_models.ModelCacheLayout.from_prevalidated_root(A0_CACHE_ROOT)
    spec = legacy_models.get_formal_model_spec(model_id)
    checkpoint_paths = spec.cache_paths(layout)
    if len(checkpoint_paths) != 1:
        raise ValueError(f"expected one checkpoint for {model_id}: {checkpoint_paths}")
    checkpoint_path = checkpoint_paths[0]
    checkpoint_sha256 = sha256_file(checkpoint_path)
    torch_device = torch.device(device)
    model = legacy_models.build_inference_model(
        model_id,
        cache_root=A0_CACHE_ROOT,
        device=torch_device,
        trusted_prevalidated_offline=True,
    )
    model.eval()

    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(torch_device)
    torch.cuda.synchronize(torch_device)
    started = time.perf_counter()
    with _strict_fp32(torch) as numeric_contract:
        for method in methods:
            for sample in samples:
                job = source_jobs[sample.image_id]
                _interventions, baseline = legacy_worker.load_precomputed_science_inputs(
                    sample, REFERENCE_ID
                )
                result = legacy_methods.decaf_part_attribution(
                    model,
                    sample.image.to(torch_device),
                    baseline.to(torch_device),
                    sample.masks.to(torch_device),
                    int(sample.target_class),
                    schedule=SCHEDULES[method],
                    epsilon=ENDPOINT_EPSILON,
                )
                meta = result.metadata
                q_plus = _tensor_numpy(meta["q_plus"])
                q_minus = _tensor_numpy(meta["q_minus"])
                sealed = components[
                    components["method"].eq(method)
                    & components["image_id"].eq(sample.image_id)
                ]
                if len(sealed) != len(sample.part_names):
                    raise ValueError(
                        f"sealed row count mismatch for {method}/{sample.image_id}"
                    )
                rows.extend(
                    _trajectory_rows(
                        dataset="funnybirds",
                        reference_run="decaf_reference_locked_v1",
                        model_id=model_id,
                        checkpoint_sha256=checkpoint_sha256,
                        image_id=sample.image_id,
                        target=int(sample.target_class),
                        method=method,
                        part_names=sample.part_names,
                        stage_t=meta["schedule"],
                        q_plus=q_plus,
                        q_minus=q_minus,
                        historical=_historical_vectors(sealed, sample.part_names),
                        counterfactual_map=REFERENCE_ID,
                        metadata={
                            **numeric_contract,
                            "legacy_model_source": spec.source,
                            "legacy_checkpoint_path": str(checkpoint_path),
                            "legacy_shard_id": str(job["shard_id"]),
                            "legacy_shard_sha256": str(job["shard_sha256"]),
                            "legacy_config_sha256": str(job["config_sha256"]),
                        },
                    )
                )
                _emit_progress(
                    event="sample_complete",
                    dataset="funnybirds",
                    model=model_id,
                    method=method,
                    image_id=sample.image_id,
                )
    torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - started
    details = {
        "reference_run": "decaf_reference_locked_v1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(torch_device)),
        "source_code_root": str(A0_SOURCE_ROOT),
        "historical_source_binding": source_binding,
        "source_plan": str(A0_PLAN),
        "source_plan_sha256": sha256_file(A0_PLAN),
        "sealed_results": str(A0_COMPONENTS),
        "sealed_results_sha256": sha256_file(A0_COMPONENTS),
        "source_shards": sorted(
            {
                str(source_jobs[image_id]["shard_path"]): str(
                    source_jobs[image_id]["shard_sha256"]
                )
                for image_id in requested
            }.items()
        ),
        "numeric_contract": numeric_contract,
    }
    return pd.DataFrame(rows, columns=NEUTRAL_COLUMNS), details


def _export_imagenet(
    *,
    model_id: str,
    selection: Mapping[str, Any],
    device: str,
    methods: Sequence[str],
    internal_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_binding = _materialize_a2_historical_source(A2_MATERIALIZED_SOURCE)
    _bind_legacy_source(Path(source_binding["import_root"]), "cmr")
    import torch
    from cmr.decaf_idsds_funnybirds_v1 import attribution as legacy_attribution
    from cmr.decaf_idsds_funnybirds_v1 import data as legacy_data
    from cmr.decaf_idsds_funnybirds_v1 import models as legacy_models

    _verify_loaded_namespace(source_binding, required_modules=A2_REQUIRED_MODULES)

    requested = list(selection["image_ids"])
    source_manifest = pd.read_parquet(A2_INPUT_MANIFEST)
    if source_manifest["image_id"].duplicated().any():
        raise ValueError("historical IDSDS input manifest has duplicate image ids")
    indexed = source_manifest.set_index("image_id", verify_integrity=True)
    missing = sorted(set(requested) - set(indexed.index.astype(str)))
    if missing:
        raise ValueError(f"IDSDS manifest is missing fixed samples: {missing}")
    selected_manifest = indexed.loc[requested].reset_index()
    expected_targets = np.asarray(selection["targets"], dtype=np.int64)
    actual_targets = selected_manifest["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(expected_targets, actual_targets):
        raise ValueError("fixed targets differ from the historical IDSDS manifest")
    dataset = legacy_data.ImageNetParquetDataset(selected_manifest, model_id=model_id)

    columns = [
        "dataset",
        "scope",
        "model",
        "method",
        "image_id",
        "label",
        "correctly_classified",
        "effects",
        "spearman",
        "finite_complete",
        "deletion_target_sha256",
        "decaf_M",
        "decaf_E",
        "decaf_C",
        "decaf_F",
        "decaf_Abs",
        "member_path",
    ]
    sealed = pd.read_parquet(
        A2_RESULTS,
        columns=columns,
        filters=[("model", "=", model_id), ("scope", "=", "science")],
    )
    sealed = sealed[
        sealed["method"].isin(methods) & sealed["image_id"].isin(requested)
    ].copy()
    if len(sealed) != len(methods) * len(requested):
        raise ValueError("sealed IDSDS row count differs from methods x fixed samples")

    torch_device = torch.device(device)
    model = legacy_models.load_idsds_model_adapter(
        model_id,
        device=torch_device,
        precision="fp32",
        source_root=A2_DATA_ROOT / "official/idsds",
        checkpoint_root=A2_DATA_ROOT / "official/idsds_checkpoints",
    )
    checkpoint_path = Path(model.model.idsds_checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != str(model.model.idsds_checkpoint_sha256):
        raise ValueError("loaded IDSDS checkpoint hash differs from its model contract")
    model.eval()

    rows: list[dict[str, Any]] = []
    patch_names = tuple(f"patch_{index:02d}" for index in range(16))
    torch.cuda.reset_peak_memory_stats(torch_device)
    torch.cuda.synchronize(torch_device)
    started = time.perf_counter()
    with _strict_fp32(torch) as numeric_contract:
        for method in methods:
            for index in range(len(dataset)):
                item = dataset[index]
                image_id = str(item["image_id"])
                target = int(item["label"])
                result = legacy_attribution.compute_decaf_attribution(
                    model,
                    item["image"].unsqueeze(0).to(torch_device),
                    torch.tensor([target], device=torch_device),
                    schedule=SCHEDULES[method],
                    epsilon=ENDPOINT_EPSILON,
                    internal_batch_size=internal_batch_size,
                )
                meta = result.metadata
                matches = sealed[
                    sealed["method"].eq(method) & sealed["image_id"].eq(image_id)
                ]
                if len(matches) != 1:
                    raise ValueError(f"sealed IDSDS row mismatch for {method}/{image_id}")
                historical_row = matches.iloc[0]
                if (
                    int(historical_row["label"]) != target
                    or not bool(historical_row["correctly_classified"])
                    or not bool(historical_row["finite_complete"])
                ):
                    raise ValueError(f"sealed IDSDS contract failed for {method}/{image_id}")
                history = {
                    name: historical_row[f"decaf_{name}"]
                    for name in ("M", "E", "C", "F", "Abs")
                }
                rows.extend(
                    _trajectory_rows(
                        dataset="imagenet1k_idsds",
                        reference_run="decaf_idsds_funnybirds_v1",
                        model_id=model_id,
                        checkpoint_sha256=checkpoint_sha256,
                        image_id=image_id,
                        target=target,
                        method=method,
                        part_names=patch_names,
                        stage_t=meta["schedule"],
                        q_plus=_tensor_numpy(meta["q_plus"])[0],
                        q_minus=_tensor_numpy(meta["q_minus"])[0],
                        historical=history,
                        historical_endpoint_d=historical_row["effects"],
                        counterfactual_map="normalized_zero_4x4_patch_deletion",
                        metadata={
                            **numeric_contract,
                            "legacy_checkpoint_path": str(checkpoint_path),
                            "legacy_member_path": str(historical_row["member_path"]),
                            "deletion_target_sha256": str(
                                historical_row["deletion_target_sha256"]
                            ),
                            "sealed_spearman": float(historical_row["spearman"]),
                            "internal_batch_size": int(internal_batch_size),
                            "source_shard": str(item["source_shard"]),
                            "source_row_index": int(item["row_index"]),
                        },
                    )
                )
                _emit_progress(
                    event="sample_complete",
                    dataset="imagenet1k_idsds",
                    model=model_id,
                    method=method,
                    image_id=image_id,
                )
    torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - started
    details = {
        "reference_run": "decaf_idsds_funnybirds_v1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(torch_device)),
        "internal_batch_size": int(internal_batch_size),
        "source_code_root": str(source_binding["import_root"]),
        "historical_source_binding": source_binding,
        "source_input_manifest": str(A2_INPUT_MANIFEST),
        "source_input_manifest_sha256": sha256_file(A2_INPUT_MANIFEST),
        "sealed_results": str(A2_RESULTS),
        "sealed_results_sha256": sha256_file(A2_RESULTS),
        "numeric_contract": numeric_contract,
    }
    return pd.DataFrame(rows, columns=NEUTRAL_COLUMNS), details


def export_attribution(
    *,
    dataset: str,
    model_id: str,
    sample_manifest: Path,
    output: Path,
    selection_manifest: Path,
    device: str = "cuda:0",
    methods: Sequence[str] = METHODS,
    internal_batch_size: int = 17,
) -> dict[str, Any]:
    """Export one model's fixed eight-sample attribution trajectory record."""

    selected_methods = tuple(methods)
    if dataset not in MODELS or model_id not in MODELS[dataset]:
        raise ValueError(f"unsupported dataset/model pair: {dataset}/{model_id}")
    if (
        not selected_methods
        or len(set(selected_methods)) != len(selected_methods)
        or any(method not in METHODS for method in selected_methods)
    ):
        raise ValueError(f"methods must be a unique subset of {METHODS}")
    if internal_batch_size <= 0:
        raise ValueError("internal_batch_size must be positive")
    selection = _read_fixed_manifest(
        sample_manifest, dataset=dataset, model_id=model_id
    )
    if dataset == "funnybirds":
        frame, details = _export_funnybirds(
            model_id=model_id,
            selection=selection,
            device=device,
            methods=selected_methods,
        )
    else:
        frame, details = _export_imagenet(
            model_id=model_id,
            selection=selection,
            device=device,
            methods=selected_methods,
            internal_batch_size=internal_batch_size,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.stem}.part{output.suffix}")
    temporary_output.unlink(missing_ok=True)
    try:
        write_trajectory_record(frame, temporary_output)
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
    written = output
    manifest = {
        "schema_version": 1,
        "experiment_family": "attribution",
        "dataset": dataset,
        "model_id": model_id,
        "device": device,
        "methods": list(selected_methods),
        "sample_count": len(selection["image_ids"]),
        "image_ids": list(selection["image_ids"]),
        "targets": list(selection["targets"]),
        "selection": selection["selection"],
        "fixed_sample_manifest": str(sample_manifest.resolve()),
        "fixed_sample_manifest_sha256": sha256_file(sample_manifest),
        "trajectory_record": str(written.resolve()),
        "trajectory_record_sha256": sha256_file(written),
        "trajectory_row_count": int(len(frame)),
        "trajectory_unit_count": int(frame["unit_id"].nunique()),
        **details,
    }
    selection_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = selection_manifest.with_name(
        f".{selection_manifest.name}.part"
    )
    temporary_manifest.unlink(missing_ok=True)
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(selection_manifest)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(MODELS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--internal-batch-size", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = export_attribution(
        dataset=args.dataset,
        model_id=args.model,
        sample_manifest=args.sample_manifest,
        output=args.output,
        selection_manifest=args.selection_manifest,
        device=args.device,
        methods=args.methods,
        internal_batch_size=args.internal_batch_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
