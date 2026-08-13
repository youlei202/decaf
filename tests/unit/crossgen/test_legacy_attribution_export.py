from __future__ import annotations

import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from decaf.core.trajectories import trajectory_scores
from tools.crossgen import legacy_attribution_export as exporter
from tools.crossgen.compare_core import compare_record
from tools.crossgen.legacy_attribution_export import (
    _read_fixed_manifest,
    _strict_fp32,
    _trajectory_rows,
)
from tools.crossgen.schema import (
    NEUTRAL_COLUMNS,
    validate_trajectory_record,
    write_trajectory_record,
)


def _synthetic_rows() -> list[dict[str, object]]:
    grid = np.asarray([0.0, 0.5, 1.0])
    q_plus = np.asarray([0.5, 0.6, 0.8])
    q_minus = np.asarray(
        [
            [0.5, 0.5, 0.4],
            [0.5, 0.61, 0.795],
        ]
    )
    endpoint = q_plus[-1] - q_minus[:, -1]
    summaries = [
        trajectory_scores(grid, q_plus - branch, effect, 0.02)
        for branch, effect in zip(q_minus, endpoint, strict=True)
    ]
    historical = {
        name: np.asarray([float(summary[name]) for summary in summaries])
        for name in ("M", "E", "C", "F", "Abs")
    }
    return _trajectory_rows(
        dataset="imagenet1k_idsds",
        reference_run="sealed",
        model_id="resnet50",
        checkpoint_sha256="a" * 64,
        image_id="image",
        target=7,
        method="decaf_3",
        part_names=("patch_00", "patch_01"),
        stage_t=grid,
        q_plus=q_plus,
        q_minus=q_minus,
        historical=historical,
        historical_endpoint_d=endpoint,
        counterfactual_map="normalized_zero_4x4_patch_deletion",
        metadata={"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
    )


def test_trajectory_rows_validate_and_recompute_exactly(tmp_path: Path) -> None:
    frame = validate_trajectory_record(
        pd.DataFrame(_synthetic_rows(), columns=NEUTRAL_COLUMNS)
    )
    assert len(frame) == 6
    assert frame["unit_id"].nunique() == 2
    assert frame.groupby("unit_id")["quadrature_weight"].sum().eq(1.0).all()

    active = frame[frame["factor_or_part_id"].eq("patch_00")]
    active_metadata = json.loads(active["metadata_json"].iloc[0])
    assert active_metadata["historical_gate"] is True
    assert active_metadata["historical_orientation"] == 1
    assert active_metadata["historical_dominant"] in {"E", "C", "F"}
    assert active_metadata["current_factor_or_part_id"] == "patch_00"
    assert active_metadata["cuda_matmul_allow_tf32"] is False
    assert active_metadata["cudnn_allow_tf32"] is False

    inactive = frame[frame["factor_or_part_id"].eq("patch_01")]
    inactive_metadata = json.loads(inactive["metadata_json"].iloc[0])
    assert inactive_metadata["historical_gate"] is False
    assert inactive_metadata["historical_orientation"] == 0

    trajectory = write_trajectory_record(frame, tmp_path / "attribution.parquet")
    result = compare_record(trajectory, tmp_path / "comparison.parquet")
    assert result["summary"]["unit_count"] == 2
    assert result["summary"]["tier_a_fraction"] == 1.0
    assert result["summary"]["hard_mismatch_fraction"] == 0.0
    assert result["summary"]["identity_agreement"] == 1.0


def test_trajectory_rows_reject_sealed_endpoint_drift() -> None:
    rows = _synthetic_rows()
    historical = {
        name: [rows[0][f"historical_{name}"], rows[3][f"historical_{name}"]]
        for name in ("M", "E", "C", "F", "Abs")
    }
    with pytest.raises(ValueError, match="historical M"):
        _trajectory_rows(
            dataset="imagenet1k_idsds",
            reference_run="sealed",
            model_id="resnet50",
            checkpoint_sha256="a" * 64,
            image_id="image",
            target=7,
            method="decaf_3",
            part_names=("patch_00", "patch_01"),
            stage_t=[0.0, 0.5, 1.0],
            q_plus=[0.5, 0.6, 0.8],
            q_minus=[[0.5, 0.5, 0.4], [0.5, 0.61, 0.795]],
            historical=historical,
            historical_endpoint_d=[0.1, 0.005],
            counterfactual_map="map",
        )


def test_fixed_manifest_requires_exact_frozen_eight(tmp_path: Path) -> None:
    path = tmp_path / "fixed.json"
    payload = {
        "dataset": "funnybirds",
        "model_id": "funnybirds_resnet50",
        "selection": "first_eight_in_frozen_candidate_order",
        "image_ids": [f"0/{index:06d}" for index in range(8)],
        "targets": [0] * 8,
    }
    path.write_text(json.dumps(payload))
    selected = _read_fixed_manifest(
        path, dataset="funnybirds", model_id="funnybirds_resnet50"
    )
    assert selected["image_ids"] == payload["image_ids"]

    payload["image_ids"][-1] = payload["image_ids"][0]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="eight unique"):
        _read_fixed_manifest(
            path, dataset="funnybirds", model_id="funnybirds_resnet50"
        )


def test_strict_fp32_is_scoped_and_restored() -> None:
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        )
    )
    with _strict_fp32(fake_torch) as contract:
        assert contract == {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        assert fake_torch.backends.cuda.matmul.allow_tf32 is False
        assert fake_torch.backends.cudnn.allow_tf32 is False
    assert fake_torch.backends.cuda.matmul.allow_tf32 is True
    assert fake_torch.backends.cudnn.allow_tf32 is True


def _write_a2_package(path: Path) -> tuple[str, str, str]:
    prefix = "code_snapshot/src/cmr/decaf_idsds_funnybirds_v1/"
    payloads = {
        f"{prefix}{name}.py": f"VALUE = {index!r}\n".encode()
        for index, name in enumerate(
            (
                *exporter.A2_REQUIRED_MODULES,
                *(
                    f"extra_{index:02d}"
                    for index in range(19 - len(exporter.A2_REQUIRED_MODULES))
                ),
            )
        )
    }
    records = [
        {
            "path": member,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for member, payload in sorted(payloads.items())
    ]
    manifest = {
        "schema_version": 1,
        "package_kind": "lightweight",
        "payload_tree_sha256": exporter._a2_payload_tree_sha256(records),
        "members": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in payloads.items():
            archive.writestr(member, payload)
        archive.writestr(exporter.A2_PACKAGE_MANIFEST_MEMBER, manifest_bytes)
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        hashlib.sha256(manifest_bytes).hexdigest(),
        manifest["payload_tree_sha256"],
    )


def test_a2_source_binding_materializes_exact_namespace_and_isolates_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "a2.zip"
    package_sha256, manifest_sha256, tree_sha256 = _write_a2_package(package)
    monkeypatch.setattr(exporter, "A2_HISTORICAL_PACKAGE", package)
    monkeypatch.setattr(exporter, "A2_HISTORICAL_PACKAGE_SHA256", package_sha256)
    monkeypatch.setattr(exporter, "A2_PACKAGE_MANIFEST_SHA256", manifest_sha256)
    monkeypatch.setattr(exporter, "A2_PACKAGE_PAYLOAD_TREE_SHA256", tree_sha256)
    exporter._a2_historical_source_binding.cache_clear()
    source = tmp_path / "source"
    names = [
        name
        for name in list(sys.modules)
        if name == "cmr" or name.startswith(f"{exporter.A2_HISTORICAL_NAMESPACE}.")
    ]
    saved_modules = {name: sys.modules.pop(name) for name in names}
    original_path = list(sys.path)
    try:
        binding = exporter._materialize_a2_historical_source(source)
        exporter._bind_legacy_source(source, "cmr")
        for module in exporter.A2_REQUIRED_MODULES:
            importlib.import_module(f"{exporter.A2_HISTORICAL_NAMESPACE}.{module}")
        exporter._verify_loaded_namespace(
            binding, required_modules=exporter.A2_REQUIRED_MODULES
        )
    finally:
        for name in list(sys.modules):
            if name == "cmr" or name.startswith(
                f"{exporter.A2_HISTORICAL_NAMESPACE}."
            ):
                sys.modules.pop(name)
        sys.modules.update(saved_modules)
        sys.path[:] = original_path
        exporter._a2_historical_source_binding.cache_clear()

    assert binding["namespace_member_count"] == 19
    assert binding["archive_member_count"] == 20
    assert binding["manifest_member_count"] == 19
    assert binding["archive_inventory_verified"] is True
    assert binding["origin_verified"] is True
    assert len(binding["loaded_module_origins"]) == 1 + len(
        exporter.A2_REQUIRED_MODULES
    )
    assert Path(binding["parent_package_shim"]["path"]).read_bytes() == (
        exporter.A2_PARENT_PACKAGE_SHIM
    )


def test_a2_materialization_rejects_stale_extra_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "a2.zip"
    package_sha256, manifest_sha256, tree_sha256 = _write_a2_package(package)
    monkeypatch.setattr(exporter, "A2_HISTORICAL_PACKAGE", package)
    monkeypatch.setattr(exporter, "A2_HISTORICAL_PACKAGE_SHA256", package_sha256)
    monkeypatch.setattr(exporter, "A2_PACKAGE_MANIFEST_SHA256", manifest_sha256)
    monkeypatch.setattr(exporter, "A2_PACKAGE_PAYLOAD_TREE_SHA256", tree_sha256)
    exporter._a2_historical_source_binding.cache_clear()
    namespace = tmp_path / "source/cmr/decaf_idsds_funnybirds_v1"
    namespace.mkdir(parents=True)
    (namespace / "stale.py").write_text("STALE = True\n")
    try:
        with pytest.raises(ValueError, match="stale or missing"):
            exporter._materialize_a2_historical_source(tmp_path / "source")
    finally:
        exporter._a2_historical_source_binding.cache_clear()


def test_a0_binding_recomputes_receipt_bound_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = tmp_path / "ready"
    source = ready / "code"
    namespace = source / "cmr/decaf_reference_locked_v1"
    namespace.mkdir(parents=True)
    (source / "cmr/__init__.py").write_text("# cmr\n")
    (namespace / "__init__.py").write_text("# namespace\n")
    cross_namespace = source / "cmr/decaf_imagenet9_v1"
    cross_namespace.mkdir()
    (cross_namespace / "decaf.py").write_text("# shared decaf\n")
    module_hashes: dict[str, str] = {}
    for name in exporter.A0_REQUIRED_MODULE_SHA256:
        path = namespace / f"{name}.py"
        path.write_text(f"# {name}\n")
        module_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree_sha256, _ = exporter._formal_python_tree_digest(source / "cmr")
    plan = ready / "plans/formal_jobs.jsonl"
    plan.parent.mkdir(parents=True)
    plan.write_text('{"job":1}\n')
    plan_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()
    plan_receipt = ready / "plans/formal_jobs.jsonl.receipt.json"
    plan_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "decaf_reference_locked_formal_job_plan",
                "plan_path": str(plan),
                "plan_sha256": plan_sha256,
                "code_sha256": tree_sha256,
                "job_count": 9184,
            }
        )
    )
    plan_receipt_sha256 = hashlib.sha256(plan_receipt.read_bytes()).hexdigest()
    deployment = ready / "deployment_receipt.json"
    deployment.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "code_root": str(source),
                "code_sha256": tree_sha256,
                "entrypoint": "cmr.decaf_reference_locked_v1.run",
                "code_changes_required": False,
                "downloads_required": False,
                "installs_required": False,
                "formal_job_plan": {
                    "path": "plans/formal_jobs.jsonl",
                    "bytes": plan.stat().st_size,
                    "sha256": plan_sha256,
                },
                "formal_job_plan_receipt": {
                    "path": "plans/formal_jobs.jsonl.receipt.json",
                    "bytes": plan_receipt.stat().st_size,
                    "sha256": plan_receipt_sha256,
                },
                "formal_job_plan_rebind": {
                    "schema_version": 1,
                    "kind": "formal_job_plan_receipt_code_rebind",
                    "code_sha256": tree_sha256,
                    "plan_sha256_unchanged": True,
                    "rebound_fields": ["code_sha256"],
                    "plan": {
                        "path": str(plan),
                        "bytes": plan.stat().st_size,
                        "sha256": plan_sha256,
                    },
                    "receipt": {
                        "path": str(plan_receipt),
                        "bytes": plan_receipt.stat().st_size,
                        "sha256": plan_receipt_sha256,
                    },
                },
            }
        )
    )
    monkeypatch.setattr(exporter, "A0_READY_ROOT", ready)
    monkeypatch.setattr(exporter, "A0_SOURCE_ROOT", source)
    monkeypatch.setattr(exporter, "A0_PLAN", plan)
    monkeypatch.setattr(exporter, "A0_FORMAL_PLAN_RECEIPT", plan_receipt)
    monkeypatch.setattr(exporter, "A0_DEPLOYMENT_RECEIPT", deployment)
    monkeypatch.setattr(
        exporter,
        "A0_DEPLOYMENT_RECEIPT_SHA256",
        hashlib.sha256(deployment.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        exporter, "A0_FORMAL_PLAN_RECEIPT_SHA256", plan_receipt_sha256
    )
    monkeypatch.setattr(exporter, "A0_FORMAL_PLAN_SHA256", plan_sha256)
    monkeypatch.setattr(exporter, "A0_FORMAL_SOURCE_SHA256", tree_sha256)
    monkeypatch.setattr(exporter, "A0_REQUIRED_MODULE_SHA256", module_hashes)
    monkeypatch.setattr(
        exporter,
        "A0_ANCHOR_FILE_SHA256",
        {
            "cmr/decaf_reference_locked_v1/__init__.py": hashlib.sha256(
                (namespace / "__init__.py").read_bytes()
            ).hexdigest(),
            "cmr/decaf_imagenet9_v1/decaf.py": hashlib.sha256(
                (cross_namespace / "decaf.py").read_bytes()
            ).hexdigest(),
        },
    )
    exporter._a0_historical_source_binding.cache_clear()
    try:
        binding = exporter._a0_historical_source_binding()
        (namespace / "methods.py").write_text("# drift\n")
        exporter._a0_historical_source_binding.cache_clear()
        with pytest.raises(ValueError, match="source-tree digest"):
            exporter._a0_historical_source_binding()
    finally:
        exporter._a0_historical_source_binding.cache_clear()

    assert binding["source_tree_sha256"] == tree_sha256
    assert binding["source_python_file_count"] == 6
    assert binding["git_head_role"] == "context_only_untracked"
    assert set(binding["required_modules"]) == set(module_hashes)
