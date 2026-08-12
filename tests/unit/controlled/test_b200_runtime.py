from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from decaf.experiments.controlled.gpu_runtime import (
    CHECKPOINT_MANIFEST_ENV,
    audit_score_trajectory,
    b200_enabled,
    build_b200_members,
    canonical_tensor_identity,
    load_b200_inventory,
    validate_checkpoint_fingerprint_records,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(
    root: Path,
    *,
    case_id: str,
    family: str,
    behavior: str,
    architecture: str,
    fingerprint: bool = False,
) -> dict[str, object]:
    checkpoint = root / f"{case_id}.pt"
    checkpoint.write_bytes(f"checkpoint:{case_id}".encode())
    return {
        "case_id": case_id,
        "family": family,
        "expected_behavior": behavior,
        "architecture": architecture,
        "model_id": f"model_{case_id}",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _digest(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "task": "invert" if family == "context_swap" else "object_shape",
        "factor": (
            "object_color"
            if family == "context_swap"
            else "wall_color"
            if family == "fragility"
            else "object_shape"
        ),
        "seed": 7,
        "fingerprint": fingerprint,
        "fingerprint_sample_ids": [0, 15],
    }


def _inventory_environment(tmp_path: Path) -> dict[str, str]:
    cases = [
        _case(
            tmp_path,
            case_id="base_r18",
            family="base",
            behavior="active",
            architecture="resnet18",
            fingerprint=True,
        ),
        _case(
            tmp_path,
            case_id="evidence_vit",
            family="evidence",
            behavior="aligned",
            architecture="small_vit",
            fingerprint=True,
        ),
        _case(
            tmp_path,
            case_id="fragility_r18",
            family="fragility",
            behavior="null",
            architecture="resnet18",
        ),
        _case(
            tmp_path,
            case_id="context_vit",
            family="context-swap",
            behavior="opposed",
            architecture="small-vit",
        ),
    ]
    manifest = tmp_path / "controlled_b200_cases.yaml"
    manifest.write_text(yaml.safe_dump({"schema_version": 1, "cases": cases}))
    return {CHECKPOINT_MANIFEST_ENV: str(manifest)}


def test_b200_gate_is_explicit() -> None:
    assert not b200_enabled({})
    assert not b200_enabled({"DECAF_B200_VERIFY": "true"})
    assert b200_enabled({"DECAF_B200_VERIFY": "1"})


def test_external_manifest_builds_only_real_cuda_members(tmp_path: Path) -> None:
    environment = _inventory_environment(tmp_path)
    inventory = load_b200_inventory(environment)
    assert {case.family for case in inventory.cases} == {
        "base",
        "evidence",
        "fragility",
        "context_swap",
    }
    assert {case.expected_behavior for case in inventory.cases} == {
        "active",
        "aligned",
        "null",
        "opposed",
    }
    members = build_b200_members({"profile": "smoke"}, environment)
    assert len(members) == 4
    assert all(member.resource == "cuda:0" for member in members)
    assert all(member.phase.startswith("b200_") for member in members)
    assert all(str(tmp_path) not in str(member.as_dict()) for member in members)


def test_external_manifest_fails_closed_on_missing_coverage(tmp_path: Path) -> None:
    environment = _inventory_environment(tmp_path)
    manifest = Path(environment[CHECKPOINT_MANIFEST_ENV])
    payload = yaml.safe_load(manifest.read_text())
    payload["cases"] = payload["cases"][:-1]
    manifest.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="family coverage incomplete"):
        load_b200_inventory(environment)


def test_cpu_score_audit_checks_conservation_and_endpoint_swap() -> None:
    alpha = (0.0, 0.5, 1.0)
    response = np.asarray(
        [
            [0.0, 0.4, 1.0],
            [0.0, -0.6, 1.0],
            [0.0, 0.5, 0.01],
        ],
        dtype=np.float32,
    )
    scores, audit = audit_score_trajectory(
        alpha,
        response,
        response[:, -1],
        epsilon=0.02,
    )
    assert audit["passed"]
    assert audit["pointwise_conservation"]["passed"]
    assert audit["integrated_conservation"]["passed"]
    assert audit["tiny_endpoint_swap"]["passed"]
    assert scores["endpoint_active"].tolist() == [True, True, False]
    assert float(scores["C"][1]) > 0.0
    assert float(scores["F"][2]) > 0.0


def test_canonical_tensor_and_two_case_fingerprint_contract(tmp_path: Path) -> None:
    tensor = np.zeros((1, 3, 32, 32), dtype=np.float32)
    identity = canonical_tensor_identity(tensor)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    def record(architecture: str) -> dict[str, object]:
        return {
            "family": "controlled",
            "case_id": f"controlled_{architecture}",
            "architecture": architecture,
            "model_id": architecture,
            "checkpoints": [
                {
                    "path": str(checkpoint),
                    "sha256": _digest(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                }
            ],
            "sample_ids": [0],
            "preprocessed_tensor": identity,
            "target_class": [1],
            "logits": [[0.0, 1.0]],
            "probabilities": [[0.25, 0.75]],
            "precision": "float32",
            "device": "cuda:0",
        }

    records = [record("resnet18"), record("small_vit")]
    assert validate_checkpoint_fingerprint_records(records) == records
    with pytest.raises(ValueError, match="exactly two"):
        validate_checkpoint_fingerprint_records(records[:1])
    records[1]["probabilities"] = [[0.2, 0.7]]
    with pytest.raises(ValueError, match="logits/probabilities"):
        validate_checkpoint_fingerprint_records(records)
