from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from decaf.core.receipts import load_member_receipt, write_member_receipt
from decaf.experiments.attribution import evaluate as attribution_evaluate
from decaf.experiments.attribution.evaluate import (
    _compute_single_gpu_queue,
    _finalize_gpu_queue_after_termination,
)
from decaf.experiments.attribution.gpu_runtime import (
    CHECKPOINT_SPECS,
    FINGERPRINT_CASES,
    CheckpointSpec,
    _normalize_idsds,
    resolve_checkpoint,
    validate_checkpoint_fingerprint_rows,
)
from decaf.experiments.attribution.plan import (
    VERIFICATION_PROFILE_MEMBER_COUNTS,
    build_plan,
)
from decaf.experiments.common import TerminationRequested, load_profile, repository_root


def test_single_b200_profiles_are_small_exact_real_compute_plans() -> None:
    expected = {
        "smoke-b200": 72,
        "large-model-smoke": 16,
        "boundary-smoke": 8,
        "smoke-resume": 5,
    }
    assert VERIFICATION_PROFILE_MEMBER_COUNTS == expected
    paths = {
        "smoke-b200": repository_root() / "configs/attribution/smoke_b200.yaml",
        "large-model-smoke": repository_root() / "configs/attribution/large-model-smoke.yaml",
        "boundary-smoke": repository_root() / "configs/attribution/boundary-smoke.yaml",
        "smoke-resume": repository_root() / "configs/attribution/smoke-resume.yaml",
    }
    for profile_key, path in paths.items():
        cli_profile = "smoke" if profile_key == "smoke-b200" else profile_key
        config = load_profile("attribution", cli_profile, path)
        plan = build_plan(config)
        assert plan["profile_key"] == profile_key
        assert plan["member_count"] == expected[profile_key]
        assert plan["execution_contract"]["requires_gpu"] is True
        assert plan["audit"]["passed"] is True
        assert {member["model_image_count"] for member in plan["members"]} == {8}
        assert {member["image_count"] for member in plan["members"]} == {8}


def test_checkpoint_resolver_requires_exact_environment_path_and_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"fixed offline checkpoint bytes"
    path = tmp_path / "fixture.pth"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint_id = "test_exact_checkpoint"
    monkeypatch.setitem(
        CHECKPOINT_SPECS,
        checkpoint_id,
        CheckpointSpec(checkpoint_id, "DECAF_TEST_CHECKPOINT", digest, path.name),
    )
    asset = resolve_checkpoint(
        checkpoint_id,
        {"DECAF_TEST_CHECKPOINT": str(path)},
    )
    assert asset.path == path.resolve()
    assert asset.sha256 == digest
    assert asset.bytes == len(payload)

    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="size changed|SHA256 mismatch"):
        resolve_checkpoint(checkpoint_id, {"DECAF_TEST_CHECKPOINT": str(path)})


def _fingerprint_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_id, dataset in FINGERPRINT_CASES:
        classes = 50 if dataset == "funnybirds" else 1_000
        probabilities = [[1.0 / classes] * classes]
        rows.append(
            {
                "family": "attribution",
                "case_id": f"attribution/{dataset}/{model_id}",
                "model_id": model_id,
                "dataset": dataset,
                "checkpoints": [
                    {
                        "path": "/offline/checkpoint.pth",
                        "sha256": "a" * 64,
                        "bytes": 1,
                    }
                ],
                "sample_ids": ["sample-0"],
                "preprocessed_tensor": {
                    "sha256": "b" * 64,
                    "dtype": "torch.float32",
                    "shape": [1, 3, 224, 224],
                    "byte_order": "little",
                    "layout": "contiguous_c_order",
                },
                "target_class": 0,
                "logits": [[0.0] * classes],
                "probabilities": probabilities,
                "precision": "fp32",
                "device": "cuda:0",
            }
        )
    return rows


def test_fingerprint_interchange_schema_is_exact_and_probability_normalized() -> None:
    rows = _fingerprint_rows()
    assert len(validate_checkpoint_fingerprint_rows(rows)) == 7
    rows[0]["probabilities"][0][0] = 0.5  # type: ignore[index]
    with pytest.raises(ValueError, match="logits/probabilities"):
        validate_checkpoint_fingerprint_rows(rows)


def test_idsds_vit_uses_the_official_half_range_normalization() -> None:
    torch = pytest.importorskip("torch")
    value = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).reshape(3, 1, 1)

    vit = _normalize_idsds(value, "vit_base_patch16_224")
    resnet = _normalize_idsds(value, "resnet50")

    assert torch.equal(vit.flatten(), torch.tensor([-1.0, 0.0, 1.0]))
    assert not torch.equal(vit, resnet)


def _queue_job(member_id: str, method_id: str, dependencies: list[str]) -> dict[str, object]:
    return {
        "member_id": member_id,
        "kind": "quality" if dependencies else "shared_deletion_targets",
        "dataset": "imagenet1k_idsds",
        "model_id": "resnet50",
        "method_id": method_id,
        "repeat": 0,
        "shard": 0,
        "receipt_path": f"receipts/members/{member_id}.json",
        "depends_on": [{"member_id": value} for value in dependencies],
    }


def _queue_runtime() -> dict[str, object]:
    return {
        "data_binding_manifest_sha256": "c" * 64,
        "checkpoint_binding_manifest_sha256": "d" * 64,
    }


def test_single_gpu_scheduler_reports_exclusive_dependency_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _queue_job("target", "__idsds_deletion__", [])
    quality = _queue_job("quality", "ig_32", ["target"])
    plan = {
        "members": [quality, target],
        "plan_contract_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    context = SimpleNamespace(path=tmp_path, resume=False)

    def complete_member(context, job, _evaluator, **_kwargs):
        path = context.path / str(job["receipt_path"])
        write_member_receipt(path, str(job["member_id"]), "completed")
        return "completed", load_member_receipt(path)

    monkeypatch.setattr(attribution_evaluate, "run_member", complete_member)
    report = _compute_single_gpu_queue(
        context,
        plan,
        _queue_runtime(),
        lambda _job, _context: None,
        backend="gpu",
    )
    global_receipt = json.loads(
        (tmp_path / "receipts/compute_members.json").read_text(encoding="utf-8")
    )
    events = global_receipt["details"]["queue_events"]
    assert report["scheduler"] == "single_gpu_dynamic_queue"
    assert global_receipt["status"] == "completed"
    assert global_receipt["details"]["exclusive_member_concurrency"] == 1
    assert [event["member_id"] for event in events if event["event"] == "start"] == [
        "target",
        "quality",
    ]


def test_sigterm_terminalizer_never_leaves_running_receipts(tmp_path: Path) -> None:
    target = _queue_job("target", "__idsds_deletion__", [])
    quality = _queue_job("quality", "ig_32", ["target"])
    plan = {
        "members": [target, quality],
        "plan_contract_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    context = SimpleNamespace(path=tmp_path)
    write_member_receipt(
        tmp_path / str(target["receipt_path"]),
        "target",
        "completed",
    )
    write_member_receipt(
        tmp_path / str(quality["receipt_path"]),
        "quality",
        "running",
    )
    _finalize_gpu_queue_after_termination(
        context,
        plan,
        _queue_runtime(),
        backend="gpu",
        error=TerminationRequested("received signal 15"),
    )
    global_receipt = json.loads(
        (tmp_path / "receipts/compute_members.json").read_text(encoding="utf-8")
    )
    assert load_member_receipt(tmp_path / str(quality["receipt_path"]))["status"] == "failed"
    assert global_receipt["status"] == "partial"
    assert global_receipt["all_processes_exited"] is True
