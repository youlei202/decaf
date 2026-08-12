"""Focused tests for the portable single-B200 release builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[2] / "scripts" / "reproduce" / "package_b200_release.py"
    spec = importlib.util.spec_from_file_location("package_b200_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passed_status() -> dict[str, object]:
    return {
        "status": "passed",
        "repository": {
            "commit": "abc123",
            "tree": "tree123",
            "branch": "gpu-verification-v1",
            "tracked_worktree_clean": True,
        },
        "final_audits": {"full_pytest": "PASS"},
        "acceptance_gates": {"controlled": True, "analysis_replay": True},
        "multi_gpu_scheduler": {"real_execution": "NOT_TESTED_SINGLE_GPU_NODE"},
        "full_paper_scale_compute_rerun": False,
    }


def test_b200_release_status_requires_every_gate_and_honest_boundaries() -> None:
    module = _module()
    module._require_passed_status(_passed_status(), "abc123", "tree123")

    failed = _passed_status()
    failed["acceptance_gates"] = {"controlled": True, "analysis_replay": False}
    with pytest.raises(RuntimeError, match="acceptance gates"):
        module._require_passed_status(failed, "abc123", "tree123")


def test_b200_release_includes_portable_resume_and_fingerprint_wrappers() -> None:
    module = _module()
    assert {
        "verification/checkpoint_fingerprint_verification.json",
        "verification/controlled_resume.json",
        "verification/imagenet9_resume.json",
        "verification/single_gpu_resume_fault_injection.json",
    }.issubset(module.REQUIRED_EVIDENCE)
    assert "reference_runs.yaml" in module.CPU_PROVENANCE_FILES
    assert "paper_artifact_provenance.csv" in module.CPU_PROVENANCE_FILES
    assert len(module.CPU_PROVENANCE_FILES) == 7
    assert "verification/final_audit/full_pytest.json" in module.REQUIRED_EVIDENCE
    assert "verification/final_audit/full_pytest.log" in module.REQUIRED_EVIDENCE


def test_b200_release_payload_rejects_private_paths_cjk_and_pdfs(tmp_path: Path) -> None:
    module = _module()
    package = tmp_path / "package"
    package.mkdir()
    report = package / "report.md"
    report.write_text("portable evidence\n", encoding="utf-8")
    assert module._validate_public_payload(package)["status"] == "passed"

    private_path = "/" + "work" + "/" + "Users" + "/example/run"
    report.write_text(f"private: {private_path}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="private absolute path"):
        module._validate_public_payload(package)
    report.write_text(chr(0x4E2D) + chr(0x6587) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="CJK"):
        module._validate_public_payload(package)
    report.write_text("portable evidence\n", encoding="utf-8")
    (package / "paper.pdf").write_bytes(b"%PDF")
    with pytest.raises(RuntimeError, match="PDF"):
        module._validate_public_payload(package)
