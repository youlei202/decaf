"""Focused tests for the portable single-B200 release builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

PRIVATE_WORK_USERS_ROOT = "/" + "work" + "/" + "Users" + "/"


def _module():
    path = Path(__file__).parents[2] / "scripts" / "reproduce" / "package_b200_release.py"
    spec = importlib.util.spec_from_file_location("package_b200_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packager_sources_do_not_embed_private_path_detection_fixture() -> None:
    root = Path(__file__).parents[2]
    for relative in (
        "scripts/reproduce/package_b200_release.py",
        "tests/unit/test_package_b200_release.py",
    ):
        assert PRIVATE_WORK_USERS_ROOT not in (root / relative).read_text(encoding="utf-8")


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
        "verification/checkpoint_fingerprints.json",
        "verification/checkpoint_fingerprint_verification.json",
        "verification/cpu_verification.json",
        "verification/controlled_resume.json",
        "verification/imagenet9_resume.json",
        "verification/single_gpu_resume_fault_injection.json",
    }.issubset(module.REQUIRED_EVIDENCE)
    assert "reference_runs.yaml" in module.CPU_PROVENANCE_FILES
    assert "paper_artifact_provenance.csv" in module.CPU_PROVENANCE_FILES
    assert len(module.CPU_PROVENANCE_FILES) == 7
    assert "verification/final_audit/full_pytest.json" in module.REQUIRED_EVIDENCE
    assert "verification/final_audit/full_pytest.log" in module.REQUIRED_EVIDENCE


def test_cpu_provenance_source_release_lineage_is_frozen() -> None:
    module = _module()
    assert module.CPU_PROVENANCE_SOURCE_ARCHIVE == {
        "name": "decaf_reproducibility_release_v1_20260811T230840Z.zip",
        "sha256": "57c2fab75a1def7ee47c0c3cf20af514fbf75a74361d1808504d158e1bfa22bd",
    }
    assert module.CPU_PROVENANCE_SOURCE_FILES == {
        "historical_git_state.json": {
            "sha256": "7bab8dc13e640116eada0e9ee0e13db98cdf9cd0a8035e39d011188b570b4e81",
            "replacement_count": 6,
        },
        "paper_artifact_provenance.csv": {
            "sha256": "65e5b7d4878b4f6d531e6aa665ab04847c5f5da97c5ece059b0f202f04e7035a",
            "replacement_count": 0,
        },
        "reference_runs.yaml": {
            "sha256": "173341af15c0b13c310180faf714090c16dc570ddb63e12d2286013c5666f234",
            "replacement_count": 53,
        },
        "server_inventory.json": {
            "sha256": "5241e3f88886b5a717157e08f570386a4dfed635bd8641f64844131372327aaf",
            "replacement_count": 41,
        },
        "source_snapshot_recovery.json": {
            "sha256": "f01a6bac3fac5a0230b4ea1f1e38a9572e69685d6c11e9a0af79778ed73cade4",
            "replacement_count": 12,
        },
        "source_snapshots.yaml": {
            "sha256": "5af425f8ad3cc0c59393802b61b07b88759ef566c49db69962ff42007e907ceb",
            "replacement_count": 10,
        },
        "historical_repository_external_drift.json": {
            "sha256": "e75651c2035e709286be363d5101bf19cb2f624709949ae34e7f03ff38b1bcea",
            "replacement_count": 1,
        },
    }
    assert [
        value["replacement_count"] for value in module.CPU_PROVENANCE_SOURCE_FILES.values()
    ] == [6, 0, 53, 41, 12, 10, 1]


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


def test_cpu_provenance_transform_is_portable_auditable_and_non_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    provenance = tmp_path / "source-provenance"
    provenance.mkdir()
    originals: dict[str, bytes] = {}
    counts = [6, 0, 53, 41, 12, 10, 1]
    fixture_identities: dict[str, dict[str, object]] = {}
    for index, (name, occurrences) in enumerate(
        zip(module.CPU_PROVENANCE_FILES, counts, strict=True)
    ):
        text = (
            "source observation without a host path\n"
            if occurrences == 0
            else "".join(
                f"source_{repeat}: {PRIVATE_WORK_USERS_ROOT}user-{index}/project/{repeat}\n"
                for repeat in range(occurrences)
            )
        )
        source = provenance / name
        source.write_text(text, encoding="utf-8")
        originals[name] = source.read_bytes()
        fixture_identities[name] = {
            "sha256": hashlib.sha256(originals[name]).hexdigest(),
            "replacement_count": occurrences,
        }
    monkeypatch.setattr(
        module,
        "CPU_PROVENANCE_SOURCE_FILES",
        fixture_identities,
    )

    package = tmp_path / "package"
    package.mkdir()
    transform = module._copy_portable_cpu_provenance(provenance, package)

    assert transform["file_count"] == 7
    assert transform["total_replacement_count"] == 123
    assert transform["source_files_modified"] is False
    assert transform["path_semantics"] == "source_host_logical_observation"
    assert transform["source_observations_only"] is True
    assert transform["external_artifacts_included"] is False
    assert transform["replacement_prefix"] == "source-host://workspace/"
    assert transform["source_release_archive"] == (module.CPU_PROVENANCE_SOURCE_ARCHIVE)
    assert len(transform["files"]) == 7
    for record in transform["files"]:
        name = Path(record["path"]).name
        source = provenance / name
        packaged = package / record["path"]
        assert source.read_bytes() == originals[name]
        assert record["source_sha256"] == hashlib.sha256(originals[name]).hexdigest()
        assert record["packaged_sha256"] == hashlib.sha256(packaged.read_bytes()).hexdigest()
        assert record["replacement_count"] == fixture_identities[name]["replacement_count"]
        text = packaged.read_text(encoding="utf-8")
        assert PRIVATE_WORK_USERS_ROOT not in text
        if record["replacement_count"]:
            assert "source-host://workspace/project/" in text
    persisted = json.loads(
        (package / "provenance/PORTABILITY_TRANSFORM.json").read_text(encoding="utf-8")
    )
    assert persisted == transform
    assert module._validate_public_payload(package)["status"] == "passed"

    changed = provenance / module.CPU_PROVENANCE_FILES[0]
    changed.write_bytes(originals[changed.name] + b"drift\n")
    with pytest.raises(RuntimeError, match="source identity differs"):
        module._copy_portable_cpu_provenance(provenance, tmp_path / "rejected")


def _b200_projection_fixture(module: object, root: Path) -> dict[str, bytes]:
    paths = module.B200_PORTABLE_EVIDENCE_FILES
    payload = {
        "case_count": 2,
        "cases": [
            {"path": f"{PRIVATE_WORK_USERS_ROOT}tester/checkpoints/first.pt"},
            {"path": f"{PRIVATE_WORK_USERS_ROOT}tester/checkpoints/second.pt"},
        ],
    }
    payload_path = root / paths[0]
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "passed",
        "case_count": 2,
        "checkpoint_fingerprints_path": "checkpoint_fingerprints.json",
        "checkpoint_fingerprints_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
    }
    report_path = root / paths[1]
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    wrapper = {
        "status": "passed",
        "steps": {"checkpoint_fingerprint": report},
    }
    wrapper_path = root / paths[2]
    wrapper_path.write_text(json.dumps(wrapper, indent=2) + "\n", encoding="utf-8")
    log_path = root / paths[3]
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        f"rootdir: {PRIVATE_WORK_USERS_ROOT}tester/GitHub/decaf\n2 passed\n",
        encoding="utf-8",
    )
    pytest_receipt = {
        "status": "passed",
        "output_log": {
            "path": paths[3],
            "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "size_bytes": log_path.stat().st_size,
        },
    }
    pytest_path = root / paths[4]
    pytest_path.write_text(json.dumps(pytest_receipt, indent=2) + "\n", encoding="utf-8")
    source_bytes = {relative: (root / relative).read_bytes() for relative in paths}
    inventory = [
        {
            "path": relative,
            "sha256": hashlib.sha256(source_bytes[relative]).hexdigest(),
            "size_bytes": len(source_bytes[relative]),
        }
        for relative in paths
    ]
    provenance = {
        "schema_version": 1,
        "status": "passed",
        "evidence_file_count": len(inventory),
        "evidence_files": inventory,
        "evidence_inventory_sha256": module._canonical_sha256(inventory),
    }
    provenance_path = root / "provenance/B200_PROVENANCE.json"
    provenance_path.parent.mkdir()
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    source_bytes["provenance/B200_PROVENANCE.json"] = provenance_path.read_bytes()
    return source_bytes


def test_b200_evidence_projection_preserves_portable_hash_closure(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "source"
    destination = tmp_path / "package" / "b200-verification"
    originals = _b200_projection_fixture(module, source)

    transform = module._project_b200_evidence_portability(source, destination)

    assert transform["file_count"] == 5
    assert transform["total_replacement_count"] == 3
    assert transform["path_semantics"] == "source_host_logical_observation"
    assert transform["source_observations_only"] is True
    assert transform["external_artifacts_included"] is False
    assert transform["source_files_modified"] is False
    assert all(transform["closure_checks"].values())
    assert [record["replacement_count"] for record in transform["files"]] == [
        2,
        0,
        0,
        1,
        0,
    ]
    assert [record["dependency_rewrites"] for record in transform["files"]] == [
        [],
        ["checkpoint_fingerprints_sha256"],
        ["steps.checkpoint_fingerprint"],
        [],
        ["output_log.sha256", "output_log.size_bytes"],
    ]
    for relative, payload in originals.items():
        assert (source / relative).read_bytes() == payload

    fingerprints = destination / module.B200_PORTABLE_EVIDENCE_FILES[0]
    log = destination / module.B200_PORTABLE_EVIDENCE_FILES[3]
    assert PRIVATE_WORK_USERS_ROOT not in fingerprints.read_text(encoding="utf-8")
    assert "source-host://workspace/checkpoints/first.pt" in fingerprints.read_text(
        encoding="utf-8"
    )
    assert "source-host://workspace/GitHub/decaf" in log.read_text(encoding="utf-8")
    report = json.loads(
        (destination / module.B200_PORTABLE_EVIDENCE_FILES[1]).read_text(encoding="utf-8")
    )
    wrapper = json.loads(
        (destination / module.B200_PORTABLE_EVIDENCE_FILES[2]).read_text(encoding="utf-8")
    )
    pytest_receipt = json.loads(
        (destination / module.B200_PORTABLE_EVIDENCE_FILES[4]).read_text(encoding="utf-8")
    )
    assert (
        report["checkpoint_fingerprints_sha256"]
        == hashlib.sha256(fingerprints.read_bytes()).hexdigest()
    )
    assert wrapper["steps"]["checkpoint_fingerprint"] == report
    assert pytest_receipt["output_log"]["sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()
    assert pytest_receipt["output_log"]["size_bytes"] == log.stat().st_size

    provenance_path = destination / "provenance/B200_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["evidence_inventory_sha256"] == module._canonical_sha256(
        provenance["evidence_files"]
    )
    assert (
        transform["raw_provenance_sha256"]
        == hashlib.sha256(originals["provenance/B200_PROVENANCE.json"]).hexdigest()
    )
    assert (
        transform["packaged_provenance_sha256"]
        == hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    )
    persisted_transform = json.loads(
        (destination / "provenance/B200_EVIDENCE_PORTABILITY_TRANSFORM.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted_transform == transform
    module._validate_packaged_evidence_projection(destination)
    assert module._validate_public_payload(destination)["status"] == "passed"


def test_b200_evidence_projection_rejects_raw_provenance_drift(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "source"
    _b200_projection_fixture(module, source)
    payload = source / module.B200_PORTABLE_EVIDENCE_FILES[0]
    payload.write_bytes(payload.read_bytes() + b"drift\n")

    with pytest.raises(RuntimeError, match="differs from provenance"):
        module._project_b200_evidence_portability(
            source,
            tmp_path / "destination",
        )


def test_zip_is_crc_checked_and_matches_package_manifest(tmp_path: Path) -> None:
    module = _module()
    package = tmp_path / "stage" / "release-name"
    package.mkdir(parents=True)
    payload = package / "repository" / "README.md"
    payload.parent.mkdir()
    payload.write_text("portable release\n", encoding="utf-8")
    manifest = module._write_manifest(package)
    destination = tmp_path / "packages" / "release.zip"

    result = module._write_zip(package, destination, manifest)

    assert result == {
        "status": "passed",
        "testzip": "passed",
        "manifest_verified": True,
        "member_count": 2,
    }
    with zipfile.ZipFile(destination) as bundle:
        assert bundle.testzip() is None
        archived = json.loads(bundle.read("release-name/PACKAGE_MANIFEST.json").decode("utf-8"))
    assert archived == manifest

    second_package = tmp_path / "second-stage" / "release-name"
    second_package.mkdir(parents=True)
    changed = second_package / "payload.txt"
    changed.write_text("before\n", encoding="utf-8")
    second_manifest = module._write_manifest(second_package)
    changed.write_text("after\n", encoding="utf-8")
    rejected = tmp_path / "packages" / "rejected.zip"
    with pytest.raises(RuntimeError, match="member differs from manifest"):
        module._write_zip(second_package, rejected, second_manifest)
    assert not rejected.exists()
    assert not (rejected.parent / f".{rejected.name}.part").exists()


def test_external_status_is_unchanged_until_verified_zip_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "repository"
    verification = tmp_path / "verification"
    release = tmp_path / "release"
    provenance = tmp_path / "provenance"
    for path in (repository, verification, provenance):
        path.mkdir()
    status_path = verification / "B200_VERIFICATION_STATUS.json"
    status_path.write_text(json.dumps(_passed_status()), encoding="utf-8")
    original_status = status_path.read_bytes()

    def command(_repository: Path, *arguments: str) -> str:
        if arguments[:2] == ("git", "status"):
            return ""
        if arguments == ("git", "rev-parse", "HEAD"):
            return "abc123"
        if arguments == ("git", "rev-parse", "HEAD^{tree}"):
            return "tree123"
        raise AssertionError(arguments)

    monkeypatch.setattr(module, "_command", command)
    monkeypatch.setattr(
        module,
        "_archive_repository",
        lambda _repository, destination, _commit: destination.mkdir(parents=True),
    )
    monkeypatch.setattr(
        module,
        "_copy_evidence",
        lambda _root, destination, *, provenance_root: destination.mkdir(parents=True),
    )
    monkeypatch.setattr(
        module,
        "_validate_public_payload",
        lambda _package: {"status": "passed", "scanned_file_count": 2},
    )

    def fail_zip(_package: Path, _destination: Path, _manifest: object) -> object:
        assert status_path.read_bytes() == original_status
        raise RuntimeError("injected ZIP failure")

    monkeypatch.setattr(module, "_write_zip", fail_zip)
    with pytest.raises(RuntimeError, match="injected ZIP failure"):
        module.build_b200_release(
            repository=repository,
            verification_root=verification,
            release_root=release,
            provenance_root=provenance,
        )
    assert status_path.read_bytes() == original_status

    observed_packaged_status: dict[str, object] = {}

    def succeed_zip(package: Path, destination: Path, _manifest: object) -> object:
        assert status_path.read_bytes() == original_status
        observed_packaged_status.update(
            json.loads(
                (package / "b200-verification/B200_VERIFICATION_STATUS.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"verified zip fixture")
        return {
            "status": "passed",
            "testzip": "passed",
            "manifest_verified": True,
            "member_count": 2,
        }

    monkeypatch.setattr(module, "_write_zip", succeed_zip)
    receipt = module.build_b200_release(
        repository=repository,
        verification_root=verification,
        release_root=release,
        provenance_root=provenance,
    )

    persisted_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert observed_packaged_status["release"] == persisted_status["release"]
    assert persisted_status["release"]["archive"].startswith("packages/")
    assert persisted_status["release"]["sha256_sidecar"].endswith(".zip.sha256")
    assert receipt["zip_self_check"]["manifest_verified"] is True
