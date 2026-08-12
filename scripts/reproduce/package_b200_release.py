#!/usr/bin/env python3
"""Build the clean, portable DECAF v2 single-B200 verification release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from decaf.experiments.common import atomic_json, atomic_text

REQUIRED_EVIDENCE = (
    "B200_VERIFICATION_STATUS.json",
    "B200_VERIFICATION_REPORT.md",
    "verification/checkpoint_fingerprint_report.json",
    "verification/checkpoint_fingerprints.json",
    "verification/checkpoint_fingerprint_verification.json",
    "verification/controlled_resume.json",
    "verification/imagenet9_resume.json",
    "verification/single_gpu_scheduler.json",
    "verification/single_gpu_resume_fault_injection.json",
    "verification/analysis_replay.json",
    "verification/cpu_verification.json",
    "verification/headline_assertions.json",
    "verification/repository_audit.json",
    "verification/final_audit/full_pytest.json",
    "verification/final_audit/full_pytest.log",
    "verification/paper_artifact_diff.csv",
    "provenance/B200_PROVENANCE.json",
)
CPU_PROVENANCE_FILES = (
    "historical_git_state.json",
    "paper_artifact_provenance.csv",
    "reference_runs.yaml",
    "server_inventory.json",
    "source_snapshot_recovery.json",
    "source_snapshots.yaml",
    "historical_repository_external_drift.json",
)
CPU_PROVENANCE_SOURCE_ARCHIVE = {
    "name": "decaf_reproducibility_release_v1_20260811T230840Z.zip",
    "sha256": "57c2fab75a1def7ee47c0c3cf20af514fbf75a74361d1808504d158e1bfa22bd",
}
CPU_PROVENANCE_SOURCE_FILES = {
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
PUBLIC_TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_FRAGMENTS = (
    "/" + "work" + "/" + "Users" + "/",
    "/" + "home" + "/",
    "/" + "Users" + "/",
    "/" + "mnt" + "/",
    "/" + "tmp" + "/",
    "C:" + "\\" + "Users" + "\\",
)
WORK_USERS_PREFIX = re.compile(r"/work/Users/[A-Za-z0-9._-]+/")
PORTABLE_WORKSPACE_PREFIX = "source-host://workspace/"
B200_PORTABLE_EVIDENCE_FILES = (
    "verification/checkpoint_fingerprints.json",
    "verification/checkpoint_fingerprint_report.json",
    "verification/checkpoint_fingerprint_verification.json",
    "verification/final_audit/full_pytest.log",
    "verification/final_audit/full_pytest.json",
)


def _command(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        arguments,
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _require_passed_status(status: dict[str, Any], commit: str, tree: str) -> None:
    if status.get("status") != "passed":
        raise RuntimeError("single-B200 verification status has not passed")
    if status.get("repository", {}).get("commit") != commit:
        raise RuntimeError("single-B200 verification is not bound to the release commit")
    if (
        status.get("repository", {}).get("tree") != tree
        or status.get("repository", {}).get("tracked_worktree_clean") is not True
        or status.get("repository", {}).get("branch") != "gpu-verification-v1"
        or status.get("final_audits", {}).get("full_pytest") != "PASS"
    ):
        raise RuntimeError("single-B200 verification is not bound to the clean release tree")
    gates = status.get("acceptance_gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise RuntimeError("one or more single-B200 acceptance gates have not passed")
    if status.get("multi_gpu_scheduler", {}).get("real_execution") != (
        "NOT_TESTED_SINGLE_GPU_NODE"
    ):
        raise RuntimeError("real multi-GPU execution boundary is not recorded honestly")
    if status.get("full_paper_scale_compute_rerun") is not False:
        raise RuntimeError("full paper-scale compute boundary is not recorded honestly")


def _archive_repository(repository: Path, destination: Path, commit: str) -> None:
    destination.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as stream:
        archive = Path(stream.name)
    try:
        subprocess.run(
            ("git", "-C", str(repository), "archive", "--format=tar", "-o", str(archive), commit),
            check=True,
        )
        with tarfile.open(archive, "r") as bundle:
            bundle.extractall(destination, filter="data")
    finally:
        archive.unlink(missing_ok=True)


def _copy_portable_cpu_provenance(
    provenance_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Copy the seven CPU manifests with an auditable path-only transform."""

    records: list[dict[str, Any]] = []
    copied: set[str] = set()
    total_replacements = 0
    if tuple(CPU_PROVENANCE_SOURCE_FILES) != CPU_PROVENANCE_FILES:
        raise AssertionError("CPU provenance source identity inventory drifted")
    target_root = destination / "provenance"
    target_root.mkdir(parents=True, exist_ok=True)
    for name in CPU_PROVENANCE_FILES:
        expected = CPU_PROVENANCE_SOURCE_FILES[name]
        source = provenance_root / name
        if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"required CPU provenance is missing: {name}")
        source_payload = source.read_bytes()
        source_sha256 = _sha256_bytes(source_payload)
        if source_sha256 != expected["sha256"]:
            raise RuntimeError(f"CPU provenance source identity differs: {name}")
        try:
            source_text = source_payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"CPU provenance is not UTF-8 text: {name}") from error
        packaged_text, replacement_count = WORK_USERS_PREFIX.subn(
            PORTABLE_WORKSPACE_PREFIX,
            source_text,
        )
        if replacement_count != expected["replacement_count"]:
            raise RuntimeError(f"CPU provenance path inventory differs: {name}")
        target = target_root / name
        atomic_text(target, packaged_text)
        if _sha256(source) != source_sha256:
            raise RuntimeError(f"CPU provenance changed while packaging: {name}")
        packaged_payload = target.read_bytes()
        packaged_sha256 = _sha256_bytes(packaged_payload)
        records.append(
            {
                "path": f"provenance/{name}",
                "source_bytes": len(source_payload),
                "source_sha256": source_sha256,
                "packaged_bytes": len(packaged_payload),
                "packaged_sha256": packaged_sha256,
                "replacement_count": replacement_count,
            }
        )
        copied.add(name)
        total_replacements += replacement_count
    if copied != set(CPU_PROVENANCE_FILES) or len(copied) != 7:
        raise RuntimeError("release package does not contain exactly seven CPU provenance files")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "transform": "work_users_account_root_to_portable_workspace",
        "replacement_prefix": PORTABLE_WORKSPACE_PREFIX,
        "path_semantics": "source_host_logical_observation",
        "source_observations_only": True,
        "external_artifacts_included": False,
        "source_release_archive": dict(CPU_PROVENANCE_SOURCE_ARCHIVE),
        "source_files_modified": False,
        "file_count": len(records),
        "total_replacement_count": total_replacements,
        "files": records,
    }
    atomic_json(target_root / "PORTABILITY_TRANSFORM.json", payload)
    return payload


def _portable_text_projection(path: Path) -> tuple[str, int]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"B200 evidence is not UTF-8 text: {path.name}") from error
    return WORK_USERS_PREFIX.subn(PORTABLE_WORKSPACE_PREFIX, source)


def _portable_evidence_record(
    source: Path,
    packaged: Path,
    *,
    relative: str,
    replacement_count: int,
    dependency_rewrites: list[str],
) -> dict[str, Any]:
    return {
        "path": relative,
        "source_bytes": source.stat().st_size,
        "source_sha256": _sha256(source),
        "packaged_bytes": packaged.stat().st_size,
        "packaged_sha256": _sha256(packaged),
        "replacement_count": replacement_count,
        "dependency_rewrites": dependency_rewrites,
    }


def _validate_packaged_evidence_projection(destination: Path) -> None:
    fingerprints = destination / B200_PORTABLE_EVIDENCE_FILES[0]
    report_path = destination / B200_PORTABLE_EVIDENCE_FILES[1]
    wrapper_path = destination / B200_PORTABLE_EVIDENCE_FILES[2]
    log_path = destination / B200_PORTABLE_EVIDENCE_FILES[3]
    pytest_path = destination / B200_PORTABLE_EVIDENCE_FILES[4]
    provenance_path = destination / "provenance/B200_PROVENANCE.json"
    report = _read_json(report_path)
    wrapper = _read_json(wrapper_path)
    pytest_receipt = _read_json(pytest_path)
    provenance = _read_json(provenance_path)
    if report.get("checkpoint_fingerprints_sha256") != _sha256(fingerprints):
        raise RuntimeError("packaged checkpoint fingerprint report is not closed")
    if wrapper.get("steps", {}).get("checkpoint_fingerprint") != report:
        raise RuntimeError("packaged checkpoint fingerprint wrapper is not closed")
    output_log = pytest_receipt.get("output_log", {})
    if (
        output_log.get("sha256") != _sha256(log_path)
        or output_log.get("size_bytes") != log_path.stat().st_size
    ):
        raise RuntimeError("packaged full-pytest receipt is not closed")
    inventory = provenance.get("evidence_files")
    if (
        not isinstance(inventory, list)
        or provenance.get("evidence_file_count") != len(inventory)
        or provenance.get("evidence_inventory_sha256") != _canonical_sha256(inventory)
    ):
        raise RuntimeError("packaged B200 provenance inventory is not closed")
    inventory_by_path = {
        str(record.get("path")): record for record in inventory if isinstance(record, dict)
    }
    if len(inventory_by_path) != len(inventory):
        raise RuntimeError("packaged B200 provenance contains duplicate paths")
    for relative in B200_PORTABLE_EVIDENCE_FILES:
        path = destination / relative
        record = inventory_by_path.get(relative)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != _sha256(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"packaged B200 provenance differs: {relative}")


def _project_b200_evidence_portability(
    verification_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Project path-bearing B200 evidence while preserving its hash closure."""

    raw_provenance_path = verification_root / "provenance/B200_PROVENANCE.json"
    raw_provenance = _read_json(raw_provenance_path)
    raw_inventory = raw_provenance.get("evidence_files")
    if (
        not isinstance(raw_inventory, list)
        or raw_provenance.get("evidence_file_count") != len(raw_inventory)
        or raw_provenance.get("evidence_inventory_sha256") != _canonical_sha256(raw_inventory)
    ):
        raise RuntimeError("raw B200 provenance inventory is not closed")
    raw_by_path = {
        str(record.get("path")): record for record in raw_inventory if isinstance(record, dict)
    }
    if len(raw_by_path) != len(raw_inventory):
        raise RuntimeError("raw B200 provenance contains duplicate paths")
    for relative in B200_PORTABLE_EVIDENCE_FILES:
        source = verification_root / relative
        record = raw_by_path.get(relative)
        if (
            not source.is_file()
            or source.is_symlink()
            or not isinstance(record, dict)
            or record.get("sha256") != _sha256(source)
            or record.get("size_bytes") != source.stat().st_size
        ):
            raise RuntimeError(f"raw B200 evidence differs from provenance: {relative}")

    source_payload = verification_root / B200_PORTABLE_EVIDENCE_FILES[0]
    packaged_payload = destination / B200_PORTABLE_EVIDENCE_FILES[0]
    payload_text, payload_replacements = _portable_text_projection(source_payload)
    atomic_text(packaged_payload, payload_text)
    _read_json(packaged_payload)

    source_report = verification_root / B200_PORTABLE_EVIDENCE_FILES[1]
    packaged_report = destination / B200_PORTABLE_EVIDENCE_FILES[1]
    report = _read_json(source_report)
    report["checkpoint_fingerprints_sha256"] = _sha256(packaged_payload)
    atomic_json(packaged_report, report)

    source_wrapper = verification_root / B200_PORTABLE_EVIDENCE_FILES[2]
    packaged_wrapper = destination / B200_PORTABLE_EVIDENCE_FILES[2]
    wrapper = _read_json(source_wrapper)
    steps = wrapper.get("steps")
    if not isinstance(steps, dict) or "checkpoint_fingerprint" not in steps:
        raise RuntimeError("checkpoint fingerprint wrapper schema differs")
    steps["checkpoint_fingerprint"] = report
    atomic_json(packaged_wrapper, wrapper)

    source_log = verification_root / B200_PORTABLE_EVIDENCE_FILES[3]
    packaged_log = destination / B200_PORTABLE_EVIDENCE_FILES[3]
    log_text, log_replacements = _portable_text_projection(source_log)
    atomic_text(packaged_log, log_text)

    source_pytest = verification_root / B200_PORTABLE_EVIDENCE_FILES[4]
    packaged_pytest = destination / B200_PORTABLE_EVIDENCE_FILES[4]
    pytest_receipt = _read_json(source_pytest)
    output_log = pytest_receipt.get("output_log")
    if not isinstance(output_log, dict):
        raise RuntimeError("full-pytest receipt output-log schema differs")
    output_log["sha256"] = _sha256(packaged_log)
    output_log["size_bytes"] = packaged_log.stat().st_size
    atomic_json(packaged_pytest, pytest_receipt)

    dependency_rewrites = {
        B200_PORTABLE_EVIDENCE_FILES[0]: [],
        B200_PORTABLE_EVIDENCE_FILES[1]: [
            "checkpoint_fingerprints_sha256",
        ],
        B200_PORTABLE_EVIDENCE_FILES[2]: [
            "steps.checkpoint_fingerprint",
        ],
        B200_PORTABLE_EVIDENCE_FILES[3]: [],
        B200_PORTABLE_EVIDENCE_FILES[4]: [
            "output_log.sha256",
            "output_log.size_bytes",
        ],
    }
    replacements = {
        B200_PORTABLE_EVIDENCE_FILES[0]: payload_replacements,
        B200_PORTABLE_EVIDENCE_FILES[1]: 0,
        B200_PORTABLE_EVIDENCE_FILES[2]: 0,
        B200_PORTABLE_EVIDENCE_FILES[3]: log_replacements,
        B200_PORTABLE_EVIDENCE_FILES[4]: 0,
    }
    records = [
        _portable_evidence_record(
            verification_root / relative,
            destination / relative,
            relative=relative,
            replacement_count=replacements[relative],
            dependency_rewrites=dependency_rewrites[relative],
        )
        for relative in B200_PORTABLE_EVIDENCE_FILES
    ]
    projected = {**raw_provenance, "evidence_files": [dict(row) for row in raw_inventory]}
    projected_by_path = {str(record["path"]): record for record in projected["evidence_files"]}
    for record in records:
        projected_by_path[record["path"]].update(
            {
                "sha256": record["packaged_sha256"],
                "size_bytes": record["packaged_bytes"],
            }
        )
    projected["evidence_inventory_sha256"] = _canonical_sha256(projected["evidence_files"])
    packaged_provenance_path = destination / "provenance/B200_PROVENANCE.json"
    atomic_json(packaged_provenance_path, projected)
    _validate_packaged_evidence_projection(destination)

    transform = {
        "schema_version": 1,
        "status": "passed",
        "transform": "coordinated_b200_evidence_path_projection",
        "replacement_prefix": PORTABLE_WORKSPACE_PREFIX,
        "path_semantics": "source_host_logical_observation",
        "source_observations_only": True,
        "external_artifacts_included": False,
        "source_files_modified": False,
        "raw_provenance_bytes": raw_provenance_path.stat().st_size,
        "raw_provenance_sha256": _sha256(raw_provenance_path),
        "raw_evidence_inventory_sha256": raw_provenance["evidence_inventory_sha256"],
        "packaged_evidence_inventory_sha256": projected["evidence_inventory_sha256"],
        "packaged_provenance_bytes": packaged_provenance_path.stat().st_size,
        "packaged_provenance_sha256": _sha256(packaged_provenance_path),
        "file_count": len(records),
        "total_replacement_count": sum(replacements.values()),
        "files": records,
        "closure_checks": {
            "raw_sources_match_provenance": True,
            "fingerprint_report_payload_hash": True,
            "fingerprint_wrapper_embedded_report": True,
            "full_pytest_receipt_log_hash_and_size": True,
            "packaged_provenance_inventory": True,
        },
    }
    transform_path = destination / "provenance/B200_EVIDENCE_PORTABILITY_TRANSFORM.json"
    atomic_json(transform_path, transform)
    return transform


def _copy_evidence(
    verification_root: Path,
    destination: Path,
    *,
    provenance_root: Path,
) -> None:
    for relative_value in REQUIRED_EVIDENCE:
        relative = Path(relative_value)
        source = verification_root / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"required B200 evidence is missing: {relative_value}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _project_b200_evidence_portability(verification_root, destination)
    analysis = _read_json(verification_root / "verification/analysis_replay.json")
    inventory = analysis.get("artifact_inventory")
    if not isinstance(inventory, list) or len(inventory) != 60:
        raise RuntimeError("analysis replay inventory is not exactly 60 files")
    encoded_inventory = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded_inventory).hexdigest() != analysis.get("artifact_inventory_sha256"):
        raise RuntimeError("analysis replay inventory digest differs")
    copied_paper_outputs = 0
    inventory_paths: set[str] = set()
    paper_paths: set[str] = set()
    for record in inventory:
        if not isinstance(record, dict):
            raise TypeError("analysis artifact inventory record is not an object")
        relative_value = record.get("relative_path")
        if not isinstance(relative_value, str):
            raise RuntimeError("analysis artifact inventory path is missing")
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_value
            or relative_value in inventory_paths
            or record.get("source_root") != "verification_root"
            or record.get("portable_path") != f"verification_root/{relative_value}"
        ):
            raise RuntimeError("analysis artifact inventory path is unsafe")
        inventory_paths.add(relative_value)
        if not relative_value.startswith("paper_outputs/"):
            continue
        source = verification_root / "verification" / relative
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != record.get("size_bytes")
            or _sha256(source) != record.get("sha256")
        ):
            raise RuntimeError(f"analysis artifact bytes differ: {relative_value}")
        target = destination / "verification" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.stat().st_size != record.get("size_bytes") or _sha256(target) != record.get(
            "sha256"
        ):
            raise RuntimeError(f"packaged analysis artifact bytes differ: {relative_value}")
        paper_paths.add(relative_value)
        copied_paper_outputs += 1
    actual_paper_paths = {
        path.relative_to(destination / "verification").as_posix()
        for path in (destination / "verification/paper_outputs").rglob("*")
        if path.is_file()
    }
    if copied_paper_outputs != 58 or len(paper_paths) != 58 or actual_paper_paths != paper_paths:
        raise RuntimeError("release package does not contain exactly 58 paper outputs")
    _copy_portable_cpu_provenance(provenance_root, destination)
    reference_destination = destination / "reference-run-manifests"
    reference_destination.mkdir(parents=True)
    repository = destination.parent / "repository"
    for source in sorted((repository / "manifests" / "reference_runs").glob("*.yaml")):
        shutil.copy2(source, reference_destination / source.name)


def _contains_cjk(text: str) -> bool:
    return any(
        0x3400 <= ord(character) <= 0x4DBF
        or 0x4E00 <= ord(character) <= 0x9FFF
        or 0xF900 <= ord(character) <= 0xFAFF
        for character in text
    )


def _validate_public_payload(package: Path) -> dict[str, Any]:
    scanned = 0
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"release contains a symlink: {path.relative_to(package)}")
        if not path.is_file():
            continue
        scanned += 1
        relative = path.relative_to(package)
        if path.suffix.lower() == ".pdf":
            raise RuntimeError(f"release contains a PDF: {relative}")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise RuntimeError(f"release contains an unexpectedly large file: {relative}")
        declared_text = path.suffix.lower() in PUBLIC_TEXT_SUFFIXES
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            if declared_text:
                raise RuntimeError(f"release text is not UTF-8: {relative}") from error
            continue
        if not declared_text and "\0" in text:
            continue
        if any(fragment in text for fragment in PRIVATE_FRAGMENTS):
            raise RuntimeError(f"release contains a private absolute path: {relative}")
        if _contains_cjk(text):
            raise RuntimeError(f"release contains CJK text: {relative}")
    return {"status": "passed", "scanned_file_count": scanned}


def _write_manifest(package: Path) -> dict[str, Any]:
    records = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json":
            records.append(
                {
                    "path": path.relative_to(package).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    payload = {
        "schema_version": 1,
        "file_count": len(records),
        "files": records,
    }
    atomic_json(package / "PACKAGE_MANIFEST.json", payload)
    return payload


def _validate_zip(
    archive: Path,
    package_name: str,
    expected_manifest: dict[str, Any],
) -> dict[str, Any]:
    prefix = f"{package_name}/"
    with zipfile.ZipFile(archive, "r") as bundle:
        corrupt_member = bundle.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"release ZIP CRC check failed: {corrupt_member}")
        members = [item for item in bundle.infolist() if not item.is_dir()]
        names = [item.filename for item in members]
        if len(names) != len(set(names)):
            raise RuntimeError("release ZIP contains duplicate members")
        expected_names = {
            f"{prefix}PACKAGE_MANIFEST.json",
            *(f"{prefix}{record['path']}" for record in expected_manifest["files"]),
        }
        if set(names) != expected_names:
            raise RuntimeError("release ZIP member inventory differs from its manifest")
        try:
            archived_manifest = json.loads(
                bundle.read(f"{prefix}PACKAGE_MANIFEST.json").decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("release ZIP package manifest is unreadable") from error
        if archived_manifest != expected_manifest:
            raise RuntimeError("release ZIP package manifest differs from staging")
        for record in expected_manifest["files"]:
            payload = bundle.read(f"{prefix}{record['path']}")
            if len(payload) != record["bytes"] or _sha256_bytes(payload) != record["sha256"]:
                raise RuntimeError(f"release ZIP member differs from manifest: {record['path']}")
    return {
        "status": "passed",
        "testzip": "passed",
        "manifest_verified": True,
        "member_count": len(names),
    }


def _write_zip(
    package: Path,
    destination: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as bundle:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(package.parent).as_posix())
        validation = _validate_zip(temporary, package.name, manifest)
        os.replace(temporary, destination)
        return validation
    finally:
        temporary.unlink(missing_ok=True)


def build_b200_release(
    *,
    repository: Path,
    verification_root: Path,
    release_root: Path,
    provenance_root: Path,
) -> dict[str, Any]:
    repository = repository.resolve()
    verification_root = verification_root.resolve()
    release_root = release_root.resolve()
    provenance_root = provenance_root.resolve()
    if _command(repository, "git", "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("B200 release packaging requires a clean worktree")
    commit = _command(repository, "git", "rev-parse", "HEAD")
    tree = _command(repository, "git", "rev-parse", "HEAD^{tree}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017
    basename = f"decaf_reproducibility_release_v2_b200_verified_{timestamp}"
    destination = release_root / "packages" / f"{basename}.zip"

    status_path = verification_root / "B200_VERIFICATION_STATUS.json"
    status = _read_json(status_path)
    _require_passed_status(status, commit, tree)
    release_fields = {
        "archive": f"packages/{destination.name}",
        "sha256_sidecar": f"packages/{destination.name}.sha256",
    }
    packaged_status = {**status, "release": release_fields}

    with tempfile.TemporaryDirectory(prefix="decaf-b200-release-") as temporary_value:
        package = Path(temporary_value) / basename
        _archive_repository(repository, package / "repository", commit)
        evidence = package / "b200-verification"
        _copy_evidence(
            verification_root,
            evidence,
            provenance_root=provenance_root,
        )
        # The package advertises its own relative release paths, while the
        # external status remains untouched until the verified ZIP is durable.
        atomic_json(evidence / "B200_VERIFICATION_STATUS.json", packaged_status)
        atomic_json(
            package / "GIT_INFO.json",
            {"schema_version": 1, "commit": commit, "tree": tree, "branch": "gpu-verification-v1"},
        )
        public_validation = _validate_public_payload(package)
        manifest = _write_manifest(package)
        zip_validation = _write_zip(package, destination, manifest)

    digest = _sha256(destination)
    sidecar = destination.with_suffix(".zip.sha256")
    atomic_text(sidecar, f"{digest}  {destination.name}\n")
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "repository_commit": commit,
        "repository_tree": tree,
        "public_payload": public_validation,
        "zip_self_check": zip_validation,
        "packaged_file_count": manifest["file_count"] + 1,
        "final_zip": str(destination),
        "sha256_sidecar": str(sidecar),
        "sha256": digest,
    }
    atomic_json(release_root / "B200_RELEASE_RECEIPT.json", receipt)
    atomic_json(status_path, packaged_status)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--verification-root", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--provenance-root", required=True, type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    receipt = build_b200_release(
        repository=arguments.repository,
        verification_root=arguments.verification_root,
        release_root=arguments.release_root,
        provenance_root=arguments.provenance_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
