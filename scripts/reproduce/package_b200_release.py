#!/usr/bin/env python3
"""Build the clean, portable DECAF v2 single-B200 verification release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    "verification/checkpoint_fingerprint_verification.json",
    "verification/controlled_resume.json",
    "verification/imagenet9_resume.json",
    "verification/single_gpu_scheduler.json",
    "verification/single_gpu_resume_fault_injection.json",
    "verification/analysis_replay.json",
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
    copied_provenance: set[str] = set()
    for name in CPU_PROVENANCE_FILES:
        source = provenance_root / name
        if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"required CPU provenance is missing: {name}")
        target = destination / "provenance" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.stat().st_size != source.stat().st_size or _sha256(target) != _sha256(source):
            raise RuntimeError(f"packaged CPU provenance bytes differ: {name}")
        copied_provenance.add(name)
    if copied_provenance != set(CPU_PROVENANCE_FILES) or len(copied_provenance) != 7:
        raise RuntimeError("release package does not contain exactly seven CPU provenance files")
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


def _write_zip(package: Path, destination: Path) -> None:
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
        os.replace(temporary, destination)
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
    status["release"] = {
        "archive": f"packages/{destination.name}",
        "sha256_sidecar": f"packages/{destination.name}.sha256",
    }
    atomic_json(status_path, status)

    with tempfile.TemporaryDirectory(prefix="decaf-b200-release-") as temporary_value:
        package = Path(temporary_value) / basename
        _archive_repository(repository, package / "repository", commit)
        evidence = package / "b200-verification"
        _copy_evidence(
            verification_root,
            evidence,
            provenance_root=provenance_root,
        )
        atomic_json(
            package / "GIT_INFO.json",
            {"schema_version": 1, "commit": commit, "tree": tree, "branch": "gpu-verification-v1"},
        )
        public_validation = _validate_public_payload(package)
        manifest = _write_manifest(package)
        _write_zip(package, destination)

    digest = _sha256(destination)
    sidecar = destination.with_suffix(".zip.sha256")
    atomic_text(sidecar, f"{digest}  {destination.name}\n")
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "repository_commit": commit,
        "repository_tree": tree,
        "public_payload": public_validation,
        "packaged_file_count": manifest["file_count"] + 1,
        "final_zip": str(destination),
        "sha256_sidecar": str(sidecar),
        "sha256": digest,
    }
    atomic_json(release_root / "B200_RELEASE_RECEIPT.json", receipt)
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
