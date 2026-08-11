#!/usr/bin/env python3
"""Build the tracked-source DECAF reproducibility release and status receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decaf.experiments.common import atomic_json, atomic_text

PROVENANCE_FILES = (
    "historical_git_state.json",
    "paper_artifact_provenance.csv",
    "reference_runs.yaml",
    "server_inventory.json",
    "source_snapshots.yaml",
)
VERIFICATION_FILES = (
    "analysis_replay.json",
    "cpu_verification.json",
    "headline_assertions.json",
    "paper_artifact_diff.csv",
    "repository_audit.json",
)


def _command(repo: Path, *arguments: str) -> str:
    process = subprocess.run(
        arguments,
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return process.stdout.strip()


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


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _require_passed_reports(verification: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = _read_json(verification / "analysis_replay.json")
    cpu = _read_json(verification / "cpu_verification.json")
    audit = _read_json(verification / "repository_audit.json")
    if analysis.get("status") != "passed":
        raise RuntimeError("analysis replay has not passed")
    if cpu.get("status") != "passed" or cpu.get("mode") != "all-cpu":
        raise RuntimeError("all-cpu verification has not passed")
    if audit.get("passed") is not True:
        raise RuntimeError("repository audit has not passed")
    if analysis.get("reference_runs_verified") != 9:
        raise RuntimeError("the release does not verify all nine reference runs")
    if analysis.get("figures_regenerated") != 12:
        raise RuntimeError("the release does not regenerate all 12 figures")
    if analysis.get("tables_regenerated") != 16:
        raise RuntimeError("the release does not regenerate all 16 tables")
    return analysis, cpu


def _git_info(repository: Path) -> dict[str, Any]:
    status = _command(repository, "git", "status", "--porcelain=v1")
    if status:
        raise RuntimeError("release packaging requires a clean tracked repository")
    return {
        "schema_version": 1,
        "repository": "decaf",
        "commit": _command(repository, "git", "rev-parse", "HEAD"),
        "tree": _command(repository, "git", "rev-parse", "HEAD^{tree}"),
        "branch": _command(repository, "git", "branch", "--show-current"),
        "commit_subject": _command(repository, "git", "show", "-s", "--format=%s", "HEAD"),
        "tracked_worktree_clean": True,
    }


def _archive_repository(repository: Path, destination: Path) -> None:
    archive = destination.parent / "repository.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(archive), "HEAD"],
        cwd=repository,
        check=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as bundle:
        bundle.extractall(destination, filter="data")
    archive.unlink()


def _write_zip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"release archive already exists: {destination}")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as bundle:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(source.parent).as_posix())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_release(
    *,
    repository: Path,
    provenance: Path,
    verification: Path,
    release_root: Path,
    historical_drift: Path | None,
) -> dict[str, Any]:
    repository = repository.resolve()
    provenance = provenance.resolve()
    verification = verification.resolve()
    release_root = release_root.resolve()
    analysis, cpu = _require_passed_reports(verification)
    git_info = _git_info(repository)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    basename = f"decaf_reproducibility_release_v1_{timestamp}"
    destination = release_root / "packages" / f"{basename}.zip"

    with tempfile.TemporaryDirectory(prefix="decaf-release-") as temporary_value:
        temporary = Path(temporary_value)
        package = temporary / basename
        _archive_repository(repository, package / "repository")
        for name in PROVENANCE_FILES:
            source = provenance / name
            if not source.is_file():
                raise FileNotFoundError(f"required provenance file is missing: {source}")
            _copy(source, package / "provenance" / name)
        if historical_drift is not None:
            if not historical_drift.is_file():
                raise FileNotFoundError(historical_drift)
            _copy(historical_drift, package / "provenance" / historical_drift.name)
        for name in VERIFICATION_FILES:
            source = verification / name
            if not source.is_file():
                raise FileNotFoundError(f"required verification file is missing: {source}")
            _copy(source, package / "verification" / name)
        atomic_json(package / "GIT_INFO.json", git_info)
        _write_zip(package, destination)

    digest = _sha256(destination)
    atomic_text(destination.with_suffix(".zip.sha256"), f"{digest}  {destination.name}\n")
    steps = cpu.get("steps", {})
    source_missing = analysis.get("source_missing_recorded", [])
    status = {
        "schema_version": 1,
        "status": ("completed_with_documented_historical_gap" if source_missing else "completed"),
        "new_repository_path": str(repository),
        "new_repository_commit": git_info["commit"],
        "analysis_replay_status": analysis["status"],
        "reference_runs_inventoried": analysis["reference_runs_verified"],
        "figures_regenerated_count": analysis["figures_regenerated"],
        "tables_regenerated_count": analysis["tables_regenerated"],
        "cpu_tests_status": cpu["status"],
        "static_plan_status": steps.get("full_plan", {}).get("status"),
        "gpu_verification_pending": True,
        "historical_source_gaps": source_missing,
        "historical_repository_modified_by_restructure": False,
        "historical_repository_external_drift_manifest": (
            str(historical_drift.resolve()) if historical_drift is not None else None
        ),
        "final_zip_path": str(destination),
        "sha256": digest,
    }
    release_root.mkdir(parents=True, exist_ok=True)
    atomic_json(release_root / "REPRODUCIBILITY_STATUS.json", status)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--historical-drift", type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    status = build_release(
        repository=arguments.repository,
        provenance=arguments.provenance,
        verification=arguments.verification,
        release_root=arguments.release_root,
        historical_drift=arguments.historical_drift,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
