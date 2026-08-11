#!/usr/bin/env python3
"""Build the tracked-source DECAF reproducibility release and status receipt."""

from __future__ import annotations

import argparse
import csv
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

import yaml

from decaf.experiments.common import atomic_json, atomic_text
from decaf.paper.manifest import load_visual_manifest
from decaf.paper.reference import load_reference_runs

PROVENANCE_FILES = (
    "historical_git_state.json",
    "paper_artifact_provenance.csv",
    "reference_runs.yaml",
    "server_inventory.json",
    "source_snapshot_recovery.json",
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


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    process = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {process.stderr.decode('utf-8', errors='replace')}"
        )
    return process.stdout


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _validate_provenance_manifests(
    provenance: Path,
    repository: Path,
) -> dict[str, Any]:
    tracked_runs = load_reference_runs(repository / "manifests" / "reference_runs")
    inventory = yaml.safe_load((provenance / "reference_runs.yaml").read_text(encoding="utf-8"))
    records = inventory.get("runs") if isinstance(inventory, dict) else None
    if (
        inventory.get("schema_version") != 1
        or inventory.get("reference_run_count") != 9
        or inventory.get("all_archives_present") is not True
        or not isinstance(records, list)
    ):
        raise RuntimeError("reference-run provenance inventory is invalid")
    indexed = {str(record.get("id")): record for record in records if isinstance(record, dict)}
    if set(indexed) != set(tracked_runs) or len(records) != len(indexed):
        raise RuntimeError("reference-run provenance IDs differ from tracked manifests")
    archive_paths: dict[str, tuple[str, int]] = {}
    for run_id, run in tracked_runs.items():
        record = indexed[run_id]
        expected = {
            "family": run.family,
            "scientific_status": run.scientific_status,
            "archive_sha256": run.archive_sha256,
            "archive_size_bytes": run.archive_size_bytes,
            "archive_member_count": run.archive_member_count,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"reference-run provenance disagrees with tracked manifest: {run_id}"
            )
        archive_path = str(record.get("archive_path", ""))
        if (
            record.get("archive_exists") is not True
            or Path(archive_path).name != run.archive_filename
        ):
            raise RuntimeError(f"reference archive provenance is invalid: {run_id}")
        archive_paths[archive_path] = (run_id, run.archive_size_bytes)
    if len(archive_paths) != 9:
        raise RuntimeError("reference archive provenance paths are not unique")

    server = _read_json(provenance / "server_inventory.json")
    server_archives = server.get("reference_archives")
    historical_state = _read_json(provenance / "historical_git_state.json")
    if (
        server.get("schema_version") != 1
        or not isinstance(server_archives, list)
        or len(server_archives) != 9
        or server.get("historical_repository", {}).get("path")
        != historical_state.get("absolute_path")
    ):
        raise RuntimeError("server inventory provenance is invalid")
    server_index = {
        str(record.get("path")): record for record in server_archives if isinstance(record, dict)
    }
    if set(server_index) != set(archive_paths) or len(server_index) != 9:
        raise RuntimeError("server and reference archive inventories disagree")
    for path, (_, size_bytes) in archive_paths.items():
        record = server_index[path]
        if (
            record.get("exists") is not True
            or record.get("kind") != "file"
            or record.get("size_bytes") != size_bytes
        ):
            raise RuntimeError(f"server archive inventory is invalid: {path}")

    visual_manifest = load_visual_manifest(repository / "paper" / "visual_manifest.yaml")
    provenance_rows: list[dict[str, str]]
    with (provenance / "paper_artifact_provenance.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        required = {
            "artifact_type",
            "artifact_number",
            "reference_runs",
            "declared_input",
            "generator_target",
        }
        if set(reader.fieldnames or ()) != required:
            raise RuntimeError("paper artifact provenance columns are invalid")
        provenance_rows = list(reader)
    seen_assets: set[str] = set()
    seen_rows: set[tuple[str, ...]] = set()
    for row in provenance_rows:
        try:
            asset_id = f"{row['artifact_type']}_{int(row['artifact_number']):02d}"
        except (KeyError, ValueError) as error:
            raise RuntimeError("paper artifact provenance contains an invalid row") from error
        asset = visual_manifest.assets.get(asset_id)
        if asset is None or row["generator_target"] != asset.tex_target:
            raise RuntimeError(f"paper artifact provenance target is invalid: {asset_id}")
        declared_runs = (
            set() if row["reference_runs"] == "none" else set(row["reference_runs"].split("+"))
        )
        if declared_runs != set(asset.run_ids):
            raise RuntimeError(f"paper artifact provenance run IDs differ: {asset_id}")
        fingerprint = tuple(row[field] for field in reader.fieldnames or ())
        if fingerprint in seen_rows:
            raise RuntimeError("paper artifact provenance contains duplicate rows")
        seen_rows.add(fingerprint)
        seen_assets.add(asset_id)
    if seen_assets != set(visual_manifest.assets):
        raise RuntimeError("paper artifact provenance does not cover all 28 assets")
    return {
        "status": "passed",
        "reference_run_count": len(indexed),
        "paper_asset_count": len(seen_assets),
        "paper_provenance_row_count": len(provenance_rows),
    }


def _write_package_manifest(package: Path) -> dict[str, Any]:
    records = []
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.relative_to(package).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "file_count": len(records),
        "files": records,
    }
    atomic_json(package / "PACKAGE_MANIFEST.json", receipt)
    return receipt


def _validate_package_manifest(package: Path) -> dict[str, Any]:
    receipt = _read_json(package / "PACKAGE_MANIFEST.json")
    records = receipt.get("files")
    if (
        receipt.get("status") != "passed"
        or not isinstance(records, list)
        or receipt.get("file_count") != len(records)
    ):
        raise RuntimeError("package manifest is invalid")
    expected_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json"
    }
    indexed = {str(record.get("path")): record for record in records if isinstance(record, dict)}
    if set(indexed) != expected_paths or len(indexed) != len(records):
        raise RuntimeError("package manifest file inventory differs from payload")
    for relative_value, record in indexed.items():
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("package manifest contains an unsafe path")
        path = package / relative
        if path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256"):
            raise RuntimeError(f"packaged file bytes differ from manifest: {relative_value}")
    return receipt


def _require_passed_reports(
    verification: Path,
    repository_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = _read_json(verification / "analysis_replay.json")
    cpu = _read_json(verification / "cpu_verification.json")
    audit = _read_json(verification / "repository_audit.json")
    for name, report in (("analysis replay", analysis), ("CPU verification", cpu)):
        if report.get("repository_commit") != repository_identity["commit"]:
            raise RuntimeError(f"{name} is not bound to the packaged commit")
        if report.get("repository_tree") != repository_identity["tree"]:
            raise RuntimeError(f"{name} is not bound to the packaged tree")
        if report.get("tracked_worktree_clean") is not True:
            raise RuntimeError(f"{name} was not produced from a clean worktree")
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
    steps = cpu.get("steps")
    if not isinstance(steps, dict):
        raise RuntimeError("all-cpu verification has no structured steps")
    for name in (
        "quality",
        "analysis_replay",
        "unit",
        "integration_cpu",
        "full_plan",
        "repository_audit",
    ):
        if not isinstance(steps.get(name), dict):
            raise RuntimeError(f"all-cpu verification omits the {name} step")
    if steps["quality"].get("status") != "passed":
        raise RuntimeError("quality gates have not passed")
    if steps["analysis_replay"].get("status") != "passed":
        raise RuntimeError("embedded analysis replay has not passed")
    if steps["unit"].get("status") != "passed":
        raise RuntimeError("unit/regression tests have not passed")
    if steps["integration_cpu"].get("status") != "passed":
        raise RuntimeError("CPU integration tests have not passed")
    full_plan = steps["full_plan"]
    if full_plan.get("status") != "passed":
        raise RuntimeError("static full-plan verification has not passed")
    if steps["repository_audit"].get("passed") is not True:
        raise RuntimeError("embedded repository audit has not passed")
    families = full_plan.get("families")
    if not isinstance(families, dict):
        raise RuntimeError("static full-plan report has no family records")
    for family in ("controlled", "imagenet9", "attribution", "covertype"):
        if not isinstance(families.get(family), dict):
            raise RuntimeError(f"static full-plan report omits {family}")
        if families[family].get("status") != "passed":
            raise RuntimeError(f"{family} static full-plan verification has not passed")
    return analysis, cpu


def _validate_source_snapshot_recovery(provenance: Path) -> dict[str, Any]:
    receipt = _read_json(provenance / "source_snapshot_recovery.json")
    if receipt.get("status") != "repaired_and_verified":
        raise RuntimeError("source snapshot recovery is not verified")
    snapshots = receipt.get("snapshots")
    if not isinstance(snapshots, dict) or len(snapshots) != 5:
        raise RuntimeError("source snapshot recovery must cover exactly five snapshots")
    for name, record in snapshots.items():
        if not isinstance(record, dict) or record.get("repaired_sha256_match") is not True:
            raise RuntimeError(f"source snapshot recovery is unverified for {name}")
        if record.get("current_source_byte_compare", {}).get("status") != "passed":
            raise RuntimeError(f"source snapshot payload comparison failed for {name}")
    return receipt


def _validate_source_snapshots(provenance: Path, recovery: dict[str, Any]) -> dict[str, Any]:
    manifest = yaml.safe_load((provenance / "source_snapshots.yaml").read_text(encoding="utf-8"))
    snapshots = manifest.get("snapshots") if isinstance(manifest, dict) else None
    expected_names = {
        "controlled",
        "endpoint_behavior_v1",
        "imagenet9",
        "attribution",
        "covertype",
    }
    if not isinstance(snapshots, dict) or set(snapshots) != expected_names:
        raise RuntimeError("source snapshot manifest inventory is incomplete")
    recovery_snapshots = recovery["snapshots"]
    verified: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        record = snapshots[name]
        if not isinstance(record, dict):
            raise RuntimeError(f"source snapshot record is invalid: {name}")
        path = Path(str(record.get("path", "")))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(f"source snapshot bytes do not match: {name}")
        recovered = recovery_snapshots.get(name)
        if (
            not isinstance(recovered, dict)
            or recovered.get("repaired_sha256") != record.get("sha256")
            or recovered.get("size_bytes") != record.get("size_bytes")
        ):
            raise RuntimeError(f"source snapshot recovery disagrees for {name}")
        verified[name] = {
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
    return {"status": "passed", "count": len(verified), "snapshots": verified}


def _validate_historical_bundle(provenance: Path, repository: Path) -> dict[str, Any]:
    state = _read_json(provenance / "historical_git_state.json")
    bundle = state.get("bundle")
    if not isinstance(bundle, dict):
        raise RuntimeError("historical git state has no bundle record")
    path = Path(str(bundle.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != bundle.get("size_bytes")
        or _sha256(path) != bundle.get("sha256")
    ):
        raise RuntimeError("historical repository bundle bytes do not match")
    _command(repository, "git", "bundle", "verify", str(path))
    return {
        "status": "passed",
        "sha256": bundle["sha256"],
        "size_bytes": bundle["size_bytes"],
    }


def _validate_historical_drift(path: Path, provenance: Path) -> dict[str, Any]:
    drift = _read_json(path)
    if drift.get("status") not in {"unchanged", "documented_external_drift"}:
        raise RuntimeError("historical repository drift is not safely classified")
    if drift.get("historical_repository_modified_by_restructure") is not False:
        raise RuntimeError("historical repository has restructuring-owned writes")
    if drift.get("head_unchanged") is not True:
        raise RuntimeError("historical repository HEAD changed after freeze")
    if drift.get("tracked_diff", {}).get("unchanged") is not True:
        raise RuntimeError("historical repository tracked diff changed after freeze")
    if drift.get("staged_diff", {}).get("unchanged") is not True:
        raise RuntimeError("historical repository staged diff changed after freeze")
    if drift.get("external_drift_detected") is True:
        if drift.get("status") != "documented_external_drift":
            raise RuntimeError("external historical drift is not documented")
        if drift.get("only_additional_untracked_paths") is not True:
            raise RuntimeError("external historical drift is broader than additive files")
        if not drift.get("added_untracked_files"):
            raise RuntimeError("external historical drift has no file evidence")

    frozen = _read_json(provenance / "historical_git_state.json")
    historical = Path(str(drift.get("historical_repository", ""))).resolve()
    if historical != Path(str(frozen.get("absolute_path", ""))).resolve():
        raise RuntimeError("historical drift repository differs from frozen state")
    status = _git_bytes(historical, "status", "--porcelain=v1", "--untracked-files=all")
    tracked = _git_bytes(historical, "diff", "--no-ext-diff", "--binary")
    staged = _git_bytes(historical, "diff", "--cached", "--no-ext-diff", "--binary")
    head = _git_bytes(historical, "rev-parse", "HEAD").decode().strip()
    fingerprint = hashlib.sha256(status + b"\0" + tracked + b"\0" + staged).hexdigest()
    if head != drift.get("current_head") or fingerprint != drift.get(
        "current_working_tree_fingerprint"
    ):
        raise RuntimeError("historical repository changed after drift observation")

    records = frozen.get("records")
    if not isinstance(records, dict):
        raise RuntimeError("frozen historical records are invalid")
    frozen_status_text = str(records["status_porcelain"]["stdout"])
    frozen_tracked_text = str(records["diff"]["stdout"])
    frozen_staged_text = str(records["diff_staged"]["stdout"])
    frozen_payloads = [
        value.encode("utf-8", errors="surrogateescape")
        for value in (frozen_status_text, frozen_tracked_text, frozen_staged_text)
    ]
    frozen_fingerprint = hashlib.sha256(b"\0".join(frozen_payloads)).hexdigest()
    frozen_head = str(records["head"]["stdout"]).strip()
    if (
        frozen_fingerprint != frozen.get("working_tree_fingerprint")
        or frozen_fingerprint != drift.get("frozen_working_tree_fingerprint")
        or frozen_head != drift.get("frozen_head")
    ):
        raise RuntimeError("historical frozen-state identity does not verify")
    if head != frozen_head:
        raise RuntimeError("historical repository live HEAD differs from freeze")
    if tracked != frozen_payloads[1]:
        raise RuntimeError("historical repository live tracked diff differs from freeze")
    if staged != frozen_payloads[2]:
        raise RuntimeError("historical repository live staged diff differs from freeze")
    if hashlib.sha256(frozen_payloads[1]).hexdigest() != drift.get("tracked_diff", {}).get(
        "frozen_sha256"
    ) or hashlib.sha256(frozen_payloads[2]).hexdigest() != drift.get("staged_diff", {}).get(
        "frozen_sha256"
    ):
        raise RuntimeError("historical frozen diff hashes do not verify")
    frozen_status = frozen_status_text.splitlines()
    current_status = status.decode("utf-8", errors="surrogateescape").splitlines()
    added_lines = sorted(set(current_status) - set(frozen_status))
    removed_lines = sorted(set(frozen_status) - set(current_status))
    added_paths = [line[3:] for line in added_lines if line.startswith("?? ")]
    only_additional_untracked = (
        not removed_lines
        and len(added_paths) == len(added_lines)
        and head == frozen_head
        and tracked == frozen_payloads[1]
        and staged == frozen_payloads[2]
    )
    external_drift = fingerprint != frozen_fingerprint
    if drift.get("external_drift_detected") is not external_drift:
        raise RuntimeError("historical external-drift classification is inconsistent")
    if drift.get("only_additional_untracked_paths") is not only_additional_untracked:
        raise RuntimeError("historical additive-drift classification is inconsistent")
    expected_status = "documented_external_drift" if external_drift else "unchanged"
    if drift.get("status") != expected_status:
        raise RuntimeError("historical drift status differs from live state")
    if removed_lines or len(added_paths) != len(added_lines):
        raise RuntimeError("historical status drift is not purely additive/untracked")
    if (
        added_lines != drift.get("added_status_lines")
        or removed_lines != drift.get("removed_status_lines")
        or len(current_status) != drift.get("current_status_line_count")
        or len(frozen_status) != drift.get("initial_status_line_count")
    ):
        raise RuntimeError("historical status delta differs from drift receipt")
    if hashlib.sha256(tracked).hexdigest() != drift.get("tracked_diff", {}).get(
        "current_sha256"
    ) or hashlib.sha256(staged).hexdigest() != drift.get("staged_diff", {}).get("current_sha256"):
        raise RuntimeError("historical diff bytes differ from drift receipt")
    if len(tracked) != drift.get("tracked_diff", {}).get("current_size_bytes") or len(
        staged
    ) != drift.get("staged_diff", {}).get("current_size_bytes"):
        raise RuntimeError("historical diff sizes differ from drift receipt")

    added_records = drift.get("added_untracked_files")
    if not isinstance(added_records, list) or sorted(
        str(record.get("path")) for record in added_records if isinstance(record, dict)
    ) != sorted(added_paths):
        raise RuntimeError("historical added-file inventory differs from status")
    for record in added_records:
        if not isinstance(record, dict):
            raise RuntimeError("historical added-file record is invalid")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("historical added-file path is not contained")
        candidate = (historical / relative).resolve()
        try:
            candidate.relative_to(historical)
        except ValueError as error:
            raise RuntimeError("historical added-file path escapes repository") from error
        if (
            not candidate.is_file()
            or candidate.stat().st_size != record.get("size_bytes")
            or _sha256(candidate) != record.get("sha256")
        ):
            raise RuntimeError(f"historical added-file bytes differ from receipt: {relative}")
    return drift


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


def _archive_repository(repository: Path, destination: Path, commit: str) -> None:
    archive = destination.parent / "repository.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(archive), commit],
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
    historical_drift: Path,
) -> dict[str, Any]:
    repository = repository.resolve()
    provenance = provenance.resolve()
    verification = verification.resolve()
    release_root = release_root.resolve()
    git_info = _git_info(repository)
    provenance_manifests = _validate_provenance_manifests(provenance, repository)
    analysis, cpu = _require_passed_reports(verification, git_info)
    recovery = _validate_source_snapshot_recovery(provenance)
    snapshots = _validate_source_snapshots(provenance, recovery)
    historical_bundle = _validate_historical_bundle(provenance, repository)
    drift = _validate_historical_drift(historical_drift.resolve(), provenance)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    basename = f"decaf_reproducibility_release_v1_{timestamp}"
    destination = release_root / "packages" / f"{basename}.zip"

    with tempfile.TemporaryDirectory(prefix="decaf-release-") as temporary_value:
        temporary = Path(temporary_value)
        package = temporary / basename
        _archive_repository(repository, package / "repository", git_info["commit"])
        for name in PROVENANCE_FILES:
            source = provenance / name
            if not source.is_file():
                raise FileNotFoundError(f"required provenance file is missing: {source}")
            _copy(source, package / "provenance" / name)
        if not historical_drift.is_file():
            raise FileNotFoundError(historical_drift)
        _copy(
            historical_drift,
            package / "provenance" / "historical_repository_external_drift.json",
        )
        for name in VERIFICATION_FILES:
            source = verification / name
            if not source.is_file():
                raise FileNotFoundError(f"required verification file is missing: {source}")
            _copy(source, package / "verification" / name)
        atomic_json(package / "GIT_INFO.json", git_info)
        package_manifest = _write_package_manifest(package)
        _validate_package_manifest(package)
        _write_zip(package, destination)

    digest = _sha256(destination)
    atomic_text(destination.with_suffix(".zip.sha256"), f"{digest}  {destination.name}\n")
    steps = cpu.get("steps", {})
    source_missing = analysis.get("source_missing_recorded", [])
    visual_manifest = load_visual_manifest(repository / "paper" / "visual_manifest.yaml")
    source_gap_records = []
    for asset in visual_manifest.assets.values():
        if asset.status != "source_missing":
            continue
        source_gap_records.append(
            {
                "asset_id": asset.asset_id,
                "missing_item": asset.generation_contract["missing_item"],
                "why_it_matters": asset.generation_contract["why_it_matters"],
                "what_remains_reproducible": asset.generation_contract["reproducible_scope"],
                "required_action": asset.generation_contract["required_recovery_action"],
            }
        )
    if sorted(source_missing) != sorted(record["asset_id"] for record in source_gap_records):
        raise RuntimeError("analysis and visual-manifest source gaps disagree")
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
        "quality_status": steps.get("quality", {}).get("status"),
        "static_plan_status": steps.get("full_plan", {}).get("status"),
        "gpu_verification_pending": True,
        "historical_source_gaps": source_gap_records,
        "source_snapshot_recovery_status": recovery["status"],
        "source_snapshots_verified_count": snapshots["count"],
        "historical_bundle_verification_status": historical_bundle["status"],
        "reference_provenance_verification_status": provenance_manifests["status"],
        "paper_provenance_assets_verified_count": provenance_manifests["paper_asset_count"],
        "packaged_file_count": package_manifest["file_count"] + 1,
        "historical_repository_modified_by_restructure": drift[
            "historical_repository_modified_by_restructure"
        ],
        "historical_repository_external_drift_detected": drift["external_drift_detected"],
        "historical_repository_external_drift_attribution": drift.get("attribution"),
        "historical_repository_external_drift_manifest": str(historical_drift.resolve()),
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
    parser.add_argument("--historical-drift", type=Path, required=True)
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
