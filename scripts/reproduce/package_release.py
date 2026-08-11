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
from collections import Counter
from collections.abc import Mapping
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
ANALYSIS_INVENTORY_KEYS = {
    "portable_path",
    "source_root",
    "relative_path",
    "sha256",
    "size_bytes",
    "role",
}
ANALYSIS_ROLE_COUNTS = {
    "generated_tex": 28,
    "canonical_csv": 27,
    "canonical_receipt": 1,
    "family_replay_receipt": 1,
    "replay_receipt": 1,
    "headline_assertions": 1,
    "paper_artifact_diff": 1,
}
PAPER_DIFF_COLUMNS = (
    "asset_id",
    "kind",
    "number",
    "manifest_status",
    "generated_path",
    "generated_sha256",
    "generated_bytes",
    "comparison_status",
    "canonical_path",
    "canonical_sha256",
    "semantic_contract_sha256",
    "schema_sha256",
    "row_count",
    "panel_cardinality",
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


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _portable_relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"{label} is not a portable relative path")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
        or not relative.parts
    ):
        raise RuntimeError(f"{label} is not a contained relative path")
    return relative


def _inventory_source(
    verification: Path,
    record: Mapping[str, Any],
) -> tuple[Path, str]:
    if set(record) != ANALYSIS_INVENTORY_KEYS:
        raise RuntimeError("analysis artifact inventory record fields differ")
    if record.get("source_root") != "verification_root":
        raise RuntimeError("analysis artifact source root is not verification_root")
    relative_value = record.get("relative_path")
    relative = _portable_relative_path(relative_value, label="analysis artifact path")
    expected_portable = f"verification_root/{relative.as_posix()}"
    if record.get("portable_path") != expected_portable:
        raise RuntimeError("analysis artifact portable path differs from its source")
    if not _is_sha256(record.get("sha256")):
        raise RuntimeError("analysis artifact SHA256 is malformed")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0:
        raise RuntimeError("analysis artifact size is not a positive integer")

    root = verification.resolve()
    candidate = root / relative
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"analysis artifact traverses a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(
            f"analysis artifact is missing or escapes its root: {relative}"
        ) from error
    if resolved != candidate or not candidate.is_file():
        raise RuntimeError(f"analysis artifact source is not a regular contained file: {relative}")
    if candidate.stat().st_size != record["size_bytes"] or _sha256(candidate) != record["sha256"]:
        raise RuntimeError(f"analysis artifact bytes differ from inventory: {relative}")
    return candidate, relative.as_posix()


def _validate_inventory_contract(
    analysis: Mapping[str, Any],
    verification: Path,
    repository: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Any, list[Any]]:
    if analysis.get("paper_outputs_root") != "verification_root/paper_outputs":
        raise RuntimeError("analysis paper-output root is not portable")
    inventory_value = analysis.get("artifact_inventory")
    if not isinstance(inventory_value, list) or len(inventory_value) != 60:
        raise RuntimeError("analysis artifact inventory must contain exactly 60 records")
    if analysis.get("artifact_inventory_count") != len(inventory_value):
        raise RuntimeError("analysis artifact inventory count differs")
    if analysis.get("artifact_inventory_sha256") != _json_sha256(inventory_value):
        raise RuntimeError("analysis artifact inventory digest differs")

    inventory: list[dict[str, Any]] = []
    relative_index: dict[str, dict[str, Any]] = {}
    portable_paths: list[str] = []
    for value in inventory_value:
        if not isinstance(value, Mapping):
            raise RuntimeError("analysis artifact inventory contains a malformed record")
        record = dict(value)
        _, relative = _inventory_source(verification, record)
        inventory.append(record)
        portable_paths.append(str(record["portable_path"]))
        if relative in relative_index:
            raise RuntimeError(f"analysis artifact path is duplicated: {relative}")
        relative_index[relative] = record
    if portable_paths != sorted(portable_paths) or len(set(portable_paths)) != 60:
        raise RuntimeError("analysis artifact inventory is not uniquely sorted")
    role_counts = Counter(str(record["role"]) for record in inventory)
    if dict(role_counts) != ANALYSIS_ROLE_COUNTS:
        raise RuntimeError(f"analysis artifact role counts differ: {dict(role_counts)}")

    singular_roles = {
        "replay_receipt_sha256": "replay_receipt",
        "family_replay_receipt_sha256": "family_replay_receipt",
        "canonical_receipt_sha256": "canonical_receipt",
        "headline_assertions_sha256": "headline_assertions",
        "paper_artifact_diff_sha256": "paper_artifact_diff",
    }
    for field, role in singular_roles.items():
        matches = [record for record in inventory if record["role"] == role]
        if len(matches) != 1 or analysis.get(field) != matches[0]["sha256"]:
            raise RuntimeError(f"analysis {role} hash is not bound to its inventory record")

    manifest = load_visual_manifest(repository / "paper" / "visual_manifest.yaml")
    assets = list(manifest.assets.values())
    asset_ids = [asset.asset_id for asset in assets]
    expected_ids = [f"figure_{number:02d}" for number in range(1, 13)] + [
        f"table_{number:02d}" for number in range(1, 17)
    ]
    if asset_ids != expected_ids:
        raise RuntimeError("tracked visual manifest does not contain the expected 28 assets")

    expected_paths_by_role: dict[str, set[str]] = {role: set() for role in ANALYSIS_ROLE_COUNTS}
    for asset in assets:
        subdirectory = "figures" if asset.kind == "figure" else "tables"
        expected_paths_by_role["generated_tex"].add(
            f"paper_outputs/generated/{subdirectory}/{Path(asset.tex_target).name}"
        )
        if asset.status != "source_missing":
            expected_paths_by_role["canonical_csv"].add(
                f"paper_outputs/canonical/{subdirectory}/{asset.asset_id}.csv"
            )
    expected_paths_by_role["canonical_receipt"] = {"paper_outputs/receipts/canonical_receipt.json"}
    expected_paths_by_role["family_replay_receipt"] = {
        "paper_outputs/receipts/family_replay_receipt.json"
    }
    expected_paths_by_role["replay_receipt"] = {"paper_outputs/receipts/replay_receipt.json"}
    expected_paths_by_role["headline_assertions"] = {"headline_assertions.json"}
    expected_paths_by_role["paper_artifact_diff"] = {"paper_artifact_diff.csv"}
    actual_paths_by_role = {
        role: {str(record["relative_path"]) for record in inventory if record["role"] == role}
        for role in ANALYSIS_ROLE_COUNTS
    }
    if actual_paths_by_role != expected_paths_by_role:
        raise RuntimeError("analysis artifact role-to-path mapping differs from contract")

    paper_root = verification / "paper_outputs"
    if not paper_root.is_dir():
        raise RuntimeError("sealed paper-output directory is missing")
    for path in paper_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"sealed paper outputs contain a symlink: {path}")
    live_paper_paths = {
        path.relative_to(verification).as_posix()
        for path in paper_root.rglob("*")
        if path.is_file()
    }
    declared_paper_paths = {
        relative for relative in relative_index if relative.startswith("paper_outputs/")
    }
    if live_paper_paths != declared_paper_paths or len(live_paper_paths) != 58:
        raise RuntimeError("sealed paper outputs are not the exact 58-file allowlist")
    return inventory, relative_index, manifest, assets


def _validate_headline_and_diff(
    analysis: Mapping[str, Any],
    verification: Path,
    assets: list[Any],
    relative_index: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    headline = _read_json(verification / "headline_assertions.json")
    assertions = headline.get("assertions")
    expected_assertion_owners = {
        str(spec["id"]): asset.asset_id for asset in assets for spec in asset.headline_assertions
    }
    if (
        headline.get("schema_version") != 1
        or headline.get("status") != "passed"
        or headline.get("assertion_count") != 27
        or headline.get("verified_count") != 27
        or headline.get("source_missing_count") != 0
        or analysis.get("headline_assertion_count") != 27
        or not isinstance(assertions, dict)
        or set(assertions) != set(expected_assertion_owners)
    ):
        raise RuntimeError("headline assertion receipt differs from the 27-claim contract")
    for assertion_id, owner in expected_assertion_owners.items():
        value = assertions[assertion_id]
        if (
            not isinstance(value, Mapping)
            or value.get("status") != "verified"
            or value.get("asset_id") != owner
        ):
            raise RuntimeError(f"headline assertion is not verified: {assertion_id}")

    with (verification / "paper_artifact_diff.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PAPER_DIFF_COLUMNS:
            raise RuntimeError("paper artifact diff columns differ")
        diff_rows = list(reader)
    asset_ids = [asset.asset_id for asset in assets]
    if [row["asset_id"] for row in diff_rows] != asset_ids:
        raise RuntimeError("paper artifact diff asset order or identity differs")
    statuses = Counter(row["comparison_status"] for row in diff_rows)
    if statuses != {
        "regenerated_semantic_geometry": 11,
        "source_missing_recorded": 1,
        "regenerated_semantic_table": 16,
    }:
        raise RuntimeError(f"paper artifact diff statuses differ: {dict(statuses)}")
    diff_by_id = {row["asset_id"]: row for row in diff_rows}
    for asset, row in zip(assets, diff_rows, strict=True):
        subdirectory = "figures" if asset.kind == "figure" else "tables"
        generated_path = f"paper_outputs/generated/{subdirectory}/{Path(asset.tex_target).name}"
        generated_record = relative_index.get(generated_path)
        expected_comparison = (
            "source_missing_recorded"
            if asset.status == "source_missing"
            else (
                "regenerated_semantic_geometry"
                if asset.kind == "figure"
                else "regenerated_semantic_table"
            )
        )
        if (
            row["kind"] != asset.kind
            or row["number"] != str(asset.number)
            or row["manifest_status"] != asset.status
            or row["generated_path"] != generated_path
            or generated_record is None
            or generated_record["role"] != "generated_tex"
            or row["generated_sha256"] != generated_record["sha256"]
            or row["generated_bytes"] != str(generated_record["size_bytes"])
            or row["comparison_status"] != expected_comparison
        ):
            raise RuntimeError(f"paper artifact diff generated record differs: {asset.asset_id}")
        if asset.status == "source_missing":
            empty_fields = (
                "canonical_path",
                "canonical_sha256",
                "semantic_contract_sha256",
                "schema_sha256",
                "row_count",
            )
            if (
                any(row[field] for field in empty_fields)
                or row["panel_cardinality"] != "{}"
                or asset.asset_id != "figure_01"
            ):
                raise RuntimeError("Figure 1 source-gap record is not explicit and canonical-free")
    assert isinstance(assertions, dict)
    return assertions, diff_by_id


def _validate_canonical_receipt(
    verification: Path,
    manifest: Any,
    assets: list[Any],
    relative_index: Mapping[str, Mapping[str, Any]],
    diff_by_id: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    canonical = _read_json(verification / "paper_outputs" / "receipts" / "canonical_receipt.json")
    canonical_rows = canonical.get("artifacts")
    canonical_ids = [asset.asset_id for asset in assets if asset.asset_id != "figure_01"]
    if (
        canonical.get("schema_version") != 1
        or canonical.get("status") != "completed"
        or canonical.get("artifact_count") != 27
        or not _is_sha256(canonical.get("schema_sha256"))
        or not _is_sha256(canonical.get("contract_set_sha256"))
        or not isinstance(canonical_rows, list)
        or [row.get("asset_id") for row in canonical_rows if isinstance(row, Mapping)]
        != canonical_ids
    ):
        raise RuntimeError("canonical receipt differs from the 27-asset contract")
    for value in canonical_rows:
        if not isinstance(value, Mapping):
            raise RuntimeError("canonical receipt contains a malformed artifact")
        item = dict(value)
        asset_id = str(item["asset_id"])
        asset = manifest.assets[asset_id]
        subdirectory = "figures" if asset.kind == "figure" else "tables"
        original_path = f"paper_data/canonical/{subdirectory}/{asset_id}.csv"
        sealed_path = f"paper_outputs/canonical/{subdirectory}/{asset_id}.csv"
        sealed_record = relative_index.get(sealed_path)
        diff_row = diff_by_id[asset_id]
        if (
            item.get("kind") != asset.kind
            or item.get("path") != original_path
            or sealed_record is None
            or sealed_record["role"] != "canonical_csv"
            or item.get("sha256") != sealed_record["sha256"]
            or item.get("size_bytes") != sealed_record["size_bytes"]
            or item.get("schema_sha256") != canonical["schema_sha256"]
            or not _is_sha256(item.get("semantic_contract_sha256"))
            or type(item.get("row_count")) is not int
            or item["row_count"] <= 0
            or not isinstance(item.get("panel_cardinality"), dict)
        ):
            raise RuntimeError(f"canonical artifact receipt differs: {asset_id}")
        if (
            diff_row["canonical_path"] != sealed_path
            or diff_row["canonical_sha256"] != item["sha256"]
            or diff_row["semantic_contract_sha256"] != item["semantic_contract_sha256"]
            or diff_row["schema_sha256"] != item["schema_sha256"]
            or diff_row["row_count"] != str(item["row_count"])
            or diff_row["panel_cardinality"]
            != json.dumps(item["panel_cardinality"], sort_keys=True, separators=(",", ":"))
        ):
            raise RuntimeError(f"paper diff and canonical receipt disagree: {asset_id}")
    return canonical


def _validate_family_receipt(verification: Path) -> dict[str, Any]:
    family = _read_json(verification / "paper_outputs" / "receipts" / "family_replay_receipt.json")
    family_rows = family.get("families")
    family_names = ("controlled", "imagenet9", "attribution", "covertype")
    family_artifact_counts = {
        "controlled": 19,
        "imagenet9": 13,
        "attribution": 39,
        "covertype": 25,
    }
    invocation = _portable_relative_path(
        family.get("invocation_path"), label="family invocation path"
    )
    if (
        family.get("schema_version") != 2
        or family.get("status") != "completed"
        or family.get("family_count") != 4
        or not isinstance(family_rows, list)
        or [row.get("family") for row in family_rows if isinstance(row, Mapping)]
        != list(family_names)
    ):
        raise RuntimeError("family replay receipt is not a completed four-family replay")
    for value in family_rows:
        if not isinstance(value, Mapping):
            raise RuntimeError("family replay receipt contains a malformed row")
        name = str(value["family"])
        stages = value.get("stages")
        artifacts = value.get("artifacts")
        if (
            value.get("status") != "completed"
            or value.get("path") != f"{invocation.as_posix()}/{name}"
            or not isinstance(artifacts, list)
            or len(artifacts) != family_artifact_counts[name]
            or not isinstance(stages, list)
            or [
                (stage.get("stage"), stage.get("status"))
                for stage in stages
                if isinstance(stage, Mapping)
            ]
            != [("analyze", "completed"), ("paper", "completed")]
        ):
            raise RuntimeError(f"family replay receipt differs: {name}")
    return family


def _validate_replay_receipt(
    verification: Path,
    repository: Path,
    assets: list[Any],
    assertions: Mapping[str, Any],
    family: Mapping[str, Any],
    canonical: Mapping[str, Any],
    relative_index: Mapping[str, Mapping[str, Any]],
) -> None:
    replay = _read_json(verification / "paper_outputs" / "receipts" / "replay_receipt.json")
    expected_replay_keys = {
        "schema_version",
        "paper_data_directory",
        "runs",
        "inputs",
        "representative_cases",
        "headline_assertions",
        "assets",
        "family_replay",
        "canonical",
    }
    runs = replay.get("runs")
    inputs = replay.get("inputs")
    replay_assets = replay.get("assets")
    representatives = replay.get("representative_cases")
    asset_ids = [asset.asset_id for asset in assets]
    if (
        set(replay) != expected_replay_keys
        or replay.get("schema_version") != 2
        or replay.get("paper_data_directory") != "paper_data"
        or not isinstance(runs, list)
        or not isinstance(inputs, list)
        or not isinstance(replay_assets, dict)
        or set(replay_assets) != set(asset_ids)
        or not isinstance(representatives, dict)
        or set(representatives) != {"figure_02", "figure_03", "figure_04"}
        or any(
            not isinstance(value, Mapping) or value.get("status") != "verified"
            for value in representatives.values()
        )
        or replay.get("headline_assertions") != assertions
        or replay.get("family_replay") != family
    ):
        raise RuntimeError("replay receipt structure or nested evidence differs")

    tracked_runs = load_reference_runs(repository / "manifests" / "reference_runs")
    expected_run_ids = sorted(tracked_runs)
    if [row.get("run_id") for row in runs if isinstance(row, Mapping)] != expected_run_ids:
        raise RuntimeError("replay receipt does not contain the exact nine reference runs")
    for value in runs:
        if not isinstance(value, Mapping):
            raise RuntimeError("replay receipt contains a malformed run")
        run = tracked_runs[str(value["run_id"])]
        expected = {
            "family": run.family,
            "scientific_status": run.scientific_status,
            "archive_filename": run.archive_filename,
            "archive_sha256": run.archive_sha256,
            "archive_size_bytes": run.archive_size_bytes,
            "archive_member_count": run.archive_member_count,
        }
        if any(value.get(field) != expected_value for field, expected_value in expected.items()):
            raise RuntimeError(f"replay run receipt differs from manifest: {run.run_id}")

    requested = {run_id: set(run.analysis_inputs) for run_id, run in tracked_runs.items()}
    for asset in assets:
        for raw in asset.raw_inputs:
            requested[raw.run_id].add(raw.member)
    expected_inputs = [
        (run_id, suffix) for run_id in sorted(requested) for suffix in sorted(requested[run_id])
    ]
    actual_inputs = [
        (str(value.get("run_id")), str(value.get("requested_suffix")))
        for value in inputs
        if isinstance(value, Mapping)
    ]
    if len(inputs) != 72 or actual_inputs != expected_inputs:
        raise RuntimeError("replay receipt does not contain the exact 72 requested inputs")

    nested_canonical_value = replay.get("canonical")
    if not isinstance(nested_canonical_value, Mapping):
        raise RuntimeError("replay receipt has no nested canonical receipt")
    nested_canonical = dict(nested_canonical_value)
    nested_path = nested_canonical.pop("path", None)
    nested_sha256 = nested_canonical.pop("sha256", None)
    nested_size = nested_canonical.pop("size_bytes", None)
    canonical_inventory = relative_index["paper_outputs/receipts/canonical_receipt.json"]
    if (
        nested_path != "paper_data/canonical/canonical_receipt.json"
        or nested_sha256 != canonical_inventory["sha256"]
        or nested_size != canonical_inventory["size_bytes"]
        or nested_canonical != canonical
    ):
        raise RuntimeError("nested and standalone canonical receipts disagree")


def _validate_analysis_artifacts(
    analysis: Mapping[str, Any],
    verification: Path,
    repository: Path,
) -> list[dict[str, Any]]:
    """Revalidate the complete portable semantic-paper evidence closure."""

    inventory, relative_index, manifest, assets = _validate_inventory_contract(
        analysis, verification, repository
    )
    assertions, diff_by_id = _validate_headline_and_diff(
        analysis, verification, assets, relative_index
    )
    canonical = _validate_canonical_receipt(
        verification, manifest, assets, relative_index, diff_by_id
    )
    family = _validate_family_receipt(verification)
    _validate_replay_receipt(
        verification,
        repository,
        assets,
        assertions,
        family,
        canonical,
        relative_index,
    )
    return inventory


def _validate_public_package_payload(package: Path) -> dict[str, Any]:
    """Reject non-portable text and PDFs in the public repository/evidence payload."""

    roots = (package / "repository", package / "verification")
    private_fragments = (
        "/" + "work" + "/" + "Users" + "/",
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "/" + "mnt" + "/",
        "/" + "tmp" + "/",
        "C:" + "\\" + "Users" + "\\",
    )
    scanned = 0
    for root in roots:
        if not root.is_dir():
            raise RuntimeError(f"public package root is missing: {root.name}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"public package contains a symlink: {path}")
            if not path.is_file():
                continue
            scanned += 1
            if path.suffix.lower() == ".pdf":
                raise RuntimeError(f"public package contains a PDF: {path}")
            if path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError(f"public package text is not UTF-8: {path}") from error
            if any(fragment in content for fragment in private_fragments):
                raise RuntimeError(f"public package contains a private absolute path: {path}")
            if any(
                0x3400 <= ord(character) <= 0x4DBF
                or 0x4E00 <= ord(character) <= 0x9FFF
                or 0xF900 <= ord(character) <= 0xFAFF
                for character in content
            ):
                raise RuntimeError(f"public package contains CJK text: {path}")
    return {"status": "passed", "scanned_file_count": scanned}


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
    if analysis.get("inputs_materialized") != 72:
        raise RuntimeError("the release does not materialize all 72 declared inputs")
    if analysis.get("paper_assets_mapped") != 28:
        raise RuntimeError("the release does not map all 28 paper assets")
    if analysis.get("figure_assets_emitted") != 12:
        raise RuntimeError("the release does not emit all 12 figure assets")
    if analysis.get("figures_regenerated") != 11:
        raise RuntimeError("the release does not regenerate 11 data-backed figures")
    if analysis.get("figures_source_missing_recorded") != 1:
        raise RuntimeError("the release does not record the sole figure source gap")
    if analysis.get("source_missing_recorded") != ["figure_01"]:
        raise RuntimeError("the release source-gap identity is not Figure 1")
    if analysis.get("tables_regenerated") != 16:
        raise RuntimeError("the release does not regenerate all 16 tables")
    if analysis.get("family_replays_completed") != 4:
        raise RuntimeError("the release does not replay all four experiment families")
    if analysis.get("canonical_assets_materialized") != 27:
        raise RuntimeError("the release does not materialize all 27 reproducible assets")
    if analysis.get("artifact_inventory_count") != 60:
        raise RuntimeError("the release analysis inventory is not exactly 60 files")
    if analysis.get("headline_assertion_count") != 27:
        raise RuntimeError("the release does not verify all 27 headline assertions")
    if analysis.get("paper_outputs_root") != "verification_root/paper_outputs":
        raise RuntimeError("the release analysis paper-output root is not portable")
    if analysis.get("headline_assertions_status") != "passed":
        raise RuntimeError("the release headline assertions have not passed")
    if analysis.get("model_inference_performed") is not False:
        raise RuntimeError("analysis replay unexpectedly performed model inference")
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
    if steps["analysis_replay"] != analysis:
        raise RuntimeError("standalone and CPU-embedded analysis reports differ")
    if steps["unit"].get("status") != "passed":
        raise RuntimeError("unit/regression tests have not passed")
    if steps["integration_cpu"].get("status") != "passed":
        raise RuntimeError("CPU integration tests have not passed")
    full_plan = steps["full_plan"]
    if full_plan.get("status") != "passed":
        raise RuntimeError("static full-plan verification has not passed")
    if steps["repository_audit"].get("passed") is not True:
        raise RuntimeError("embedded repository audit has not passed")
    if steps["repository_audit"] != audit:
        raise RuntimeError("standalone and CPU-embedded repository audits differ")
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
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    analysis_artifacts = _validate_analysis_artifacts(analysis, verification, repository)
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
        for record in analysis_artifacts:
            source, relative_value = _inventory_source(verification, record)
            destination_artifact = package / "verification" / relative_value
            _copy(source, destination_artifact)
            if (
                destination_artifact.stat().st_size != record["size_bytes"]
                or _sha256(destination_artifact) != record["sha256"]
            ):
                raise RuntimeError(f"packaged analysis artifact bytes differ: {relative_value}")
        packaged_paper_outputs = {
            path.relative_to(package / "verification").as_posix()
            for path in (package / "verification" / "paper_outputs").rglob("*")
            if path.is_file()
        }
        if len(packaged_paper_outputs) != 58:
            raise RuntimeError("release package does not contain exactly 58 paper outputs")
        atomic_json(package / "GIT_INFO.json", git_info)
        public_payload = _validate_public_package_payload(package)
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
        "paper_assets_mapped_count": analysis["paper_assets_mapped"],
        "figures_regenerated_count": analysis["figures_regenerated"],
        "figures_source_missing_recorded_count": analysis["figures_source_missing_recorded"],
        "tables_regenerated_count": analysis["tables_regenerated"],
        "paper_artifacts_packaged_count": len(analysis_artifacts),
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
        "public_payload_portability_status": public_payload["status"],
        "public_payload_scanned_file_count": public_payload["scanned_file_count"],
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
