"""Run reproducibility gates and emit machine-readable verification reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from decaf.audit import audit_repository
from decaf.experiments.common import (
    atomic_json,
    atomic_text,
    parse_devices,
    repository_root,
    utc_now,
)
from decaf.paper.analysis_replay import replay_paper_data
from decaf.paper.manifest import load_visual_manifest
from decaf.paper.render import PaperRenderError, render_all, validate_rendered_asset

MODES = (
    "all-cpu",
    "analysis-replay",
    "checkpoint-fingerprint",
    "unit",
    "integration-cpu",
    "full-plan",
    "quality",
    "repository-audit",
)
FAMILIES = ("controlled", "imagenet9", "attribution", "covertype")
FINGERPRINT_CASE_COUNTS = {"controlled": 2, "imagenet9": 3, "attribution": 7}


class VerificationFailure(RuntimeError):
    """Raised when a required reproducibility gate fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _gpu_environment(torch: Any, device: Any) -> dict[str, Any]:
    """Record the exact local software and CUDA device used for fingerprints."""

    import importlib.metadata
    import platform

    versions: dict[str, str | None] = {}
    for distribution in ("torch", "torchvision", "timm", "captum", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "libraries": versions,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_index": int(device.index or 0),
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "unavailable")),
        "device_total_memory_bytes": int(properties.total_memory),
    }


def _validate_fingerprint_case(case: Mapping[str, Any]) -> None:
    required = {
        "family",
        "case_id",
        "model_id",
        "checkpoints",
        "sample_ids",
        "preprocessed_tensor",
        "target_class",
        "logits",
        "probabilities",
        "precision",
        "device",
    }
    missing = sorted(required - set(case))
    if missing:
        raise VerificationFailure(f"fingerprint case is missing fields: {missing}")
    checkpoints = case["checkpoints"]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise VerificationFailure(f"fingerprint case has no checkpoints: {case['case_id']}")
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise VerificationFailure("fingerprint checkpoint is not an object")
        path = Path(str(checkpoint.get("path", "")))
        expected = str(checkpoint.get("sha256", ""))
        if not path.is_file() or len(expected) != 64 or _sha256(path) != expected:
            raise VerificationFailure(f"fingerprint checkpoint bytes drifted: {path}")
        if int(checkpoint.get("bytes", -1)) != path.stat().st_size:
            raise VerificationFailure(f"fingerprint checkpoint size drifted: {path}")
    tensor = case["preprocessed_tensor"]
    if not isinstance(tensor, Mapping) or not {
        "sha256",
        "dtype",
        "shape",
        "byte_order",
        "layout",
    }.issubset(tensor):
        raise VerificationFailure(f"fingerprint tensor contract is incomplete: {case['case_id']}")
    logits = np.asarray(case["logits"], dtype=np.float64)
    probabilities = np.asarray(case["probabilities"], dtype=np.float64)
    if (
        logits.ndim != 2
        or probabilities.shape != logits.shape
        or logits.shape[0] < 1
        or not np.isfinite(logits).all()
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-5, rtol=0.0)
    ):
        raise VerificationFailure(f"fingerprint outputs are invalid: {case['case_id']}")


def run_checkpoint_fingerprint(
    repo: Path,
    verification: Path,
    devices: Sequence[int],
) -> dict[str, Any]:
    """Load the exact offline checkpoints and record real CUDA forward fingerprints."""

    if tuple(devices) != (0,):
        raise VerificationFailure("single-B200 fingerprint verification requires --devices 0")
    try:
        import torch
    except ImportError as error:
        message = "checkpoint fingerprints require the persistent GPU PyTorch"
        raise VerificationFailure(message) from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise VerificationFailure(
            "checkpoint fingerprints require exactly one CUDA device visible to PyTorch"
        )
    device = torch.device("cuda:0")
    environment = _gpu_environment(torch, device)
    if "B200" not in str(environment["device_name"]):
        raise VerificationFailure(f"expected an NVIDIA B200, received {environment['device_name']}")

    collectors = {
        "controlled": (
            "decaf.experiments.controlled.gpu_runtime",
            "collect_checkpoint_fingerprints",
        ),
        "imagenet9": (
            "decaf.experiments.imagenet9.gpu_runtime",
            "collect_checkpoint_fingerprints",
        ),
        "attribution": (
            "decaf.experiments.attribution.gpu_runtime",
            "collect_checkpoint_fingerprints",
        ),
    }
    import importlib

    cases: list[dict[str, Any]] = []
    coverage: dict[str, int] = {}
    for family, (module_name, function_name) in collectors.items():
        module = importlib.import_module(module_name)
        collector = getattr(module, function_name, None)
        if not callable(collector):
            raise VerificationFailure(f"{module_name} does not export {function_name}")
        family_cases = list(collector(device=device))
        if len(family_cases) != FINGERPRINT_CASE_COUNTS[family]:
            raise VerificationFailure(
                f"{family} produced {len(family_cases)} fingerprint cases, "
                f"expected {FINGERPRINT_CASE_COUNTS[family]}"
            )
        for case in family_cases:
            _validate_fingerprint_case(case)
        coverage[family] = len(family_cases)
        cases.extend(dict(case) for case in family_cases)
        torch.cuda.empty_cache()
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)) or len(cases) != sum(FINGERPRINT_CASE_COUNTS.values()):
        raise VerificationFailure("fingerprint case coverage is duplicated or incomplete")

    payload = {
        "schema_version": 1,
        "status": "passed",
        "repository": _git_identity(repo),
        "environment": environment,
        "tensor_hash_contract": {
            "algorithm": "sha256",
            "source": "C-contiguous tensor bytes after CPU conversion",
            "byte_order": "little-endian",
            "shape_and_dtype_recorded_separately": True,
        },
        "coverage": coverage,
        "case_count": len(cases),
        "cases": cases,
    }
    fingerprints_path = verification / "checkpoint_fingerprints.json"
    atomic_json(fingerprints_path, payload)
    report = {
        "schema_version": 1,
        "status": "passed",
        "checkpoint_fingerprints_path": "checkpoint_fingerprints.json",
        "checkpoint_fingerprints_sha256": _sha256(fingerprints_path),
        "case_set_sha256": _canonical_json_sha256(case_ids),
        "coverage": coverage,
        "case_count": len(cases),
        "device": environment,
        "checks": {
            "exact_case_coverage": True,
            "checkpoint_bytes_verified": True,
            "preprocessed_tensor_hashes_recorded": True,
            "finite_logits": True,
            "normalized_probabilities": True,
            "single_b200": True,
        },
    }
    atomic_json(verification / "checkpoint_fingerprint_report.json", report)
    return report


def _portable_command(command: Sequence[str]) -> list[str]:
    """Hide environment-specific interpreter paths from public receipts."""

    return [
        "python" if index == 0 and argument == sys.executable else str(argument)
        for index, argument in enumerate(command)
    ]


def _run_command(command: Sequence[str], *, cwd: Path, echo_output: bool = True) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(
        tuple(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if echo_output:
        sys.stdout.write(process.stdout)
    report = {
        "command": _portable_command(command),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "exit_code": process.returncode,
        "status": "passed" if process.returncode == 0 else "failed",
        "output": process.stdout,
    }
    if process.returncode:
        raise VerificationFailure(
            "command failed with exit code "
            f"{process.returncode}: {' '.join(_portable_command(command))}"
        )
    return report


def _git_identity(repo: Path) -> dict[str, Any]:
    """Bind verification evidence to the exact repository tree under test."""

    def git(*arguments: str) -> str:
        process = subprocess.run(
            ("git", *arguments),
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
        return process.stdout.strip()

    status = git("status", "--porcelain=v1")
    return {
        "repository_commit": git("rev-parse", "HEAD"),
        "repository_tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": not status,
    }


def _write_artifact_diff(
    repo: Path,
    generated_paths: Sequence[Path],
    verification: Path,
    canonical_receipt: Mapping[str, Any],
    generated_root: Path,
) -> dict[str, Any]:
    """Write an exact 28-asset, canonical-data-bound render inventory."""

    manifest = load_visual_manifest(repo / "paper" / "visual_manifest.yaml")
    generated_root = generated_root.resolve()
    if len(generated_paths) != len(manifest.assets):
        raise VerificationFailure(
            f"paper render emitted {len(generated_paths)} paths, expected {len(manifest.assets)}"
        )
    by_name = {path.name: path for path in generated_paths}
    if len(by_name) != len(generated_paths):
        raise VerificationFailure("paper render emitted duplicate destination names")
    canonical_rows = canonical_receipt.get("artifacts")
    if (
        canonical_receipt.get("status") != "completed"
        or canonical_receipt.get("artifact_count") != 27
        or not isinstance(canonical_rows, list)
        or len(canonical_rows) != 27
    ):
        raise VerificationFailure("canonical receipt does not contain exactly 27 artifacts")
    canonical_by_id = {
        str(item.get("asset_id")): item for item in canonical_rows if isinstance(item, Mapping)
    }
    if len(canonical_by_id) != 27:
        raise VerificationFailure("canonical receipt contains duplicate/malformed asset IDs")
    rows: list[dict[str, Any]] = []
    for asset in manifest.assets.values():
        expected_name = Path(asset.tex_target).name
        path = by_name.get(expected_name)
        exists = path is not None and path.is_file()
        classification = "missing"
        if exists and path is not None:
            try:
                classification = validate_rendered_asset(asset, path.read_text(encoding="utf-8"))
            except (PaperRenderError, UnicodeDecodeError):
                classification = "invalid_rendered_asset"
        canonical = canonical_by_id.get(asset.asset_id)
        if asset.status == "source_missing":
            if canonical is not None:
                raise VerificationFailure(
                    f"source-missing {asset.asset_id} unexpectedly has canonical data"
                )
            canonical = {}
        elif canonical is None:
            raise VerificationFailure(f"{asset.asset_id} has no canonical-data receipt")
        relative = ""
        if path is not None:
            try:
                relative = path.resolve().relative_to(generated_root).as_posix()
            except ValueError as error:
                raise VerificationFailure(
                    f"generated asset escapes generated root: {path}"
                ) from error
        exported_generated = f"paper_outputs/generated/{relative}" if relative else ""
        canonical_path = str(canonical.get("path", ""))
        exported_canonical = ""
        if canonical_path:
            prefix = "paper_data/canonical/"
            if not canonical_path.startswith(prefix):
                raise VerificationFailure(
                    f"{asset.asset_id} canonical path is outside the registered root"
                )
            exported_canonical = "paper_outputs/canonical/" + canonical_path[len(prefix) :]
        rows.append(
            {
                "asset_id": asset.asset_id,
                "kind": asset.kind,
                "number": asset.number,
                "manifest_status": asset.status,
                "generated_path": exported_generated,
                "generated_sha256": _sha256(path) if exists and path is not None else "",
                "generated_bytes": path.stat().st_size if exists and path is not None else "",
                "comparison_status": classification,
                "canonical_path": exported_canonical,
                "canonical_sha256": canonical.get("sha256", ""),
                "semantic_contract_sha256": canonical.get("semantic_contract_sha256", ""),
                "schema_sha256": canonical.get("schema_sha256", ""),
                "row_count": canonical.get("row_count", ""),
                "panel_cardinality": json.dumps(
                    canonical.get("panel_cardinality", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    destination = verification / "paper_artifact_diff.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(rows[0])
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(destination, buffer.getvalue())
    invalid = [
        row["asset_id"]
        for row in rows
        if row["comparison_status"] in {"missing", "invalid_rendered_asset"}
    ]
    if invalid:
        raise VerificationFailure(f"paper assets were not data-rendered: {invalid}")
    summary = {
        "paper_assets_mapped": len(rows),
        "figure_assets_emitted": sum(row["kind"] == "figure" for row in rows),
        "figures_regenerated": sum(
            row["comparison_status"] == "regenerated_semantic_geometry" for row in rows
        ),
        "figures_source_missing_recorded": sum(
            row["comparison_status"] == "source_missing_recorded" for row in rows
        ),
        "tables_regenerated": sum(
            row["comparison_status"] == "regenerated_semantic_table" for row in rows
        ),
        "source_missing_recorded": [
            row["asset_id"] for row in rows if row["comparison_status"] == "source_missing_recorded"
        ],
    }
    expected = {
        "paper_assets_mapped": 28,
        "figure_assets_emitted": 12,
        "figures_regenerated": 11,
        "figures_source_missing_recorded": 1,
        "tables_regenerated": 16,
        "source_missing_recorded": ["figure_01"],
    }
    if summary != expected:
        raise VerificationFailure(f"paper artifact summary differs from contract: {summary}")
    return summary


def _repository_relative(path: Path, repo: Path) -> str | None:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return None


def _inventory_row(path: Path, root: Path, root_name: str, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise VerificationFailure(f"artifact escapes {root_name}: {path}") from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise VerificationFailure(f"artifact is missing or empty: {path}")
    return {
        "portable_path": f"{root_name}/{relative}",
        "source_root": root_name,
        "relative_path": relative,
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
        "role": role,
    }


def _assert_portable_evidence(paths: Sequence[Path]) -> None:
    forbidden = (
        "/" + "work" + "/" + "Users" + "/",
        "/" + "home" + "/",
        "/" + "tmp" + "/",
        "C:" + "\\" + "Users" + "\\",
    )
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise VerificationFailure(f"paper evidence is not UTF-8 text: {path}") from error
        matched = next((value for value in forbidden if value in content), None)
        if matched is not None:
            raise VerificationFailure(
                f"paper evidence contains a private absolute-path fragment: {path}"
            )
        if any(
            0x3400 <= ord(character) <= 0x4DBF
            or 0x4E00 <= ord(character) <= 0x9FFF
            or 0xF900 <= ord(character) <= 0xFAFF
            for character in content
        ):
            raise VerificationFailure(f"paper evidence contains CJK text: {path}")


def _legacy_analysis_artifact_inventory(
    *,
    replay_root: Path,
    generated_root: Path,
    verification: Path,
    generated_paths: Sequence[Path],
    receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _inventory_row(path, generated_root, "generated_root", "generated_tex")
        for path in generated_paths
    ]
    canonical = receipt.get("canonical")
    if not isinstance(canonical, Mapping):
        raise VerificationFailure("analysis replay has no canonical receipt")
    canonical_rows = canonical.get("artifacts")
    if not isinstance(canonical_rows, list) or len(canonical_rows) != 27:
        raise VerificationFailure("analysis replay canonical inventory is not exactly 27")
    for item in canonical_rows:
        if not isinstance(item, Mapping):
            raise VerificationFailure("canonical artifact receipt is malformed")
        path = replay_root / str(item.get("path", ""))
        row = _inventory_row(path, replay_root, "replay_root", "canonical_csv")
        if row["sha256"] != item.get("sha256") or row["size_bytes"] != item.get("size_bytes"):
            raise VerificationFailure(f"canonical artifact bytes drifted: {path}")
        rows.append(row)
    receipt_files = (
        (
            replay_root / str(canonical.get("path", "")),
            replay_root,
            "replay_root",
            "canonical_receipt",
            canonical,
        ),
        (
            replay_root / "family_replays" / "family_replay_receipt.json",
            replay_root,
            "replay_root",
            "family_replay_receipt",
            None,
        ),
        (
            replay_root / "replay_receipt.json",
            replay_root,
            "replay_root",
            "replay_receipt",
            None,
        ),
        (
            verification / "headline_assertions.json",
            verification,
            "verification_root",
            "headline_assertions",
            None,
        ),
        (
            verification / "paper_artifact_diff.csv",
            verification,
            "verification_root",
            "paper_artifact_diff",
            None,
        ),
    )
    for path, root, root_name, role, expected in receipt_files:
        row = _inventory_row(path, root, root_name, role)
        if expected is not None and (
            row["sha256"] != expected.get("sha256")
            or row["size_bytes"] != expected.get("size_bytes")
        ):
            raise VerificationFailure(f"recorded receipt bytes drifted: {path}")
        rows.append(row)
    portable_paths = [str(row["portable_path"]) for row in rows]
    if len(rows) != 60 or len(set(portable_paths)) != 60:
        raise VerificationFailure(
            f"analysis artifact inventory must contain 60 unique paths, received {len(rows)}"
        )
    return sorted(rows, key=lambda row: str(row["portable_path"]))


def _seal_paper_outputs(
    *,
    replay_root: Path,
    generated_root: Path,
    verification: Path,
    generated_paths: Sequence[Path],
    receipt: Mapping[str, Any],
) -> list[tuple[Path, str]]:
    """Copy the exact public evidence allowlist under the verification root."""

    export_root = verification / "paper_outputs"
    if export_root.exists():
        shutil.rmtree(export_root)
    exports: list[tuple[Path, str]] = []

    def copy(source: Path, destination: Path, role: str) -> None:
        if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
            raise VerificationFailure(f"paper evidence source is unsafe: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _sha256(source) != _sha256(destination):
            raise VerificationFailure(f"paper evidence copy drifted: {destination}")
        exports.append((destination, role))

    for path in generated_paths:
        try:
            relative = path.resolve().relative_to(generated_root.resolve())
        except ValueError as error:
            raise VerificationFailure(f"generated asset escapes generated root: {path}") from error
        copy(path, export_root / "generated" / relative, "generated_tex")

    canonical = receipt.get("canonical")
    if not isinstance(canonical, Mapping):
        raise VerificationFailure("analysis replay has no canonical receipt")
    canonical_rows = canonical.get("artifacts")
    if not isinstance(canonical_rows, list) or len(canonical_rows) != 27:
        raise VerificationFailure("analysis replay canonical inventory is not exactly 27")
    for item in canonical_rows:
        if not isinstance(item, Mapping):
            raise VerificationFailure("canonical artifact receipt is malformed")
        relative = str(item.get("path", ""))
        prefix = "paper_data/canonical/"
        if not relative.startswith(prefix):
            raise VerificationFailure(f"canonical artifact path is outside its root: {relative}")
        source = replay_root / relative
        if _sha256(source) != item.get("sha256") or source.stat().st_size != item.get("size_bytes"):
            raise VerificationFailure(f"canonical artifact bytes drifted: {source}")
        copy(source, export_root / "canonical" / relative[len(prefix) :], "canonical_csv")

    receipt_files = (
        (
            replay_root / str(canonical.get("path", "")),
            export_root / "receipts" / "canonical_receipt.json",
            "canonical_receipt",
        ),
        (
            replay_root / "family_replays" / "family_replay_receipt.json",
            export_root / "receipts" / "family_replay_receipt.json",
            "family_replay_receipt",
        ),
        (
            replay_root / "replay_receipt.json",
            export_root / "receipts" / "replay_receipt.json",
            "replay_receipt",
        ),
    )
    for source, destination, role in receipt_files:
        copy(source, destination, role)
    if len(exports) != 58:
        raise VerificationFailure(
            f"sealed paper outputs must contain 58 files, received {len(exports)}"
        )
    return exports


def _analysis_artifact_inventory(
    *,
    verification: Path,
    sealed_outputs: Sequence[tuple[Path, str]],
) -> list[dict[str, Any]]:
    rows = [
        _inventory_row(path, verification, "verification_root", role)
        for path, role in sealed_outputs
    ]
    rows.extend(
        [
            _inventory_row(
                verification / "headline_assertions.json",
                verification,
                "verification_root",
                "headline_assertions",
            ),
            _inventory_row(
                verification / "paper_artifact_diff.csv",
                verification,
                "verification_root",
                "paper_artifact_diff",
            ),
        ]
    )
    portable_paths = [str(row["portable_path"]) for row in rows]
    if len(rows) != 60 or len(set(portable_paths)) != 60:
        raise VerificationFailure(
            f"analysis artifact inventory must contain 60 unique paths, received {len(rows)}"
        )
    return sorted(rows, key=lambda row: str(row["portable_path"]))


def run_analysis_replay(
    repo: Path,
    verification: Path,
    reference_roots: Sequence[str] | None,
    generated_root: Path,
) -> dict[str, Any]:
    """Verify sealed archives, recompute assertions, and regenerate all TeX assets."""

    replay_root = verification / "replay"
    receipt = replay_paper_data(
        replay_root,
        reference_root=reference_roots,
        repo_root=repo,
    )
    paths = render_all(replay_root, repo_root=repo, generated_root=generated_root)
    assertions = dict(receipt["headline_assertions"])
    unacceptable = {
        name: value
        for name, value in assertions.items()
        if value.get("status") not in {"verified", "source_missing", "generated"}
    }
    if unacceptable:
        raise VerificationFailure(
            "headline assertions did not fail closed: "
            + ", ".join(f"{name}={value.get('status')}" for name, value in unacceptable.items())
        )
    canonical = receipt.get("canonical")
    if not isinstance(canonical, Mapping):
        raise VerificationFailure("analysis replay did not publish canonical artifacts")
    artifact_summary = _write_artifact_diff(
        repo,
        paths,
        verification,
        canonical,
        generated_root,
    )
    headline_report = {
        "schema_version": 1,
        "status": "passed",
        "assertion_count": len(assertions),
        "verified_count": sum(value.get("status") == "verified" for value in assertions.values()),
        "source_missing_count": sum(
            value.get("status") == "source_missing" for value in assertions.values()
        ),
        "assertions": assertions,
    }
    atomic_json(verification / "headline_assertions.json", headline_report)
    sealed_outputs = _seal_paper_outputs(
        replay_root=replay_root,
        generated_root=generated_root,
        verification=verification,
        generated_paths=paths,
        receipt=receipt,
    )
    _assert_portable_evidence(
        [path for path, _ in sealed_outputs]
        + [
            verification / "headline_assertions.json",
            verification / "paper_artifact_diff.csv",
        ]
    )
    inventory = _analysis_artifact_inventory(
        verification=verification,
        sealed_outputs=sealed_outputs,
    )
    hashes_by_role = {str(item["role"]): str(item["sha256"]) for item in inventory}
    inventory_sha256 = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    family_replay = receipt.get("family_replay")
    if not isinstance(family_replay, Mapping):
        raise VerificationFailure("analysis replay did not publish family receipts")
    replay_relative = _repository_relative(replay_root, repo)
    generated_relative = _repository_relative(generated_root, repo)
    report = {
        "schema_version": 2,
        "status": "passed",
        "completed_at": utc_now(),
        "replay_root": (
            f"repository_root/{replay_relative}"
            if replay_relative is not None
            else "verification_root/replay"
        ),
        "replay_root_relative_to_repository": replay_relative,
        "generated_root": (
            f"repository_root/{generated_relative}"
            if generated_relative is not None
            else "generated_root"
        ),
        "generated_root_relative_to_repository": generated_relative,
        "paper_outputs_root": "verification_root/paper_outputs",
        "reference_runs_verified": len(receipt["runs"]),
        "inputs_materialized": len(receipt["inputs"]),
        "family_replays_completed": family_replay.get("family_count"),
        "canonical_assets_materialized": canonical.get("artifact_count"),
        **artifact_summary,
        "headline_assertion_count": len(assertions),
        "headline_assertions_status": "passed",
        "model_inference_performed": False,
        "replay_receipt_sha256": hashes_by_role["replay_receipt"],
        "family_replay_receipt_sha256": hashes_by_role["family_replay_receipt"],
        "canonical_receipt_sha256": hashes_by_role["canonical_receipt"],
        "headline_assertions_sha256": hashes_by_role["headline_assertions"],
        "paper_artifact_diff_sha256": hashes_by_role["paper_artifact_diff"],
        "artifact_inventory_count": len(inventory),
        "artifact_inventory_sha256": inventory_sha256,
        "artifact_inventory": inventory,
        **_git_identity(repo),
    }
    atomic_json(verification / "analysis_replay.json", report)
    return report


def run_unit(repo: Path) -> dict[str, Any]:
    """Run model-agnostic and paper-regression tests."""

    targets = [
        relative for relative in ("tests/unit", "tests/regression") if (repo / relative).exists()
    ]
    return _run_command([sys.executable, "-m", "pytest", "-q", *targets], cwd=repo)


def run_integration_cpu(repo: Path) -> dict[str, Any]:
    """Run the real CPU integration suite."""

    target = repo / "tests" / "integration"
    if not target.is_dir() or not any(target.glob("test_*.py")):
        raise VerificationFailure("CPU integration tests have not been implemented")
    report = _run_command(
        [sys.executable, "-m", "pytest", "-q", "tests/integration"],
        cwd=repo,
    )
    report["gpu_real_shard_verification"] = "pending"
    return report


def _assertion_status(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and isinstance(value.get("passed"), bool):
        return bool(value["passed"])
    return None


def _summarize_plan_report(family: str, report: dict[str, Any]) -> dict[str, Any]:
    """Parse a planner receipt, fail on false assertions, and omit its large body."""

    output = str(report.get("output", ""))
    try:
        plan = json.loads(output)
    except (TypeError, json.JSONDecodeError) as error:
        raise VerificationFailure(f"{family} plan-only output is not one JSON document") from error
    if not isinstance(plan, dict):
        raise VerificationFailure(f"{family} plan-only output is not an object")

    assertion_source = plan.get("assertions")
    if not isinstance(assertion_source, dict):
        assertion_source = plan.get("audit")
    if not isinstance(assertion_source, dict):
        raise VerificationFailure(f"{family} plan has no assertion/audit mapping")

    assertions = {
        name: status
        for name, value in assertion_source.items()
        if (status := _assertion_status(value)) is not None
    }
    errors = assertion_source.get("errors", [])
    failed = sorted(name for name, passed in assertions.items() if not passed)
    if isinstance(errors, list) and errors:
        failed.append("errors")
    if not assertions:
        raise VerificationFailure(f"{family} plan exposes no boolean assertions")
    if failed:
        raise VerificationFailure(f"{family} plan assertions failed: {', '.join(failed)}")

    actual_members = plan.get("member_count")
    expected_members = plan.get("expected_member_count")
    if (
        isinstance(actual_members, int)
        and isinstance(expected_members, int)
        and actual_members != expected_members
    ):
        raise VerificationFailure(
            f"{family} planned {actual_members} members, expected {expected_members}"
        )

    counts = plan.get("scientific_counts")
    if not isinstance(counts, dict):
        counts = plan.get("counts")
    if not isinstance(counts, dict):
        counts = {}
    if isinstance(actual_members, int):
        counts = {
            **counts,
            "member_count": actual_members,
            "expected_member_count": expected_members,
        }

    return {
        "status": "passed",
        "command": report["command"],
        "elapsed_seconds": report["elapsed_seconds"],
        "exit_code": report["exit_code"],
        "output_bytes": len(output.encode("utf-8")),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "counts": counts,
        "assertion_count": len(assertions),
        "assertions": assertions,
    }


def run_full_plan(repo: Path) -> dict[str, Any]:
    """Run all paper-profile planners without starting computation."""

    reports: dict[str, Any] = {}
    for family in FAMILIES:
        raw_report = _run_command(
            [
                sys.executable,
                "-m",
                f"decaf.experiments.{family}.cli",
                "--profile",
                "paper",
                "--plan-only",
            ],
            cwd=repo,
            echo_output=False,
        )
        reports[family] = _summarize_plan_report(family, raw_report)
    return {"status": "passed", "families": reports}


def run_quality(repo: Path) -> dict[str, Any]:
    """Run formatting, lint, import, and public shell syntax gates."""

    reports = {
        "ruff_check": _run_command(
            [sys.executable, "-m", "ruff", "check", "."],
            cwd=repo,
        ),
        "ruff_format": _run_command(
            [sys.executable, "-m", "ruff", "format", "--check", "."],
            cwd=repo,
        ),
        "static_imports": _run_command(
            [
                sys.executable,
                "-c",
                (
                    "import decaf, decaf.audit, decaf.verification; "
                    "import decaf.paper.analysis_replay, decaf.paper.render; "
                    "import decaf.experiments.controlled.cli; "
                    "import decaf.experiments.imagenet9.cli; "
                    "import decaf.experiments.attribution.cli; "
                    "import decaf.experiments.covertype.cli"
                ),
            ],
            cwd=repo,
        ),
    }
    tracked = _run_command(["git", "ls-files", "--", "*.sh"], cwd=repo)
    shell_scripts = sorted(line for line in str(tracked["output"]).splitlines() if line)
    for relative in shell_scripts:
        _run_command(["bash", "-n", relative], cwd=repo)
    reports["shell_syntax"] = {
        "status": "passed",
        "checked_count": len(shell_scripts),
        "checked_paths": shell_scripts,
    }
    return {"status": "passed", "checks": reports}


def run_repository_audit(repo: Path, verification: Path) -> dict[str, Any]:
    """Scan the public checkout for release-forbidden content."""

    report = audit_repository(repo)
    atomic_json(verification / "repository_audit.json", report)
    if not report["passed"]:
        raise VerificationFailure(f"repository audit found {report['finding_count']} issue(s)")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="all-cpu")
    parser.add_argument(
        "--reference-root",
        "--reference-runs",
        action="append",
        dest="reference_roots",
        help="Archive file or recursive archive-search root; repeat as needed",
    )
    parser.add_argument("--output", type=Path, help="Verification report directory")
    parser.add_argument("--generated-root", type=Path, help="Generated TeX root")
    parser.add_argument(
        "--devices",
        default=(0,),
        type=parse_devices,
        help="Comma-separated physical CUDA IDs; checkpoint fingerprinting requires 0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = repository_root()
    verification = (args.output or repo / "verification").resolve()
    generated = (args.generated_root or repo / "paper" / "generated").resolve()
    verification.mkdir(parents=True, exist_ok=True)
    report_name = (
        "checkpoint_fingerprint_verification.json"
        if args.mode == "checkpoint-fingerprint"
        else "cpu_verification.json"
    )
    steps: dict[str, Any] = {}
    started_at = utc_now()
    identity = _git_identity(repo)
    try:
        if args.mode == "all-cpu" and not identity["tracked_worktree_clean"]:
            raise VerificationFailure("all-cpu verification requires a clean repository")
        if args.mode in {"quality", "all-cpu"}:
            steps["quality"] = run_quality(repo)
        if args.mode in {"analysis-replay", "all-cpu"}:
            steps["analysis_replay"] = run_analysis_replay(
                repo,
                verification,
                args.reference_roots,
                generated,
            )
        if args.mode in {"unit", "all-cpu"}:
            steps["unit"] = run_unit(repo)
        if args.mode in {"integration-cpu", "all-cpu"}:
            steps["integration_cpu"] = run_integration_cpu(repo)
        if args.mode in {"full-plan", "all-cpu"}:
            steps["full_plan"] = run_full_plan(repo)
        if args.mode in {"repository-audit", "all-cpu"}:
            steps["repository_audit"] = run_repository_audit(repo, verification)
        if args.mode == "checkpoint-fingerprint":
            steps["checkpoint_fingerprint"] = run_checkpoint_fingerprint(
                repo,
                verification,
                args.devices,
            )
    except Exception as error:
        report = {
            "schema_version": 1,
            "mode": args.mode,
            "status": "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "gpu_real_shard_verification": (
                "failed" if args.mode == "checkpoint-fingerprint" else "pending"
            ),
            **identity,
            "steps": steps,
            "error": f"{type(error).__name__}: {error}",
        }
        atomic_json(verification / report_name, report)
        raise
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "status": "passed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "gpu_real_shard_verification": (
            "checkpoint_fingerprints_passed" if args.mode == "checkpoint-fingerprint" else "pending"
        ),
        **identity,
        "steps": steps,
    }
    atomic_json(verification / report_name, report)
    print(f"verification_status={report['status']}")
    print(f"gpu_real_shard_verification={report['gpu_real_shard_verification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VerificationFailure",
    "build_parser",
    "main",
    "run_analysis_replay",
    "run_checkpoint_fingerprint",
    "run_full_plan",
    "run_integration_cpu",
    "run_quality",
    "run_repository_audit",
    "run_unit",
]
