from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml

from decaf.paper.manifest import load_visual_manifest
from decaf.paper.reference import load_reference_runs
from decaf.paper.semantic import (
    CANONICAL_COLUMNS,
    CANONICAL_SCHEMA_SHA256,
    semantic_contract,
    semantic_contract_sha256,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "reproduce" / "package_release.py"
SPEC = spec_from_file_location("decaf_package_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PACKAGE_RELEASE = module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE_RELEASE)
_archive_repository = PACKAGE_RELEASE._archive_repository
_validate_analysis_artifacts = PACKAGE_RELEASE._validate_analysis_artifacts
_validate_canonical_receipt = PACKAGE_RELEASE._validate_canonical_receipt
_require_passed_reports = PACKAGE_RELEASE._require_passed_reports
_validate_historical_drift = PACKAGE_RELEASE._validate_historical_drift
_validate_package_manifest = PACKAGE_RELEASE._validate_package_manifest
_validate_provenance_manifests = PACKAGE_RELEASE._validate_provenance_manifests
_validate_public_package_payload = PACKAGE_RELEASE._validate_public_package_payload
_validate_source_snapshot_recovery = PACKAGE_RELEASE._validate_source_snapshot_recovery
_validate_source_snapshots = PACKAGE_RELEASE._validate_source_snapshots
_write_package_manifest = PACKAGE_RELEASE._write_package_manifest

REPOSITORY_IDENTITY = {
    "commit": "a" * 40,
    "tree": "b" * 40,
    "tracked_worktree_clean": True,
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
    ).stdout


def _passed_steps(
    analysis: dict[str, object] | None = None,
) -> dict[str, object]:
    families = {
        family: {"status": "passed"}
        for family in ("controlled", "imagenet9", "attribution", "covertype")
    }
    return {
        "quality": {"status": "passed"},
        "analysis_replay": analysis or {"status": "passed"},
        "unit": {"status": "passed"},
        "integration_cpu": {"status": "passed"},
        "full_plan": {"status": "passed", "families": families},
        "repository_audit": {"passed": True},
    }


def _bound_report(payload: dict[str, object]) -> dict[str, object]:
    return {
        **payload,
        "repository_commit": REPOSITORY_IDENTITY["commit"],
        "repository_tree": REPOSITORY_IDENTITY["tree"],
        "tracked_worktree_clean": True,
    }


def _analysis_report() -> dict[str, object]:
    return _bound_report(
        {
            "status": "passed",
            "reference_runs_verified": 9,
            "inputs_materialized": 72,
            "paper_assets_mapped": 28,
            "figure_assets_emitted": 12,
            "figures_regenerated": 11,
            "figures_source_missing_recorded": 1,
            "source_missing_recorded": ["figure_01"],
            "tables_regenerated": 16,
            "family_replays_completed": 4,
            "canonical_assets_materialized": 27,
            "artifact_inventory_count": 60,
            "headline_assertion_count": 27,
            "headline_assertions_status": "passed",
            "model_inference_performed": False,
            "paper_outputs_root": "verification_root/paper_outputs",
        }
    )


def _artifact_record(
    verification: Path,
    relative: str,
    role: str,
) -> dict[str, object]:
    path = verification / relative
    return {
        "portable_path": f"verification_root/{relative}",
        "source_root": "verification_root",
        "relative_path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "role": role,
    }


def _analysis_artifact_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    verification = tmp_path / "verification"
    verification.mkdir(parents=True)
    manifest = load_visual_manifest(REPOSITORY / "paper/visual_manifest.yaml")
    schema_sha256 = CANONICAL_SCHEMA_SHA256
    canonical_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    generated_paths: list[str] = []
    canonical_paths: list[str] = []

    for asset in manifest.assets.values():
        subdirectory = "figures" if asset.kind == "figure" else "tables"
        generated_relative = f"paper_outputs/generated/{subdirectory}/{Path(asset.tex_target).name}"
        generated = verification / generated_relative
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(f"% generated {asset.asset_id}\n", encoding="utf-8")
        generated_paths.append(generated_relative)
        generated_sha256 = hashlib.sha256(generated.read_bytes()).hexdigest()

        canonical_relative = ""
        canonical_sha256 = ""
        semantic_sha256 = ""
        row_count: int | str = ""
        panel_cardinality: dict[str, int] = {}
        if asset.status != "source_missing":
            contract = semantic_contract(asset)
            if "panels" in contract:
                panel_cardinality = {
                    str(panel): int(count) for panel, count in contract["panels"].items()
                }
            else:
                panel_cardinality = {
                    "table_body": int(contract.get("exact_rows", contract.get("minimum_rows", 1)))
                }
            canonical_relative = f"paper_outputs/canonical/{subdirectory}/{asset.asset_id}.csv"
            canonical = verification / canonical_relative
            canonical.parent.mkdir(parents=True, exist_ok=True)
            source_sha256 = hashlib.sha256(f"source:{asset.asset_id}".encode()).hexdigest()
            rows = []
            for panel, count in panel_cardinality.items():
                for index in range(count):
                    rows.append(
                        {
                            "artifact_id": asset.asset_id,
                            "panel_id": panel,
                            "series": "fixture",
                            "x": index,
                            "y": float(index + 1),
                            "estimate": float(index + 1),
                            "ci_low": float(index),
                            "ci_high": float(index + 2),
                            "n": 1,
                            "source_sha256": source_sha256,
                            "record_json": json.dumps(
                                {
                                    "artifact_id": asset.asset_id,
                                    "index": index,
                                    "panel_id": panel,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    )
            with canonical.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=CANONICAL_COLUMNS,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            canonical_paths.append(canonical_relative)
            canonical_sha256 = hashlib.sha256(canonical.read_bytes()).hexdigest()
            semantic_sha256 = semantic_contract_sha256(asset)
            row_count = len(rows)
            canonical_rows.append(
                {
                    "asset_id": asset.asset_id,
                    "kind": asset.kind,
                    "path": (f"paper_data/canonical/{subdirectory}/{asset.asset_id}.csv"),
                    "sha256": canonical_sha256,
                    "size_bytes": canonical.stat().st_size,
                    "semantic_contract_sha256": semantic_sha256,
                    "schema_sha256": schema_sha256,
                    "row_count": row_count,
                    "panel_count": len(panel_cardinality),
                    "panel_cardinality": panel_cardinality,
                    "source_sha256s": [source_sha256],
                    "resolved_source_sha256s": [source_sha256],
                    "source_lineage": {source_sha256: [source_sha256]},
                    "representative_case_ids": list(contract.get("representative_case_ids", [])),
                }
            )
        comparison = (
            "source_missing_recorded"
            if asset.status == "source_missing"
            else (
                "regenerated_semantic_geometry"
                if asset.kind == "figure"
                else "regenerated_semantic_table"
            )
        )
        diff_rows.append(
            {
                "asset_id": asset.asset_id,
                "kind": asset.kind,
                "number": asset.number,
                "manifest_status": asset.status,
                "generated_path": generated_relative,
                "generated_sha256": generated_sha256,
                "generated_bytes": generated.stat().st_size,
                "comparison_status": comparison,
                "canonical_path": canonical_relative,
                "canonical_sha256": canonical_sha256,
                "semantic_contract_sha256": semantic_sha256,
                "schema_sha256": schema_sha256 if canonical_relative else "",
                "row_count": row_count,
                "panel_cardinality": json.dumps(
                    panel_cardinality, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    canonical_receipt = {
        "schema_version": 1,
        "status": "completed",
        "required_columns": list(CANONICAL_COLUMNS),
        "schema_sha256": schema_sha256,
        "artifact_count": len(canonical_rows),
        "contract_set_sha256": hashlib.sha256(
            json.dumps(
                {
                    asset.asset_id: semantic_contract_sha256(asset)
                    for asset in manifest.assets.values()
                    if asset.status != "source_missing"
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "artifacts": canonical_rows,
    }
    canonical_receipt_path = verification / "paper_outputs/receipts/canonical_receipt.json"
    canonical_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(canonical_receipt_path, canonical_receipt)

    family_counts = {
        "controlled": 19,
        "imagenet9": 13,
        "attribution": 39,
        "covertype": 25,
    }
    invocation_path = "family_replays/invocation-test"
    family_receipt = {
        "schema_version": 2,
        "status": "completed",
        "invocation_path": invocation_path,
        "family_count": 4,
        "families": [
            {
                "family": family,
                "path": f"{invocation_path}/{family}",
                "status": "completed",
                "stages": [
                    {"stage": "analyze", "status": "completed"},
                    {"stage": "paper", "status": "completed"},
                ],
                "analysis": {},
                "paper": {},
                "contract": {},
                "artifacts": [{} for _ in range(count)],
            }
            for family, count in family_counts.items()
        ],
    }
    family_receipt_path = verification / "paper_outputs/receipts/family_replay_receipt.json"
    _write_json(family_receipt_path, family_receipt)

    assertions = {
        str(spec["id"]): {
            "asset_id": asset.asset_id,
            "status": "verified",
        }
        for asset in manifest.assets.values()
        for spec in asset.headline_assertions
    }
    headline = {
        "schema_version": 1,
        "status": "passed",
        "assertion_count": len(assertions),
        "verified_count": len(assertions),
        "source_missing_count": 0,
        "assertions": assertions,
    }
    _write_json(verification / "headline_assertions.json", headline)

    with (verification / "paper_artifact_diff.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=PACKAGE_RELEASE.PAPER_DIFF_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(diff_rows)

    tracked_runs = load_reference_runs(REPOSITORY / "manifests/reference_runs")
    runs = [
        {
            "run_id": run.run_id,
            "family": run.family,
            "scientific_status": run.scientific_status,
            "archive_filename": run.archive_filename,
            "archive_sha256": run.archive_sha256,
            "archive_size_bytes": run.archive_size_bytes,
            "archive_member_count": run.archive_member_count,
        }
        for run in (tracked_runs[run_id] for run_id in sorted(tracked_runs))
    ]
    requested = {run_id: set(run.analysis_inputs) for run_id, run in tracked_runs.items()}
    for asset in manifest.assets.values():
        for raw in asset.raw_inputs:
            requested[raw.run_id].add(raw.member)
    inputs = [
        {"run_id": run_id, "requested_suffix": suffix}
        for run_id in sorted(requested)
        for suffix in sorted(requested[run_id])
    ]
    nested_canonical = {
        **canonical_receipt,
        "path": "paper_data/canonical/canonical_receipt.json",
        "sha256": hashlib.sha256(canonical_receipt_path.read_bytes()).hexdigest(),
        "size_bytes": canonical_receipt_path.stat().st_size,
    }
    replay_receipt = {
        "schema_version": 2,
        "paper_data_directory": "paper_data",
        "runs": runs,
        "inputs": inputs,
        "representative_cases": {
            asset_id: {"status": "verified"} for asset_id in ("figure_02", "figure_03", "figure_04")
        },
        "headline_assertions": assertions,
        "assets": {asset_id: {} for asset_id in manifest.assets},
        "family_replay": family_receipt,
        "canonical": nested_canonical,
    }
    replay_receipt_path = verification / "paper_outputs/receipts/replay_receipt.json"
    _write_json(replay_receipt_path, replay_receipt)

    inventory = [
        *[
            _artifact_record(verification, relative, "generated_tex")
            for relative in generated_paths
        ],
        *[
            _artifact_record(verification, relative, "canonical_csv")
            for relative in canonical_paths
        ],
        _artifact_record(
            verification,
            "paper_outputs/receipts/canonical_receipt.json",
            "canonical_receipt",
        ),
        _artifact_record(
            verification,
            "paper_outputs/receipts/family_replay_receipt.json",
            "family_replay_receipt",
        ),
        _artifact_record(
            verification,
            "paper_outputs/receipts/replay_receipt.json",
            "replay_receipt",
        ),
        _artifact_record(verification, "headline_assertions.json", "headline_assertions"),
        _artifact_record(verification, "paper_artifact_diff.csv", "paper_artifact_diff"),
    ]
    inventory.sort(key=lambda row: str(row["portable_path"]))
    hashes_by_role = {str(row["role"]): str(row["sha256"]) for row in inventory}
    analysis = {
        **_analysis_report(),
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "replay_receipt_sha256": hashes_by_role["replay_receipt"],
        "family_replay_receipt_sha256": hashes_by_role["family_replay_receipt"],
        "canonical_receipt_sha256": hashes_by_role["canonical_receipt"],
        "headline_assertions_sha256": hashes_by_role["headline_assertions"],
        "paper_artifact_diff_sha256": hashes_by_role["paper_artifact_diff"],
    }
    return verification, analysis


def test_release_reports_require_every_structured_gate(tmp_path: Path) -> None:
    analysis_report = _analysis_report()
    _write_json(tmp_path / "analysis_replay.json", analysis_report)
    _write_json(
        tmp_path / "cpu_verification.json",
        _bound_report(
            {
                "status": "passed",
                "mode": "all-cpu",
                "steps": _passed_steps(analysis_report),
            }
        ),
    )
    _write_json(tmp_path / "repository_audit.json", {"passed": True})

    analysis, cpu = _require_passed_reports(tmp_path, REPOSITORY_IDENTITY)

    assert analysis["reference_runs_verified"] == 9
    assert cpu["steps"]["quality"]["status"] == "passed"


def test_release_reports_fail_when_a_family_plan_is_absent(tmp_path: Path) -> None:
    analysis_report = _analysis_report()
    steps = _passed_steps(analysis_report)
    full_plan = steps["full_plan"]
    assert isinstance(full_plan, dict)
    families = full_plan["families"]
    assert isinstance(families, dict)
    del families["attribution"]
    _write_json(tmp_path / "analysis_replay.json", analysis_report)
    _write_json(
        tmp_path / "cpu_verification.json",
        _bound_report({"status": "passed", "mode": "all-cpu", "steps": steps}),
    )
    _write_json(tmp_path / "repository_audit.json", {"passed": True})

    with pytest.raises(RuntimeError, match="omits attribution"):
        _require_passed_reports(tmp_path, REPOSITORY_IDENTITY)


def test_release_reports_reject_a_stale_repository_identity(tmp_path: Path) -> None:
    analysis_report = _analysis_report()
    _write_json(tmp_path / "analysis_replay.json", analysis_report)
    stale_cpu = _bound_report(
        {
            "status": "passed",
            "mode": "all-cpu",
            "steps": _passed_steps(analysis_report),
        }
    )
    stale_cpu["repository_commit"] = "c" * 40
    _write_json(tmp_path / "cpu_verification.json", stale_cpu)
    _write_json(tmp_path / "repository_audit.json", {"passed": True})

    with pytest.raises(RuntimeError, match="CPU verification.*packaged commit"):
        _require_passed_reports(tmp_path, REPOSITORY_IDENTITY)


def test_analysis_artifact_inventory_closes_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    verification, analysis = _analysis_artifact_fixture(tmp_path)

    inventory = _validate_analysis_artifacts(analysis, verification, REPOSITORY)

    assert len(inventory) == 60
    assert (
        sum(str(record["relative_path"]).startswith("paper_outputs/") for record in inventory) == 58
    )
    canonical = verification / "paper_outputs/canonical/figures/figure_02.csv"
    canonical.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bytes differ from inventory"):
        _validate_analysis_artifacts(analysis, verification, REPOSITORY)


def test_canonical_receipt_rejects_hash_consistent_semantic_tamper(
    tmp_path: Path,
) -> None:
    verification, analysis = _analysis_artifact_fixture(tmp_path)
    manifest = load_visual_manifest(REPOSITORY / "paper/visual_manifest.yaml")
    assets = list(manifest.assets.values())
    canonical_path = verification / "paper_outputs/canonical/figures/figure_02.csv"
    source_sha256 = hashlib.sha256(b"forged-source").hexdigest()
    forged_row = {column: "" for column in CANONICAL_COLUMNS}
    forged_row.update(
        {
            "artifact_id": "figure_02",
            "panel_id": "forged",
            "series": "fixture",
            "x": 0,
            "y": 1.0,
            "estimate": 1.0,
            "ci_low": 0.0,
            "ci_high": 2.0,
            "n": 1,
            "source_sha256": source_sha256,
            "record_json": "{}",
        }
    )
    with canonical_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=CANONICAL_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(forged_row)
    forged_sha256 = hashlib.sha256(canonical_path.read_bytes()).hexdigest()

    receipt_path = verification / "paper_outputs/receipts/canonical_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    item = next(row for row in receipt["artifacts"] if row["asset_id"] == "figure_02")
    item.update(
        {
            "sha256": forged_sha256,
            "size_bytes": canonical_path.stat().st_size,
            "row_count": 1,
            "panel_count": 1,
            "panel_cardinality": {"forged": 1},
            "source_sha256s": [source_sha256],
            "resolved_source_sha256s": [source_sha256],
            "source_lineage": {source_sha256: [source_sha256]},
        }
    )
    _write_json(receipt_path, receipt)

    relative_index = {
        str(record["relative_path"]): dict(record) for record in analysis["artifact_inventory"]
    }
    canonical_record = relative_index["paper_outputs/canonical/figures/figure_02.csv"]
    canonical_record.update(
        {
            "sha256": forged_sha256,
            "size_bytes": canonical_path.stat().st_size,
        }
    )
    with (verification / "paper_artifact_diff.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        diff_by_id = {row["asset_id"]: row for row in csv.DictReader(stream)}
    diff_by_id["figure_02"].update(
        {
            "canonical_sha256": forged_sha256,
            "row_count": "1",
            "panel_cardinality": '{"forged":1}',
        }
    )

    with pytest.raises(RuntimeError, match="panel structure drifted"):
        _validate_canonical_receipt(
            verification,
            manifest,
            assets,
            relative_index,
            diff_by_id,
        )


def test_analysis_artifacts_reject_extra_symlink_and_escaping_path(
    tmp_path: Path,
) -> None:
    extra_verification, extra_analysis = _analysis_artifact_fixture(tmp_path / "extra")
    (extra_verification / "paper_outputs/extra.txt").write_text("stale\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact 58-file allowlist"):
        _validate_analysis_artifacts(extra_analysis, extra_verification, REPOSITORY)

    link_verification, link_analysis = _analysis_artifact_fixture(tmp_path / "link")
    (link_verification / "paper_outputs/link.txt").symlink_to(
        link_verification / "headline_assertions.json"
    )
    with pytest.raises(RuntimeError, match="contain a symlink"):
        _validate_analysis_artifacts(link_analysis, link_verification, REPOSITORY)

    path_verification, path_analysis = _analysis_artifact_fixture(tmp_path / "path")
    forged = json.loads(json.dumps(path_analysis))
    forged_record = forged["artifact_inventory"][0]
    forged_record["relative_path"] = "../escape"
    forged_record["portable_path"] = "verification_root/../escape"
    forged["artifact_inventory_sha256"] = hashlib.sha256(
        json.dumps(forged["artifact_inventory"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="contained relative path"):
        _validate_analysis_artifacts(forged, path_verification, REPOSITORY)


def test_public_package_payload_rejects_private_paths_and_pdfs(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    repository = package / "repository"
    verification = package / "verification"
    repository.mkdir(parents=True)
    verification.mkdir()
    (repository / "README.md").write_text("portable\n", encoding="utf-8")
    (verification / "report.json").write_text('{"status":"passed"}\n', encoding="utf-8")

    assert _validate_public_package_payload(package)["status"] == "passed"
    (verification / "report.json").write_text(
        '{"path":"/' + "home" + '/private"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="private absolute path"):
        _validate_public_package_payload(package)
    (verification / "report.json").write_text('{"status":"passed"}\n', encoding="utf-8")
    (verification / "report.json").write_text("\u673a\u5bc6\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contains CJK text"):
        _validate_public_package_payload(package)
    (verification / "report.json").write_text('{"status":"passed"}\n', encoding="utf-8")
    notice = repository / "NOTICE"
    notice.write_text(
        "cache=/" + "work" + "/" + "Lei" + "/private\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="private absolute path"):
        _validate_public_package_payload(package)
    notice.write_text("portable\n", encoding="utf-8")
    (repository / "forbidden.pdf").write_bytes(b"%PDF")
    with pytest.raises(RuntimeError, match="contains a PDF"):
        _validate_public_package_payload(package)


def test_drift_receipt_recomputes_live_tree_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical"
    historical.mkdir()
    _git(historical, "init", "-q")
    (historical / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    _git(historical, "add", "tracked.txt")
    _git(
        historical,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    frozen_status = _git(historical, "status", "--porcelain=v1", "--untracked-files=all")
    frozen_tracked = _git(historical, "diff", "--no-ext-diff", "--binary")
    frozen_staged = _git(historical, "diff", "--cached", "--no-ext-diff", "--binary")
    frozen_head = _git(historical, "rev-parse", "HEAD").decode().strip()
    frozen_fingerprint = hashlib.sha256(
        frozen_status + b"\0" + frozen_tracked + b"\0" + frozen_staged
    ).hexdigest()
    _write_json(
        tmp_path / "historical_git_state.json",
        {
            "absolute_path": str(historical),
            "captured_at": "2026-01-01T00:00:00+00:00",
            "working_tree_fingerprint": frozen_fingerprint,
            "records": {
                "head": {"stdout": frozen_head + "\n"},
                "status_porcelain": {"stdout": frozen_status.decode()},
                "diff": {"stdout": frozen_tracked.decode()},
                "diff_staged": {"stdout": frozen_staged.decode()},
            },
        },
    )

    external = historical / "external.txt"
    external.write_text("external\n", encoding="utf-8")
    current_status = _git(historical, "status", "--porcelain=v1", "--untracked-files=all")
    current_tracked = _git(historical, "diff", "--no-ext-diff", "--binary")
    current_staged = _git(historical, "diff", "--cached", "--no-ext-diff", "--binary")
    current_fingerprint = hashlib.sha256(
        current_status + b"\0" + current_tracked + b"\0" + current_staged
    ).hexdigest()
    empty_sha = hashlib.sha256(b"").hexdigest()
    drift_path = tmp_path / "drift.json"
    _write_json(
        drift_path,
        {
            "status": "documented_external_drift",
            "historical_repository": str(historical),
            "historical_repository_modified_by_restructure": False,
            "head_unchanged": True,
            "frozen_head": frozen_head,
            "current_head": frozen_head,
            "frozen_working_tree_fingerprint": frozen_fingerprint,
            "current_working_tree_fingerprint": current_fingerprint,
            "tracked_diff": {
                "unchanged": True,
                "frozen_sha256": empty_sha,
                "current_sha256": empty_sha,
                "current_size_bytes": 0,
            },
            "staged_diff": {
                "unchanged": True,
                "frozen_sha256": empty_sha,
                "current_sha256": empty_sha,
                "current_size_bytes": 0,
            },
            "external_drift_detected": True,
            "only_additional_untracked_paths": True,
            "initial_status_line_count": 0,
            "current_status_line_count": 1,
            "added_status_lines": ["?? external.txt"],
            "removed_status_lines": [],
            "added_untracked_files": [
                {
                    "path": "external.txt",
                    "size_bytes": external.stat().st_size,
                    "sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
                }
            ],
        },
    )

    def forge_current_observation() -> None:
        report = json.loads(drift_path.read_text(encoding="utf-8"))
        live_status = _git(historical, "status", "--porcelain=v1", "--untracked-files=all")
        live_tracked = _git(historical, "diff", "--no-ext-diff", "--binary")
        live_staged = _git(historical, "diff", "--cached", "--no-ext-diff", "--binary")
        live_lines = live_status.decode().splitlines()
        report["current_head"] = _git(historical, "rev-parse", "HEAD").decode().strip()
        report["current_working_tree_fingerprint"] = hashlib.sha256(
            live_status + b"\0" + live_tracked + b"\0" + live_staged
        ).hexdigest()
        report["current_status_line_count"] = len(live_lines)
        report["added_status_lines"] = sorted(set(live_lines))
        report["removed_status_lines"] = []
        report["tracked_diff"]["current_sha256"] = hashlib.sha256(live_tracked).hexdigest()
        report["tracked_diff"]["current_size_bytes"] = len(live_tracked)
        report["staged_diff"]["current_sha256"] = hashlib.sha256(live_staged).hexdigest()
        report["staged_diff"]["current_size_bytes"] = len(live_staged)
        _write_json(drift_path, report)

    assert _validate_historical_drift(drift_path, tmp_path)["external_drift_detected"] is True
    external.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="added-file bytes"):
        _validate_historical_drift(drift_path, tmp_path)
    external.write_text("external\n", encoding="utf-8")

    (historical / "tracked.txt").write_text("changed\n", encoding="utf-8")
    forge_current_observation()
    with pytest.raises(RuntimeError, match="live tracked diff differs"):
        _validate_historical_drift(drift_path, tmp_path)

    (historical / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    _git(
        historical,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "unexpected head",
    )
    forge_current_observation()
    with pytest.raises(RuntimeError, match="live HEAD differs"):
        _validate_historical_drift(drift_path, tmp_path)


def test_snapshot_recovery_receipt_verifies_live_archives(tmp_path: Path) -> None:
    recovery_records = {
        name: {
            "repaired_sha256_match": True,
            "current_source_byte_compare": {"status": "passed"},
        }
        for name in (
            "controlled",
            "endpoint_behavior_v1",
            "imagenet9",
            "attribution",
            "covertype",
        )
    }
    snapshot_manifest = {"schema_version": 1, "snapshots": {}}
    for name, recovery in recovery_records.items():
        path = tmp_path / f"{name}.tar.gz"
        path.write_bytes(f"snapshot:{name}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        recovery["repaired_sha256"] = digest
        recovery["size_bytes"] = path.stat().st_size
        snapshot_manifest["snapshots"][name] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    _write_json(
        tmp_path / "source_snapshot_recovery.json",
        {"status": "repaired_and_verified", "snapshots": recovery_records},
    )
    (tmp_path / "source_snapshots.yaml").write_text(
        yaml.safe_dump(snapshot_manifest),
        encoding="utf-8",
    )

    recovery = _validate_source_snapshot_recovery(tmp_path)
    assert recovery["status"] == "repaired_and_verified"
    assert _validate_source_snapshots(tmp_path, recovery)["count"] == 5


def test_repository_archive_uses_captured_commit_not_symbolic_head(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    source = repository / "value.txt"
    source.write_text("captured\n", encoding="utf-8")
    _git(repository, "add", "value.txt")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "captured",
    )
    captured = _git(repository, "rev-parse", "HEAD").decode().strip()
    source.write_text("later\n", encoding="utf-8")
    _git(repository, "add", "value.txt")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "later",
    )

    destination = tmp_path / "nested" / "archive"
    _archive_repository(repository, destination, captured)

    assert (destination / "value.txt").read_text(encoding="utf-8") == "captured\n"


def test_provenance_manifests_cross_check_tracked_contracts(tmp_path: Path) -> None:
    tracked_runs = load_reference_runs(REPOSITORY / "manifests/reference_runs")
    run_records = []
    server_archives = []
    for run in tracked_runs.values():
        archive_path = f"/sealed/{run.archive_filename}"
        run_records.append(
            {
                "id": run.run_id,
                "family": run.family,
                "scientific_status": run.scientific_status,
                "archive_path": archive_path,
                "archive_exists": True,
                "archive_sha256": run.archive_sha256,
                "archive_size_bytes": run.archive_size_bytes,
                "archive_member_count": run.archive_member_count,
            }
        )
        server_archives.append(
            {
                "path": archive_path,
                "exists": True,
                "kind": "file",
                "size_bytes": run.archive_size_bytes,
            }
        )
    (tmp_path / "reference_runs.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reference_run_count": 9,
                "all_archives_present": True,
                "runs": run_records,
            }
        ),
        encoding="utf-8",
    )
    _write_json(tmp_path / "historical_git_state.json", {"absolute_path": "/historical"})
    _write_json(
        tmp_path / "server_inventory.json",
        {
            "schema_version": 1,
            "historical_repository": {"path": "/historical"},
            "reference_archives": server_archives,
        },
    )
    visual = load_visual_manifest(REPOSITORY / "paper/visual_manifest.yaml")
    with (tmp_path / "paper_artifact_provenance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "artifact_type",
            "artifact_number",
            "reference_runs",
            "declared_input",
            "generator_target",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for asset in visual.assets.values():
            writer.writerow(
                {
                    "artifact_type": asset.kind,
                    "artifact_number": asset.number,
                    "reference_runs": "+".join(asset.run_ids) or "none",
                    "declared_input": (
                        asset.raw_inputs[0].member if asset.raw_inputs else "paper-only source"
                    ),
                    "generator_target": asset.tex_target,
                }
            )

    assert _validate_provenance_manifests(tmp_path, REPOSITORY) == {
        "status": "passed",
        "reference_run_count": 9,
        "paper_asset_count": 28,
        "paper_provenance_row_count": 28,
    }

    inventory = yaml.safe_load((tmp_path / "reference_runs.yaml").read_text(encoding="utf-8"))
    inventory["runs"][0]["archive_sha256"] = "0" * 64
    (tmp_path / "reference_runs.yaml").write_text(
        yaml.safe_dump(inventory),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tracked manifest"):
        _validate_provenance_manifests(tmp_path, REPOSITORY)


def test_package_manifest_rejects_payload_tamper(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "repository").mkdir(parents=True)
    payload = package / "repository" / "README.md"
    payload.write_text("release\n", encoding="utf-8")

    receipt = _write_package_manifest(package)

    assert receipt["file_count"] == 1
    assert _validate_package_manifest(package)["status"] == "passed"
    payload.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bytes differ"):
        _validate_package_manifest(package)
