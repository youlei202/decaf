from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from decaf.paper.manifest import load_visual_manifest
from decaf.paper.semantic import (
    CANONICAL_SCHEMA_SHA256,
    canonical_cardinality_sha256,
    canonical_cardinality_text,
    semantic_contract,
    semantic_contract_sha256,
)
from decaf.verification import (
    VerificationFailure,
    _assert_portable_evidence,
    _inventory_row,
    _run_command,
    _summarize_plan_report,
    _write_artifact_diff,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def _semantic_fixture(tmp_path: Path) -> tuple[list[Path], dict[str, object]]:
    manifest = load_visual_manifest(REPOSITORY / "paper" / "visual_manifest.yaml")
    generated: list[Path] = []
    canonical: list[dict[str, object]] = []
    for asset in manifest.assets.values():
        subdirectory = "figures" if asset.kind == "figure" else "tables"
        path = tmp_path / subdirectory / Path(asset.tex_target).name
        path.parent.mkdir(parents=True, exist_ok=True)
        if asset.status == "source_missing":
            content = "% DECAF_SOURCE_MISSING asset=figure_01\n" + r"status=source\_missing"
        else:
            contract = semantic_contract(asset)
            if asset.kind == "figure":
                cardinality = {str(name): int(value) for name, value in contract["panels"].items()}
                kind = "GEOMETRY"
                label = "points"
                body = r"\begin{picture}(10,10)\put(1,1){\circle*{1}}\end{picture}"
            else:
                rows = int(contract.get("exact_rows", contract.get("minimum_rows", 1)))
                cardinality = {"table_body": rows}
                kind = "TABLE"
                label = "rows"
                body = r"\begin{tabular}{r}1\\\end{tabular}"
            count = sum(cardinality.values())
            content = (
                f"% DECAF_SEMANTIC_{kind} asset={asset.asset_id} "
                f"contract_sha256={semantic_contract_sha256(asset)} "
                f"schema_sha256={CANONICAL_SCHEMA_SHA256} "
                f"canonical_sha256={'a' * 64} panels={len(cardinality)} "
                f"data_{label}={count} "
                f"cardinality_sha256={canonical_cardinality_sha256(cardinality)} "
                f"cardinality={canonical_cardinality_text(cardinality)}\n{body}"
            )
            canonical.append(
                {
                    "asset_id": asset.asset_id,
                    "kind": asset.kind,
                    "path": f"paper_data/canonical/{subdirectory}/{asset.asset_id}.csv",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                    "semantic_contract_sha256": semantic_contract_sha256(asset),
                    "schema_sha256": CANONICAL_SCHEMA_SHA256,
                    "row_count": count,
                    "panel_count": len(cardinality),
                    "panel_cardinality": cardinality,
                }
            )
        path.write_text(content, encoding="utf-8")
        generated.append(path)
    return generated, {
        "schema_version": 1,
        "status": "completed",
        "artifact_count": 27,
        "artifacts": canonical,
    }


def test_artifact_diff_requires_exact_semantic_28_outputs(tmp_path: Path) -> None:
    generated, canonical = _semantic_fixture(tmp_path)

    summary = _write_artifact_diff(
        REPOSITORY,
        generated,
        tmp_path / "verification",
        canonical,
        tmp_path,
    )

    assert summary == {
        "paper_assets_mapped": 28,
        "figure_assets_emitted": 12,
        "figures_regenerated": 11,
        "figures_source_missing_recorded": 1,
        "tables_regenerated": 16,
        "source_missing_recorded": ["figure_01"],
    }
    rows = list(
        csv.DictReader(
            (tmp_path / "verification" / "paper_artifact_diff.csv").open(encoding="utf-8")
        )
    )
    assert len(rows) == 28
    assert all(row["generated_sha256"] for row in rows)
    assert all(row["generated_path"].startswith("paper_outputs/generated/") for row in rows)
    assert {row["comparison_status"] for row in rows} == {
        "regenerated_semantic_geometry",
        "regenerated_semantic_table",
        "source_missing_recorded",
    }


def test_artifact_diff_rejects_generic_provenance_box(tmp_path: Path) -> None:
    generated, canonical = _semantic_fixture(tmp_path)
    target = next(path for path in generated if path.name == "figure_02.tex")
    target.write_text(r"\begin{figure}\fbox{summary}\end{figure}", encoding="utf-8")

    with pytest.raises(VerificationFailure, match="not data-rendered"):
        _write_artifact_diff(
            REPOSITORY,
            generated,
            tmp_path / "verification",
            canonical,
            tmp_path,
        )


def test_inventory_paths_are_portable_and_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "verification"
    artifact = root / "paper_outputs" / "generated" / "figure_01.tex"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("evidence", encoding="utf-8")

    row = _inventory_row(artifact, root, "verification_root", "generated_tex")

    assert row["portable_path"] == ("verification_root/paper_outputs/generated/figure_01.tex")
    assert not Path(str(row["relative_path"])).is_absolute()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(VerificationFailure, match="escapes verification_root"):
        _inventory_row(outside, root, "verification_root", "bad")


@pytest.mark.parametrize("content", ["/home/private/result.csv", "private note: 机密"])
def test_portable_evidence_scan_rejects_private_text(tmp_path: Path, content: str) -> None:
    artifact = tmp_path / "evidence.txt"
    artifact.write_text(content, encoding="utf-8")

    with pytest.raises(VerificationFailure, match="private|CJK"):
        _assert_portable_evidence([artifact])


def test_command_receipt_normalizes_python_interpreter_path(tmp_path: Path) -> None:
    report = _run_command(
        [sys.executable, "-c", "print('portable')"], cwd=tmp_path, echo_output=False
    )

    assert report["command"] == ["python", "-c", "print('portable')"]
    assert str(Path(sys.executable).parent) not in json.dumps(report["command"])


def test_repository_audit_report_is_json_safe() -> None:
    from decaf.audit import audit_repository

    json.dumps(audit_repository(REPOSITORY))


def test_plan_summary_keeps_counts_and_assertions_without_large_output() -> None:
    plan = {
        "assertions": {
            "coverage": True,
            "count": {"actual": 3, "expected": 3, "passed": True},
        },
        "counts": {"jobs": 3},
    }
    report = {
        "command": ["python", "-m", "planner"],
        "elapsed_seconds": 0.5,
        "exit_code": 0,
        "status": "passed",
        "output": json.dumps(plan),
    }

    summary = _summarize_plan_report("example", report)

    assert summary["counts"] == {"jobs": 3}
    assert summary["assertion_count"] == 2
    assert summary["assertions"] == {"coverage": True, "count": True}
    assert summary["output_bytes"] > 0
    assert len(summary["output_sha256"]) == 64
    assert "output" not in summary


def test_plan_summary_fails_closed_on_false_audit() -> None:
    report = {
        "command": ["python", "-m", "planner"],
        "elapsed_seconds": 0.5,
        "exit_code": 0,
        "status": "passed",
        "output": json.dumps({"audit": {"passed": False, "errors": ["bad"]}}),
    }

    with pytest.raises(VerificationFailure, match="assertions failed"):
        _summarize_plan_report("example", report)
