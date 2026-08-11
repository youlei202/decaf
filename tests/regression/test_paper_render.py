from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from decaf.paper.manifest import load_visual_manifest
from decaf.paper.reference import sha256_file
from decaf.paper.render import (
    PaperRenderError,
    build_parser,
    render_data_asset,
    render_source_missing_asset,
    validate_rendered_asset,
)
from decaf.paper.semantic import (
    CANONICAL_COLUMNS,
    CANONICAL_SCHEMA_SHA256,
    canonical_asset_path,
    semantic_contract_sha256,
)

REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST = load_visual_manifest(REPOSITORY / "paper" / "visual_manifest.yaml")


def _canonical_context(
    tmp_path: Path,
    asset_id: str,
    cardinality: dict[str, int],
) -> tuple[object, dict[str, object], Path]:
    asset = MANIFEST.assets[asset_id]
    paper_data = tmp_path / "paper_data"
    rows: list[dict[str, object]] = []
    offset = 0
    for panel_id, count in cardinality.items():
        for index in range(count):
            if asset_id == "table_12":
                record = {
                    "module": f"T{index}",
                    "models": index + 1,
                    "design": "sealed design",
                    "strengths_or_regimes": "all",
                }
            else:
                record = {"panel": panel_id, "observation": index}
            rows.append(
                {
                    "artifact_id": asset_id,
                    "panel_id": panel_id,
                    "series": "series-a",
                    "x": index,
                    "y": offset + index / 10,
                    "estimate": offset + index / 10,
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n": 1,
                    "source_sha256": "a" * 64,
                    "record_json": json.dumps(record, sort_keys=True, separators=(",", ":")),
                }
            )
        offset += count
    frame = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    path = canonical_asset_path(paper_data, asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    item = {
        "asset_id": asset_id,
        "kind": asset.kind,
        "path": path.relative_to(tmp_path).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "semantic_contract_sha256": semantic_contract_sha256(asset),
        "schema_sha256": CANONICAL_SCHEMA_SHA256,
        "row_count": len(frame),
        "panel_count": len(cardinality),
        "panel_cardinality": cardinality,
        "source_sha256s": ["a" * 64],
        "resolved_source_sha256s": ["a" * 64],
        "source_lineage": {"a" * 64: ["a" * 64]},
        "representative_case_ids": [],
    }
    context: dict[str, object] = {
        "_paper_data_root": str(paper_data),
        "canonical": {
            "schema_version": 1,
            "status": "completed",
            "artifact_count": 1,
            "artifacts": [item],
        },
    }
    return asset, context, path


def test_public_paper_wrapper_argument_aliases() -> None:
    arguments = build_parser().parse_args(
        ["--reference-runs", "one:two", "--output", "paper/generated"]
    )

    assert arguments.reference_roots == ["one:two"]
    assert arguments.generated_root == "paper/generated"
    assert arguments.replay_root is None


def test_figure_renderer_uses_validated_asset_semantics(tmp_path: Path) -> None:
    cardinality = {
        "matched_abs": 12,
        "false_null_order": 2,
        "false_null_evidence": 1,
    }
    asset, context, _ = _canonical_context(tmp_path, "figure_02", cardinality)

    content = render_data_asset(asset, context, context)  # type: ignore[arg-type]

    assert validate_rendered_asset(asset, content) == "regenerated_semantic_geometry"  # type: ignore[arg-type]
    assert "panels=3 data_points=15" in content
    assert content.count(r"\begin{picture}") == 3
    assert r"\qbezier" in content
    assert "canonical_sha256=" in content


def test_table_renderer_emits_every_canonical_semantic_row(tmp_path: Path) -> None:
    asset, context, _ = _canonical_context(tmp_path, "table_12", {"table_body": 2})

    content = render_data_asset(asset, context, context)  # type: ignore[arg-type]

    assert validate_rendered_asset(asset, content) == "regenerated_semantic_table"  # type: ignore[arg-type]
    assert "panels=1 data_rows=2" in content
    assert r"\begin{tabular}" in content
    assert "sealed design" in content


def test_canonical_post_receipt_tamper_fails_closed(tmp_path: Path) -> None:
    asset, context, path = _canonical_context(tmp_path, "table_12", {"table_body": 2})
    path.write_text(path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")

    with pytest.raises(PaperRenderError, match="canonical bytes drifted"):
        render_data_asset(asset, context, context)  # type: ignore[arg-type]


def test_figure_one_is_the_only_explicit_source_gap() -> None:
    gaps = [asset for asset in MANIFEST.assets.values() if asset.status == "source_missing"]

    assert [asset.asset_id for asset in gaps] == ["figure_01"]
    content = render_source_missing_asset(gaps[0], {}, {})
    assert validate_rendered_asset(gaps[0], content) == "source_missing_recorded"
    assert "% DECAF_SOURCE_MISSING asset=figure_01" in content
    assert r"status=source\_missing" in content


def test_provenance_box_cannot_pass_as_regenerated_figure() -> None:
    asset = MANIFEST.assets["figure_02"]

    with pytest.raises(PaperRenderError, match="semantic render marker"):
        validate_rendered_asset(
            asset,
            r"\begin{figure}\fbox{sealed replay summary}\end{figure}",
        )


def test_rendering_context_without_canonical_data_fails_closed() -> None:
    asset = MANIFEST.assets["figure_02"]

    with pytest.raises(PaperRenderError, match="canonical replay data"):
        render_data_asset(asset, {"inputs": []}, {})
