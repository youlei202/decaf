from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from decaf.paper.manifest import (
    ManifestError,
    import_generator,
    load_representative_cases,
    load_visual_manifest,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def test_visual_manifest_covers_every_numbered_asset() -> None:
    manifest = load_visual_manifest(REPOSITORY / "paper" / "visual_manifest.yaml")

    assert len(manifest.assets) == 28
    assert {asset.asset_id for asset in manifest.assets.values() if asset.kind == "figure"} == {
        f"figure_{number:02d}" for number in range(1, 13)
    }
    assert {asset.asset_id for asset in manifest.assets.values() if asset.kind == "table"} == {
        f"table_{number:02d}" for number in range(1, 17)
    }
    assert all(callable(import_generator(asset.generator)) for asset in manifest.assets.values())
    assertions = [
        assertion for asset in manifest.assets.values() for assertion in asset.headline_assertions
    ]
    assert len(assertions) == 27
    assert all("expected" in assertion for assertion in assertions)
    assert all(asset.generation_contract["operation"] for asset in manifest.assets.values())


def test_missing_conceptual_source_is_explicit() -> None:
    manifest = load_visual_manifest(REPOSITORY / "paper" / "visual_manifest.yaml")
    figure = manifest.assets["figure_01"]

    assert figure.status == "source_missing"
    assert figure.raw_inputs == ()
    assert figure.source_note
    assert figure.generation_contract["status"] == "source_missing"
    assert figure.generation_contract["missing_item"]
    assert figure.generation_contract["why_it_matters"]
    assert figure.generation_contract["reproducible_scope"]
    assert figure.generation_contract["required_recovery_action"]
    assert figure.headline_assertions == ()
    assert figure.tex_target.endswith("figure_01.tex")


def test_prose_contract_cannot_masquerade_as_a_headline_assertion(
    tmp_path: Path,
) -> None:
    source = REPOSITORY / "paper" / "visual_manifest.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    assertion = payload["assets"]["figure_02"]["headline_assertions"][0]
    assertion.pop("expected")
    candidate = tmp_path / "visual_manifest.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ManifestError, match="numerical expectation"):
        load_visual_manifest(candidate)


def test_representative_case_manifest_freezes_figures_two_through_four() -> None:
    payload = load_representative_cases(REPOSITORY / "paper" / "representative_cases.yaml")
    resolved = payload["cases"]["figure_02"]["resolved"]

    assert resolved["rows_per_factor"] == 6
    assert resolved["intended_factor"]["mean_auc_abs"] == 0.05079873173186311
    assert resolved["endpoint_null_factor"]["mean_auc_abs"] == 0.05268779850012135
    assert len(resolved["intended_factor"]["source_rows"]) == 6
    assert len(resolved["endpoint_null_factor"]["source_rows"]) == 6
