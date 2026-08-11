from __future__ import annotations

from pathlib import Path

from decaf.paper.manifest import import_generator, load_visual_manifest
from decaf.paper.reference import load_reference_runs

REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTED_ARCHIVE_HASHES = {
    "C0": "2126b7fcf720e367ca6dd6ed7c467c45ef20199364685d10372a96efa7ebf559",
    "C1": "387c5a572249110a31698d384d942a8e1adf542c7c0b1a9f3a0c5d453102a8a7",
    "C2": "26bc5bc4a9efd5e23d6b34372f5ebce9c3563bb4d5659bc847c4344c67fe7ede",
    "I9": "3bae5ac670f6731d8a7832c3f9d7051e308a3f322c6192068bc11868be3821cc",
    "A0": "7ecad798213d41662749625692618c615135776009ece187fd6a72adf067d420",
    "A1": "0522de042bc71e679d2d724c1fe9242c812bb31b3dd9f562f18d70e7b8b4819b",
    "A2": "f68ed1fec48b39403fb677492283066f853722f466ce703edd5b468d59cc93a4",
    "A3": "b3b6c7abd41be23fdd47e36cbbc38bc0cb18f106cb6180340d76054e363a9ed5",
    "T0": "e9acaf30491dcdf654fdfb691df915e19d75e9c19d0ffe2546312d0d34f87927",
}


def test_sealed_archive_identities_are_frozen() -> None:
    runs = load_reference_runs(REPOSITORY / "manifests" / "reference_runs")

    assert {run_id: run.archive_sha256 for run_id, run in runs.items()} == (EXPECTED_ARCHIVE_HASHES)
    assert all(run.archive_size_bytes > 0 for run in runs.values())
    assert all(run.archive_member_count > 0 for run in runs.values())
    assert all(run.analysis_inputs for run in runs.values())


def test_visual_assets_only_use_declared_reference_runs_and_importable_generators() -> None:
    runs = load_reference_runs(REPOSITORY / "manifests" / "reference_runs")
    manifest = load_visual_manifest(REPOSITORY / "paper" / "visual_manifest.yaml")

    for asset in manifest.assets.values():
        assert set(asset.run_ids) <= set(runs)
        assert callable(import_generator(asset.generator))
        assert asset.generation_contract["operation"]
        assert all("expected" in item for item in asset.headline_assertions)
        assert not asset.tex_target.endswith(".pdf")


def test_public_paper_manifests_do_not_embed_private_locations() -> None:
    paths = [
        REPOSITORY / "paper" / "visual_manifest.yaml",
        REPOSITORY / "paper" / "representative_cases.yaml",
        *(REPOSITORY / "manifests" / "reference_runs").glob("*.yaml"),
    ]
    private_prefix = "/" + "work" + "/" + "Users" + "/"

    assert all(private_prefix not in path.read_text(encoding="utf-8") for path in paths)
