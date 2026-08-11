from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import decaf.paper.semantic as semantic
from decaf.paper.manifest import load_visual_manifest

REPOSITORY = Path(__file__).resolve().parents[2]


def test_every_nonmissing_asset_has_an_explicit_semantic_contract() -> None:
    manifest = load_visual_manifest(REPOSITORY / "paper" / "visual_manifest.yaml")
    contracts = {
        asset.asset_id: semantic.semantic_contract(asset)
        for asset in manifest.assets.values()
        if asset.status != "source_missing"
    }

    assert len(contracts) == 27
    assert set(contracts) == {
        *(f"figure_{number:02d}" for number in range(2, 13)),
        *(f"table_{number:02d}" for number in range(1, 17)),
    }
    assert all(contract["raw_inputs"] for contract in contracts.values())
    assert all("operation" in contract for contract in contracts.values())


def test_source_lineage_rejects_unknown_well_formed_hash(tmp_path: Path) -> None:
    raw_hash = "a" * 64
    family_hash = "b" * 64
    raw = semantic.RawInputs(
        tmp_path,
        {("run", "member.csv"): {"sha256": raw_hash, "size_bytes": 1}},
    )
    families = {
        "controlled": semantic.FamilyOutputs(
            tmp_path,
            {"derived.csv": {"sha256": family_hash, "size_bytes": 1}},
        )
    }
    closure = semantic._source_lineage_closure(raw, families)
    combined = semantic._combined_hash((raw_hash, family_hash))
    valid = pd.DataFrame({"source_sha256": [raw_hash, family_hash, combined]})

    assert semantic._resolve_source_lineage("asset", valid, closure)[combined] == [
        raw_hash,
        family_hash,
    ]
    invalid = pd.DataFrame({"source_sha256": ["c" * 64]})
    with pytest.raises(semantic.SemanticDataError, match="outside sealed replay closure"):
        semantic._resolve_source_lineage("asset", invalid, closure)


def test_canonical_record_values_normalize_paths_recursively() -> None:
    value = {"source": "/home/private/run/result.csv", "nested": ["safe", 3]}

    assert semantic._safe(value) == {
        "source": "result.csv",
        "nested": ["safe", 3],
    }
    with pytest.raises(semantic.SemanticDataError, match="embedded private path"):
        semantic._safe("loaded from /tmp/private/result.csv")
