"""CPU regressions for the verification-only ImageNet-9 exact-unit bridge."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from decaf.experiments.imagenet9.evaluate import evaluate_response_frame
from tools.crossgen import imagenet9_e2e as e2e
from tools.crossgen import legacy_imagenet9_export as legacy
from tools.crossgen.schema import read_trajectory_record


def test_historical_package_provenance_validates_manifest_and_source_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = legacy.HISTORICAL_PACKAGE_PREFIX
    source_names = ("__init__", "data", "decaf", "models", "reveal", "run")
    payloads = {
        f"code/cmr/decaf_imagenet9_v1/{name}.py": f"# {name}\n".encode()
        for name in source_names
    }
    members = [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in sorted(payloads.items())
    ]
    manifest = {
        "schema_version": 1,
        "namespace": prefix,
        "lightweight": True,
        "source_layout": "code/cmr",
        "recorded_member_count": len(members),
        "manifest_self_entry": "excluded_by_design_to_avoid_self_hash_recursion",
        "members": members,
    }
    archive = tmp_path / "historical.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            legacy.HISTORICAL_PACKAGE_MANIFEST_MEMBER,
            json.dumps(manifest, sort_keys=True),
        )
        for path, payload in payloads.items():
            stream.writestr(f"{prefix}/{path}", payload)

    monkeypatch.setattr(legacy, "HISTORICAL_PACKAGE", archive)
    monkeypatch.setattr(
        legacy,
        "HISTORICAL_PACKAGE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    legacy._historical_package_provenance.cache_clear()
    try:
        provenance = legacy._historical_package_provenance()
    finally:
        legacy._historical_package_provenance.cache_clear()

    assert provenance["path"] == str(archive.resolve())
    assert provenance["recorded_member_count"] == len(members)
    assert set(provenance["source_members"]) == set(payloads)
    assert provenance["source_authority"].startswith("sha256-verified")

    monkeypatch.setattr(legacy, "HISTORICAL_PACKAGE_SHA256", "0" * 64)
    legacy._historical_package_provenance.cache_clear()
    try:
        with pytest.raises(ValueError, match="package SHA-256 changed"):
            legacy._historical_package_provenance()
    finally:
        legacy._historical_package_provenance.cache_clear()


def test_typed_selection_expands_both_registered_pair_types() -> None:
    wide = pd.DataFrame(
        [
            {
                "pair_id": f"0{index}/sample_{index}",
                "true_in9_class": index % 9,
                "mixed_same_path": f"same/{index}.jpg",
                "mixed_rand_path": f"rand/{index}.jpg",
                "mixed_next_path": f"next/{index}.jpg",
            }
            for index in range(8)
        ]
    )
    typed = legacy.typed_selection(wide)
    assert len(typed) == 16
    assert typed["source_pair_id"].nunique() == 8
    assert typed["pair_type"].value_counts().to_dict() == {
        "same_next": 8,
        "same_rand": 8,
    }


def _synthetic_bridge(
    binding: legacy.ModelBinding,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    legacy_rows: list[dict[str, object]] = []
    current_rows: list[dict[str, object]] = []
    for source_index in range(8):
        source_pair_id = f"0{source_index}/sample_{source_index}"
        for pair_type in legacy.PAIR_TYPES:
            pair_id = f"{source_pair_id}__{pair_type}"
            for reveal_index, reveal_path in enumerate(legacy.REVEAL_PATHS):
                endpoint = 0.20 + source_index * 0.01 + reveal_index * 0.02
                for stage_index, alpha in enumerate(legacy.ALPHA):
                    target_response = endpoint * float(alpha)
                    positive = 0.5 + target_response / 2.0
                    negative = 0.5 - target_response / 2.0
                    response = positive - negative
                    legacy_rows.append(
                        {
                            "historical_model_id": binding.historical_model_id,
                            "model_id": binding.current_model_id,
                            "checkpoint_sha256": binding.checkpoint_sha256,
                            "source_pair_id": source_pair_id,
                            "pair_id": pair_id,
                            "pair_type": pair_type,
                            "class_id": source_index % 9,
                            "reveal_path": reveal_path,
                            "stage_index": stage_index,
                            "alpha": float(alpha),
                            "score_plus": positive,
                            "score_minus": negative,
                            "response": response,
                        }
                    )
                    current_rows.append(
                        {
                            "pair_id": pair_id,
                            "pair_type": pair_type,
                            "model_id": binding.current_model_id,
                            "reveal_path": reveal_path,
                            "stage_index": stage_index,
                            "alpha": float(alpha),
                            "response": response,
                            "checkpoint_sha256": binding.checkpoint_sha256,
                        }
                    )
    legacy_frame = pd.DataFrame(legacy_rows)
    current_frame = pd.DataFrame(current_rows)
    sealed = evaluate_response_frame(current_frame, epsilon=legacy.EPSILON).rename(
        columns={"reveal_path": "path"}
    )
    sealed["pair_id"] = sealed["pair_id"].str.rsplit("__", n=1).str[0]
    sealed["model_id"] = binding.historical_model_id
    sealed["epsilon"] = legacy.EPSILON
    return legacy_frame, current_frame, sealed


def test_neutral_and_e2e_comparisons_are_exact_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = legacy.ModelBinding(
        historical_model_id="historical_model",
        current_model_id="current_model",
        checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_sha256="a" * 64,
    )
    legacy_frame, current_frame, sealed = _synthetic_bridge(binding)
    monkeypatch.setattr(legacy, "MODEL_BINDINGS", (binding,))
    monkeypatch.setattr(e2e, "MODEL_BINDINGS", (binding,))
    monkeypatch.setattr(legacy, "sealed_summaries", lambda selected: sealed.copy())
    monkeypatch.setattr(e2e, "sealed_summaries", lambda selected: sealed.copy())

    neutral_path = legacy.build_neutral_record(legacy_frame, tmp_path)
    neutral = read_trajectory_record(neutral_path)
    assert len(neutral) == 432
    assert neutral["unit_id"].nunique() == 48

    legacy_path = tmp_path / "legacy.parquet"
    current_path = tmp_path / "current.parquet"
    legacy_frame.to_parquet(legacy_path, index=False)
    current_frame.to_parquet(current_path, index=False)
    result = e2e.compare_e2e(current_path, legacy_path, tmp_path)
    summary = result["summary"]
    assert summary["status"] == "PASS_CORE_AND_E2E"
    assert summary["unit_count"] == 48
    assert summary["tier_a_fraction"] == 1.0
    assert summary["hard_mismatch_fraction"] == 0.0
    assert summary["stage_response"]["maximum_absolute_error"] == 0.0

    core = e2e._compare_core_atomic(neutral_path, tmp_path)
    assert core["summary"]["tier_a_fraction"] == 1.0
    assert core["summary_path"].is_file()
    assert e2e._combine_core_and_e2e_status(core, result) == "PASS_CORE_AND_E2E"
    result["summary"]["acceptance"]["identity_exact"] = False
    assert e2e._combine_core_and_e2e_status(core, result) == "FAIL_NUMERICAL"
    assert not list(tmp_path.rglob("*.part.*"))
