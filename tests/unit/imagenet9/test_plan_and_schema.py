"""Static plan, registry, and paired-data contract tests for ImageNet-9."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from decaf.experiments.common import load_profile
from decaf.experiments.imagenet9.data import resolve_dataset_root
from decaf.experiments.imagenet9.evaluate import build_formal_plan
from decaf.experiments.imagenet9.models import deep_model_registry, model_registry
from decaf.experiments.imagenet9.pairs import (
    normalize_wide_manifest,
    shard_assignments,
    validate_pair_manifest,
)


def test_paper_plan_has_sealed_cardinalities_and_unique_artifacts() -> None:
    config = load_profile("imagenet9", "paper")
    plan = build_formal_plan(config)

    assert plan["counts"] == {
        "models": 72,
        "off_the_shelf_models": 24,
        "fine_tuned_models": 48,
        "deep_benchmark_models": 32,
        "deep_pairs": 768,
        "score_pairs": 1644,
        "expanded_deep_pairs": 1536,
        "expanded_score_pairs": 3288,
        "seeds": 2,
        "score_shards": 4,
        "deep_shards": 2,
        "methods": 6,
        "training_jobs": 48,
        "scan_jobs": 864,
        "baseline_jobs": 384,
        "jobs": 1296,
    }
    assert all(plan["assertions"].values())
    outputs = [job["output"] for job in plan["jobs"]]
    receipts = [job["receipt"] for job in plan["jobs"]]
    assert len(outputs) == len(set(outputs))
    assert len(receipts) == len(set(receipts))
    assert not plan["gpu_execution_verified"]
    training_ids = {job["job_id"] for job in plan["jobs"] if job["kind"] == "finetune"}
    assert len(training_ids) == 48
    for job in plan["jobs"]:
        if job["kind"] != "finetune" and job["model_id"].startswith("ft_"):
            assert job["depends_on"] == [f"train__{job['model_id']}"]


def test_model_subsets_are_deterministic() -> None:
    config = load_profile("imagenet9", "paper")
    records = model_registry(config)
    deep = deep_model_registry(config, records)

    assert len(records) == 72
    assert len(deep) == 32
    assert len({record.model_id for record in records}) == 72
    assert len({record.checkpoint_key for record in records}) == 72
    fine_tuned = [record for record in records if record.source == "experiment"]
    assert all(record.model_id.startswith("ft_") for record in fine_tuned)
    assert all(record.checkpoint_key == f"{record.model_id}/best.pt" for record in fine_tuned)


def test_pair_manifest_rejects_absolute_paths_and_duplicate_ids() -> None:
    valid = pd.DataFrame(
        [
            {
                "pair_id": "p0",
                "pair_type": "same_rand",
                "original_path": "original/a.png",
                "counterfactual_path": "mixed_rand/a.png",
                "class_id": 0,
            }
        ]
    )
    assert len(validate_pair_manifest(valid, expected_rows=1)) == 1

    absolute = valid.copy()
    absolute.loc[0, "original_path"] = str(Path("/restricted/a.png"))
    with pytest.raises(ValueError, match="relative"):
        validate_pair_manifest(absolute)

    duplicated = pd.concat([valid, valid], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_pair_manifest(duplicated)


def test_sharding_is_stable_and_validated() -> None:
    assert shard_assignments(["b", "a", "c"], 2) == {"a": 0, "b": 0, "c": 1}
    with pytest.raises(ValueError, match="positive"):
        shard_assignments(["a"], 0)
    with pytest.raises(ValueError, match="unique"):
        shard_assignments(["a", "a"], 1)


def test_wide_manifest_expands_source_rows_without_changing_shard_unit(tmp_path: Path) -> None:
    root = tmp_path / "imagenet9"
    frame = pd.DataFrame(
        {
            "pair_id": ["base-0"],
            "true_in9_class": [3],
            "mixed_same_path": [root / "official" / "mixed_same" / "x.png"],
            "mixed_rand_path": [root / "official" / "mixed_rand" / "x.png"],
            "mixed_next_path": [root / "official" / "mixed_next" / "x.png"],
        }
    )
    expanded = normalize_wide_manifest(frame, dataset_root=root, expected_rows=1)

    assert len(expanded) == 2
    assert set(expanded["pair_type"]) == {"same_rand", "same_next"}
    assert set(expanded["source_pair_id"]) == {"base-0"}
    assert all(not Path(value).is_absolute() for value in expanded["original_path"])


def test_dataset_root_accepts_direct_or_parent_layout(tmp_path: Path) -> None:
    config = load_profile("imagenet9", "smoke")
    direct = tmp_path / "direct"
    (direct / "manifests").mkdir(parents=True)
    nested = tmp_path / "parent" / "imagenet9"
    (nested / "manifests").mkdir(parents=True)

    assert resolve_dataset_root(config, environment={"DECAF_DATA_ROOT": str(direct)}) == direct
    assert (
        resolve_dataset_root(config, environment={"DECAF_DATA_ROOT": str(tmp_path / "parent")})
        == nested
    )
