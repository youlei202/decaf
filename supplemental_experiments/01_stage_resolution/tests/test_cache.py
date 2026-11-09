"""Cache integrity/coverage tests use temporary analytic fixtures only."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("stage_resolution_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture_plan():
    pairs = [
        {
            "pair_id": "real-schema-fixture",
            "source_pair_id": "fixture-source",
            "pair_type": "same_rand",
            "class_id": 0,
        }
    ]
    task = {
        "task_id": "model__blend__000000_000001",
        "model_id": "model",
        "path": "blend",
        "start": 0,
        "stop": 1,
    }
    plan = {
        "plan_sha256": "test-only-contract",
        "models": [{"model_id": "model"}],
        "pairs": pairs,
        "tasks": [task],
        "config": {"stage_counts": [3, 5, 9], "epsilon": 0.02},
    }
    return plan, task


def complete_cache(output, plan, task):
    path = output / "cache" / f"{task['task_id']}.npz"
    grid = np.linspace(0, 1, 9)
    runner.atomic_npz(
        path,
        grid=grid,
        q_plus=(0.5 + 0.4 * grid)[None],
        q_minus=np.full((1, 9), 0.5),
        pair_ids=np.array(["real-schema-fixture"]),
    )
    runner.atomic_json(
        path.with_suffix(".json"),
        {
            "task": task,
            "plan_sha256": plan["plan_sha256"],
            "cache_sha256": runner.file_hash(path),
            "model_input_evaluations": 18,
        },
    )
    return path


def test_resume_rejects_corruption_and_wrong_contract(tmp_path):
    plan, task = fixture_plan()
    assert runner.verified_cache(tmp_path, plan, task) is None
    path = complete_cache(tmp_path, plan, task)
    assert runner.verified_cache(tmp_path, plan, task) == path
    with pytest.raises(ValueError, match="contract mismatch"):
        runner.verified_cache(tmp_path, {**plan, "plan_sha256": "different"}, task)
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        runner.verified_cache(tmp_path, plan, task)


def test_analysis_requires_complete_support_then_replays_cache(tmp_path):
    plan, task = fixture_plan()
    runner.atomic_json(tmp_path / "manifest.json", plan)
    with pytest.raises(RuntimeError, match="cache shards missing"):
        runner.analyze(tmp_path)
    assert not json.loads((tmp_path / "analysis_status.json").read_text())["complete"]
    assert not (tmp_path / "analysis" / "model_metrics.csv").exists()
    complete_cache(tmp_path, plan, task)
    runner.analyze(tmp_path)
    status = json.loads((tmp_path / "analysis_status.json").read_text())
    assert status["complete"] and status["model_input_evaluations"] == 18
    metrics = pd.read_csv(tmp_path / "analysis" / "model_metrics.csv")
    np.testing.assert_allclose(metrics.loc[metrics.component == "E", "mean"], 0.2)
    ranks = pd.read_csv(tmp_path / "analysis" / "model_rank_stability.csv")
    assert ranks.model_rank_spearman.isna().all()


@pytest.mark.parametrize("wide", [False, True])
def test_split_filter_respects_wide_source_and_typed_pair_keys(tmp_path, wide):
    rows = []
    for source, split in [("foreground_a", "deep"), ("foreground_b", "validation")]:
        if wide:
            rows.append(
                {
                    "pair_id": source,
                    "true_in9_class": 0,
                    "split": split,
                    "mixed_same_path": source + "/same.jpg",
                    "mixed_rand_path": source + "/rand.jpg",
                    "mixed_next_path": source + "/next.jpg",
                }
            )
        else:
            for pair_type in ("same_rand", "same_next"):
                rows.append(
                    {
                        "pair_id": source + "__" + pair_type,
                        "source_pair_id": source,
                        "pair_type": pair_type,
                        "class_id": 0,
                        "split": split,
                        "original_path": source + "/same.jpg",
                        "counterfactual_path": source + "/" + pair_type + ".jpg",
                    }
                )
    path = tmp_path / "pairs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    selected = runner.selected_pairs(
        {
            "pair_manifest": str(path),
            "dataset_root": str(tmp_path),
            "split": "deep",
            "pair_types": ["same_rand", "same_next"],
            "max_source_pairs": None,
        }
    )
    assert len(selected) == 2
    assert set(selected.source_pair_id) == {"foreground_a"}
    assert set(selected.pair_type) == {"same_rand", "same_next"}


def test_legacy_densenet_keys_are_normalized_without_silent_collisions():
    state = {"features.denseblock1.denselayer1.norm.1.weight": "fixture"}
    expected = {"features.denseblock1.denselayer1.norm1.weight": "fixture"}
    assert runner.canonical_state_dict(state, "densenet121") == expected
    assert runner.canonical_state_dict(state, "resnet18") == state
    with pytest.raises(ValueError, match="duplicate checkpoint key"):
        runner.canonical_state_dict({**state, **expected}, "densenet121")
