from __future__ import annotations

from decaf.experiments.common import load_profile
from decaf.experiments.controlled.cli import build_plan
from decaf.experiments.controlled.train import (
    assert_c0_has_no_training_jobs,
    c1_factory_training_jobs,
    selected_c1_checkpoints,
)


def test_paper_plan_locks_every_registered_controlled_count() -> None:
    config = load_profile("controlled", "paper")
    plan = build_plan(config)
    assert plan["scientific_counts"] == {
        "base_models": 30,
        "base_model_factor_units": 180,
        "evidence_checkpoints": 52,
        "causal_checkpoints": 18,
        "fragility_checkpoints": 18,
        "endpoint_behavior_checkpoints": 88,
        "endpoint_behavior_training_jobs": 44,
        "endpoint_behavior_units_per_pass": 158,
        "endpoint_behavior_units_total": 316,
        "contradiction_models": 30,
        "contradiction_evaluation_units": 30,
        "scheduled_members": 600,
    }
    assert all(assertion["passed"] for assertion in plan["assertions"].values())
    assert plan["contracts"]["c0_no_retraining"] is True


def test_plan_paths_receipts_and_dependencies_are_unique() -> None:
    plan = build_plan(load_profile("controlled", "paper"))
    members = plan["members"]
    identifiers = [member["member_id"] for member in members]
    outputs = [member["output"] for member in members]
    receipts = [f"receipts/members/{identifier}.json" for identifier in identifiers]
    assert len(identifiers) == len(set(identifiers))
    assert len(outputs) == len(set(outputs))
    assert len(receipts) == len(set(receipts))
    known = set(identifiers)
    assert all(set(member["dependencies"]) <= known for member in members)
    assert not any(member["phase"].startswith("c0_train") for member in members)
    c1_training = {member["member_id"] for member in members if member["phase"] == "c1_train"}
    assert len(c1_training) == 44
    assert all(
        len(member["dependencies"]) == 1 and member["dependencies"][0] in c1_training
        for member in members
        if member["phase"] == "c1_measure"
    )


def test_c1_exact_checkpoint_selection_and_factory_jobs() -> None:
    section = load_profile("controlled", "paper")["endpoint_behavior"]
    checkpoints = selected_c1_checkpoints(section)
    counts = {
        module: sum(row["module"] == module for row in checkpoints) for module in ("E", "C", "F")
    }
    assert counts == {"E": 52, "C": 18, "F": 18}
    assert "e__p095__small_vit__seed_5102__epoch_019" in {row["model_id"] for row in checkpoints}
    jobs = c1_factory_training_jobs(section)
    assert len(jobs) == 44
    assert_c0_has_no_training_jobs(jobs)


def test_smoke_plan_is_small_but_uses_the_same_schema() -> None:
    plan = build_plan(load_profile("controlled", "smoke"))
    assert plan["scientific_counts"]["scheduled_members"] == 11
    assert plan["scientific_counts"]["endpoint_behavior_checkpoints"] == 3
    assert plan["scientific_counts"]["endpoint_behavior_training_jobs"] == 3
