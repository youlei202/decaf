from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from decaf.experiments.attribution.methods import BOUNDARY_METHODS
from decaf.experiments.attribution.plan import (
    PROFILE_MEMBER_COUNTS,
    TARGET_METHODS,
    build_plan,
)
from decaf.experiments.common import load_profile, repository_root


def _config(profile: str) -> dict[str, object]:
    explicit: Path | None = None
    if profile == "large-model":
        explicit = repository_root() / "configs/attribution/large_model.yaml"
    return load_profile("attribution", profile, explicit)


def test_all_attribution_profiles_have_exact_collision_free_plans() -> None:
    expected = {
        "smoke": 1,
        "main": 888,
        "paper": 3349,
        "large-model": 56,
        "boundary": 1088,
    }
    assert PROFILE_MEMBER_COUNTS == expected
    for profile, member_count in expected.items():
        plan = build_plan(_config(profile))
        assert plan["member_count"] == member_count
        assert plan["audit"]["passed"] is True
        assert plan["endpoint_m_stage"] == "analyze"
        assert not any(job["method_id"] == "endpoint_m" for job in plan["members"])
        assert len({job["member_id"] for job in plan["members"]}) == member_count
        assert len({job["output_path"] for job in plan["members"]}) == member_count
        assert len({job["receipt_path"] for job in plan["members"]}) == member_count


def test_boundary_contract_adds_endpoint_m_only_in_analysis() -> None:
    plan = build_plan(_config("boundary"))
    compute_methods = {
        job["method_id"] for job in plan["members"] if job["method_id"] not in TARGET_METHODS
    }
    assert len(BOUNDARY_METHODS) == 15
    assert len(compute_methods) == 14
    assert "endpoint_m" in BOUNDARY_METHODS
    assert "endpoint_m" not in compute_methods
    quality_jobs = [job for job in plan["members"] if job["scope"] == "partimagenet_boundary"]
    assert quality_jobs
    assert all(len(job["depends_on"]) == 3 for job in quality_jobs)


def test_paper_shards_cover_each_registered_model_support_exactly_once() -> None:
    plan = build_plan(_config("paper"))
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for job in plan["members"]:
        key = (job["scope"], job["model_id"], job["method_id"], job["repeat"])
        groups[key].append(job)
    for jobs in groups.values():
        ordered = sorted(jobs, key=lambda job: int(job["shard"]))
        assert ordered[0]["image_start"] == 0
        assert ordered[-1]["image_stop"] == ordered[0]["model_image_count"]
        assert sum(int(job["image_count"]) for job in ordered) == ordered[0]["model_image_count"]
        assert all(
            left["image_stop"] == right["image_start"]
            for left, right in zip(ordered, ordered[1:], strict=False)
        )

    expected_support = {
        "idsds_primary": {
            "resnet50": 7_663,
            "vgg16": 7_189,
            "vit_base_patch16_224": 8_285,
        },
        "idsds_full50k": {
            "resnet50": 38_460,
            "vgg16": 36_042,
            "vit_base_patch16_224": 41_374,
        },
    }
    for scope, counts in expected_support.items():
        observed = {
            str(job["model_id"]): int(job["model_image_count"])
            for job in plan["members"]
            if job["scope"] == scope
        }
        assert observed == counts
