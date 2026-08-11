"""End-to-end CPU smoke run for the ImageNet-9 run-directory contract."""

from __future__ import annotations

import hashlib
import json

from decaf.experiments.imagenet9.cli import main


def test_imagenet9_smoke_all_and_resume(tmp_path: object) -> None:
    output = tmp_path / "imagenet9-smoke"  # type: ignore[operator]
    arguments = ["--profile", "smoke", "--stage", "all", "--output", str(output)]

    assert main(arguments) == 0
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["completed_stages"] == ["prepare", "compute", "analyze", "paper"]
    assert (output / "paper_data" / "manifest.json").is_file()
    assert (output / "metrics" / "decaf_scores.csv").is_file()
    plan = json.loads((output / "manifests" / "plan.json").read_text(encoding="utf-8"))
    planned_members = {job["job_id"] for job in plan["jobs"] if job["kind"] != "finetune"}
    receipts = {
        payload["job_id"]: payload
        for path in (output / "receipts" / "members").glob("*.json")
        if (payload := json.loads(path.read_text(encoding="utf-8")))
    }
    assert set(receipts) == planned_members
    jobs = {job["job_id"]: job for job in plan["jobs"]}
    for job_id, receipt in receipts.items():
        encoded = json.dumps(jobs[job_id], sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert receipt["job_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert len(list((output / "raw" / "scans").glob("*.parquet"))) == 2
    assert len(list((output / "raw" / "baselines").glob("*.parquet"))) == 1

    assert main([*arguments, "--resume"]) == 0
    resumed = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert resumed["status"] == "completed"
    assert resumed["completed_stages"] == ["prepare", "compute", "analyze", "paper"]
