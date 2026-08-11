from __future__ import annotations

import json
from pathlib import Path

from decaf.core.manifests import sha256_file
from decaf.experiments.attribution.cli import main


def test_attribution_cpu_oracle_smoke_and_member_resume(tmp_path: Path) -> None:
    output = tmp_path / "attribution-smoke"
    command = ["--profile", "smoke", "--stage", "all", "--output", str(output)]
    assert main(command) == 0
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    members = list((output / "receipts/members").rglob("*.json"))
    assert len(members) == 1
    member = json.loads(members[0].read_text(encoding="utf-8"))
    assert member["status"] == "completed"
    result_path = output / member["details"]["output_path"]
    original_hash = sha256_file(result_path)
    audit = json.loads(
        (output / "metrics/endpoint_m/source_audit.json").read_text(encoding="utf-8")
    )
    assert audit["generated_in_stage"] == "analyze"
    assert audit["inference_performed"] is False
    assert audit["passed"] is True
    tables = json.loads((output / "paper_data/attribution_tables.json").read_text(encoding="utf-8"))
    assert tables["registered_tables"] == [2, 3, 4, 6, 7, 8, 9, 10, 11]
    assert all(table["schema_only"] for table in tables["tables"])
    assert len(list((output / "paper_data").glob("table_*.csv"))) == 9
    assert len(list((output / "paper_data").glob("table_*.tex"))) == 9

    (output / "receipts/compute.json").unlink()
    resume = [
        "--profile",
        "smoke",
        "--stage",
        "compute",
        "--output",
        str(output),
        "--resume",
    ]
    assert main(resume) == 0
    compute = json.loads((output / "receipts/compute.json").read_text(encoding="utf-8"))
    assert compute["details"]["completed_members"] == 0
    assert compute["details"]["resumed_members"] == 1
    assert sha256_file(result_path) == original_hash
