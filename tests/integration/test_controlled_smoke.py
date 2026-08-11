from __future__ import annotations

import json
from pathlib import Path

from decaf.experiments.controlled.cli import main


def test_controlled_cpu_oracle_smoke_runs_all_stages(tmp_path: Path) -> None:
    output = tmp_path / "controlled-smoke"
    assert main(["--profile", "smoke", "--stage", "all", "--output", str(output)]) == 0
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert len(list((output / "receipts" / "members").glob("*.json"))) == 11
    assert (output / "metrics" / "controlled_smoke_metrics.csv").is_file()
    assert (output / "paper_data" / "controlled" / "controlled_smoke_panel.csv").is_file()
    receipt = json.loads(
        (output / "paper_data" / "controlled" / "controlled_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["scope"] == "cpu_score_oracle"
    assert receipt["gpu_real_shard_verification"] == "pending"
