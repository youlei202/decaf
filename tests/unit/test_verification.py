from __future__ import annotations

import csv
import json
from pathlib import Path

from decaf.verification import _write_artifact_diff


def test_artifact_diff_requires_all_28_outputs(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    generated: list[Path] = []
    for kind, count in (("figures", 12), ("tables", 16)):
        singular = kind[:-1]
        for number in range(1, count + 1):
            path = tmp_path / kind / f"{singular}_{number:02d}.tex"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("generated\n", encoding="utf-8")
            generated.append(path)
    summary = _write_artifact_diff(repo, generated, tmp_path / "verification")
    assert summary["figures_regenerated"] == 12
    assert summary["tables_regenerated"] == 16
    rows = list(
        csv.DictReader(
            (tmp_path / "verification" / "paper_artifact_diff.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert len(rows) == 28
    assert all(row["sha256"] for row in rows)


def test_repository_audit_report_is_json_safe() -> None:
    from decaf.audit import audit_repository

    repo = Path(__file__).resolve().parents[2]
    json.dumps(audit_repository(repo))
