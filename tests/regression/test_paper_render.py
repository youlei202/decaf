from __future__ import annotations

import json
from pathlib import Path

from decaf.paper.render import build_parser, render_all

REPOSITORY = Path(__file__).resolve().parents[2]


def test_public_paper_wrapper_argument_aliases() -> None:
    arguments = build_parser().parse_args(
        ["--reference-runs", "one:two", "--output", "paper/generated"]
    )

    assert arguments.reference_roots == ["one:two"]
    assert arguments.generated_root == "paper/generated"
    assert arguments.replay_root is None


def test_render_all_generates_twenty_eight_self_contained_tex_assets(tmp_path: Path) -> None:
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    (replay_root / "replay_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [],
                "inputs": [],
                "representative_cases": {},
                "headline_assertions": {},
            }
        ),
        encoding="utf-8",
    )
    generated = tmp_path / "generated"

    paths = render_all(replay_root, repo_root=REPOSITORY, generated_root=generated)

    assert len(paths) == 28
    assert len(list(generated.rglob("*.tex"))) == 28
    assert not list(generated.rglob("*.pdf"))
    assert all("external PDF" in path.read_text(encoding="utf-8") for path in paths)
    assert r"source\_missing" in (generated / "figures" / "figure_01.tex").read_text(
        encoding="utf-8"
    )
    artifact_diff = replay_root / "verification" / "paper_artifact_diff.csv"
    assert artifact_diff.is_file()
    assert len(artifact_diff.read_text(encoding="utf-8").splitlines()) == 29
