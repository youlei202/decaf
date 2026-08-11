import subprocess
from pathlib import Path

from decaf.audit import audit_repository


def test_clean_repository_passes(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("English text.\n", encoding="utf-8")

    report = audit_repository(tmp_path)

    assert report["passed"] is True
    assert report["finding_count"] == 0


def test_audit_detects_cjk_private_path_pdf_and_prompt(tmp_path: Path) -> None:
    private_root = "/" + "work" + "/" + "Users" + "/" + "leiyo"
    (tmp_path / ("RUN" + "_notes.md")).write_text(
        "\u5f00\u59cb " + private_root + "\n",
        encoding="utf-8",
    )
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")

    report = audit_repository(tmp_path)

    rules = {finding["rule"] for finding in report["findings"]}
    assert {"development_prompt", "english_only", "pdf_artifact", "private_path"} <= rules


def test_audit_detects_equation_definition_outside_core(tmp_path: Path) -> None:
    module = tmp_path / "src" / "decaf" / "experiments" / "demo.py"
    module.parent.mkdir(parents=True)
    module.write_text("def decompose():\n    return None\n", encoding="utf-8")

    report = audit_repository(tmp_path)

    assert any(item["rule"] == "duplicate_equations" for item in report["findings"])


def test_audit_scans_tracked_files_inside_generated_directories(
    tmp_path: Path,
) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    generated = tmp_path / "verification" / ("RUN" + "_private.md")
    generated.parent.mkdir()
    private_root = "/" + "work" + "/" + "Users" + "/" + "leiyo"
    generated.write_text(private_root + "\n", encoding="utf-8")
    subprocess.run(
        ("git", "-C", str(tmp_path), "add", "-f", "verification/RUN_private.md"),
        check=True,
    )

    report = audit_repository(tmp_path)

    findings = {(finding["rule"], finding["path"]) for finding in report["findings"]}
    assert ("development_prompt", "verification/RUN_private.md") in findings
    assert ("private_path", "verification/RUN_private.md") in findings
