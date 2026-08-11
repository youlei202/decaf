"""Static checks that keep the public repository release-safe."""

from __future__ import annotations

import ast
import os
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "runs",
    "verification",
}
MAX_PUBLIC_FILE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class AuditFinding:
    """One actionable repository-audit finding."""

    rule: str
    path: str
    detail: str


def _private_markers() -> tuple[str, ...]:
    unix_root = "/" + "work" + "/" + "Users" + "/" + "leiyo"
    legacy_root = "/" + "work" + "/" + "Lei"
    return unix_root, legacy_root


def _prompt_prefixes() -> tuple[str, ...]:
    separator = "_"
    return tuple(name + separator for name in ("FIX", "PREPARE", "RUN"))


def iter_public_files(root: Path) -> Iterable[Path]:
    """Yield tracked files plus non-generated, non-cache repository files."""

    resolved = root.resolve()
    tracked: set[Path] = set()
    try:
        top_level = subprocess.run(
            ("git", "-C", str(resolved), "rev-parse", "--show-toplevel"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top_level).resolve() == resolved:
            output = subprocess.run(
                ("git", "-C", str(resolved), "ls-files", "-z"),
                check=True,
                capture_output=True,
            ).stdout
            for value in output.split(b"\0"):
                if not value:
                    continue
                path = resolved / value.decode("utf-8", errors="surrogateescape")
                if path.is_file():
                    tracked.add(path)
    except (OSError, subprocess.CalledProcessError):
        pass

    files = set(tracked)
    for directory, directory_names, file_names in os.walk(resolved):
        directory_path = Path(directory)
        relative_parts = directory_path.relative_to(resolved).parts
        if any(part in SKIP_DIRECTORIES for part in relative_parts):
            directory_names.clear()
            continue
        directory_names[:] = [name for name in directory_names if name not in SKIP_DIRECTORIES]
        files.update(directory_path / name for name in file_names)

    yield from sorted(files, key=lambda path: path.relative_to(resolved).as_posix())


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _text_findings(path: Path, root: Path) -> list[AuditFinding]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    relative = _relative(path, root)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [AuditFinding("utf8", relative, "public text file is not valid UTF-8")]
    findings: list[AuditFinding] = []
    if any("\u3400" <= character <= "\u9fff" for character in text):
        findings.append(AuditFinding("english_only", relative, "contains a CJK character"))
    for marker in _private_markers():
        if marker in text:
            findings.append(
                AuditFinding("private_path", relative, "contains a private absolute path")
            )
    return findings


def _duplicate_equation_findings(path: Path, root: Path) -> list[AuditFinding]:
    if path.suffix != ".py":
        return []
    relative = _relative(path, root)
    allowed = {
        "src/decaf/core/decomposition.py",
        "src/decaf/core/quadrature.py",
        "src/decaf/core/trajectories.py",
    }
    if relative in allowed:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    equation_names = {"compute_components", "decompose", "integrate_components"}
    return [
        AuditFinding(
            "duplicate_equations",
            relative,
            f"defines reserved core function {node.name!r}",
        )
        for node in ast.walk(tree)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name in equation_names
    ]


def audit_repository(root: Path) -> dict[str, object]:
    """Return a deterministic audit report for a repository root."""

    resolved = root.resolve()
    findings: list[AuditFinding] = []
    files = list(iter_public_files(resolved))
    for path in files:
        relative = _relative(path, resolved)
        if path.suffix.lower() == ".pdf":
            findings.append(AuditFinding("pdf_artifact", relative, "PDF files are forbidden"))
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            findings.append(
                AuditFinding(
                    "large_file",
                    relative,
                    f"file exceeds {MAX_PUBLIC_FILE_BYTES} bytes",
                )
            )
        if path.name.startswith(_prompt_prefixes()):
            findings.append(
                AuditFinding("development_prompt", relative, "prompt-style file is forbidden")
            )
        findings.extend(_text_findings(path, resolved))
        findings.extend(_duplicate_equation_findings(path, resolved))
    payload = [asdict(item) for item in sorted(findings, key=lambda item: (item.rule, item.path))]
    return {
        "passed": not payload,
        "root": ".",
        "scanned_file_count": len(files),
        "finding_count": len(payload),
        "findings": payload,
    }


__all__ = ["AuditFinding", "audit_repository", "iter_public_files"]
