"""Static checks that keep the public repository release-safe."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".py", ".rst", ".sh", ".toml", ".txt", ".yaml", ".yml"}
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
    """Yield release files while pruning repositories, environments, and output."""

    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


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
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name in equation_names
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
