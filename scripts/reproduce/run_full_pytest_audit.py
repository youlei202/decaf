#!/usr/bin/env python3
"""Run the complete pytest suite and atomically bind its log to Git identity."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_COMMAND = ("python", "-m", "pytest")
UTC_TIMEZONE = getattr(__import__("datetime"), "UTC", timezone.utc)  # noqa: UP017


def _utc_now() -> str:
    return datetime.now(UTC_TIMEZONE).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def passed_test_count(output: str) -> int:
    """Extract the terminal pytest passed count, rejecting ambiguous logs."""

    values = [int(value) for value in re.findall(r"(?m)(\d+) passed(?:,|\s+in\b)", output)]
    if not values:
        raise ValueError("pytest output has no terminal passed-test summary")
    return values[-1]


def build_receipt(
    *,
    commit: str,
    tree: str,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    exit_code: int,
    passed_tests: int,
    log_path: Path,
) -> dict[str, Any]:
    """Build the portable receipt schema consumed by finalization."""

    return {
        "schema_version": 1,
        "status": "passed" if exit_code == 0 else "failed",
        "command": list(REQUIRED_COMMAND),
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "passed_tests": passed_tests,
        "repository_commit": commit,
        "repository_tree": tree,
        "tracked_worktree_clean": True,
        "output_log": {
            "path": "verification/final_audit/full_pytest.log",
            "streams": "stdout+stderr",
            "size_bytes": log_path.stat().st_size,
            "sha256": _sha256(log_path),
        },
    }


def run(repository: Path, verification_root: Path) -> dict[str, Any]:
    """Execute the full suite from a clean worktree and write its receipt."""

    repository = repository.resolve()
    verification_root = verification_root.resolve()
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("full pytest audit requires a clean worktree")
    commit = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    output_directory = verification_root / "verification" / "final_audit"
    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = output_directory / "full_pytest.log"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".full_pytest.log.", suffix=".part", dir=output_directory
    )
    started_at = _utc_now()
    started_clock = time.monotonic()
    try:
        with os.fdopen(descriptor, "wb") as log:
            process = subprocess.Popen(
                (sys.executable, "-m", "pytest"),
                cwd=repository,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if process.stdout is None:  # pragma: no cover - Popen contract
                raise AssertionError("pytest stdout pipe is unavailable")
            for chunk in iter(lambda: process.stdout.read(64 * 1024), b""):
                log.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            exit_code = process.wait()
            log.flush()
            os.fsync(log.fileno())
        os.replace(temporary_name, log_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    finished_at = _utc_now()
    elapsed_seconds = round(time.monotonic() - started_clock, 6)
    output = log_path.read_text(encoding="utf-8")
    passed_tests = passed_test_count(output)
    if (
        _git(repository, "rev-parse", "HEAD") != commit
        or _git(repository, "rev-parse", "HEAD^{tree}") != tree
    ):
        raise RuntimeError("repository identity changed during full pytest")
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("full pytest changed the repository worktree")
    receipt = build_receipt(
        commit=commit,
        tree=tree,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
        exit_code=exit_code,
        passed_tests=passed_tests,
        log_path=log_path,
    )
    sys.path.insert(0, str(repository / "src"))
    from decaf.experiments.common import atomic_json

    atomic_json(output_directory / "full_pytest.json", receipt)
    if exit_code != 0:
        raise RuntimeError(f"full pytest failed with exit code {exit_code}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = run(arguments.repository, arguments.verification_root)
    print(f"full pytest audit passed: {receipt['passed_tests']} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
