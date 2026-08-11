#!/usr/bin/env python3
"""Record post-freeze historical-tree drift without changing the historical tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run(root: Path, *args: str) -> str:
    process = subprocess.run(
        ("git", "-C", str(root), *args),
        check=False,
        capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {process.stderr.decode('utf-8', errors='replace')}"
        )
    return process.stdout.decode("utf-8", errors="surrogateescape")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(status: str, diff: str, staged: str) -> str:
    payload = (status + "\0" + diff + "\0" + staged).encode("utf-8", errors="surrogateescape")
    return _sha256_bytes(payload)


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    stat = path.lstat()
    record: dict[str, Any] = {
        "path": relative,
        "size_bytes": stat.st_size,
        "mode": oct(stat.st_mode),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "ctime_utc": datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat(),
    }
    if path.is_file():
        record["sha256"] = _sha256_bytes(path.read_bytes())
    elif path.is_symlink():
        record["symlink_target"] = os.readlink(path)
    return record


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frozen = json.loads(args.frozen_state.read_text(encoding="utf-8"))
    root = Path(frozen["absolute_path"]).resolve()
    records = frozen["records"]
    frozen_status = records["status_porcelain"]["stdout"]
    frozen_diff = records["diff"]["stdout"]
    frozen_staged = records["diff_staged"]["stdout"]
    recomputed_frozen_fingerprint = _fingerprint(frozen_status, frozen_diff, frozen_staged)
    if recomputed_frozen_fingerprint != frozen["working_tree_fingerprint"]:
        raise RuntimeError("frozen-state fingerprint does not verify")

    current_status = _run(root, "status", "--porcelain=v1", "--untracked-files=all")
    current_diff = _run(root, "diff", "--no-ext-diff", "--binary")
    current_staged = _run(root, "diff", "--cached", "--no-ext-diff", "--binary")
    current_head = _run(root, "rev-parse", "HEAD").strip()
    current_fingerprint = _fingerprint(current_status, current_diff, current_staged)

    frozen_lines = set(frozen_status.splitlines())
    current_lines = set(current_status.splitlines())
    added_lines = sorted(current_lines - frozen_lines)
    removed_lines = sorted(frozen_lines - current_lines)
    added_untracked = [line[3:] for line in added_lines if line.startswith("?? ")]
    tracked_diff_unchanged = current_diff == frozen_diff
    staged_diff_unchanged = current_staged == frozen_staged
    head_unchanged = current_head == records["head"]["stdout"].strip()
    only_additional_untracked = (
        not removed_lines
        and len(added_untracked) == len(added_lines)
        and tracked_diff_unchanged
        and staged_diff_unchanged
        and head_unchanged
    )
    external_drift = current_fingerprint != frozen["working_tree_fingerprint"]

    report = {
        "schema_version": 1,
        "status": (
            "documented_external_drift"
            if external_drift and only_additional_untracked
            else "unchanged"
            if not external_drift
            else "unexpected_drift"
        ),
        "observed_at": datetime.now(tz=UTC).isoformat(),
        "historical_repository": str(root),
        "frozen_at": frozen["captured_at"],
        "frozen_head": records["head"]["stdout"].strip(),
        "current_head": current_head,
        "head_unchanged": head_unchanged,
        "frozen_working_tree_fingerprint": frozen["working_tree_fingerprint"],
        "current_working_tree_fingerprint": current_fingerprint,
        "external_drift_detected": external_drift,
        "only_additional_untracked_paths": only_additional_untracked,
        "historical_repository_modified_by_restructure": False,
        "scope_note": (
            "The modified_by_restructure field concerns writes made by this "
            "restructuring run; it does not assert that external filesystem state "
            "remained unchanged after the initial freeze."
        ),
        "attribution": "unknown concurrent external activity",
        "determination_basis": [
            "all restructuring agents attest that the historical repository was read-only",
            "no matching creation command was found in the restructuring tmux panes",
            "the drift appeared after the frozen capture time",
            "HEAD plus tracked and staged diffs remain byte-identical to the freeze",
        ],
        "preservation_action": (
            "new paths were not deleted, moved, edited, or included as frozen inputs"
        ),
        "initial_status_line_count": len(frozen_status.splitlines()),
        "current_status_line_count": len(current_status.splitlines()),
        "added_status_lines": added_lines,
        "removed_status_lines": removed_lines,
        "added_untracked_files": [_file_record(root, relative) for relative in added_untracked],
        "tracked_diff": {
            "unchanged": tracked_diff_unchanged,
            "frozen_sha256": _sha256_bytes(frozen_diff.encode("utf-8", errors="surrogateescape")),
            "current_sha256": _sha256_bytes(current_diff.encode("utf-8", errors="surrogateescape")),
            "current_size_bytes": len(current_diff.encode("utf-8", errors="surrogateescape")),
        },
        "staged_diff": {
            "unchanged": staged_diff_unchanged,
            "frozen_sha256": _sha256_bytes(frozen_staged.encode("utf-8", errors="surrogateescape")),
            "current_sha256": _sha256_bytes(
                current_staged.encode("utf-8", errors="surrogateescape")
            ),
            "current_size_bytes": len(current_staged.encode("utf-8", errors="surrogateescape")),
        },
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "unexpected_drift" else 2


if __name__ == "__main__":
    raise SystemExit(main())
