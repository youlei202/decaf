"""Run reproducibility gates and emit machine-readable verification reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
import time
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import Any

from decaf.audit import audit_repository
from decaf.experiments.common import atomic_json, atomic_text, repository_root, utc_now
from decaf.paper.analysis_replay import replay_paper_data
from decaf.paper.manifest import load_visual_manifest
from decaf.paper.render import render_all

MODES = (
    "all-cpu",
    "analysis-replay",
    "unit",
    "integration-cpu",
    "full-plan",
    "repository-audit",
)
FAMILIES = ("controlled", "imagenet9", "attribution", "covertype")


class VerificationFailure(RuntimeError):
    """Raised when a required reproducibility gate fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(command: Sequence[str], *, cwd: Path) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(
        tuple(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sys.stdout.write(process.stdout)
    report = {
        "command": list(command),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "exit_code": process.returncode,
        "status": "passed" if process.returncode == 0 else "failed",
        "output": process.stdout,
    }
    if process.returncode:
        raise VerificationFailure(
            f"command failed with exit code {process.returncode}: {' '.join(command)}"
        )
    return report


def _write_artifact_diff(
    repo: Path,
    generated_paths: Sequence[Path],
    verification: Path,
) -> dict[str, Any]:
    manifest = load_visual_manifest(repo / "paper" / "visual_manifest.yaml")
    by_name = {path.name: path for path in generated_paths}
    rows: list[dict[str, Any]] = []
    for asset in manifest.assets.values():
        expected_name = Path(asset.tex_target).name
        path = by_name.get(expected_name)
        exists = path is not None and path.is_file()
        source_missing = asset.status == "source_missing"
        rows.append(
            {
                "asset_id": asset.asset_id,
                "kind": asset.kind,
                "number": asset.number,
                "generated_path": (
                    path.relative_to(repo).as_posix()
                    if path is not None and path.is_relative_to(repo)
                    else str(path or "")
                ),
                "sha256": _sha256(path) if exists and path is not None else "",
                "status": (
                    "source_missing_recorded"
                    if exists and source_missing
                    else "regenerated"
                    if exists
                    else "missing"
                ),
            }
        )
    destination = verification / "paper_artifact_diff.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = ("asset_id", "kind", "number", "generated_path", "sha256", "status")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(destination, buffer.getvalue())
    missing = [row["asset_id"] for row in rows if row["status"] == "missing"]
    if missing:
        raise VerificationFailure(f"paper assets were not regenerated: {missing}")
    return {
        "figures_regenerated": sum(row["kind"] == "figure" for row in rows),
        "tables_regenerated": sum(row["kind"] == "table" for row in rows),
        "source_missing_recorded": [
            row["asset_id"] for row in rows if row["status"] == "source_missing_recorded"
        ],
    }


def run_analysis_replay(
    repo: Path,
    verification: Path,
    reference_roots: Sequence[str] | None,
    generated_root: Path,
) -> dict[str, Any]:
    """Verify sealed archives, recompute assertions, and regenerate all TeX assets."""

    replay_root = verification / "replay"
    receipt = replay_paper_data(
        replay_root,
        reference_root=reference_roots,
        repo_root=repo,
    )
    paths = render_all(replay_root, repo_root=repo, generated_root=generated_root)
    assertions = dict(receipt["headline_assertions"])
    unacceptable = {
        name: value
        for name, value in assertions.items()
        if value.get("status") not in {"verified", "source_missing", "generated"}
    }
    if unacceptable:
        raise VerificationFailure(
            "headline assertions did not fail closed: "
            + ", ".join(
                f"{name}={value.get('status')}" for name, value in unacceptable.items()
            )
        )
    artifact_summary = _write_artifact_diff(repo, paths, verification)
    headline_report = {
        "schema_version": 1,
        "status": "passed",
        "assertion_count": len(assertions),
        "verified_count": sum(
            value.get("status") == "verified" for value in assertions.values()
        ),
        "source_missing_count": sum(
            value.get("status") == "source_missing" for value in assertions.values()
        ),
        "assertions": assertions,
    }
    atomic_json(verification / "headline_assertions.json", headline_report)
    report = {
        "schema_version": 1,
        "status": "passed",
        "completed_at": utc_now(),
        "reference_runs_verified": len(receipt["runs"]),
        "inputs_materialized": len(receipt["inputs"]),
        **artifact_summary,
        "headline_assertion_count": len(assertions),
        "headline_assertions_status": "passed",
        "model_inference_performed": False,
    }
    atomic_json(verification / "analysis_replay.json", report)
    return report


def run_unit(repo: Path) -> dict[str, Any]:
    """Run model-agnostic and paper-regression tests."""

    targets = [
        relative
        for relative in ("tests/unit", "tests/regression")
        if (repo / relative).exists()
    ]
    return _run_command([sys.executable, "-m", "pytest", "-q", *targets], cwd=repo)


def run_integration_cpu(repo: Path) -> dict[str, Any]:
    """Run the real CPU integration suite."""

    target = repo / "tests" / "integration"
    if not target.is_dir() or not any(target.glob("test_*.py")):
        raise VerificationFailure("CPU integration tests have not been implemented")
    report = _run_command(
        [sys.executable, "-m", "pytest", "-q", "tests/integration"],
        cwd=repo,
    )
    report["gpu_real_shard_verification"] = "pending"
    return report


def run_full_plan(repo: Path) -> dict[str, Any]:
    """Run all paper-profile planners without starting computation."""

    reports: dict[str, Any] = {}
    for family in FAMILIES:
        reports[family] = _run_command(
            [
                sys.executable,
                "-m",
                f"decaf.experiments.{family}.cli",
                "--profile",
                "paper",
                "--plan-only",
            ],
            cwd=repo,
        )
    return {"status": "passed", "families": reports}


def run_repository_audit(repo: Path, verification: Path) -> dict[str, Any]:
    """Scan the public checkout for release-forbidden content."""

    report = audit_repository(repo)
    atomic_json(verification / "repository_audit.json", report)
    if not report["passed"]:
        raise VerificationFailure(f"repository audit found {report['finding_count']} issue(s)")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="all-cpu")
    parser.add_argument(
        "--reference-root",
        "--reference-runs",
        action="append",
        dest="reference_roots",
        help="Archive file or recursive archive-search root; repeat as needed",
    )
    parser.add_argument("--output", type=Path, help="Verification report directory")
    parser.add_argument("--generated-root", type=Path, help="Generated TeX root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = repository_root()
    verification = (args.output or repo / "verification").resolve()
    generated = (args.generated_root or repo / "paper" / "generated").resolve()
    verification.mkdir(parents=True, exist_ok=True)
    steps: dict[str, Any] = {}
    started_at = utc_now()
    try:
        if args.mode in {"analysis-replay", "all-cpu"}:
            steps["analysis_replay"] = run_analysis_replay(
                repo,
                verification,
                args.reference_roots,
                generated,
            )
        if args.mode in {"unit", "all-cpu"}:
            steps["unit"] = run_unit(repo)
        if args.mode in {"integration-cpu", "all-cpu"}:
            steps["integration_cpu"] = run_integration_cpu(repo)
        if args.mode in {"full-plan", "all-cpu"}:
            steps["full_plan"] = run_full_plan(repo)
        if args.mode in {"repository-audit", "all-cpu"}:
            steps["repository_audit"] = run_repository_audit(repo, verification)
    except Exception as error:
        report = {
            "schema_version": 1,
            "mode": args.mode,
            "status": "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "gpu_real_shard_verification": "pending",
            "steps": steps,
            "error": f"{type(error).__name__}: {error}",
        }
        atomic_json(verification / "cpu_verification.json", report)
        raise
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "status": "passed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "gpu_real_shard_verification": "pending",
        "steps": steps,
    }
    atomic_json(verification / "cpu_verification.json", report)
    print(f"verification_status={report['status']}")
    print("gpu_real_shard_verification=pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VerificationFailure",
    "build_parser",
    "main",
    "run_analysis_replay",
    "run_full_plan",
    "run_integration_cpu",
    "run_repository_audit",
    "run_unit",
]
