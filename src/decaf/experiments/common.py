"""Shared command-line and run-lifecycle infrastructure for experiment families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Keep the GPU verifier importable under its pinned Python 3.10 environment.
# Ruff targets the public Python 3.11 package and would otherwise require
# ``datetime.UTC``, which does not exist in that runtime.
UTC_TIMEZONE = getattr(__import__("datetime"), "UTC", timezone.utc)  # noqa: UP017

VALID_STAGES = ("prepare", "compute", "analyze", "paper", "all")
TERMINAL_STATUSES = (
    "partial",
    "failed",
    "completed",
    "completed_with_optional_failures",
)
StageHandler = Callable[["RunContext"], Mapping[str, Any] | None]


class TerminationRequested(RuntimeError):
    """Raised when the process receives a normal termination request."""


def parse_devices(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, duplicate-free CUDA device list."""

    raw = tuple(part.strip() for part in value.split(","))
    if not raw or any(not part for part in raw):
        raise argparse.ArgumentTypeError("devices must be a comma-separated list of integers")
    try:
        devices = tuple(int(part) for part in raw)
    except ValueError as error:
        message = "devices must be a comma-separated list of integers"
        raise argparse.ArgumentTypeError(message) from error
    if any(device < 0 for device in devices) or len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("devices must contain unique non-negative integers")
    return devices


def utc_now() -> str:
    """Return a stable UTC timestamp."""

    return datetime.now(UTC_TIMEZONE).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_now_precise() -> str:
    return datetime.now(UTC_TIMEZONE).isoformat(timespec="microseconds").replace("+00:00", "Z")


def repository_root() -> Path:
    """Locate the checkout root without relying on the caller's working directory."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not locate the repository root")


def atomic_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    """Atomically write formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def available_cpu_count() -> int:
    """Return the usable CPU count, including a cgroup-v2 quota when present."""

    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
    quota_path = Path("/sys/fs/cgroup/cpu.max")
    if not quota_path.is_file():
        return max(1, affinity)
    quota, period = quota_path.read_text(encoding="utf-8").strip().split()
    if quota == "max":
        return max(1, affinity)
    quota_count = max(1, int(int(quota) / int(period)))
    return max(1, min(affinity, quota_count))


def bounded_workers(requested: int | None = None) -> int:
    """Bound CPU work to the task's 32-worker ceiling and the runtime quota."""

    ceiling = available_cpu_count()
    if requested is not None:
        ceiling = min(ceiling, requested)
    return max(1, min(32, ceiling))


def configure_cpu_runtime(workers: int) -> None:
    """Prevent nested numerical-library thread pools from oversubscribing workers."""

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")
    os.environ["DECAF_MAX_WORKERS"] = str(workers)


def load_profile(experiment: str, profile: str, explicit: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate an experiment configuration."""

    path = explicit or repository_root() / "configs" / experiment / f"{profile}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    if payload.get("experiment") != experiment:
        raise ValueError(f"configuration experiment must be {experiment!r}: {path}")
    payload["_source"] = str(path.resolve())
    return payload


def make_parser(
    experiment: str,
    *,
    profiles: Sequence[str] = ("smoke", "paper"),
) -> argparse.ArgumentParser:
    """Create the uniform experiment parser."""

    parser = argparse.ArgumentParser(prog=f"decaf-{experiment}")
    parser.add_argument("--stage", choices=VALID_STAGES, default="all")
    parser.add_argument("--profile", choices=tuple(profiles), default=profiles[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--devices",
        type=parse_devices,
        help="Comma-separated physical CUDA device IDs; single-B200 verification uses 0",
    )
    return parser


def default_output(experiment: str, run_id: str) -> Path:
    """Resolve a public, environment-configurable run directory."""

    configured = os.environ.get("DECAF_RESULTS_ROOT")
    root = Path(configured).expanduser() if configured else repository_root() / "runs"
    return root / experiment / run_id


def make_run_id(experiment: str, profile: str) -> str:
    """Create a sortable run identifier."""

    stamp = datetime.now(UTC_TIMEZONE).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment}-{profile}-{stamp}"


@dataclass
class RunContext:
    """Paths, configuration, and atomic receipts for one experiment run."""

    experiment: str
    profile: str
    stage: str
    path: Path
    config: dict[str, Any]
    workers: int
    resume: bool

    @classmethod
    def create(
        cls,
        *,
        experiment: str,
        profile: str,
        stage: str,
        output: Path,
        config: dict[str, Any],
        workers: int,
        resume: bool,
    ) -> RunContext:
        context = cls(
            experiment=experiment,
            profile=profile,
            stage=stage,
            path=output.resolve(),
            config=config,
            workers=workers,
            resume=resume,
        )
        context._initialize()
        return context

    @property
    def run_receipt_path(self) -> Path:
        return self.path / "run.json"

    def _initialize(self) -> None:
        if self.path.exists() and not self.resume and any(self.path.iterdir()):
            raise FileExistsError(
                f"run directory is not empty; pass --resume to reuse it: {self.path}"
            )
        for relative in (
            "manifests",
            "raw",
            "metrics",
            "paper_data",
            "receipts",
            "logs",
        ):
            (self.path / relative).mkdir(parents=True, exist_ok=True)
        config_copy = {key: value for key, value in self.config.items() if key != "_source"}
        atomic_text(self.path / "config.yaml", yaml.safe_dump(config_copy, sort_keys=False))
        atomic_json(
            self.path / "environment.json",
            {
                "available_cpus": available_cpu_count(),
                "executable": sys.executable,
                "max_workers": self.workers,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "recorded_at": utc_now(),
                "thread_limits": {
                    key: os.environ.get(key)
                    for key in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
        )
        for name in ("data.json", "checkpoints.json"):
            path = self.path / "manifests" / name
            if not path.exists():
                atomic_json(path, {"items": [], "schema_version": 1})
        jobs = self.path / "manifests" / "jobs.jsonl"
        if not jobs.exists():
            atomic_text(jobs, "")
        previous = self.read_run_receipt()
        started_at = previous.get("started_at", utc_now()) if self.resume else utc_now()
        self.set_status("running", started_at=started_at)

    def read_run_receipt(self) -> dict[str, Any]:
        if not self.run_receipt_path.is_file():
            return {}
        return json.loads(self.run_receipt_path.read_text(encoding="utf-8"))

    def set_status(self, status: str, **extra: Any) -> None:
        if status != "running" and status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid global run status: {status}")
        previous = self.read_run_receipt()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "experiment": self.experiment,
            "profile": self.profile,
            "requested_stage": self.stage,
            "run_id": self.path.name,
            "status": status,
            "started_at": previous.get("started_at", utc_now()),
            "updated_at": utc_now(),
        }
        payload.update(extra)
        if status in TERMINAL_STATUSES:
            payload["finished_at"] = utc_now()
        atomic_json(self.run_receipt_path, payload)

    def stage_receipt(self, stage: str) -> Path:
        return self.path / "receipts" / f"{stage}.json"

    def stage_completed(self, stage: str) -> bool:
        path = self.stage_receipt(stage)
        if not path.is_file():
            return False
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"

    def record_stage(
        self,
        stage: str,
        status: str,
        *,
        started_at: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        atomic_json(
            self.stage_receipt(stage),
            {
                "schema_version": 1,
                "stage": stage,
                "status": status,
                "started_at": started_at,
                "finished_at": utc_now(),
                "details": dict(details or {}),
            },
        )

    def append_job(self, record: Mapping[str, Any]) -> None:
        path = self.path / "manifests" / "jobs.jsonl"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        line = json.dumps(dict(record), sort_keys=True, separators=(",", ":"))
        atomic_text(path, existing + line + "\n")


def requested_stages(stage: str) -> tuple[str, ...]:
    """Expand the all stage into the canonical stage order."""

    return ("prepare", "compute", "analyze", "paper") if stage == "all" else (stage,)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_artifact_inventory(context: RunContext) -> list[dict[str, Any]]:
    """Bind the immutable run evidence checked by a resume validator.

    Lifecycle files rewritten by opening/closing a resumed CLI invocation are
    deliberately excluded.  Resume receipts are excluded so later stage
    validations do not invalidate the earlier compute proof.
    """

    volatile = {"config.yaml", "environment.json", "run.json"}
    records: list[dict[str, Any]] = []
    for path in sorted(context.path.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(context.path).as_posix()
        if (
            relative in volatile
            or relative.startswith("logs/")
            or relative.startswith("receipts/resume/")
        ):
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise RuntimeError("resume validation produced an empty artifact inventory")
    return records


def _compute_member_inventory(context: RunContext) -> list[dict[str, str]] | None:
    """Return the canonical member-output inventory when a plan exposes it."""

    plan_path = context.path / "manifests" / "plan.json"
    if not plan_path.is_file():
        return None
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    members = plan.get("members") if isinstance(plan, Mapping) else None
    if not isinstance(members, list) or not members:
        return None
    records: list[dict[str, str]] = []
    for raw in sorted(
        members,
        key=lambda value: str(value.get("member_id", "")) if isinstance(value, Mapping) else "",
    ):
        if not isinstance(raw, Mapping):
            return None
        member_id = raw.get("member_id")
        output_value = raw.get("output_path")
        receipt_value = raw.get("receipt_path")
        identities = (member_id, output_value, receipt_value)
        if not all(isinstance(value, str) and value for value in identities):
            return None
        output_path = context.path / str(output_value)
        receipt_path = context.path / str(receipt_value)
        if not output_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"resumed compute member is incomplete: {member_id}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        details = receipt.get("details") if isinstance(receipt, Mapping) else None
        if (
            receipt.get("status") != "completed"
            or receipt.get("member_id") != member_id
            or not isinstance(details, Mapping)
            or details.get("output_path") != output_value
            or details.get("output_sha256") != _sha256_file(output_path)
        ):
            raise RuntimeError(f"resumed compute member lineage drifted: {member_id}")
        records.append(
            {
                "member_id": str(member_id),
                "output_path": str(output_value),
                "output_sha256": str(details["output_sha256"]),
                "receipt_path": str(receipt_value),
            }
        )
    return records


def _record_resume_validation(
    context: RunContext,
    stage: str,
    *,
    started_at: str,
    details: Mapping[str, Any] | None,
) -> None:
    source_receipt = context.stage_receipt(stage)
    if not source_receipt.is_file():
        raise FileNotFoundError(f"resumed {stage} stage has no source receipt")
    inventory: list[dict[str, Any]] = (
        _compute_member_inventory(context) if stage == "compute" else None
    ) or _resume_artifact_inventory(context)
    encoded_inventory = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    validation_details = dict(details or {})
    count_values = {
        int(validation_details[key])
        for key in (
            "member_count",
            "validated_members",
            "completed_members",
            "configured_members",
        )
        if key in validation_details and validation_details[key] is not None
    }
    if len(count_values) > 1:
        raise ValueError("resume validator returned contradictory member counts")
    member_count = count_values.pop() if count_values else 0
    if member_count < 0:
        raise ValueError("resume validator returned a negative member count")
    atomic_json(
        context.path / "receipts" / "resume" / f"{stage}.json",
        {
            "schema_version": 1,
            "stage": stage,
            "status": "completed",
            "validation_started_at": started_at,
            "validation_finished_at": _utc_now_precise(),
            "member_count": member_count,
            "resumed_members": member_count,
            "reexecuted": 0,
            "source_compute_receipt_sha256": (
                _sha256_file(source_receipt) if stage == "compute" else None
            ),
            "source_stage_receipt_sha256": _sha256_file(source_receipt),
            "artifact_inventory": inventory,
            "artifact_inventory_sha256": hashlib.sha256(encoded_inventory).hexdigest(),
            "validator_details": validation_details,
        },
    )


def execute_run(
    context: RunContext,
    handlers: Mapping[str, StageHandler],
    resume_validators: Mapping[str, StageHandler] | None = None,
) -> int:
    """Execute stages with resumable, terminal-state-safe receipts."""

    completed: list[str] = []
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_termination(signum: int, _frame: Any) -> None:
        signal.signal(signum, signal.SIG_IGN)
        raise TerminationRequested(f"received signal {signum}")

    signal.signal(signal.SIGTERM, request_termination)
    try:
        for stage in requested_stages(context.stage):
            if context.resume and context.stage_completed(stage):
                validator = (resume_validators or {}).get(stage)
                if validator is not None:
                    validation_started_at = _utc_now_precise()
                    details = validator(context)
                    _record_resume_validation(
                        context,
                        stage,
                        started_at=validation_started_at,
                        details=details,
                    )
                completed.append(stage)
                continue
            handler = handlers.get(stage)
            if handler is None:
                raise KeyError(f"no handler registered for stage {stage!r}")
            started_at = utc_now()
            started_clock = time.monotonic()
            try:
                details = dict(handler(context) or {})
            except Exception as error:
                context.record_stage(
                    stage,
                    "failed",
                    started_at=started_at,
                    details={"error": f"{type(error).__name__}: {error}"},
                )
                raise
            details["elapsed_seconds"] = round(time.monotonic() - started_clock, 6)
            context.record_stage(stage, "completed", started_at=started_at, details=details)
            completed.append(stage)
        context.set_status("completed", completed_stages=completed)
        return 0
    except Exception as error:
        context.set_status(
            "failed",
            completed_stages=completed,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def write_plan(plan: Mapping[str, Any], destination: Path | None = None) -> None:
    """Print a static plan and optionally persist it atomically."""

    serializable = dict(plan)
    text = json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(text)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "plan.json", serializable)


def remove_tree_if_empty(path: Path) -> None:
    """Remove an abandoned empty output directory created by a caller."""

    if path.is_dir() and not any(path.iterdir()):
        shutil.rmtree(path)


def run_cli(
    *,
    experiment: str,
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    handlers: Mapping[str, StageHandler],
    resume_validators: Mapping[str, StageHandler] | None = None,
) -> int:
    """Run a family CLI after uniform planning and lifecycle setup."""

    config = load_profile(experiment, args.profile, args.config)
    if args.devices is not None:
        device_text = ",".join(str(device) for device in args.devices)
        configured = os.environ.get("CUDA_VISIBLE_DEVICES")
        if configured not in {None, "", device_text}:
            raise ValueError(
                "--devices conflicts with the existing CUDA_VISIBLE_DEVICES setting: "
                f"{device_text!r} != {configured!r}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = device_text
        os.environ["DECAF_DEVICES"] = device_text
    if args.plan_only:
        write_plan(plan, args.output)
        return 0
    run_id = args.run_id or make_run_id(experiment, args.profile)
    output = args.output or default_output(experiment, run_id)
    workers = bounded_workers(args.max_workers or config.get("max_workers"))
    configure_cpu_runtime(workers)
    context = RunContext.create(
        experiment=experiment,
        profile=args.profile,
        stage=args.stage,
        output=output,
        config=config,
        workers=workers,
        resume=args.resume,
    )
    return execute_run(context, handlers, resume_validators)


__all__ = [
    "RunContext",
    "StageHandler",
    "TerminationRequested",
    "atomic_json",
    "atomic_text",
    "available_cpu_count",
    "bounded_workers",
    "configure_cpu_runtime",
    "execute_run",
    "load_profile",
    "make_parser",
    "parse_devices",
    "repository_root",
    "requested_stages",
    "run_cli",
    "utc_now",
    "write_plan",
]
