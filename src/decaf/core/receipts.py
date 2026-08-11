"""Atomic member and global run receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decaf.core.manifests import atomic_write_json, read_json, to_jsonable

MEMBER_STATUSES = frozenset({"running", "failed", "completed", "skipped"})
GLOBAL_STATUSES = frozenset(
    {
        "running",
        "partial",
        "failed",
        "completed",
        "completed_with_optional_failures",
    }
)


def utc_now() -> str:
    """Return a stable UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def build_member_receipt(
    member_id: str,
    status: str,
    *,
    optional: bool = False,
    started_at: str | None = None,
    finished_at: str | None = None,
    details: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a validated member-level receipt payload."""

    identifier = _identifier(member_id, name="member_id")
    if status not in MEMBER_STATUSES:
        raise ValueError(f"invalid member status: {status}")
    if not isinstance(optional, bool):
        raise TypeError("optional must be a boolean")
    start = utc_now() if started_at is None else _identifier(started_at, name="started_at")
    if status == "running":
        if finished_at is not None:
            raise ValueError("a running member cannot have finished_at")
        finish = None
    else:
        finish = (
            utc_now()
            if finished_at is None
            else _identifier(
                finished_at,
                name="finished_at",
            )
        )
    if status == "failed" and (not isinstance(error, str) or not error.strip()):
        raise ValueError("a failed member receipt requires a non-empty error")
    if status != "failed" and error is not None:
        raise ValueError("only a failed member receipt may contain an error")
    if details is not None and not isinstance(details, Mapping):
        raise TypeError("details must be a mapping")
    return {
        "schema_version": 1,
        "kind": "member",
        "member_id": identifier,
        "optional": optional,
        "status": status,
        "started_at": start,
        "finished_at": finish,
        "details": to_jsonable(dict(details or {})),
        "error": error.strip() if error is not None else None,
    }


def write_member_receipt(
    path: str | Path,
    member_id: str,
    status: str,
    **kwargs: Any,
) -> Path:
    """Atomically write a member receipt."""

    return atomic_write_json(path, build_member_receipt(member_id, status, **kwargs))


def load_member_receipt(path: str | Path) -> dict[str, Any]:
    """Read and minimally validate a member receipt."""

    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("kind") != "member":
        raise ValueError("receipt is not a member receipt")
    _identifier(payload.get("member_id"), name="member_id")
    if payload.get("status") not in MEMBER_STATUSES:
        raise ValueError("member receipt has an invalid status")
    return payload


def _member_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_identifier(value, name=name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique")
    return result


def aggregate_global_status(
    member_receipts: Mapping[str, Mapping[str, Any]],
    *,
    expected_members: Sequence[str] | None = None,
    optional_members: Sequence[str] = (),
    all_processes_exited: bool = False,
) -> str:
    """Derive a global status from atomic member receipts."""

    if not isinstance(all_processes_exited, bool):
        raise TypeError("all_processes_exited must be a boolean")
    receipts = dict(member_receipts)
    expected = (
        tuple(receipts)
        if expected_members is None
        else _member_ids(expected_members, name="expected_members")
    )
    if not expected:
        raise ValueError("expected_members must be non-empty")
    optional = set(_member_ids(optional_members, name="optional_members"))
    expected_set = set(expected)
    if not optional.issubset(expected_set):
        raise ValueError("optional_members must be a subset of expected_members")
    unexpected = set(receipts) - expected_set
    if unexpected:
        raise ValueError(f"unexpected member receipts: {sorted(unexpected)}")

    statuses: dict[str, str] = {}
    for member_id, receipt in receipts.items():
        if not isinstance(receipt, Mapping):
            raise TypeError(f"receipt for {member_id} must be a mapping")
        embedded_id = receipt.get("member_id", member_id)
        if embedded_id != member_id:
            raise ValueError(f"receipt member_id mismatch for {member_id}")
        status = receipt.get("status")
        if status not in MEMBER_STATUSES:
            raise ValueError(f"invalid status for member {member_id}: {status}")
        statuses[member_id] = status

    missing = expected_set - set(statuses)
    running = {member for member, status in statuses.items() if status == "running"}
    if not all_processes_exited and (missing or running):
        return "running"

    required = expected_set - optional
    required_completed = {member for member in required if statuses.get(member) == "completed"}
    required_unfinished = required - required_completed
    if required_unfinished:
        return "partial" if required_completed else "failed"

    optional_incomplete = {member for member in optional if statuses.get(member) != "completed"}
    if optional_incomplete:
        return "completed_with_optional_failures"
    return "completed"


def write_global_receipt(
    path: str | Path,
    run_id: str,
    member_receipts: Mapping[str, Mapping[str, Any]],
    *,
    expected_members: Sequence[str] | None = None,
    optional_members: Sequence[str] = (),
    all_processes_exited: bool = False,
    details: Mapping[str, Any] | None = None,
) -> Path:
    """Aggregate members and atomically write the global receipt."""

    identifier = _identifier(run_id, name="run_id")
    status = aggregate_global_status(
        member_receipts,
        expected_members=expected_members,
        optional_members=optional_members,
        all_processes_exited=all_processes_exited,
    )
    if all_processes_exited and status == "running":
        raise AssertionError("global receipt cannot remain running after process exit")
    payload = {
        "schema_version": 1,
        "kind": "global",
        "run_id": identifier,
        "status": status,
        "all_processes_exited": all_processes_exited,
        "updated_at": utc_now(),
        "member_count": len(member_receipts),
        "members": {
            member_id: {
                "status": receipt.get("status"),
                "optional": member_id in set(optional_members),
            }
            for member_id, receipt in sorted(member_receipts.items())
        },
        "details": to_jsonable(dict(details or {})),
    }
    return atomic_write_json(path, payload)


def finalize_global_receipt(
    path: str | Path,
    run_id: str,
    member_receipts: Mapping[str, Mapping[str, Any]],
    **kwargs: Any,
) -> Path:
    """Write a terminal global receipt after all worker processes exit."""

    return write_global_receipt(
        path,
        run_id,
        member_receipts,
        all_processes_exited=True,
        **kwargs,
    )
