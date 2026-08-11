"""Discovery, verification, and materialization of sealed reference archives."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REFERENCE_ROOT_ENV = "DECAF_REFERENCE_RUNS_ROOT"


class ReferenceError(RuntimeError):
    """Raised when a sealed reference archive cannot be resolved safely."""


@dataclass(frozen=True)
class ReferenceRun:
    """The public, location-independent identity of a sealed run."""

    run_id: str
    family: str
    scientific_status: str
    archive_filename: str
    archive_sha256: str
    archive_size_bytes: int
    archive_member_count: int
    analysis_inputs: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedInput:
    """Receipt for one extracted machine-readable archive member."""

    run_id: str
    requested_suffix: str
    resolved_member: str
    relative_path: str
    sha256: str
    size_bytes: int
    row_count: int | None


def _load_run(path: Path) -> ReferenceRun:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReferenceError(f"cannot load reference manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("archive"), Mapping):
        raise ReferenceError(f"reference manifest {path} is malformed")
    archive = payload["archive"]
    digest = str(archive.get("sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ReferenceError(f"reference manifest {path} has an invalid SHA256")
    return ReferenceRun(
        run_id=str(payload["run_id"]),
        family=str(payload["family"]),
        scientific_status=str(payload["scientific_status"]),
        archive_filename=str(archive["filename"]),
        archive_sha256=digest,
        archive_size_bytes=int(archive["size_bytes"]),
        archive_member_count=int(archive["member_count"]),
        analysis_inputs=tuple(str(item) for item in payload.get("analysis_inputs", ())),
    )


def load_reference_runs(directory: str | Path) -> dict[str, ReferenceRun]:
    """Load every sealed-run manifest in a directory."""

    root = Path(directory)
    runs: dict[str, ReferenceRun] = {}
    for path in sorted(root.glob("*.yaml")):
        run = _load_run(path)
        if run.run_id in runs:
            raise ReferenceError(f"duplicate reference run ID: {run.run_id}")
        runs[run.run_id] = run
    expected = {"C0", "C1", "C2", "I9", "A0", "A1", "A2", "A3", "T0"}
    if set(runs) != expected:
        raise ReferenceError(f"reference run set differs from the sealed contract: {sorted(runs)}")
    return runs


def reference_roots(explicit: str | Path | Sequence[str | Path] | None = None) -> tuple[Path, ...]:
    """Resolve search roots from a CLI value or ``DECAF_REFERENCE_RUNS_ROOT``."""

    if explicit is None:
        configured = os.environ.get(REFERENCE_ROOT_ENV)
        if not configured:
            raise ReferenceError(f"set {REFERENCE_ROOT_ENV} or pass --reference-root")
        values: Sequence[str | Path] = tuple(part for part in configured.split(os.pathsep) if part)
    elif isinstance(explicit, (str, Path)):
        values = (explicit,)
    else:
        values = explicit
    roots = tuple(Path(value).expanduser().resolve() for value in values)
    if not roots:
        raise ReferenceError("at least one reference search root is required")
    missing = [str(root) for root in roots if not root.exists()]
    if missing:
        raise ReferenceError(f"reference search roots do not exist: {missing}")
    return roots


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_archive(run: ReferenceRun, roots: Iterable[str | Path]) -> Path:
    """Recursively discover and verify a sealed archive below one or more roots."""

    candidates: set[Path] = set()
    for value in roots:
        root = Path(value).expanduser().resolve()
        if root.is_file() and root.name == run.archive_filename:
            candidates.add(root)
        elif root.is_dir():
            candidates.update(path.resolve() for path in root.rglob(run.archive_filename))
    if not candidates:
        raise ReferenceError(f"archive not found recursively: {run.archive_filename}")
    valid: list[Path] = []
    failures: list[str] = []
    for path in sorted(candidates, key=lambda item: (len(item.parts), str(item))):
        if path.stat().st_size != run.archive_size_bytes:
            failures.append(f"{path}: size mismatch")
            continue
        if sha256_file(path) != run.archive_sha256:
            failures.append(f"{path}: SHA256 mismatch")
            continue
        valid.append(path)
    if not valid:
        raise ReferenceError(f"no discovered copy passed the sealed checksum: {failures}")
    return valid[0]


def verify_archive(path: str | Path, run: ReferenceRun) -> None:
    """Verify archive bytes and central-directory member count."""

    archive = Path(path)
    if archive.stat().st_size != run.archive_size_bytes:
        raise ReferenceError(f"archive size mismatch for {run.run_id}")
    if sha256_file(archive) != run.archive_sha256:
        raise ReferenceError(f"archive SHA256 mismatch for {run.run_id}")
    with zipfile.ZipFile(archive) as bundle:
        count = len(bundle.infolist())
    if count != run.archive_member_count:
        raise ReferenceError(
            f"archive member count mismatch for {run.run_id}: "
            f"expected {run.archive_member_count}, got {count}"
        )


def _normalize_member(value: str) -> str:
    normalized = value.replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)
    if not normalized or ".." in path.parts:
        raise ReferenceError(f"unsafe archive member suffix: {value}")
    return path.as_posix()


def resolve_member(names: Iterable[str], requested_suffix: str) -> str:
    """Resolve historical ZIP prefixes by a unique, path-boundary suffix."""

    suffix = _normalize_member(requested_suffix)
    files = [name.replace("\\", "/") for name in names if name and not name.endswith("/")]
    exact = [name for name in files if name == suffix]
    if exact:
        return exact[0]
    matches = [name for name in files if name.endswith(f"/{suffix}")]
    if not matches:
        raise ReferenceError(f"archive member suffix not found: {suffix}")
    shortest = min(len(PurePosixPath(name).parts) for name in matches)
    preferred = sorted(name for name in matches if len(PurePosixPath(name).parts) == shortest)
    if len(preferred) != 1:
        raise ReferenceError(f"archive member suffix is ambiguous: {suffix} -> {preferred}")
    return preferred[0]


def _row_count(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet

            return int(parquet.ParquetFile(path).metadata.num_rows)
        except ImportError as exc:
            raise ReferenceError("pyarrow is required to count Parquet rows") from exc
    return None


def materialize_inputs(
    run: ReferenceRun,
    archive_path: str | Path,
    requested_members: Iterable[str],
    paper_data_root: str | Path,
) -> list[MaterializedInput]:
    """Copy selected machine-readable members and return immutable receipts."""

    archive = Path(archive_path)
    output = Path(paper_data_root)
    receipts: list[MaterializedInput] = []
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for requested in sorted(set(requested_members)):
            suffix = _normalize_member(requested)
            resolved = resolve_member(names, suffix)
            destination = output / run.run_id / Path(*PurePosixPath(suffix).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.part")
            with bundle.open(resolved) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)
            receipts.append(
                MaterializedInput(
                    run_id=run.run_id,
                    requested_suffix=suffix,
                    resolved_member=resolved,
                    relative_path=destination.relative_to(output).as_posix(),
                    sha256=sha256_file(destination),
                    size_bytes=destination.stat().st_size,
                    row_count=_row_count(destination),
                )
            )
    return receipts


def receipt_dict(receipt: MaterializedInput) -> dict[str, Any]:
    """Convert a materialization receipt into a JSON-safe mapping."""

    return {
        "run_id": receipt.run_id,
        "requested_suffix": receipt.requested_suffix,
        "resolved_member": receipt.resolved_member,
        "relative_path": receipt.relative_path,
        "sha256": receipt.sha256,
        "size_bytes": receipt.size_bytes,
        "row_count": receipt.row_count,
    }


__all__ = [
    "REFERENCE_ROOT_ENV",
    "MaterializedInput",
    "ReferenceError",
    "ReferenceRun",
    "discover_archive",
    "load_reference_runs",
    "materialize_inputs",
    "receipt_dict",
    "reference_roots",
    "resolve_member",
    "sha256_file",
    "verify_archive",
]
