"""Deterministic file manifests and atomic JSON persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert supported scientific scalar containers to strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("JSON payloads cannot contain non-finite floats")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            result[key] = to_jsonable(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a payload deterministically with a trailing newline."""

    normalized = to_jsonable(payload)
    text = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{text}\n".encode()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically replace a JSON file using a temporary sibling and fsync."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor: int | None = None
        try:
            directory_descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            os.fsync(directory_descriptor)
        finally:
            if directory_descriptor is not None:
                os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a regular file."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(
    paths: Sequence[str | Path],
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Build a deterministic manifest of files contained by root."""

    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError("manifest root must be a directory")
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for item in paths:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root_path / candidate
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"manifest member is not a regular file: {resolved}")
        try:
            relative = resolved.relative_to(root_path)
        except ValueError as error:
            raise ValueError(f"manifest member is outside root: {resolved}") from error
        if resolved in seen:
            raise ValueError(f"duplicate manifest member: {relative.as_posix()}")
        seen.add(resolved)
        records.append(
            {
                "path": relative.as_posix(),
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    records.sort(key=lambda record: record["path"])
    return {
        "schema_version": 1,
        "file_count": len(records),
        "total_size": sum(record["size"] for record in records),
        "files": records,
    }


def verify_file_manifest(
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Verify manifest members without accepting paths outside root."""

    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("manifest files must be a list")
    root_path = Path(root).resolve(strict=True)
    errors: list[str] = []
    checked = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each manifest member must be an object")
        relative_value = record.get("path")
        if not isinstance(relative_value, str) or not relative_value:
            raise ValueError("manifest member path must be a non-empty string")
        relative = Path(relative_value)
        if relative.is_absolute():
            raise ValueError("manifest member paths must be relative")
        candidate = (root_path / relative).resolve()
        try:
            candidate.relative_to(root_path)
        except ValueError as error:
            raise ValueError(f"manifest member escapes root: {relative_value}") from error
        checked += 1
        if not candidate.is_file():
            errors.append(f"missing:{relative_value}")
            continue
        if candidate.stat().st_size != record.get("size"):
            errors.append(f"size:{relative_value}")
        if sha256_file(candidate) != record.get("sha256"):
            errors.append(f"sha256:{relative_value}")
    result = {
        "passed": not errors,
        "checked": checked,
        "errors": errors,
    }
    if raise_on_error and errors:
        raise AssertionError(f"file manifest verification failed: {errors}")
    return result


def write_file_manifest(
    path: str | Path,
    members: Sequence[str | Path],
    *,
    root: str | Path,
) -> Path:
    """Build and atomically write a file manifest."""

    return atomic_write_json(path, build_file_manifest(members, root=root))
