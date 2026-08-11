"""Deterministic Covertype preparation and documented offline smoke fixture."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_covtype, make_classification
from sklearn.model_selection import train_test_split

from decaf.experiments.common import atomic_json

_REAL_CACHE_SOURCE = "sklearn.datasets.fetch_covtype"
_CACHE_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetSplit:
    """One immutable natural-data split."""

    X: np.ndarray
    y: np.ndarray
    source_index: np.ndarray


@dataclass(frozen=True)
class CovertypeDataset:
    """Prepared binary Covertype task with train-only standardization."""

    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    source_kind: str
    fallback_reason: str | None
    fingerprint: str


def _fixture(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    fixture = config["data"]["fixture"]
    samples = int(fixture["samples"])
    features = int(fixture.get("features", 54))
    seed = int(fixture["seed"])
    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=int(fixture.get("informative", 18)),
        n_redundant=int(fixture.get("redundant", 6)),
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=float(fixture.get("class_sep", 1.1)),
        flip_y=float(fixture.get("flip_y", 0.02)),
        random_state=seed,
    )
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int8),
        np.arange(samples, dtype=np.int64),
        "deterministic_synthetic_covtype_fixture",
        "sklearn Covertype cache unavailable; smoke fixture explicitly configured",
    )


def _balanced_covtype(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_config = config["data"]
    X_source, target_source = fetch_covtype(
        download_if_missing=bool(data_config.get("download_if_missing", False)),
        return_X_y=True,
    )
    target = np.asarray(target_source, dtype=np.int64)
    selected = np.flatnonzero(np.isin(target, (1, 2)))
    class_one = selected[target[selected] == 1]
    class_two = selected[target[selected] == 2]
    maximum = int(data_config.get("max_total_samples", len(selected)))
    per_class = min(len(class_one), len(class_two), maximum // 2)
    if per_class < 2:
        raise ValueError("Covertype classes 1 and 2 do not have enough examples")
    rng = np.random.default_rng(int(data_config["split_seed"]))
    chosen = np.concatenate(
        (
            rng.choice(class_one, size=per_class, replace=False),
            rng.choice(class_two, size=per_class, replace=False),
        )
    )
    chosen = chosen[rng.permutation(len(chosen))]
    X = np.asarray(X_source[chosen], dtype=np.float64)
    y = np.asarray(target[chosen] == 2, dtype=np.int8)
    return X, y, np.asarray(chosen, dtype=np.int64)


def _source(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str | None]:
    mode = str(config["data"].get("source", "sklearn_covtype"))
    if mode == "synthetic_fixture":
        return _fixture(config)
    if mode not in {"sklearn_covtype", "cached_covtype_or_fixture"}:
        raise ValueError(f"unknown Covertype source mode: {mode}")
    try:
        X, y, indices = _balanced_covtype(config)
        return X, y, indices, "sklearn.datasets.fetch_covtype", None
    except Exception as error:
        if mode != "cached_covtype_or_fixture" or not bool(
            config["data"].get("allow_fixture_fallback", False)
        ):
            raise
        X, y, indices, source_kind, _ = _fixture(config)
        reason = f"{type(error).__name__}: cached sklearn Covertype was unavailable"
        return X, y, indices, source_kind, reason


def _fingerprint(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _cache_relative_path(value: Any, *, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Covertype cache {field} must be a root-relative path")
    return relative


def _fixed_shard_rows(cache_config: Mapping[str, Any]) -> dict[str, int]:
    configured = cache_config.get("fixed_shard_rows")
    if not isinstance(configured, Mapping):
        raise ValueError("Covertype cache fixed_shard_rows must be a mapping")
    rows: dict[str, int] = {}
    for split in _CACHE_SPLITS:
        value = configured.get(split)
        if isinstance(value, bool) or not isinstance(value, int) or value < 4 or value % 2:
            raise ValueError(
                f"Covertype fixed shard row count for {split} must be an even integer >= 4"
            )
        rows[split] = value
    return rows


def _cached_split(
    payload: Mapping[str, Any],
    *,
    split: str,
    rows: int,
) -> DatasetSplit:
    required = {
        f"{split}__X",
        f"{split}__y",
        f"{split}__cover_type",
        f"{split}__indices",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Covertype cache is missing {split} arrays: {missing}")

    X_source = np.asarray(payload[f"{split}__X"], dtype=np.float64)
    signed_y = np.asarray(payload[f"{split}__y"], dtype=np.int8)
    cover_type = np.asarray(payload[f"{split}__cover_type"], dtype=np.int8)
    source_index = np.asarray(payload[f"{split}__indices"], dtype=np.int64)
    if X_source.ndim != 2 or X_source.shape[1] != 54:
        raise ValueError(f"Covertype cache {split} features must have shape (rows, 54)")
    if any(array.ndim != 1 for array in (signed_y, cover_type, source_index)):
        raise ValueError(f"Covertype cache {split} labels and indices must be one-dimensional")
    if not (len(X_source) == len(signed_y) == len(cover_type) == len(source_index)):
        raise ValueError(f"Covertype cache {split} arrays have inconsistent row counts")
    if len(np.unique(source_index)) != len(source_index):
        raise ValueError(f"Covertype cache {split} source indices are not unique")
    if set(np.unique(signed_y).tolist()) != {-1, 1}:
        raise ValueError(f"Covertype cache {split} labels must be exactly -1 and +1")
    if set(np.unique(cover_type).tolist()) != {1, 2}:
        raise ValueError(f"Covertype cache {split} cover types must be exactly 1 and 2")
    expected_signed = np.where(cover_type == 2, 1, -1).astype(np.int8)
    if not np.array_equal(signed_y, expected_signed):
        raise ValueError(f"Covertype cache {split} label mapping is inconsistent")

    per_class = rows // 2
    selected_parts: list[np.ndarray] = []
    for label in (-1, 1):
        candidates = np.flatnonzero(signed_y == label)
        if len(candidates) < per_class:
            raise ValueError(f"Covertype cache {split} does not contain {per_class} rows per class")
        order = np.argsort(source_index[candidates], kind="stable")
        selected_parts.append(candidates[order[:per_class]])
    selected = np.concatenate(selected_parts)
    selected = selected[np.argsort(source_index[selected], kind="stable")]

    X = np.ascontiguousarray(X_source[selected], dtype=np.float64)
    y = np.ascontiguousarray(signed_y[selected] > 0, dtype=np.int8)
    indices = np.ascontiguousarray(source_index[selected], dtype=np.int64)
    if not np.isfinite(X).all():
        raise ValueError(f"Covertype cache {split} fixed shard contains non-finite features")
    if not np.isin(X[:, 10:], (0.0, 1.0)).all():
        raise ValueError(f"Covertype cache {split} fixed shard changed binary features")
    if np.bincount(y, minlength=2).tolist() != [per_class, per_class]:
        raise ValueError(f"Covertype cache {split} fixed shard is not balanced")
    return DatasetSplit(X=X, y=y, source_index=indices)


def _cached_real_splits(
    config: dict[str, Any],
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit, dict[str, Any]]:
    data_config = config["data"]
    if bool(data_config.get("allow_fixture_fallback", False)):
        raise ValueError("real Covertype cache mode forbids synthetic fixture fallback")
    cache_config = data_config.get("cache")
    if not isinstance(cache_config, Mapping):
        raise ValueError("real Covertype cache mode requires a cache mapping")

    root_env = str(cache_config.get("root_env", "DECAF_DATA_ROOT"))
    root_value = os.environ.get(root_env)
    if not root_value:
        raise FileNotFoundError(
            f"{root_env} must point to the pinned real Covertype cache directory; "
            "synthetic fallback is disabled"
        )
    root = Path(root_value).expanduser().resolve()
    archive_relative = _cache_relative_path(cache_config.get("archive"), field="archive")
    manifest_relative = _cache_relative_path(cache_config.get("manifest"), field="manifest")
    archive_path = (root / archive_relative).resolve()
    source_manifest_path = (root / manifest_relative).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"pinned real Covertype cache does not exist: {archive_path}")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"pinned real Covertype cache manifest does not exist: {source_manifest_path}"
        )

    archive_sha256 = _sha256(archive_path)
    expected_archive_sha256 = str(cache_config.get("archive_sha256", ""))
    if len(expected_archive_sha256) != 64 or archive_sha256 != expected_archive_sha256:
        raise ValueError(
            "real Covertype cache SHA-256 mismatch: "
            f"expected {expected_archive_sha256 or '<missing>'}, observed {archive_sha256}"
        )
    source_manifest_sha256 = _sha256(source_manifest_path)
    expected_manifest_sha256 = str(cache_config.get("manifest_sha256", ""))
    if len(expected_manifest_sha256) != 64 or source_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "real Covertype cache manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256 or '<missing>'}, "
            f"observed {source_manifest_sha256}"
        )

    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("real Covertype cache manifest is not valid JSON") from error
    if not isinstance(source_manifest, dict):
        raise ValueError("real Covertype cache manifest must be a JSON object")
    expected_logical_fingerprint = str(cache_config.get("logical_fingerprint", ""))
    required_manifest_values = {
        "kind": "decaf_covertype_v1_data",
        "dataset": "covertype_1_vs_2",
        "archive_name": archive_path.name,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_path.stat().st_size,
        "fingerprint": expected_logical_fingerprint,
        "standardization": "train_only_population_ddof0_first_10",
    }
    for field, expected in required_manifest_values.items():
        if source_manifest.get(field) != expected:
            raise ValueError(
                f"real Covertype cache manifest {field} mismatch: "
                f"expected {expected!r}, observed {source_manifest.get(field)!r}"
            )
    if len(expected_logical_fingerprint) != 64:
        raise ValueError("real Covertype cache logical_fingerprint must be pinned")
    invariants = source_manifest.get("invariants")
    if not isinstance(invariants, dict) or invariants.get("passed") is not True:
        raise ValueError("real Covertype cache manifest invariants did not pass")

    rows = _fixed_shard_rows(cache_config)
    declared_split_sizes = source_manifest.get("split_sizes")
    if not isinstance(declared_split_sizes, dict):
        raise ValueError("real Covertype cache manifest does not declare split_sizes")
    with np.load(archive_path, allow_pickle=False) as payload:
        splits = tuple(
            _cached_split(payload, split=split, rows=rows[split]) for split in _CACHE_SPLITS
        )
        for split in _CACHE_SPLITS:
            cached_rows = np.asarray(payload[f"{split}__indices"]).shape[0]
            if declared_split_sizes.get(split) != cached_rows:
                raise ValueError(f"real Covertype cache manifest {split} row count mismatch")

    all_indices = np.concatenate([dataset_split.source_index for dataset_split in splits])
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("real Covertype fixed shard overlaps across train/validation/test")
    source_index_fingerprint = _fingerprint(
        [dataset_split.source_index for dataset_split in splits]
    )
    receipt = {
        "archive_relative_path": archive_relative.as_posix(),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_path.stat().st_size,
        "cache_kind": str(source_manifest["kind"]),
        "logical_fingerprint": expected_logical_fingerprint,
        "manifest_relative_path": manifest_relative.as_posix(),
        "manifest_sha256": source_manifest_sha256,
        "root_environment_variable": root_env,
        "source_dataset": _REAL_CACHE_SOURCE,
        "transport": "pinned_npz_cache",
        "fixed_shard": {
            "class_count_per_split": {split: rows[split] // 2 for split in _CACHE_SPLITS},
            "rows": rows,
            "selection": "lowest_source_indices_per_class_within_frozen_split",
            "source_index_fingerprint": source_index_fingerprint,
        },
    }
    train, validation, test = splits
    return train, validation, test, receipt


def _split_and_standardize(
    X: np.ndarray,
    y: np.ndarray,
    source_index: np.ndarray,
    *,
    seed: int,
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    positions = np.arange(len(y), dtype=np.int64)
    train_pos, remainder = train_test_split(positions, test_size=0.4, random_state=seed, stratify=y)
    validation_pos, test_pos = train_test_split(
        remainder, test_size=0.5, random_state=seed, stratify=y[remainder]
    )
    continuous = min(10, X.shape[1])
    mean = np.mean(X[train_pos, :continuous], axis=0, dtype=np.float64)
    scale = np.std(X[train_pos, :continuous], axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)

    def make(positions_: np.ndarray) -> DatasetSplit:
        values = np.array(X[positions_], dtype=np.float64, copy=True)
        values[:, :continuous] = (values[:, :continuous] - mean) / scale
        return DatasetSplit(
            X=np.ascontiguousarray(values),
            y=np.ascontiguousarray(y[positions_], dtype=np.int8),
            source_index=np.ascontiguousarray(source_index[positions_], dtype=np.int64),
        )

    return make(train_pos), make(validation_pos), make(test_pos)


def _atomic_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(run_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Prepare, persist, and manifest one deterministic dataset."""

    source_receipt: dict[str, Any] | None = None
    mode = str(config["data"].get("source", "sklearn_covtype"))
    if mode == "sklearn_covtype_cache":
        train, validation, test, source_receipt = _cached_real_splits(config)
        source_kind = _REAL_CACHE_SOURCE
        fallback_reason = None
    else:
        X, y, source_index, source_kind, fallback_reason = _source(config)
        train, validation, test = _split_and_standardize(
            X, y, source_index, seed=int(config["data"]["split_seed"])
        )
    fingerprint = _fingerprint(
        [
            train.X,
            train.y,
            train.source_index,
            validation.X,
            validation.y,
            validation.source_index,
            test.X,
            test.y,
            test.source_index,
        ]
    )
    if source_receipt is not None:
        expected_shard_fingerprint = str(config["data"]["cache"].get("fixed_shard_fingerprint", ""))
        if len(expected_shard_fingerprint) != 64 or fingerprint != expected_shard_fingerprint:
            raise ValueError(
                "real Covertype fixed-shard fingerprint mismatch: "
                f"expected {expected_shard_fingerprint or '<missing>'}, observed {fingerprint}"
            )
        source_receipt["fixed_shard"]["fingerprint"] = fingerprint
    path = run_path / "raw" / "covertype_data.npz"
    _atomic_npz(
        path,
        {
            "train_X": train.X,
            "train_y": train.y,
            "train_source_index": train.source_index,
            "validation_X": validation.X,
            "validation_y": validation.y,
            "validation_source_index": validation.source_index,
            "test_X": test.X,
            "test_y": test.y,
            "test_source_index": test.source_index,
            "source_kind": np.asarray(source_kind),
            "fallback_reason": np.asarray(fallback_reason or ""),
            "fingerprint": np.asarray(fingerprint),
        },
    )
    manifest = {
        "schema_version": 1,
        "dataset": "Covertype binary cover types 1 versus 2",
        "source_kind": source_kind,
        "source_archive": source_receipt,
        "fallback_reason": fallback_reason,
        "fixture_is_smoke_only": source_kind == "deterministic_synthetic_covtype_fixture",
        "natural_feature_count": int(train.X.shape[1]),
        "rows": {
            "train": len(train.y),
            "validation": len(validation.y),
            "test": len(test.y),
        },
        "class_counts": {
            "train": np.bincount(train.y, minlength=2).tolist(),
            "validation": np.bincount(validation.y, minlength=2).tolist(),
            "test": np.bincount(test.y, minlength=2).tolist(),
        },
        "fingerprint": fingerprint,
        "artifact": {
            "relative_path": "raw/covertype_data.npz",
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        },
    }
    atomic_json(run_path / "manifests" / "data.json", manifest)
    return manifest


def load_dataset(run_path: Path) -> CovertypeDataset:
    """Load the prepared cache without pickle support."""

    path = run_path / "raw" / "covertype_data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"prepared Covertype cache does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:

        def split(name: str) -> DatasetSplit:
            return DatasetSplit(
                X=np.asarray(payload[f"{name}_X"], dtype=np.float64),
                y=np.asarray(payload[f"{name}_y"], dtype=np.int8),
                source_index=np.asarray(payload[f"{name}_source_index"], dtype=np.int64),
            )

        fallback = str(payload["fallback_reason"].item())
        return CovertypeDataset(
            train=split("train"),
            validation=split("validation"),
            test=split("test"),
            source_kind=str(payload["source_kind"].item()),
            fallback_reason=fallback or None,
            fingerprint=str(payload["fingerprint"].item()),
        )


__all__ = [
    "CovertypeDataset",
    "DatasetSplit",
    "load_dataset",
    "prepare_dataset",
]
