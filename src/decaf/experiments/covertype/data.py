"""Deterministic Covertype preparation and documented offline smoke fixture."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_covtype, make_classification
from sklearn.model_selection import train_test_split

from decaf.experiments.common import atomic_json


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
