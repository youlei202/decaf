"""Controlled model registries and checkpoint identity validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from decaf.core.manifests import sha256_file

BASE_TASKS = ("object_color", "object_shape", "wall_color", "color_shape_xor", "context_gate")
ARCHITECTURES = ("resnet18", "small_vit")
BASE_SEEDS = (3101, 3102, 3103)
CONTRADICTION_TASKS = ("direct", "gate", "invert")
CONTRADICTION_SEEDS = (7101, 7102, 7103, 7104, 7105)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """Portable model identity used by plans and member receipts."""

    model_id: str
    family: str
    task: str
    architecture: str
    seed: int
    module: str | None = None
    factor: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "task": self.task,
            "architecture": self.architecture,
            "seed": self.seed,
            "module": self.module,
            "factor": self.factor,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def model_id(task: str, architecture: str, seed: int) -> str:
    """Return the frozen C0/C2 model identifier format."""

    return f"{task}__{architecture}__seed_{int(seed)}"


def expected_base_models(
    tasks: Sequence[str] = BASE_TASKS,
    architectures: Sequence[str] = ARCHITECTURES,
    seeds: Sequence[int] = BASE_SEEDS,
) -> tuple[ModelRecord, ...]:
    return tuple(
        ModelRecord(model_id(task, architecture, seed), "C0", task, architecture, int(seed))
        for task in tasks
        for architecture in architectures
        for seed in seeds
    )


def expected_contradiction_models(
    tasks: Sequence[str] = CONTRADICTION_TASKS,
    architectures: Sequence[str] = ARCHITECTURES,
    seeds: Sequence[int] = CONTRADICTION_SEEDS,
) -> tuple[ModelRecord, ...]:
    return tuple(
        ModelRecord(model_id(task, architecture, seed), "C2", task, architecture, int(seed))
        for task in tasks
        for architecture in architectures
        for seed in seeds
    )


def validate_c0_manifest(frame: pd.DataFrame, *, expected_count: int = 30) -> pd.DataFrame:
    """Validate the sealed no-retraining C0 checkpoint/cache registry."""

    required = {
        "model_id",
        "task_name",
        "architecture",
        "seed",
        "checkpoint_path",
        "checkpoint_sha256",
        "probability_cache_path",
        "probability_cache_sha256",
        "qualified",
        "available",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"C0 model manifest is missing columns: {sorted(missing)}")
    rows = frame.copy()
    if len(rows) != int(expected_count) or rows["model_id"].nunique() != len(rows):
        raise ValueError(f"C0 manifest must contain {expected_count} unique base models")
    if not _truthy(rows["qualified"]).all() or not _truthy(rows["available"]).all():
        raise ValueError("C0 manifest contains an unavailable or unqualified frozen model")
    for column in ("checkpoint_sha256", "probability_cache_sha256"):
        if (
            not rows[column]
            .astype(str)
            .str.lower()
            .map(lambda value: bool(SHA256_PATTERN.fullmatch(value)))
            .all()
        ):
            raise ValueError(f"C0 manifest contains an invalid {column}")
    rows.attrs["no_retraining"] = True
    return rows.sort_values("model_id", kind="mergesort").reset_index(drop=True)


def validate_c1_manifest(
    frame: pd.DataFrame,
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Select and validate the 88 frozen C1 measurement checkpoints."""

    required = {
        "model_id",
        "module",
        "variant",
        "architecture",
        "seed",
        "checkpoint_path",
        "checkpoint_sha256",
        "selected_for_b200",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"C1 model manifest is missing columns: {sorted(missing)}")
    selected = frame.loc[_truthy(frame["selected_for_b200"])].copy()
    counts = selected.groupby("module", sort=True)["model_id"].nunique().to_dict()
    expected_source = {"E": 52, "C": 18, "F": 18} if expected_counts is None else expected_counts
    expected = {str(key): int(value) for key, value in expected_source.items()}
    if counts != expected or len(selected) != sum(expected.values()):
        raise ValueError(
            f"C1 selected checkpoint counts changed: expected {expected}, found {counts}"
        )
    if selected["model_id"].duplicated().any():
        raise ValueError("C1 selected model IDs must be unique")
    if (
        not selected["checkpoint_sha256"]
        .astype(str)
        .str.lower()
        .map(lambda value: bool(SHA256_PATTERN.fullmatch(value)))
        .all()
    ):
        raise ValueError("C1 manifest contains an invalid checkpoint SHA256")
    return selected.sort_values("model_id", kind="mergesort").reset_index(drop=True)


def validate_c2_model_grid(
    frame: pd.DataFrame,
    *,
    expected_count: int = 30,
    expected_records: Sequence[ModelRecord] | None = None,
) -> pd.DataFrame:
    """Validate the complete task/architecture/seed contradiction grid."""

    required = {"model_id", "task", "architecture", "seed"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"C2 model table is missing columns: {sorted(missing)}")
    rows = frame.copy()
    keys = set(
        zip(
            rows["task"].astype(str),
            rows["architecture"].astype(str),
            rows["seed"].astype(int),
            strict=True,
        )
    )
    registry = (
        expected_contradiction_models() if expected_records is None else tuple(expected_records)
    )
    expected = {(record.task, record.architecture, record.seed) for record in registry}
    if (
        len(rows) != int(expected_count)
        or keys != expected
        or rows["model_id"].nunique() != len(rows)
    ):
        raise ValueError(
            f"C2 table does not contain the registered {int(expected_count)}-model grid"
        )
    return rows.sort_values("model_id", kind="mergesort").reset_index(drop=True)


def verify_checkpoint_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str = "checkpoint",
) -> Path:
    """Verify one user-supplied checkpoint without deserializing it."""

    source = Path(path).resolve()
    digest = str(expected_sha256).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} has an invalid registered SHA256")
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    if sha256_file(source) != digest:
        raise ValueError(f"{label} SHA256 mismatch")
    return source


def _manifest_checkpoint_path(manifest: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else manifest.parent / path


def validate_c1_checkpoint_bundle(
    manifest_path: str | Path,
    expected_checkpoints: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Validate exact C1 identities, producer jobs, and checkpoint bytes."""

    source = Path(manifest_path).resolve(strict=True)
    expected_by_id = {str(row["model_id"]): row for row in expected_checkpoints}
    expected_counts = {
        module: sum(str(row["module"]) == module for row in expected_checkpoints)
        for module in ("E", "C", "F")
    }
    frame = pd.read_csv(source)
    if "producer_member_id" not in frame:
        raise ValueError("C1 model manifest is missing columns: ['producer_member_id']")
    rows = validate_c1_manifest(frame, expected_counts=expected_counts)
    if set(rows["model_id"].astype(str)) != set(expected_by_id):
        raise ValueError("C1 manifest does not contain the configured selected checkpoint IDs")
    if rows["checkpoint_path"].astype(str).duplicated().any():
        raise ValueError("C1 checkpoint paths must be unique")
    for row in rows.itertuples(index=False):
        model_id_value = str(row.model_id)
        expected = expected_by_id[model_id_value]
        identity = (
            str(row.module) == str(expected["module"])
            and str(row.variant) == str(expected["variant"])
            and str(row.architecture) == str(expected["architecture"])
            and int(row.seed) == int(expected["seed"])
        )
        if not identity:
            raise ValueError(f"C1 checkpoint metadata mismatch: {model_id_value}")
        expected_producer = (
            f"c1_train__{expected['trajectory_id']}"
            if str(expected["module"]) == "E"
            else f"c1_train__{model_id_value}"
        )
        if str(row.producer_member_id) != expected_producer:
            raise ValueError(f"C1 checkpoint producer mismatch: {model_id_value}")
        verify_checkpoint_file(
            _manifest_checkpoint_path(source, row.checkpoint_path),
            str(row.checkpoint_sha256),
            label=f"C1 {model_id_value} checkpoint",
        )
    rows.attrs["byte_identity_verified"] = True
    return rows


def validate_c2_checkpoint_bundle(
    manifest_path: str | Path,
    expected_records: Sequence[ModelRecord] | None = None,
) -> pd.DataFrame:
    """Validate the exact C2 grid, producer jobs, and checkpoint bytes."""

    source = Path(manifest_path).resolve(strict=True)
    registry = (
        expected_contradiction_models() if expected_records is None else tuple(expected_records)
    )
    frame = pd.read_csv(source)
    required = {"checkpoint_path", "checkpoint_sha256", "producer_member_id"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"C2 model manifest is missing columns: {sorted(missing)}")
    rows = validate_c2_model_grid(
        frame,
        expected_count=len(registry),
        expected_records=registry,
    )
    expected_by_id = {record.model_id: record for record in registry}
    if set(rows["model_id"].astype(str)) != set(expected_by_id):
        raise ValueError("C2 manifest does not contain the configured model IDs")
    if rows["checkpoint_path"].astype(str).duplicated().any():
        raise ValueError("C2 checkpoint paths must be unique")
    for row in rows.itertuples(index=False):
        model_id_value = str(row.model_id)
        if str(row.producer_member_id) != f"c2_train__{model_id_value}":
            raise ValueError(f"C2 checkpoint producer mismatch: {model_id_value}")
        verify_checkpoint_file(
            _manifest_checkpoint_path(source, row.checkpoint_path),
            str(row.checkpoint_sha256),
            label=f"C2 {model_id_value} checkpoint",
        )
    rows.attrs["byte_identity_verified"] = True
    return rows


def validate_c0_no_retraining_bundle(manifest_path: str | Path) -> pd.DataFrame:
    """Validate all C0 checkpoint/cache bytes; never trains or rewrites them."""

    source = Path(manifest_path).resolve()
    rows = validate_c0_manifest(pd.read_csv(source))
    for row in rows.itertuples(index=False):
        for label, raw_path, digest in (
            ("checkpoint", row.checkpoint_path, row.checkpoint_sha256),
            ("probability cache", row.probability_cache_path, row.probability_cache_sha256),
        ):
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = source.parent / path
            verify_checkpoint_file(path, str(digest), label=f"C0 {row.model_id} {label}")
    return rows


__all__ = [
    "ARCHITECTURES",
    "BASE_SEEDS",
    "BASE_TASKS",
    "CONTRADICTION_SEEDS",
    "CONTRADICTION_TASKS",
    "ModelRecord",
    "expected_base_models",
    "expected_contradiction_models",
    "model_id",
    "validate_c0_manifest",
    "validate_c0_no_retraining_bundle",
    "validate_c1_checkpoint_bundle",
    "validate_c1_manifest",
    "validate_c2_checkpoint_bundle",
    "validate_c2_model_grid",
    "verify_checkpoint_file",
]
