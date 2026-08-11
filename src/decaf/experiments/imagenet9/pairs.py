"""Typed paired-variant manifests for the ImageNet-9 protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

PAIR_TYPES: Final = ("same_rand", "same_next")
REQUIRED_PAIR_COLUMNS: Final = {
    "pair_id",
    "pair_type",
    "original_path",
    "counterfactual_path",
    "class_id",
}
WIDE_PAIR_COLUMNS: Final = {
    "pair_id",
    "true_in9_class",
    "mixed_same_path",
    "mixed_rand_path",
    "mixed_next_path",
}


def _portable_path(value: object, dataset_root: Path | None) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        return path.as_posix()
    if dataset_root is not None:
        try:
            return path.resolve().relative_to(dataset_root.resolve()).as_posix()
        except ValueError:
            pass
    for anchor in ("official", "training", "shards"):
        if anchor in path.parts:
            return Path(*path.parts[path.parts.index(anchor) :]).as_posix()
    raise ValueError("absolute image path cannot be relocated below the dataset root")


def normalize_wide_manifest(
    frame: pd.DataFrame,
    *,
    dataset_root: Path | None = None,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Expand the sealed wide Backgrounds Challenge manifest into typed pairs."""

    missing = sorted(WIDE_PAIR_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"wide paired-variant manifest is missing columns: {missing}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"paired-variant row count mismatch: expected {expected_rows}, got {len(frame)}"
        )
    rows: list[dict[str, object]] = []
    for source_row_index, source in enumerate(frame.to_dict("records")):
        for pair_type, counterfactual_column in (
            ("same_rand", "mixed_rand_path"),
            ("same_next", "mixed_next_path"),
        ):
            rows.append(
                {
                    "pair_id": f"{source['pair_id']}__{pair_type}",
                    "source_pair_id": str(source["pair_id"]),
                    "source_row_index": source_row_index,
                    "pair_type": pair_type,
                    "original_path": _portable_path(source["mixed_same_path"], dataset_root),
                    "counterfactual_path": _portable_path(
                        source[counterfactual_column], dataset_root
                    ),
                    "class_id": int(source["true_in9_class"]),
                }
            )
    return validate_pair_manifest(pd.DataFrame(rows), expected_rows=2 * len(frame))


def validate_pair_manifest(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Validate key, type, and cardinality invariants without loading images."""

    missing = sorted(REQUIRED_PAIR_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"paired-variant manifest is missing columns: {missing}")
    normalized = frame.copy()
    normalized["pair_id"] = normalized["pair_id"].astype(str)
    normalized["pair_type"] = normalized["pair_type"].astype(str)
    if normalized["pair_id"].duplicated().any():
        raise ValueError("paired-variant manifest contains duplicate pair IDs")
    invalid_types = sorted(set(normalized["pair_type"]) - set(PAIR_TYPES))
    if invalid_types:
        raise ValueError(f"paired-variant manifest has unsupported pair types: {invalid_types}")
    if expected_rows is not None and len(normalized) != expected_rows:
        raise ValueError(
            f"paired-variant row count mismatch: expected {expected_rows}, got {len(normalized)}"
        )
    for column in ("original_path", "counterfactual_path"):
        if normalized[column].astype(str).str.startswith("/").any():
            raise ValueError(f"{column} must contain dataset-relative paths")
    return normalized.sort_values("pair_id", kind="stable").reset_index(drop=True)


def load_pair_manifest(
    path: str | Path,
    *,
    expected_rows: int | None = None,
    dataset_root: Path | None = None,
) -> pd.DataFrame:
    """Load CSV or Parquet pairs and validate the public schema."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"pair manifest does not exist: {source}")
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("pair manifest must be CSV or Parquet")
    if WIDE_PAIR_COLUMNS <= set(frame.columns):
        return normalize_wide_manifest(
            frame,
            dataset_root=dataset_root,
            expected_rows=expected_rows,
        )
    return validate_pair_manifest(frame, expected_rows=expected_rows)


def shard_assignments(pair_ids: list[str], shard_size: int) -> dict[str, int]:
    """Assign sorted pair IDs to deterministic contiguous shards."""

    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair IDs must be unique before sharding")
    return {pair_id: index // shard_size for index, pair_id in enumerate(sorted(pair_ids))}


__all__ = [
    "PAIR_TYPES",
    "REQUIRED_PAIR_COLUMNS",
    "load_pair_manifest",
    "normalize_wide_manifest",
    "shard_assignments",
    "validate_pair_manifest",
]
