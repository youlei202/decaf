"""Dataset-root resolution and manifest fingerprint checks for ImageNet-9."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from decaf.experiments.imagenet9.pairs import validate_pair_manifest


def sha256_file(path: Path) -> str:
    """Hash a manifest without loading image data."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_root(
    config: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the dataset below the configured public environment variable."""

    data = config["data"]
    variable = str(data["root_environment"])
    values = os.environ if environment is None else environment
    configured = values.get(variable)
    if not configured:
        raise RuntimeError(f"set {variable} to the external dataset root")
    configured_root = Path(configured).expanduser()
    nested = configured_root / str(data["dataset_subdirectory"])
    if (nested / "manifests").is_dir():
        return nested.resolve()
    if (configured_root / "manifests").is_dir():
        return configured_root.resolve()
    return nested.resolve()


def validate_split_fingerprints(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the sealed score/deep split fingerprints."""

    data = config["data"]
    checked: list[dict[str, Any]] = []
    for key, relative in (
        ("paired_manifest_sha256", "manifests/paired_variants.parquet"),
        ("score_split_sha256", "manifests/score_split.parquet"),
        ("deep_split_sha256", "manifests/deep_split.parquet"),
    ):
        expected = str(data[key])
        if expected == "smoke-fixture":
            checked.append({"path": relative, "status": "fixture"})
            continue
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"registered ImageNet-9 split is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"split fingerprint mismatch for {relative}")
        checked.append({"path": relative, "sha256": actual, "status": "verified"})
    return {"schema_version": 1, "splits": checked}


def smoke_pair_frame() -> pd.DataFrame:
    """Create a deterministic path/schema fixture without restricted images."""

    rows = [
        {
            "pair_id": f"smoke-{index:02d}",
            "pair_type": "same_rand",
            "original_path": f"original/{index:02d}.png",
            "counterfactual_path": f"mixed_rand/{index:02d}.png",
            "class_id": index % 2,
        }
        for index in range(4)
    ]
    return validate_pair_manifest(pd.DataFrame(rows), expected_rows=4)


__all__ = [
    "resolve_dataset_root",
    "sha256_file",
    "smoke_pair_frame",
    "validate_split_fingerprints",
]
