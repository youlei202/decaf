from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decaf.experiments.common import load_profile
from decaf.experiments.controlled.data import (
    canonical_factor,
    deterministic_sample_ids,
    exact_counterfactual_rows,
    object_color_map,
    resolve_shapes3d_root,
    validate_factor_table,
    validate_shapes3d_asset,
    wall_color_map,
)
from decaf.experiments.controlled.models import (
    expected_base_models,
    expected_contradiction_models,
    validate_c0_manifest,
    validate_c1_manifest,
    validate_c2_model_grid,
)
from decaf.experiments.controlled.train import selected_c1_checkpoints


def test_dataset_resolution_is_explicit_and_manifest_is_sanitized(tmp_path: Path) -> None:
    root = tmp_path / "datasets" / "3d_shapes"
    root.mkdir(parents=True)
    assert (
        resolve_shapes3d_root(environment={"DECAF_DATA_ROOT": str(tmp_path / "datasets")}) == root
    )
    with pytest.raises(RuntimeError, match="DECAF_DATA_ROOT"):
        resolve_shapes3d_root(environment={})
    source = root / "3dshapes.h5"
    source.write_bytes(b"tiny-loader-fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    asset = validate_shapes3d_asset(
        root, expected_sha256=digest, expected_bytes=source.stat().st_size
    )
    record = asset.public_record()
    assert record["logical_root"] == "${DECAF_DATA_ROOT}/3d_shapes"
    assert str(tmp_path) not in str(record)


def test_factor_loader_and_exact_counterfactual_oracle() -> None:
    factors = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
        ],
        dtype=np.int64,
    )
    table = validate_factor_table(factors, expected_rows=3)
    mapped = exact_counterfactual_rows(table, [0, 1, 2], "orientation", seed=17)
    changed = mapped != table
    assert np.all(changed.sum(axis=1) == 1)
    assert np.all(changed[:, 5])
    assert canonical_factor("orientation") == "object_orientation"
    assert np.array_equal(object_color_map(object_color_map(np.arange(10), 1), 1), np.arange(10))
    assert np.array_equal(wall_color_map(wall_color_map(np.arange(10), 2), 2), np.arange(10))


def test_deterministic_sample_ids_respect_exclusions() -> None:
    left = deterministic_sample_ids(20, 8, seed=11, excluded=[0, 1, 2])
    right = deterministic_sample_ids(20, 8, seed=11, excluded=[0, 1, 2])
    assert np.array_equal(left, right)
    assert not set(left) & {0, 1, 2}


def test_frozen_model_manifest_validators_lock_all_three_grids() -> None:
    base_rows = []
    for record in expected_base_models():
        base_rows.append(
            {
                "model_id": record.model_id,
                "task_name": record.task,
                "architecture": record.architecture,
                "seed": record.seed,
                "checkpoint_path": f"{record.model_id}.pt",
                "checkpoint_sha256": "a" * 64,
                "probability_cache_path": f"{record.model_id}.npy",
                "probability_cache_sha256": "b" * 64,
                "qualified": True,
                "available": True,
            }
        )
    validated = validate_c0_manifest(pd.DataFrame(base_rows))
    assert len(validated) == 30
    assert validated.attrs["no_retraining"] is True

    section = load_profile("controlled", "paper")["endpoint_behavior"]
    selected = selected_c1_checkpoints(section)
    c1 = pd.DataFrame(
        {
            "model_id": [row["model_id"] for row in selected],
            "module": [row["module"] for row in selected],
            "variant": [row["variant"] for row in selected],
            "architecture": [row["architecture"] for row in selected],
            "seed": [row["seed"] for row in selected],
            "checkpoint_path": [f"{row['model_id']}.pt" for row in selected],
            "checkpoint_sha256": ["c" * 64] * len(selected),
            "selected_for_b200": [True] * len(selected),
        }
    )
    assert len(validate_c1_manifest(c1)) == 88

    c2 = pd.DataFrame([record.as_dict() for record in expected_contradiction_models()])
    assert len(validate_c2_model_grid(c2)) == 30
