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
    validate_c1_checkpoint_bundle,
    validate_c1_manifest,
    validate_c2_checkpoint_bundle,
    validate_c2_model_grid,
)
from decaf.experiments.controlled.train import (
    c1_checkpoint_producers,
    selected_c1_checkpoints,
)


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


def test_c1_c2_checkpoint_bundles_verify_producers_and_bytes(tmp_path: Path) -> None:
    config = load_profile("controlled", "smoke")
    section = config["endpoint_behavior"]
    selected = selected_c1_checkpoints(section)
    producers = c1_checkpoint_producers(section)
    c1_rows = []
    for row in selected:
        checkpoint = tmp_path / f"{row['model_id']}.pt"
        checkpoint.write_bytes(str(row["model_id"]).encode("utf-8"))
        c1_rows.append(
            {
                "model_id": row["model_id"],
                "module": row["module"],
                "variant": row["variant"],
                "architecture": row["architecture"],
                "seed": row["seed"],
                "checkpoint_path": checkpoint.name,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "selected_for_b200": True,
                "producer_member_id": producers[str(row["model_id"])],
            }
        )
    c1_manifest = tmp_path / "c1.csv"
    pd.DataFrame(c1_rows).to_csv(c1_manifest, index=False)
    c1_validated = validate_c1_checkpoint_bundle(c1_manifest, selected)
    assert len(c1_validated) == 3
    assert c1_validated.attrs["byte_identity_verified"] is True
    validate_c1_checkpoint_bundle(
        c1_manifest,
        selected,
        expected_registry_sha256=c1_validated.attrs["logical_registry_sha256"],
    )
    with pytest.raises(ValueError, match="registry SHA256 mismatch"):
        validate_c1_checkpoint_bundle(
            c1_manifest,
            selected,
            expected_registry_sha256="0" * 64,
        )

    contradiction = config["contradiction"]
    c2_registry = expected_contradiction_models(
        tuple(contradiction["tasks"]),
        tuple(contradiction["architectures"]),
        tuple(contradiction["seeds"]),
    )
    c2_rows = []
    for record in c2_registry:
        checkpoint = tmp_path / f"{record.model_id}.pt"
        checkpoint.write_bytes(record.model_id.encode("utf-8"))
        c2_rows.append(
            {
                **record.as_dict(),
                "checkpoint_path": checkpoint.name,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "producer_member_id": f"c2_train__{record.model_id}",
            }
        )
    c2_manifest = tmp_path / "c2.csv"
    pd.DataFrame(c2_rows).to_csv(c2_manifest, index=False)
    c2_validated = validate_c2_checkpoint_bundle(c2_manifest, c2_registry)
    assert len(c2_validated) == 1
    assert c2_validated.attrs["byte_identity_verified"] is True
    validate_c2_checkpoint_bundle(
        c2_manifest,
        c2_registry,
        expected_registry_sha256=c2_validated.attrs["logical_registry_sha256"],
    )

    (tmp_path / c1_rows[0]["checkpoint_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_c1_checkpoint_bundle(c1_manifest, selected)
