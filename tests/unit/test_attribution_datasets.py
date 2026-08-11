from __future__ import annotations

import numpy as np
import pandas as pd

from decaf.experiments.attribution.datasets.funnybirds import validate_support
from decaf.experiments.attribution.datasets.idsds import (
    PATCH_COUNT,
    aggregate_patch_scores,
    grid_patch_masks,
    validate_manifest,
)
from decaf.experiments.attribution.datasets.partimagenet import (
    validate_part_attribution,
)


def test_idsds_grid_is_an_exact_sixteen_patch_partition() -> None:
    masks = grid_patch_masks(8, 12)
    assert masks.shape == (PATCH_COUNT, 8, 12)
    np.testing.assert_array_equal(masks.sum(axis=0), np.ones((8, 12)))
    attribution = np.arange(96, dtype=np.float64).reshape(8, 12)
    scores = aggregate_patch_scores(attribution)
    assert scores.shape == (1, PATCH_COUNT)
    assert scores.sum() == attribution.sum()


def test_dataset_schemas_are_cpu_safe_and_reject_duplicate_keys() -> None:
    manifest = pd.DataFrame(
        {
            "image_id": ["a"],
            "label": [3],
            "source_shard": ["s"],
            "row_index": [0],
            "image_filename": ["a.jpg"],
            "wnid": ["n00000003"],
        }
    )
    assert len(validate_manifest(manifest, expected_rows=1)) == 1
    support = pd.DataFrame(
        {
            "dataset": ["funnybirds"],
            "model": ["funnybirds_resnet50"],
            "image_id": ["a"],
            "correctly_classified": [True],
            "included": [True],
            "exclusion_reason": [""],
        }
    )
    assert len(validate_support(support)) == 1
    parts = pd.DataFrame(
        {
            "dataset": ["partimagenet"],
            "image_id": ["a"],
            "model": ["resnet50"],
            "method": ["decaf_5"],
            "part_group": ["head"],
            "attribution_score": [0.25],
        }
    )
    assert len(validate_part_attribution(parts)) == 1
