from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from decaf.core.manifests import sha256_file
from decaf.core.receipts import write_member_receipt
from decaf.experiments.attribution.evaluate import (
    _validate_completed_member,
    atomic_parquet,
    oracle_member,
)


def test_oracle_primary_scores_are_unsigned_e_for_negative_endpoints() -> None:
    frame = oracle_member({"scope": "oracle", "method_id": "decaf_5"}, None)

    saw_negative_endpoint_with_positive_e = False
    for row in frame.itertuples(index=False):
        patch_scores = np.asarray(row.patch_scores, dtype=np.float64)
        unsigned_e = np.asarray(row.decaf_E, dtype=np.float64)
        endpoint = np.asarray(row.endpoint_effects, dtype=np.float64)
        np.testing.assert_allclose(patch_scores, unsigned_e, rtol=0.0, atol=0.0)
        assert bool((patch_scores >= 0.0).all())
        saw_negative_endpoint_with_positive_e |= bool(
            ((endpoint < 0.0) & (patch_scores > 0.0)).any()
        )

    assert saw_negative_endpoint_with_positive_e


def test_resume_rejects_stale_downstream_dependency_after_target_recompute(
    tmp_path,
) -> None:
    target_output = tmp_path / "raw/members/target.parquet"
    target_receipt = tmp_path / "receipts/members/target.json"
    quality_output = tmp_path / "raw/members/quality.parquet"
    quality_receipt = tmp_path / "receipts/members/quality.json"
    atomic_parquet(pd.DataFrame({"value": [1.0]}), target_output)
    first_hash = sha256_file(target_output)
    dependency = {
        "member_id": "target",
        "job_sha256": "a" * 64,
        "output_path": "raw/members/target.parquet",
        "receipt_path": "receipts/members/target.json",
        "relationship": "shared_deletion_or_heldout_target_shard",
    }
    old_record = {
        "member_id": "target",
        "job_sha256": "a" * 64,
        "output_sha256": first_hash,
        "relationship": dependency["relationship"],
    }
    atomic_parquet(pd.DataFrame({"value": [2.0]}), quality_output)
    write_member_receipt(
        quality_receipt,
        "quality",
        "completed",
        details={"dependency_outputs": [old_record]},
    )

    atomic_parquet(pd.DataFrame({"value": [3.0]}), target_output)
    current_hash = sha256_file(target_output)
    assert current_hash != first_hash
    write_member_receipt(
        target_receipt,
        "target",
        "completed",
        details={"job_sha256": "a" * 64, "output_sha256": current_hash},
    )
    context = SimpleNamespace(path=tmp_path)
    quality_job = {
        "member_id": "quality",
        "output_path": "raw/members/quality.parquet",
        "receipt_path": "receipts/members/quality.json",
        "depends_on": [dependency],
    }
    with pytest.raises(RuntimeError, match="dependency lineage drifted"):
        _validate_completed_member(context, quality_job, {})
