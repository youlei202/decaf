"""Exact support and receipt checks for externally materialized ImageNet-9 jobs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from decaf.experiments.common import RunContext
from decaf.experiments.imagenet9 import evaluate


def _context(path: Path) -> RunContext:
    for relative in ("manifests", "raw", "checkpoints", "receipts/members"):
        (path / relative).mkdir(parents=True, exist_ok=True)
    return RunContext(
        experiment="imagenet9",
        profile="paper",
        stage="compute",
        path=path,
        config={"experiment_grid": {"alpha": [0.0, 0.5, 1.0]}},
        workers=1,
        resume=False,
    )


def _pair_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["pair0__same_rand", "pair0__same_next"],
            "pair_type": ["same_rand", "same_next"],
            "source_pair_id": ["pair0", "pair0"],
            "source_row_index": [0, 0],
        }
    )


def test_materialized_members_bind_dependencies_receipts_and_exact_support(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    pairs = _pair_manifest()
    pairs.to_csv(context.path / "manifests" / "score_pairs.csv", index=False)
    pairs.to_csv(context.path / "manifests" / "deep_pairs.csv", index=False)
    train = {
        "job_id": "train__model",
        "kind": "finetune",
        "model_id": "model",
        "depends_on": [],
        "dependency_outputs": [],
        "config_sha256": "config",
        "dataset_split_sha256": "split",
        "output": "checkpoints/model/best.pt",
        "receipt": "receipts/members/train__model.json",
    }
    scan = {
        "job_id": "scan__model__blend__000",
        "kind": "decaf_scan",
        "model_id": "model",
        "reveal_path": "blend",
        "source_row_start": 0,
        "source_row_stop": 1,
        "depends_on": [train["job_id"]],
        "dependency_outputs": [str(train["output"])],
        "config_sha256": "config",
        "dataset_split_sha256": "split",
        "output": "raw/scans/scan.parquet",
        "receipt": "receipts/members/scan.json",
    }
    baseline = {
        "job_id": "baseline__model__method__000",
        "kind": "saliency_baseline",
        "model_id": "model",
        "method_id": "method",
        "source_row_start": 0,
        "source_row_stop": 1,
        "depends_on": [train["job_id"]],
        "dependency_outputs": [str(train["output"])],
        "config_sha256": "config",
        "dataset_split_sha256": "deep-split",
        "output": "raw/baselines/baseline.parquet",
        "receipt": "receipts/members/baseline.json",
    }
    plan = {"jobs": [train, scan, baseline]}

    train_output = context.path / str(train["output"])
    train_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.write_bytes(b"sealed checkpoint")
    evaluate._write_member_receipt(context, train, train_output, 1)

    response_rows = []
    for pair_id, pair_type in zip(pairs["pair_id"], pairs["pair_type"], strict=True):
        for stage, alpha in enumerate((0.0, 0.5, 1.0)):
            response_rows.append(
                {
                    "pair_id": pair_id,
                    "pair_type": pair_type,
                    "model_id": "model",
                    "reveal_path": "blend",
                    "stage_index": stage,
                    "alpha": alpha,
                    "response": alpha,
                }
            )
    response = pd.DataFrame(response_rows)
    scan_output = context.path / str(scan["output"])
    evaluate._atomic_parquet(scan_output, response)
    evaluate._write_member_receipt(context, scan, scan_output, len(response))

    baseline_frame = pd.DataFrame(
        {
            "pair_id": pairs["pair_id"],
            "pair_type": pairs["pair_type"],
            "model_id": "model",
            "method_id": "method",
            "score": [0.1, 0.2],
        }
    )
    baseline_output = context.path / str(baseline["output"])
    evaluate._atomic_parquet(baseline_output, baseline_frame)
    evaluate._write_member_receipt(context, baseline, baseline_output, len(baseline_frame))

    responses, baselines = evaluate._load_materialized_members(context, plan)
    assert len(responses) == 6
    assert len(baselines) == 2

    train_output.write_bytes(b"different checkpoint")
    evaluate._write_member_receipt(context, train, train_output, 1)
    with pytest.raises(ValueError, match="planned output"):
        evaluate._load_materialized_members(context, plan)
    train_output.write_bytes(b"sealed checkpoint")
    evaluate._write_member_receipt(context, train, train_output, 1)

    baseline_frame.loc[0, "pair_type"] = "wrong"
    evaluate._atomic_parquet(baseline_output, baseline_frame)
    evaluate._write_member_receipt(context, baseline, baseline_output, len(baseline_frame))
    with pytest.raises(ValueError, match="wrong pair labels"):
        evaluate._load_materialized_members(context, plan)

    evaluate._atomic_parquet(baseline_output, baselines)
    evaluate._write_member_receipt(context, baseline, baseline_output, len(baselines))
    irregular_rows = []
    for pair_id, pair_type, alpha_grid in (
        ("pair0__same_rand", "same_rand", (0.0, 0.25, 0.75, 1.0)),
        ("pair0__same_next", "same_next", (0.0, 1.0)),
    ):
        for stage, alpha in enumerate(alpha_grid):
            irregular_rows.append(
                {
                    "pair_id": pair_id,
                    "pair_type": pair_type,
                    "model_id": "model",
                    "reveal_path": "blend",
                    "stage_index": stage,
                    "alpha": alpha,
                    "response": alpha,
                }
            )
    irregular = pd.DataFrame(irregular_rows)
    evaluate._atomic_parquet(scan_output, irregular)
    evaluate._write_member_receipt(context, scan, scan_output, len(irregular))
    with pytest.raises(ValueError, match="configured (stage|alpha) grid"):
        evaluate._load_materialized_members(context, plan)
