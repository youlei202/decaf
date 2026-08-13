from __future__ import annotations

import json

import numpy as np
import pandas as pd

from decaf.core.trajectories import trajectory_scores
from tools.crossgen.compare_core import compare_record
from tools.crossgen.schema import NEUTRAL_COLUMNS, trapezoid_weights, write_trajectory_record


def _record() -> pd.DataFrame:
    rows = []
    specs = [
        ("active", np.array([0.0, 0.2, -0.1, 0.3]), 0.3),
        ("null", np.array([0.0, -0.01, 0.015, 0.005]), 0.005),
    ]
    grid = np.linspace(0.0, 1.0, 4)
    weights = trapezoid_weights(grid)
    for unit_id, response, endpoint in specs:
        historical = trajectory_scores(grid, response, endpoint, 0.02)
        for stage_index, (stage_t, weight, stage_r) in enumerate(
            zip(grid, weights, response, strict=True)
        ):
            rows.append(
                {
                    "experiment_family": "test",
                    "reference_run": "T",
                    "unit_id": unit_id,
                    "model_id": "model",
                    "checkpoint_sha256": "a" * 64,
                    "sample_or_pair_id": unit_id,
                    "factor_or_part_id": "factor",
                    "counterfactual_map": "map",
                    "protocol": "path",
                    "protocol_seed": 7,
                    "stage_index": stage_index,
                    "stage_t": stage_t,
                    "quadrature_weight": weight,
                    "endpoint_epsilon": 0.02,
                    "endpoint_score_plus": endpoint + 0.4,
                    "endpoint_score_minus": 0.4,
                    "endpoint_d": endpoint,
                    "stage_score_plus": stage_r + 0.4,
                    "stage_score_minus": 0.4,
                    "stage_r": stage_r,
                    **{
                        f"historical_{name}": float(np.asarray(historical[name]))
                        for name in ("M", "E", "C", "F", "Abs")
                    },
                    "metadata_json": json.dumps(
                        {
                            "historical_gate": abs(endpoint) >= 0.02,
                            "historical_orientation": (
                                int(np.sign(endpoint)) if abs(endpoint) >= 0.02 else 0
                            ),
                            "identity_match": True,
                            "current_model_id": "model",
                            "current_checkpoint_sha256": "a" * 64,
                            "current_sample_or_pair_id": unit_id,
                            "current_factor_or_part_id": "factor",
                            "current_counterfactual_map": "map",
                            "current_protocol": "path",
                        }
                    ),
                }
            )
    return pd.DataFrame(rows, columns=NEUTRAL_COLUMNS)


def test_compare_record_exact_current_core(tmp_path):
    trajectory = tmp_path / "trajectory.parquet"
    output = tmp_path / "comparison.csv"
    write_trajectory_record(_record(), trajectory)

    result = compare_record(trajectory, output)

    assert output.is_file()
    assert result["summary"]["unit_count"] == 2
    assert result["summary"]["tier_a_fraction"] == 1.0
    assert result["summary"]["tier_b_fraction"] == 0.0
    assert result["summary"]["hard_mismatch_fraction"] == 0.0
    assert result["summary"]["numeric_identity_fraction"] == 1.0


def test_schema_rejects_response_disagreement(tmp_path):
    frame = _record()
    frame.loc[0, "stage_r"] += 0.1

    with np.testing.assert_raises_regex(ValueError, "disagrees"):
        write_trajectory_record(frame, tmp_path / "bad.parquet")


def test_schema_rejects_noncontiguous_stages_and_wrong_weights(tmp_path):
    frame = _record()
    frame.loc[frame["unit_id"].eq("active") & frame["stage_index"].eq(2), "stage_index"] = 4
    with np.testing.assert_raises_regex(ValueError, "contiguous from zero"):
        write_trajectory_record(frame, tmp_path / "bad-stages.parquet")

    frame = _record()
    frame.loc[frame["unit_id"].eq("active"), "quadrature_weight"] = 0.25
    with np.testing.assert_raises_regex(ValueError, "differs from the grid"):
        write_trajectory_record(frame, tmp_path / "bad-weights.parquet")


def test_compare_record_fails_closed_on_missing_current_identity(tmp_path):
    frame = _record()
    frame["metadata_json"] = frame["metadata_json"].map(
        lambda raw: json.dumps(
            {
                key: value
                for key, value in json.loads(raw).items()
                if key != "current_checkpoint_sha256"
            }
        )
    )
    trajectory = tmp_path / "trajectory.parquet"
    output = tmp_path / "comparison.csv"
    write_trajectory_record(frame, trajectory)

    result = compare_record(trajectory, output)

    assert result["summary"]["identity_agreement"] == 0.0
    assert result["summary"]["hard_mismatch_fraction"] == 1.0


def test_schema_rejects_metadata_that_changes_within_a_unit(tmp_path):
    frame = _record()
    row = frame.index[frame["unit_id"].eq("active")][0]
    payload = json.loads(frame.loc[row, "metadata_json"])
    payload["identity_match"] = False
    frame.loc[row, "metadata_json"] = json.dumps(payload)

    with np.testing.assert_raises_regex(ValueError, "metadata_json changes within unit"):
        write_trajectory_record(frame, tmp_path / "bad-metadata.parquet")
