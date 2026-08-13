import json

import numpy as np
import pandas as pd
import pytest

from tools.crossgen.c0_runtime_diagnostic import _diagnostic_error_scopes
from tools.crossgen.legacy_controlled_export import (
    C0_CANDIDATES,
    C2_ARCHITECTURES,
    C2_TASKS,
    C2_WALL_MAPS,
    _c0_candidate_qualification_record,
    _c0_direct_summary,
    _c0_partition_candidate_audits,
    _c0_qualified_coverage,
    _c0_replay_prefix_count,
    _c0_trajectory_rows,
    _validate_c0_component_contract,
    export_c2,
)
from tools.crossgen.schema import (
    NEUTRAL_COLUMNS,
    read_trajectory_record,
    sha256_file,
    validate_trajectory_record,
)


def _write_c2_job(root, task: str, architecture: str, wall_map: int) -> None:
    seed = 7101
    job_id = f"eval__{task}__{architecture}__seed_{seed}__wall_{wall_map}"
    model_id = f"{task}__{architecture}__seed_{seed}"
    job = root / "jobs" / "evaluation" / job_id
    job.mkdir(parents=True)
    row = {
        "model_id": model_id,
        "task": task,
        "architecture": architecture,
        "seed": seed,
        "pair_id": 0,
        "base_id": 8,
        "direction": 1,
        "wall_map": wall_map,
        "object_map": 1,
        "endpoint_fact_id": 8,
        "endpoint_cf_id": 9,
        "swap_fact_id": 10,
        "swap_cf_id": 11,
        "q_endpoint_fact": 0.7,
        "q_endpoint_cf": 0.3,
        "q_swap_fact": 0.3,
        "q_swap_cf": 0.5,
        "endpoint_delta": 0.4,
        "swap_delta": -0.2,
        "correct_E": 0.4,
        "correct_C": 0.0,
        "correct_F": 0.0,
        "correct_Abs": 0.4,
        "swap_E": 0.0,
        "swap_C": 0.2,
        "swap_F": 0.0,
        "swap_Abs": 0.2,
    }
    pd.DataFrame([row]).to_parquet(job / "samples.parquet", index=False)
    receipt = {
        "architecture": architecture,
        "checkpoint_sha256": "a" * 64,
        "completed": True,
        "job_id": job_id,
        "map_semantics_valid": True,
        "model_id": model_id,
        "seed": seed,
        "task": task,
    }
    (job / "receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )


def test_export_c2_builds_registered_two_stage_units(tmp_path) -> None:
    root = tmp_path / "formal"
    for task in C2_TASKS:
        for architecture in C2_ARCHITECTURES:
            for wall_map in C2_WALL_MAPS:
                _write_c2_job(root, task, architecture, wall_map)

    output = tmp_path / "controlled_c2.parquet"
    manifest_path = tmp_path / "controlled_c2_selection.json"
    result = export_c2(root, output, selection_manifest=manifest_path)
    frame = read_trajectory_record(output)

    assert result["unit_count"] == 12
    assert result["row_count"] == 24
    assert frame["unit_id"].nunique() == 12
    grids = frame.groupby("unit_id")["stage_t"].apply(list)
    assert grids.apply(lambda values: values == [0.0, 1.0]).all()
    assert frame["quadrature_weight"].eq(0.5).all()
    assert frame["historical_M"].eq(0.4).all()
    assert frame["historical_E"].eq(0.2).all()
    assert frame["historical_C"].eq(0.1).all()
    assert frame["historical_F"].eq(0.0).all()
    assert np.allclose(frame["historical_Abs"], 0.3)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["unit_count"] == 12
    assert manifest["row_count"] == 24
    assert len(manifest["sources"]) == 12
    assert len(manifest["trajectory_record_sha256"]) == 64


def test_c0_candidates_have_registered_semantic_coverage() -> None:
    assert len(C0_CANDIDATES) == 8
    assert {item["architecture"] for item in C0_CANDIDATES} == {
        "resnet18",
        "small_vit",
    }
    assert {item["task"] for item in C0_CANDIDATES} == {
        "color_shape_xor",
        "context_gate",
    }
    assert {item["cf_map_seed"] for item in C0_CANDIDATES} == {
        "20260882",
        "20260883",
    }
    assert sum(item["expected_state"] == "active" for item in C0_CANDIDATES) == 4
    assert sum(item["expected_state"] == "null" for item in C0_CANDIDATES) == 4
    assert all(item["noise_seed"] == 20260884 for item in C0_CANDIDATES)
    identities = {(item["model_id"], item["base_id"]) for item in C0_CANDIDATES}
    assert len(identities) == len(C0_CANDIDATES)


def test_c0_candidate_qualification_excludes_fixed_failures_without_replacement(
    tmp_path,
) -> None:
    failed_errors = {75310: 0.002671495532892166, 222025: 0.0394017958}
    audits = []
    for candidate in C0_CANDIDATES:
        maximum_error = failed_errors.get(int(candidate["base_id"]), 1.0e-8)
        passed = maximum_error <= 5.0e-4
        audits.append(
            {
                "model_id": candidate["model_id"],
                "base_id": candidate["base_id"],
                "factor": candidate["factor"],
                "protocol": "linear_lambda_0.000",
                "audit": {
                    "maximum_absolute_error": maximum_error,
                    "absolute_errors": {"auc_abs_info": maximum_error},
                    "recomputed": {"auc_abs_info": 0.0},
                    "sealed": {"auc_abs_info": maximum_error},
                    "passed": passed,
                    "tolerance": 5.0e-4,
                },
            }
        )

    qualified, excluded = _c0_partition_candidate_audits(C0_CANDIDATES, audits)

    assert len(qualified) == 6
    assert len(excluded) == 2
    assert {(item["model_id"], item["base_id"]) for item in excluded} == {
        ("context_gate__resnet18__seed_3101", 75310),
        ("context_gate__small_vit__seed_3101", 222025),
    }
    assert all(
        item["reason_code"] == "UNRESOLVED_HISTORICAL_RUNTIME_METADATA"
        and "excluded_without_replacement" in item["reason"]
        and item["audit"]["absolute_errors"]
        and item["audit"]["recomputed"]
        and item["audit"]["sealed"]
        for item in excluded
    )
    assert {item["base_id"]: item["maximum_absolute_error"] for item in excluded} == (
        failed_errors
    )

    rows = [
        {
            "unit_id": f"{item['model_id']}:{item['base_id']}",
            "historical_E": 0.1,
            "historical_C": 0.1,
        }
        for item in qualified
    ]
    coverage = _c0_qualified_coverage(qualified, rows)
    assert coverage["passed"] is True
    assert coverage["observed"]["architectures"] == ["resnet18", "small_vit"]
    assert coverage["observed"]["active_count"] == 4
    assert coverage["observed"]["null_count"] == 2
    assert coverage["observed"]["counterfactual_maps"] == ["20260882", "20260883"]
    assert all(coverage["checks"].values())

    diagnostic = tmp_path / "c0_all_selection_aggregate_audit.json"
    diagnostic.write_text('{"candidate_count":8}\n', encoding="utf-8")
    diagnostic_sha256 = sha256_file(diagnostic)
    record = _c0_candidate_qualification_record(
        qualified=qualified,
        excluded=excluded,
        coverage=coverage,
        status="STRICT_AGGREGATE_QUALIFIED_WITH_EXCLUSIONS",
        diagnostic_path=diagnostic,
        diagnostic_sha256=diagnostic_sha256,
    )
    assert record["candidate_count"] == 8
    assert record["selected_count"] == 6
    assert record["excluded_count"] == 2
    assert record["status"] == "STRICT_AGGREGATE_QUALIFIED_WITH_EXCLUSIONS"
    assert record["coverage"]["passed"] is True
    assert record["diagnostic"] == {
        "path": str(diagnostic.resolve()),
        "sha256": diagnostic_sha256,
    }
    assert record["diagnostic_sha256"] == sha256_file(diagnostic)


def test_c0_direct_summary_uses_historical_components_and_alpha_trapezoid() -> None:
    class FakeSweep:
        @staticmethod
        def _component_matrices(endpoint, delta, epsilon):
            assert epsilon == 1.0e-4
            projected = np.sign(endpoint)[:, None] * delta
            return {
                "abs": np.abs(delta),
                "align": np.maximum(projected, 0.0),
                "opp": np.maximum(-projected, 0.0),
                "null": np.zeros_like(delta),
            }

    result = _c0_direct_summary(
        FakeSweep,
        endpoint_d=1.0,
        response=np.asarray([0.0, -1.0, 1.0]),
        alpha=np.asarray([0.0, 0.5, 1.0]),
        epsilon=1.0e-4,
    )

    assert result == {"M": 1.0, "E": 0.25, "C": 0.5, "F": 0.0, "Abs": 0.75}


def test_c0_component_preflight_rejects_runtime_without_null() -> None:
    class StaleSweep:
        @staticmethod
        def _component_matrices(endpoint, delta, epsilon):
            del endpoint, delta, epsilon
            raise AttributeError("EndpointAlignedResponse has no attribute 'null'")

    with pytest.raises(RuntimeError, match="EndpointAlignedResponse.null"):
        _validate_c0_component_contract(StaleSweep)


def test_c0_replay_prefix_preserves_historical_flattened_forward_batch() -> None:
    positions = np.asarray([5, 12], dtype=np.int64)

    resnet_count = _c0_replay_prefix_count(
        dynamic_count=4096,
        stack_size=12,
        historical_batch_size=2048,
        selected_positions=positions,
    )
    vit_count = _c0_replay_prefix_count(
        dynamic_count=4096,
        stack_size=12,
        historical_batch_size=1024,
        selected_positions=positions,
    )

    assert resnet_count == 171
    assert resnet_count * 12 == 2052
    assert vit_count == 86
    assert vit_count * 12 == 1032
    historical_layout = np.arange(4096 * 12).reshape(4096, 12)
    retained_layout = historical_layout[:resnet_count].reshape(-1)
    assert np.array_equal(retained_layout[:2048], np.arange(2048))
    selected_only_layout = historical_layout[positions].reshape(-1)
    assert selected_only_layout[12:24].tolist() == list(range(144, 156))
    for position in positions:
        selected_offsets = np.arange(position * 12, (position + 1) * 12)
        assert np.all(selected_offsets < 1024)


def test_c0_runtime_diagnostic_separates_formal_and_extended_error_scopes() -> None:
    sealed = {
        "endpoint_abs": 0.5,
        "auc_abs_info": 0.003,
        "auc_align_info": 0.001,
        "auc_opp_info": 0.0003,
        "auc_null_info": 0.002,
        "auc_align_alpha": 0.21,
    }
    recomputed = {
        "endpoint_abs": 0.5,
        "auc_abs_info": 0.000001,
        "auc_align_info": 0.000001,
        "auc_opp_info": 0.000001,
        "auc_null_info": 0.000001,
        "auc_align_alpha": 0.13,
    }

    scopes = _diagnostic_error_scopes(sealed, recomputed)

    assert scopes["formal_common_maximum_absolute_error"] == pytest.approx(0.002999)
    assert scopes["extended_alpha_maximum_absolute_error"] == pytest.approx(0.08)
    assert "auc_align_alpha" not in scopes["formal_common_absolute_errors"]
    assert set(scopes["extended_alpha_absolute_errors"]) == {"auc_align_alpha"}


def test_c0_rows_keep_sealed_scores_in_metadata_and_direct_response_in_schema(
    tmp_path,
) -> None:
    selection = C0_CANDIDATES[0]
    endpoint_row = pd.Series(
        {
            "map_name": "seed_20260882",
            "delta_endpoint": 0.4,
            "endpoint_active": True,
            "factual_probability": 0.7,
            "counterfactual_probability": 0.3,
        }
    )
    rows = _c0_trajectory_rows(
        selection=selection,
        endpoint_row=endpoint_row,
        response=np.asarray([0.0, 0.4]),
        alpha=np.asarray([0.0, 1.0]),
        historical={"M": 0.4, "E": 0.2, "C": 0.0, "F": 0.0, "Abs": 0.2},
        checkpoint_sha256="a" * 64,
        endpoint_path=tmp_path / "endpoint.parquet",
        sealed_summary_path=tmp_path / "summary.parquet",
        sealed_audit={"passed": True},
        position=5,
        reference_run="C0:test",
    )
    frame = validate_trajectory_record(pd.DataFrame(rows, columns=NEUTRAL_COLUMNS))
    metadata = json.loads(frame["metadata_json"].iloc[0])

    assert frame["stage_r"].tolist() == [0.0, 0.4]
    assert frame["endpoint_score_plus"].isna().all()
    assert frame["stage_score_plus"].isna().all()
    assert metadata["sealed_endpoint_score_plus"] == 0.7
    assert metadata["quantity_provenance"]["stage_r"].startswith("regenerated")
