"""Tests for reusable statistical and provenance primitives."""

from __future__ import annotations

import json

import numpy as np
import pytest

from decaf.core.bootstrap import bootstrap_mean, paired_bootstrap_mean_difference
from decaf.core.manifests import (
    atomic_write_json,
    build_file_manifest,
    verify_file_manifest,
)
from decaf.core.metrics import pearson_correlation, safe_ratio
from decaf.core.receipts import (
    aggregate_global_status,
    build_member_receipt,
    finalize_global_receipt,
    write_member_receipt,
)


def test_bootstrap_is_deterministic_and_paired() -> None:
    values = np.arange(12, dtype=np.float32)
    first = bootstrap_mean(values, n_resamples=200, seed=19, batch_size=31)
    second = bootstrap_mean(values, n_resamples=200, seed=19, batch_size=17)

    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)
    paired = paired_bootstrap_mean_difference(
        values + 2.0,
        values,
        n_resamples=100,
        seed=3,
    )
    assert float(paired.estimate) == pytest.approx(2.0)
    assert paired.as_dict()["estimate"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_resamples": 1},
        {"confidence_level": 1.0},
        {"seed": -1},
        {"batch_size": 0},
    ],
)
def test_bootstrap_validation_errors(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        bootstrap_mean([1.0, 2.0], **kwargs)


def test_metrics_validate_degenerate_inputs() -> None:
    np.testing.assert_array_equal(
        safe_ratio([2.0, 1.0], [2.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    assert pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="constant"):
        pearson_correlation([1.0, 1.0], [2.0, 3.0])


def test_atomic_manifest_round_trip_and_tamper_detection(tmp_path: object) -> None:
    root = tmp_path  # pytest supplies pathlib.Path
    data_path = root / "data.txt"  # type: ignore[operator]
    data_path.write_text("sealed\n", encoding="utf-8")
    manifest = build_file_manifest(["data.txt"], root=root)
    manifest_path = root / "manifest.json"  # type: ignore[operator]
    atomic_write_json(manifest_path, manifest)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert verify_file_manifest(manifest, root=root)["passed"]
    assert not list(root.glob(".manifest.json.*.tmp"))

    data_path.write_text("changed\n", encoding="utf-8")
    audit = verify_file_manifest(manifest, root=root)
    assert not audit["passed"]
    assert any(error.startswith(("size:", "sha256:")) for error in audit["errors"])


def test_atomic_member_and_terminal_global_receipts(tmp_path: object) -> None:
    root = tmp_path
    completed = build_member_receipt("required-a", "completed")
    running = build_member_receipt("required-b", "running")
    optional_failure = build_member_receipt(
        "optional-c",
        "failed",
        optional=True,
        error="optional method unavailable",
    )

    assert (
        aggregate_global_status(
            {"required-a": completed, "required-b": running},
            expected_members=("required-a", "required-b"),
        )
        == "running"
    )
    assert (
        aggregate_global_status(
            {"required-a": completed, "required-b": running},
            expected_members=("required-a", "required-b"),
            all_processes_exited=True,
        )
        == "partial"
    )
    assert (
        aggregate_global_status(
            {"required-a": completed, "optional-c": optional_failure},
            expected_members=("required-a", "optional-c"),
            optional_members=("optional-c",),
            all_processes_exited=True,
        )
        == "completed_with_optional_failures"
    )

    member_path = root / "member.json"  # type: ignore[operator]
    write_member_receipt(member_path, "required-a", "completed")
    assert json.loads(member_path.read_text(encoding="utf-8"))["status"] == "completed"
    global_path = root / "global.json"  # type: ignore[operator]
    finalize_global_receipt(
        global_path,
        "run-1",
        {"required-a": completed, "required-b": running},
        expected_members=("required-a", "required-b"),
    )
    global_payload = json.loads(global_path.read_text(encoding="utf-8"))
    assert global_payload["status"] == "partial"
    assert global_payload["status"] != "running"


def test_receipt_validation_errors() -> None:
    with pytest.raises(ValueError, match="requires"):
        build_member_receipt("worker", "failed")
    with pytest.raises(ValueError, match="subset"):
        aggregate_global_status(
            {"worker": build_member_receipt("worker", "completed")},
            expected_members=("worker",),
            optional_members=("unknown",),
        )
