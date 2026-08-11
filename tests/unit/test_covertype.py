"""Unit tests for the Covertype formal plan, mechanisms, and analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decaf.experiments.covertype.analyze import canonical_fragility_correlation
from decaf.experiments.covertype.data import prepare_dataset
from decaf.experiments.covertype.evaluate import (
    FORMAL_MODEL_FAMILIES,
    FORMAL_SEEDS,
    build_formal_plan,
    formal_specs,
)
from decaf.experiments.covertype.mechanisms import (
    legal_query_features,
    realize_module_c,
    realize_module_f,
)


def test_formal_plan_proves_exact_135_model_cartesian_product() -> None:
    plan = build_formal_plan()
    specs = formal_specs()
    assert plan["counts"] == {
        "module_c_models": 90,
        "module_f_models": 45,
        "total_models": 135,
        "model_families": 5,
        "seeds": 3,
    }
    assert len(specs) == len({spec.model_id for spec in specs}) == 135
    assert sum(spec.module == "C" for spec in specs) == 90
    assert sum(spec.module == "F" for spec in specs) == 45
    assert {spec.model_family for spec in specs} == set(FORMAL_MODEL_FAMILIES)
    assert {spec.seed for spec in specs} == set(FORMAL_SEEDS)
    assert {(spec.regime, spec.strength) for spec in specs if spec.module == "C"} == {
        (mechanism, strength)
        for mechanism in ("direct", "gate", "invert")
        for strength in (0.75, 0.95)
    }
    assert {spec.regime for spec in specs if spec.module == "F"} == {
        "robust",
        "mild",
        "fragile",
    }
    assert plan["audit"] == {
        "unique_model_ids": True,
        "module_c_count": 90,
        "module_f_count": 45,
    }
    assert plan["baseline_plan"]["tree_shap_interaction_models"] == 54
    assert plan["baseline_plan"]["tree_shap_interaction_shard_jobs"] == 216
    assert plan["baseline_plan"]["optional_kernel_shap_models"] == 45


def test_contextual_mechanisms_share_registered_endpoint_streams() -> None:
    y = np.tile(np.array([0, 1], dtype=np.int8), 1000)
    direct = realize_module_c(y, strength=0.75, mechanism="direct", seed=7701, split="test")
    gate = realize_module_c(y, strength=0.75, mechanism="gate", seed=7701, split="test")
    invert = realize_module_c(y, strength=0.75, mechanism="invert", seed=7701, split="test")
    np.testing.assert_array_equal(direct.context, gate.context)
    np.testing.assert_array_equal(direct.context, invert.context)
    np.testing.assert_array_equal(direct.endpoint_factor, gate.endpoint_factor)
    np.testing.assert_array_equal(direct.endpoint_factor, invert.endpoint_factor)
    assert np.mean(direct.alternate_factor == (2 * y - 1)) == pytest.approx(0.75, abs=0.04)
    assert np.mean(gate.alternate_factor == (2 * y - 1)) == pytest.approx(0.50, abs=0.04)
    assert np.mean(invert.alternate_factor == (2 * y - 1)) == pytest.approx(0.25, abs=0.04)


def test_fragility_regimes_share_endpoint_and_increase_alternate_alignment() -> None:
    y = np.tile(np.array([0, 1], dtype=np.int8), 2000)
    values = [
        realize_module_f(y, regime=regime, seed=7701, split="test")
        for regime in ("robust", "mild", "fragile")
    ]
    for candidate in values[1:]:
        np.testing.assert_array_equal(candidate.context, values[0].context)
        np.testing.assert_array_equal(candidate.endpoint_factor, values[0].endpoint_factor)
    signed_y = 2 * y - 1
    rates = [float(np.mean(value.alternate_factor == signed_y)) for value in values]
    assert rates == sorted(rates)
    assert rates[0] == pytest.approx(0.50, abs=0.03)
    assert rates[1] == pytest.approx(0.70, abs=0.03)
    assert rates[2] == pytest.approx(0.95, abs=0.03)


def test_legal_queries_reject_interpolated_contexts() -> None:
    X = np.zeros((3, 54), dtype=np.float64)
    query = legal_query_features(X, context=1, factor=-1)
    assert query.shape == (3, 56)
    np.testing.assert_array_equal(query[:, -2], np.ones(3))
    np.testing.assert_array_equal(query[:, -1], -np.ones(3))
    with pytest.raises(ValueError, match="must be -1 or"):
        legal_query_features(X, context=0, factor=1)


def test_canonical_analysis_names_and_correlates_endpoint_null_outcome() -> None:
    frame = pd.DataFrame(
        {
            "F": [0.01, 0.02, 0.04, 0.08],
            "null_context_prediction_change_rate": [0.0, 0.1, 0.4, 0.9],
        }
    )
    result = canonical_fragility_correlation(frame)
    assert result["expression"] == ("correlation(F, null_context_prediction_change_rate)")
    assert result["component"] == "F"
    assert result["outcome"] == "null_context_prediction_change_rate"
    assert result["spearman"] == pytest.approx(1.0)


def _real_cache_config(*, allow_fixture_fallback: bool = False) -> dict[str, object]:
    return {
        "data": {
            "source": "sklearn_covtype_cache",
            "allow_fixture_fallback": allow_fixture_fallback,
            "cache": {
                "root_env": "DECAF_DATA_ROOT",
                "archive": "covertype.npz",
                "manifest": "covertype.manifest.json",
                "archive_sha256": "0" * 64,
                "manifest_sha256": "0" * 64,
                "logical_fingerprint": "0" * 64,
                "fixed_shard_fingerprint": "0" * 64,
                "fixed_shard_rows": {"train": 4, "validation": 4, "test": 4},
            },
        }
    }


def test_real_cache_mode_requires_explicit_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DECAF_DATA_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="DECAF_DATA_ROOT.*synthetic fallback is disabled"):
        prepare_dataset(tmp_path / "run", _real_cache_config())


def test_real_cache_mode_rejects_fixture_fallback_even_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DECAF_DATA_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="forbids synthetic fixture fallback"):
        prepare_dataset(tmp_path / "run", _real_cache_config(allow_fixture_fallback=True))


def test_real_cache_mode_rejects_changed_archive_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DECAF_DATA_ROOT", str(tmp_path))
    (tmp_path / "covertype.npz").write_bytes(b"not the pinned cache")
    (tmp_path / "covertype.manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache SHA-256 mismatch"):
        prepare_dataset(tmp_path / "run", _real_cache_config())
