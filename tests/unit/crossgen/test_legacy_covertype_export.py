from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools.crossgen import legacy_covertype_export as exporter
from tools.crossgen.schema import trapezoid_weights


def test_registered_grid_trapezoid_weights() -> None:
    weights = trapezoid_weights(np.linspace(0.0, 1.0, 11))
    assert np.allclose(weights[[0, -1]], 0.05)
    assert np.allclose(weights[1:-1], 0.1)
    assert np.isclose(weights.sum(), 1.0)


def test_stable_selection_is_unique_and_repeatable() -> None:
    source_indices = np.asarray([19, 3, 101, 7, 23, 11], dtype=np.int64)
    first = exporter._stable_positions(source_indices, 4)
    second = exporter._stable_positions(source_indices.copy(), 4)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 4
    assert set(first).issubset(set(range(len(source_indices))))


def test_sealed_source_binding_validates_complete_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_names = (
        "__init__",
        "behaviors",
        "compatibility",
        "config",
        "data",
        "decaf",
        "mechanisms",
        "models",
    )
    payloads = {
        f"code/src/cmr/decaf_covertype_v1/{name}.py": f"# {name}\n".encode()
        for name in module_names
    }
    files = [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in sorted(payloads.items())
    ]
    manifest = {
        "schema_version": 1,
        "namespace": "decaf_covertype_v1",
        "lightweight": True,
        "files": files,
    }
    archive = tmp_path / "covertype.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            exporter.HISTORICAL_PACKAGE_MANIFEST_MEMBER,
            json.dumps(manifest, sort_keys=True),
        )
        for path, payload in payloads.items():
            stream.writestr(path, payload)

    monkeypatch.setattr(exporter, "HISTORICAL_PACKAGE", archive)
    monkeypatch.setattr(
        exporter,
        "HISTORICAL_PACKAGE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    exporter._historical_source_binding.cache_clear()
    try:
        binding = exporter._historical_source_binding()
        materialized = exporter._materialize_historical_source(tmp_path / "source")
    finally:
        exporter._historical_source_binding.cache_clear()

    assert binding["path"] == str(archive.resolve())
    assert binding["namespace_member_count"] == len(payloads)
    assert set(binding["required_modules"]) == set(module_names)
    assert binding["archive_inventory_verified"] is True
    assert binding["origin_verified"] is False
    shim = Path(materialized["parent_package_shim"]["path"])
    assert shim.read_bytes() == exporter.PARENT_PACKAGE_SHIM
    assert materialized["parent_package_shim"]["historical_source"] is False


def test_load_legacy_isolated_to_materialized_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_payloads = {
        "__init__": "",
        "behaviors": (
            "def contextual_direction_behavior(*args, **kwargs): return None\n"
            "def endpoint_null_fragility_behavior(*args, **kwargs): return None\n"
        ),
        "compatibility": "def config_sha256_compatible(*args, **kwargs): return True\n",
        "config": (
            "def config_sha256(*args, **kwargs): return 'digest'\n"
            "def load_config(*args, **kwargs): return {}\n"
        ),
        "data": (
            "def data_fingerprint(*args, **kwargs): return 'fingerprint'\n"
            "def load_data_bundle(*args, **kwargs): return None\n"
        ),
        "decaf": "def query_responses(*args, **kwargs): return None\n",
        "mechanisms": (
            "def load_module_c_bundle(*args, **kwargs): return None\n"
            "def load_module_f_bundle(*args, **kwargs): return None\n"
            "def mechanism_fingerprint(*args, **kwargs): return 'fingerprint'\n"
        ),
        "models": "def predict_positive(*args, **kwargs): return None\n",
    }
    payloads = {
        f"code/src/cmr/decaf_covertype_v1/{name}.py": source.encode()
        for name, source in module_payloads.items()
    }
    files = [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in sorted(payloads.items())
    ]
    manifest = {
        "schema_version": 1,
        "namespace": "decaf_covertype_v1",
        "lightweight": True,
        "files": files,
    }
    archive = tmp_path / "covertype-runtime.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            exporter.HISTORICAL_PACKAGE_MANIFEST_MEMBER,
            json.dumps(manifest, sort_keys=True),
        )
        for path, payload in payloads.items():
            stream.writestr(path, payload)

    monkeypatch.setattr(exporter, "HISTORICAL_PACKAGE", archive)
    monkeypatch.setattr(
        exporter,
        "HISTORICAL_PACKAGE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    exporter._historical_source_binding.cache_clear()
    materialized = exporter._materialize_historical_source(tmp_path / "source")
    conflicting = [
        name
        for name in sys.modules
        if name == "cmr" or name.startswith("cmr.decaf_covertype_v1")
    ]
    for name in conflicting:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        loaded = exporter._load_legacy(tmp_path, source_binding=materialized)
        assert set(loaded) == {
            "config_sha256",
            "config_sha256_compatible",
            "contextual_direction_behavior",
            "data_fingerprint",
            "endpoint_null_fragility_behavior",
            "load_config",
            "load_data_bundle",
            "load_module_c_bundle",
            "load_module_f_bundle",
            "mechanism_fingerprint",
            "predict_positive",
            "query_responses",
        }
        assert materialized["origin_verified"] is True
        assert len(materialized["loaded_module_origins"]) == len(module_payloads)
        assert all(
            str(origin).startswith(str(tmp_path / "source"))
            for origin in materialized["loaded_module_origins"].values()
        )
    finally:
        exporter._historical_source_binding.cache_clear()
        for name in list(sys.modules):
            if name == "cmr" or name.startswith("cmr.decaf_covertype_v1"):
                sys.modules.pop(name, None)
        source_root = str(tmp_path / "source")
        while source_root in sys.path:
            sys.path.remove(source_root)
