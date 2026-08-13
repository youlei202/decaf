"""Verification-only exporters for exact historical Controlled responses.

Historical tables are read-only inputs.  This module exports model responses
and identities; current ``decaf.core`` performs every new decomposition.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.crossgen.schema import (
    NEUTRAL_COLUMNS,
    sha256_file,
    trapezoid_weights,
    write_trajectory_record,
)

C2_REFERENCE_SHA256 = "26bc5bc4a9efd5e23d6b34372f5ebce9c3563bb4d5659bc847c4344c67fe7ede"
C2_TASKS = ("direct", "gate", "invert")
C2_ARCHITECTURES = ("resnet18", "small_vit")
C2_WALL_MAPS = (1, 2)
C2_ENDPOINT_EPSILON = 1.0e-4
C2_MIXTURE_ETA = 0.5
C1_REFERENCE_SHA256 = "387c5a572249110a31698d384d942a8e1adf542c7c0b1a9f3a0c5d453102a8a7"
C1_ENDPOINT_EPSILON = 1.0e-4
C0_REFERENCE_SHA256 = "2126b7fcf720e367ca6dd6ed7c467c45ef20199364685d10372a96efa7ebf559"
C0_EFFECTIVE_CONFIG_SHA256 = "8e5883dbf951ca8dbe0541e7a8d86fc74b70cdd65583013777f28dc652987ee1"
C0_SOURCE_CONFIG_SHA256 = "7f8988663c1646333c674b50dd2717a1194b9268a9825b98d70e974acd0aec48"
C0_ENDPOINT_EPSILON = 1.0e-4
C0_NOISE_SEEDS = (20260884, 20260885, 20260886)
C0_PROTOCOL_FAMILY = "linear"
C0_PROTOCOL_VALUE = 0.0
C0_SEALED_AUDIT_ATOL = 5.0e-4
C0_SEALED_REPRODUCTION_SOURCE = "protocol_robust_main_v2/reproduction/src"
C0_SEALED_ENDPOINT_ALIGNED_SHA256 = (
    "d90e46139f45306bd2b97523c1cb7c7a4af02ab83e48295713d0326b2167a907"
)
DEFAULT_HISTORICAL_ROOT = Path("/work/Users/leiyo/GitHub/covariance-matched-markov-revelation")
DEFAULT_C0_RESULTS_ROOT = DEFAULT_HISTORICAL_ROOT / "results/protocol_robust_main_v2"
DEFAULT_C0_CONFIG = DEFAULT_C0_RESULTS_ROOT / "config_snapshot/effective_config.yaml"
DEFAULT_C0_SOURCE_CONFIG = DEFAULT_HISTORICAL_ROOT / "configs/protocol_robust_main_b200_4mig.yaml"
DEFAULT_C1_RESULTS_ROOT = DEFAULT_HISTORICAL_ROOT / "results/endpoint_behavior_v1_measurement"
DEFAULT_C1_CONFIG = DEFAULT_HISTORICAL_ROOT / "configs/endpoint_behavior_v1_b200.yaml"
DEFAULT_C2_ROOT = Path(
    "/work/Users/leiyo/GitHub/covariance-matched-markov-revelation/results/context_swap_c_v1/formal"
)

C0_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "model_id": "context_gate__resnet18__seed_3101",
        "task": "context_gate",
        "architecture": "resnet18",
        "model_seed": 3101,
        "base_id": 372490,
        "factor": "wall_color",
        "cf_map_seed": "20260882",
        "noise_seed": 20260884,
        "expected_state": "active",
        "coverage": "context_gate,resnet18,map_20260882,active,mixed_candidate",
    },
    {
        "model_id": "context_gate__resnet18__seed_3101",
        "task": "context_gate",
        "architecture": "resnet18",
        "model_seed": 3101,
        "base_id": 75310,
        "factor": "wall_color",
        "cf_map_seed": "20260883",
        "noise_seed": 20260884,
        "expected_state": "null",
        "coverage": "context_gate,resnet18,map_20260883,null",
    },
    {
        "model_id": "context_gate__small_vit__seed_3101",
        "task": "context_gate",
        "architecture": "small_vit",
        "model_seed": 3101,
        "base_id": 170664,
        "factor": "object_color",
        "cf_map_seed": "20260882",
        "noise_seed": 20260884,
        "expected_state": "active",
        "coverage": "context_gate,small_vit,map_20260882,active",
    },
    {
        "model_id": "context_gate__small_vit__seed_3101",
        "task": "context_gate",
        "architecture": "small_vit",
        "model_seed": 3101,
        "base_id": 222025,
        "factor": "object_color",
        "cf_map_seed": "20260883",
        "noise_seed": 20260884,
        "expected_state": "null",
        "coverage": "context_gate,small_vit,map_20260883,null,mixed_candidate",
    },
    {
        "model_id": "color_shape_xor__resnet18__seed_3101",
        "task": "color_shape_xor",
        "architecture": "resnet18",
        "model_seed": 3101,
        "base_id": 74606,
        "factor": "object_color",
        "cf_map_seed": "20260882",
        "noise_seed": 20260884,
        "expected_state": "active",
        "coverage": "color_shape_xor,resnet18,map_20260882,active",
    },
    {
        "model_id": "color_shape_xor__resnet18__seed_3101",
        "task": "color_shape_xor",
        "architecture": "resnet18",
        "model_seed": 3101,
        "base_id": 313393,
        "factor": "object_color",
        "cf_map_seed": "20260882",
        "noise_seed": 20260884,
        "expected_state": "null",
        "coverage": "color_shape_xor,resnet18,map_20260882,null",
    },
    {
        "model_id": "color_shape_xor__small_vit__seed_3101",
        "task": "color_shape_xor",
        "architecture": "small_vit",
        "model_seed": 3101,
        "base_id": 313393,
        "factor": "object_color",
        "cf_map_seed": "20260883",
        "noise_seed": 20260884,
        "expected_state": "active",
        "coverage": "color_shape_xor,small_vit,map_20260883,active,mixed_candidate",
    },
    {
        "model_id": "color_shape_xor__small_vit__seed_3101",
        "task": "color_shape_xor",
        "architecture": "small_vit",
        "model_seed": 3101,
        "base_id": 356967,
        "factor": "object_color",
        "cf_map_seed": "20260883",
        "noise_seed": 20260884,
        "expected_state": "null",
        "coverage": "color_shape_xor,small_vit,map_20260883,null",
    },
)

C1_SELECTION: tuple[dict[str, Any], ...] = (
    {
        "job_id": "e__p095__resnet18__seed_5101__epoch_001__object_shape",
        "map_index": 0,
        "noise_seed": 6201,
        "sample_id": 346270,
        "expected_state": "active",
        "coverage": "evidence:p095,resnet18,early,shape",
    },
    {
        "job_id": "e__p095__resnet18__seed_5101__epoch_001__wall_color",
        "map_index": 0,
        "noise_seed": 6201,
        "sample_id": 271706,
        "expected_state": "mixed",
        "coverage": "evidence:p095,resnet18,early,wall",
    },
    {
        "job_id": "e__p095__resnet18__seed_5101__epoch_001__object_shape",
        "map_index": 1,
        "noise_seed": 6202,
        "sample_id": 148098,
        "expected_state": "mixed",
        "coverage": "evidence:second-map",
    },
    {
        "job_id": "e__p099__small_vit__seed_5101__epoch_030__object_shape",
        "map_index": 0,
        "noise_seed": 6201,
        "sample_id": 88281,
        "expected_state": "mixed",
        "coverage": "evidence:p099,small_vit,late,shape",
    },
    {
        "job_id": "e__p099__small_vit__seed_5101__epoch_030__wall_color",
        "map_index": 0,
        "noise_seed": 6201,
        "sample_id": 280996,
        "expected_state": "mixed",
        "coverage": "evidence:p099,small_vit,late,wall",
    },
    {
        "job_id": "f__robust__small_vit__seed_5301__floor_color",
        "map_index": 0,
        "noise_seed": 6201,
        "sample_id": 443884,
        "expected_state": "null",
        "coverage": "fragility:robust,small_vit,null",
    },
    {
        "job_id": "f__neutral__small_vit__seed_5301__floor_color",
        "map_index": 0,
        "noise_seed": 6201,
        "sample_id": 443884,
        "expected_state": "null",
        "coverage": "fragility:neutral,small_vit,null",
    },
    {
        "job_id": "f__fragile__small_vit__seed_5301__floor_color",
        "map_index": 0,
        "noise_seed": 6201,
        "sample_id": 123063,
        "expected_state": "active",
        "coverage": "fragility:fragile,small_vit,active",
    },
    {
        "job_id": "f__fragile__small_vit__seed_5301__floor_color",
        "map_index": 1,
        "noise_seed": 6202,
        "sample_id": 54611,
        "expected_state": "mixed",
        "coverage": "fragility:fragile,second-map,mixed",
    },
    {
        "job_id": "f__fragile__resnet18__seed_5301__floor_color",
        "map_index": 0,
        "noise_seed": 6201,
        "sample_id": 443884,
        "expected_state": "mixed",
        "coverage": "fragility:fragile,resnet18,endpoint-evidence",
    },
)


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise FileNotFoundError(f"{label} is missing, empty, or unsafe: {path}")
    return path


def _read_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(_regular_file(path, "JSON receipt").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"receipt must contain a JSON object: {path}")
    return payload


def _dominant(e: float, c: float, f: float) -> str:
    values = {"E": e, "C": c, "F": f}
    maximum = max(values.values())
    return "|".join(name for name, value in values.items() if value == maximum)


def _validate_c0_component_contract(main_sweep: Any) -> dict[str, Any]:
    """Exercise the exact four-way decomposition contract before CUDA work."""

    endpoint = np.asarray([0.0, 1.0, -1.0], dtype=np.float64)
    delta = np.asarray(
        [
            [0.25, -0.75, 0.0],
            [0.25, -0.75, 0.0],
            [0.25, -0.75, 0.0],
        ],
        dtype=np.float64,
    )
    try:
        components = main_sweep._component_matrices(
            endpoint,
            delta,
            C0_ENDPOINT_EPSILON,
        )
    except Exception as exc:
        raise RuntimeError(
            "C0 sealed historical decomposition preflight failed; the runtime "
            "must expose EndpointAlignedResponse.null"
        ) from exc
    required = {"abs", "align", "opp", "null"}
    if set(components) != required:
        raise ValueError(
            "C0 sealed historical decomposition keys changed: "
            f"expected={sorted(required)}, observed={sorted(components)}"
        )
    arrays = {name: np.asarray(components[name], dtype=np.float64) for name in required}
    if any(value.shape != delta.shape for value in arrays.values()):
        raise ValueError("C0 sealed historical decomposition shapes changed")
    if not np.allclose(
        arrays["abs"],
        arrays["align"] + arrays["opp"] + arrays["null"],
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError("C0 sealed historical decomposition lost conservation")
    if not np.array_equal(arrays["null"][0], np.abs(delta[0])):
        raise ValueError("C0 endpoint-null routing changed")
    if np.any(arrays["align"][0]) or np.any(arrays["opp"][0]):
        raise ValueError("C0 endpoint-null response leaked into active components")
    if np.any(arrays["null"][1:]):
        raise ValueError("C0 endpoint-active response leaked into null")
    if not np.array_equal(arrays["align"][1], np.asarray([0.25, 0.0, 0.0])):
        raise ValueError("C0 positive-endpoint aligned routing changed")
    if not np.array_equal(arrays["opp"][1], np.asarray([0.0, 0.75, 0.0])):
        raise ValueError("C0 positive-endpoint opposed routing changed")
    if not np.array_equal(arrays["align"][2], np.asarray([0.0, 0.75, 0.0])):
        raise ValueError("C0 negative-endpoint aligned routing changed")
    if not np.array_equal(arrays["opp"][2], np.asarray([0.25, 0.0, 0.0])):
        raise ValueError("C0 negative-endpoint opposed routing changed")
    return {
        "endpoint_values": endpoint.tolist(),
        "component_keys": sorted(required),
        "conservation_passed": True,
        "null_routing_passed": True,
        "active_orientation_routing_passed": True,
    }


def _c0_replay_prefix_count(
    *,
    dynamic_count: int,
    stack_size: int,
    historical_batch_size: int,
    selected_positions: np.ndarray,
) -> int:
    """Retain the historical flattened batch containing every selected unit."""

    count = int(dynamic_count)
    width = int(stack_size)
    batch = int(historical_batch_size)
    positions = np.asarray(selected_positions, dtype=np.int64)
    if count < 1 or width < 2 or batch < 1:
        raise ValueError("C0 replay layout dimensions must be positive")
    if positions.ndim != 1 or positions.size < 1:
        raise ValueError("C0 replay positions must be a non-empty vector")
    if np.any(positions < 0) or np.any(positions >= count):
        raise ValueError("C0 replay positions are outside the dynamic lock")
    final_selected_flat = (int(np.max(positions)) + 1) * width
    historical_flat_count = count * width
    retained_flat_stop = min(
        ((final_selected_flat + batch - 1) // batch) * batch,
        historical_flat_count,
    )
    return min(count, (retained_flat_stop + width - 1) // width)


def _c0_candidate_identity(candidate: Mapping[str, Any]) -> tuple[str, int]:
    return str(candidate["model_id"]), int(candidate["base_id"])


def _c0_partition_candidate_audits(
    candidates: tuple[dict[str, Any], ...],
    aggregate_audits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Qualify fixed candidates by the sealed six-repeat common-AUC audit."""

    candidate_identities = [_c0_candidate_identity(item) for item in candidates]
    if len(set(candidate_identities)) != len(candidate_identities):
        raise ValueError("C0 candidate identities are not unique")
    audit_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for item in aggregate_audits:
        identity = _c0_candidate_identity(item)
        if identity in audit_by_identity:
            raise ValueError(f"C0 aggregate audit identity is repeated: {identity}")
        audit_by_identity[identity] = item
    if set(audit_by_identity) != set(candidate_identities):
        missing = sorted(set(candidate_identities) - set(audit_by_identity))
        unexpected = sorted(set(audit_by_identity) - set(candidate_identities))
        raise ValueError(
            "C0 aggregate audits do not exactly cover the fixed candidates: "
            f"missing={missing}, unexpected={unexpected}"
        )

    qualified: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        identity = _c0_candidate_identity(candidate)
        aggregate_record = audit_by_identity[identity]
        if str(aggregate_record["factor"]) != str(candidate["factor"]):
            raise ValueError(f"C0 aggregate audit factor changed: {identity}")
        audit = dict(aggregate_record["audit"])
        maximum_error = float(audit["maximum_absolute_error"])
        tolerance = float(audit["tolerance"])
        if tolerance != C0_SEALED_AUDIT_ATOL:
            raise ValueError(f"C0 aggregate audit tolerance changed: {identity}")
        qualifies = maximum_error <= C0_SEALED_AUDIT_ATOL
        if bool(audit["passed"]) != qualifies:
            raise ValueError(f"C0 aggregate audit pass flag is inconsistent: {identity}")
        record = {
            **dict(candidate),
            "protocol": str(aggregate_record["protocol"]),
            "audit": audit,
            "maximum_absolute_error": maximum_error,
            "qualification_tolerance": C0_SEALED_AUDIT_ATOL,
            "qualified": qualifies,
        }
        if qualifies:
            qualified.append(record)
        else:
            record["reason_code"] = "UNRESOLVED_HISTORICAL_RUNTIME_METADATA"
            record["reason"] = (
                "UNRESOLVED_HISTORICAL_RUNTIME_METADATA: "
                "excluded_without_replacement; sealed six-repeat common-information "
                f"maximum_absolute_error={maximum_error:.17g} exceeds "
                f"qualification_tolerance={C0_SEALED_AUDIT_ATOL:.17g}"
            )
            excluded.append(record)
    return qualified, excluded


def _c0_qualified_coverage(
    qualified: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the predeclared C0 coverage gate on qualified candidates only."""

    architectures = sorted({str(item["architecture"]) for item in qualified})
    maps = sorted({str(item["cf_map_seed"]) for item in qualified})
    active_count = sum(str(item["expected_state"]) == "active" for item in qualified)
    null_count = sum(str(item["expected_state"]) == "null" for item in qualified)
    unit_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unit_rows.setdefault(str(row["unit_id"]), row)
    mixed_units = sorted(
        unit_id
        for unit_id, row in unit_rows.items()
        if float(row["historical_E"]) > 1.0e-5
        and float(row["historical_C"]) > 1.0e-5
    )
    checks = {
        "two_registered_architectures": architectures == ["resnet18", "small_vit"],
        "at_least_two_active": active_count >= 2,
        "at_least_two_null": null_count >= 2,
        "at_least_one_mixed_E_C": len(mixed_units) >= 1,
        "both_registered_counterfactual_maps": maps == ["20260882", "20260883"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "architectures": architectures,
            "active_count": active_count,
            "null_count": null_count,
            "mixed_unit_count": len(mixed_units),
            "mixed_units": mixed_units,
            "counterfactual_maps": maps,
        },
        "requirements": {
            "architectures": ["resnet18", "small_vit"],
            "minimum_active": 2,
            "minimum_null": 2,
            "minimum_mixed_E_C": 1,
            "counterfactual_maps": ["20260882", "20260883"],
            "mixed_threshold_each": 1.0e-5,
        },
    }


def _c0_candidate_qualification_record(
    *,
    qualified: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    coverage: Mapping[str, Any],
    status: str,
    diagnostic_path: Path,
    diagnostic_sha256: str,
) -> dict[str, Any]:
    if len(diagnostic_sha256) != 64:
        raise ValueError("C0 candidate diagnostic SHA-256 is malformed")
    diagnostic = {
        "path": str(diagnostic_path.resolve()),
        "sha256": diagnostic_sha256,
    }
    return {
        "candidate_count": len(C0_CANDIDATES),
        "selected_count": len(qualified),
        "excluded_count": len(excluded),
        "status": status,
        "qualification_fraction": f"{len(qualified)}/{len(C0_CANDIDATES)}",
        "tolerance": C0_SEALED_AUDIT_ATOL,
        "rule": (
            "sealed six-repeat common-information maximum_absolute_error <= 5e-4; "
            "failed fixed candidates are excluded without replacement"
        ),
        "selected": qualified,
        "excluded": excluded,
        "diagnostic": diagnostic,
        "diagnostic_path": diagnostic["path"],
        "diagnostic_sha256": diagnostic["sha256"],
        "coverage": dict(coverage),
    }


def _historical_c0_modules(
    historical_root: Path,
    package_path: Path,
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    import datetime as datetime_module
    import hashlib
    import importlib
    import importlib.machinery
    import importlib.util
    import types
    import zipfile

    root = historical_root.expanduser().resolve()
    source = root / "src"
    package = _regular_file(package_path.expanduser().resolve(), "C0 package")
    if root.is_symlink() or not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"historical repository is missing or unsafe: {root}")
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError(f"historical source tree is missing or unsafe: {source}")
    if not hasattr(datetime_module, "UTC"):
        datetime_module.UTC = (  # type: ignore[attr-defined]
            datetime_module.timezone.utc  # noqa: UP017 - Python 3.10 compatibility
        )
    if importlib.util.find_spec("h5py") is None:
        h5py_stub = types.ModuleType("h5py")

        def unavailable_h5py(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("h5py is unavailable in this verification environment")

        h5py_stub.File = unavailable_h5py  # type: ignore[attr-defined]
        sys.modules["h5py"] = h5py_stub

    loaded_protocol = sorted(
        name
        for name in sys.modules
        if name == "protocol_robust" or name.startswith("protocol_robust.")
    )
    if loaded_protocol:
        raise RuntimeError(
            "C0 historical runtime requires a fresh process before importing the "
            f"sealed package; already loaded: {loaded_protocol}"
        )
    loaded_cmr_children = sorted(name for name in sys.modules if name.startswith("cmr."))
    if loaded_cmr_children:
        raise RuntimeError(
            "C0 historical runtime requires a fresh process before extending cmr "
            f"with the sealed package; already loaded: {loaded_cmr_children}"
        )

    required_members = (
        "protocol_robust/main_common.py",
        "protocol_robust/main_sweep.py",
        "protocol_robust/model_adapter.py",
        "protocol_robust/v11_channels.py",
        "protocol_robust/channel.py",
        "protocol_robust/endpoint_aligned.py",
        "cmr/interfaces/revelation_protocol.py",
        "cmr/models/model_factory.py",
        "cmr/utils/checkpoints.py",
        "cmr/utils/reproducibility.py",
    )
    member_sha256: dict[str, str] = {}
    member_source: dict[str, bytes] = {}
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        for relative in required_members:
            member = f"{C0_SEALED_REPRODUCTION_SOURCE}/{relative}"
            if member not in names:
                raise FileNotFoundError(f"C0 sealed reproduction member is missing: {member}")
            payload = archive.read(member)
            member_source[relative] = payload
            member_sha256[relative] = hashlib.sha256(payload).hexdigest()
    if member_sha256["protocol_robust/endpoint_aligned.py"] != C0_SEALED_ENDPOINT_ALIGNED_SHA256:
        raise ValueError("C0 sealed EndpointAlignedResponse source changed")

    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    cmr = importlib.import_module("cmr")
    cmr_origin = Path(str(getattr(cmr, "__file__", ""))).resolve()
    expected_cmr_origin = (source / "cmr/__init__.py").resolve()
    if cmr_origin != expected_cmr_origin:
        raise RuntimeError(
            f"C0 fallback cmr package did not originate in the historical repository: {cmr_origin}"
        )

    sealed_source = f"{package}/{C0_SEALED_REPRODUCTION_SOURCE}"
    sealed_cmr = f"{sealed_source}/cmr"
    cmr_paths = cmr.__path__  # type: ignore[attr-defined]
    if sealed_cmr not in cmr_paths:
        cmr_paths.insert(0, sealed_cmr)
    if sealed_source not in sys.path:
        sys.path.insert(0, sealed_source)

    # ``cmr.interfaces`` and ``cmr.utils`` are namespace packages in the
    # historical tree.  Python's zip importer otherwise merges those
    # namespaces with the read-only worktree and can silently load dependency
    # modules from there.  Execute the exact sealed bytes up front so every
    # scientific/runtime dependency used below has a single package-bound
    # origin.
    for namespace in ("cmr.interfaces", "cmr.utils"):
        relative_namespace = namespace.removeprefix("cmr.")
        module = types.ModuleType(namespace)
        module.__package__ = namespace
        module.__path__ = [f"{sealed_cmr}/{relative_namespace}"]  # type: ignore[attr-defined]
        specification = importlib.machinery.ModuleSpec(namespace, loader=None, is_package=True)
        specification.submodule_search_locations = module.__path__  # type: ignore[attr-defined]
        module.__spec__ = specification
        sys.modules[namespace] = module
        setattr(cmr, relative_namespace, module)

    for module_name, relative in (
        ("cmr.interfaces.revelation_protocol", "cmr/interfaces/revelation_protocol.py"),
        ("cmr.utils.reproducibility", "cmr/utils/reproducibility.py"),
        ("cmr.utils.checkpoints", "cmr/utils/checkpoints.py"),
    ):
        parent_name, _, child_name = module_name.rpartition(".")
        origin = f"{sealed_source}/{relative}"
        module = types.ModuleType(module_name)
        module.__file__ = origin
        module.__package__ = parent_name
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None, origin=origin)
        sys.modules[module_name] = module
        setattr(sys.modules[parent_name], child_name, module)
        exec(compile(member_source[relative], origin, "exec"), module.__dict__)

    main_common = importlib.import_module("protocol_robust.main_common")
    main_sweep = importlib.import_module("protocol_robust.main_sweep")
    model_adapter = importlib.import_module("protocol_robust.model_adapter")
    v11_channels = importlib.import_module("protocol_robust.v11_channels")
    channel = importlib.import_module("protocol_robust.channel")
    endpoint_aligned = importlib.import_module("protocol_robust.endpoint_aligned")
    model_factory = importlib.import_module("cmr.models.model_factory")

    sealed_modules = {
        "protocol_robust.main_common": main_common,
        "protocol_robust.main_sweep": main_sweep,
        "protocol_robust.model_adapter": model_adapter,
        "protocol_robust.v11_channels": v11_channels,
        "protocol_robust.channel": channel,
        "protocol_robust.endpoint_aligned": endpoint_aligned,
        "cmr.interfaces.revelation_protocol": sys.modules[
            "cmr.interfaces.revelation_protocol"
        ],
        "cmr.models.model_factory": model_factory,
        "cmr.utils.checkpoints": sys.modules["cmr.utils.checkpoints"],
        "cmr.utils.reproducibility": sys.modules["cmr.utils.reproducibility"],
    }
    module_origins = {
        name: str(getattr(module, "__file__", "")) for name, module in sealed_modules.items()
    }
    invalid_origins = {
        name: origin
        for name, origin in module_origins.items()
        if not origin.startswith(f"{sealed_source}/")
    }
    if invalid_origins:
        raise RuntimeError(
            f"C0 runtime imported modules outside the sealed reproduction source: {invalid_origins}"
        )
    dependency_origins = {
        name: str(getattr(module, "__file__", ""))
        for name, module in sorted(sys.modules.items())
        if name.startswith(("cmr.", "protocol_robust."))
        and getattr(module, "__file__", None)
    }
    invalid_dependency_origins = {
        name: origin
        for name, origin in dependency_origins.items()
        if not origin.startswith(f"{sealed_source}/")
    }
    if invalid_dependency_origins:
        raise RuntimeError(
            "C0 runtime imported transitive execution dependencies outside the sealed "
            f"reproduction source: {invalid_dependency_origins}"
        )
    annotations = getattr(endpoint_aligned.EndpointAlignedResponse, "__annotations__", {})
    if "null" not in annotations:
        raise RuntimeError("C0 sealed EndpointAlignedResponse lacks the required null field")
    component_preflight = _validate_c0_component_contract(main_sweep)

    runtime_provenance = {
        "source": "sealed delivery package reproduction/src",
        "sealed_source_prefix": sealed_source,
        "historical_cmr_fallback_origin": str(cmr_origin),
        "module_origins": module_origins,
        "transitive_dependency_origins": dependency_origins,
        "member_sha256": member_sha256,
        "endpoint_aligned_null_field_present": True,
        "component_contract_preflight": component_preflight,
    }

    return (
        main_common,
        main_sweep,
        model_adapter,
        v11_channels,
        channel.CovarianceGeometryFamily,
        runtime_provenance,
    )


def _delivery_expected_sha(
    inputs: Mapping[str, Any],
    path: Path,
    label: str,
) -> str:
    expected = inputs.get(str(path.resolve()))
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"C0 delivery receipt does not bind {label}: {path}")
    observed = sha256_file(_regular_file(path, label))
    if observed != expected:
        raise ValueError(f"C0 sealed {label} SHA-256 mismatch: {path}")
    return observed


def _c0_registered_assets(
    results_root: Path,
    effective_config: Path,
    source_config: Path,
) -> dict[str, Any]:
    delivery_path = _regular_file(
        results_root / "runtime/main_delivery.json",
        "C0 delivery receipt",
    )
    delivery = _read_receipt(delivery_path)
    request = delivery.get("request_identity")
    if not isinstance(request, Mapping) or not isinstance(request.get("inputs"), Mapping):
        raise ValueError(f"C0 delivery receipt lacks sealed inputs: {delivery_path}")
    inputs = request["inputs"]
    if sha256_file(effective_config) != C0_EFFECTIVE_CONFIG_SHA256:
        raise ValueError("C0 effective config snapshot changed")
    if sha256_file(source_config) != C0_SOURCE_CONFIG_SHA256:
        raise ValueError("C0 authoritative source config changed")
    _delivery_expected_sha(inputs, effective_config, "effective config snapshot")
    _delivery_expected_sha(inputs, source_config, "authoritative source config")

    package = delivery.get("package")
    if not isinstance(package, Mapping):
        raise ValueError(f"C0 delivery receipt lacks package identity: {delivery_path}")
    package_path = _regular_file(Path(str(package.get("path", ""))), "C0 package")
    package_sha = str(package.get("sha256", ""))
    if package_sha != C0_REFERENCE_SHA256 or sha256_file(package_path) != package_sha:
        raise ValueError("C0 registered package SHA-256 changed")

    endpoint_phase_path = _regular_file(
        results_root / "endpoint/endpoint_phase.json",
        "C0 endpoint receipt",
    )
    endpoint_phase = _read_receipt(endpoint_phase_path)
    if endpoint_phase.get("status") != "complete":
        raise ValueError(f"C0 endpoint receipt is not complete: {endpoint_phase_path}")
    endpoint_path = _regular_file(
        results_root / "endpoint/endpoint_sample_effects_selected.parquet",
        "C0 endpoint sample table",
    )
    outputs = endpoint_phase.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"C0 endpoint receipt lacks outputs: {endpoint_phase_path}")
    endpoint_relative = "endpoint/endpoint_sample_effects_selected.parquet"
    endpoint_sha = str(outputs.get(endpoint_relative, ""))
    if sha256_file(endpoint_path) != endpoint_sha:
        raise ValueError(f"C0 endpoint sample table SHA-256 mismatch: {endpoint_path}")
    _delivery_expected_sha(inputs, endpoint_path, "endpoint sample table")
    return {
        "delivery": delivery,
        "delivery_inputs": inputs,
        "delivery_path": delivery_path,
        "endpoint_path": endpoint_path,
        "endpoint_sha256": endpoint_sha,
        "endpoint_phase_path": endpoint_phase_path,
        "package_path": package_path,
        "package_sha256": package_sha,
    }


def _c0_endpoint_rows(endpoint_path: Path) -> pd.DataFrame:
    models = sorted({str(item["model_id"]) for item in C0_CANDIDATES})
    base_ids = sorted({int(item["base_id"]) for item in C0_CANDIDATES})
    columns = [
        "model_id",
        "task",
        "architecture",
        "model_seed",
        "factor",
        "cf_map_seed",
        "map_name",
        "sample_id",
        "base_id",
        "counterfactual_id",
        "in_dynamic_lock",
        "factual_probability",
        "counterfactual_probability",
        "delta_endpoint",
        "endpoint_active",
        "endpoint_null",
        "endpoint_sign",
        "sign_masked",
    ]
    frame = pd.read_parquet(
        endpoint_path,
        columns=columns,
        filters=[("model_id", "in", models), ("base_id", "in", base_ids)],
        use_threads=False,
    )
    frame["cf_map_seed"] = frame["cf_map_seed"].astype(str)
    return frame


def _c0_endpoint_row(
    frame: pd.DataFrame,
    selection: Mapping[str, Any],
    map_seed: str,
    *,
    check_expected_state: bool = False,
) -> pd.Series:
    selected = frame.loc[
        frame["model_id"].astype(str).eq(str(selection["model_id"]))
        & frame["base_id"].astype(int).eq(int(selection["base_id"]))
        & frame["factor"].astype(str).eq(str(selection["factor"]))
        & frame["cf_map_seed"].astype(str).eq(str(map_seed))
    ]
    if len(selected) != 1:
        raise ValueError(
            "C0 endpoint identity is not unique: "
            f"model={selection['model_id']}, base={selection['base_id']}, "
            f"factor={selection['factor']}, map={map_seed}, rows={len(selected)}"
        )
    row = selected.iloc[0]
    expected = {
        "task": selection["task"],
        "architecture": selection["architecture"],
        "model_seed": selection["model_seed"],
        "sample_id": selection["base_id"],
    }
    for key, value in expected.items():
        if str(row[key]) != str(value):
            raise ValueError(f"C0 endpoint identity changed for {key}: {dict(selection)}")
    if not bool(row["in_dynamic_lock"]):
        raise ValueError(f"C0 selected endpoint is outside the dynamic lock: {dict(selection)}")
    active = bool(row["endpoint_active"])
    null = bool(row["endpoint_null"])
    if active == null:
        raise ValueError(f"C0 endpoint gate flags are inconsistent: {dict(selection)}")
    if check_expected_state:
        expected_state = str(selection["expected_state"])
        if (expected_state == "active" and not active) or (expected_state == "null" and not null):
            raise ValueError(f"C0 registered endpoint state changed: {dict(selection)}")
    plus = float(row["factual_probability"])
    minus = float(row["counterfactual_probability"])
    endpoint_d = float(row["delta_endpoint"])
    if not np.isclose(plus - minus, endpoint_d, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"C0 sealed endpoint arithmetic changed: {dict(selection)}")
    return row


def _c0_direct_summary(
    main_sweep: Any,
    endpoint_d: float,
    response: np.ndarray,
    alpha: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    curve = np.asarray(response, dtype=np.float64)
    grid = np.asarray(alpha, dtype=np.float64)
    if curve.shape != grid.shape:
        raise ValueError("C0 response and alpha grid shapes disagree")
    components = main_sweep._component_matrices(
        np.asarray([endpoint_d], dtype=np.float64),
        curve.reshape(1, -1),
        epsilon,
    )
    result = {
        "M": abs(float(endpoint_d)),
        "E": float(np.trapezoid(components["align"][0], x=grid)),
        "C": float(np.trapezoid(components["opp"][0], x=grid)),
        "F": float(np.trapezoid(components["null"][0], x=grid)),
        "Abs": float(np.trapezoid(components["abs"][0], x=grid)),
    }
    if not np.isclose(
        result["Abs"],
        result["E"] + result["C"] + result["F"],
        atol=1.0e-12,
        rtol=1.0e-10,
    ):
        raise ValueError("C0 regenerated historical decomposition lost conservation")
    return result


def _c0_trajectory_rows(
    *,
    selection: Mapping[str, Any],
    endpoint_row: pd.Series,
    response: np.ndarray,
    alpha: np.ndarray,
    historical: Mapping[str, float],
    checkpoint_sha256: str,
    endpoint_path: Path,
    sealed_summary_path: Path,
    sealed_audit: Mapping[str, Any],
    position: int,
    reference_run: str,
) -> list[dict[str, Any]]:
    weights = trapezoid_weights(alpha)
    model_id = str(selection["model_id"])
    base_id = str(int(selection["base_id"]))
    factor = str(selection["factor"])
    map_name = str(endpoint_row["map_name"])
    endpoint_d = float(endpoint_row["delta_endpoint"])
    gate = bool(endpoint_row["endpoint_active"])
    protocol = f"{C0_PROTOCOL_FAMILY}_lambda_{C0_PROTOCOL_VALUE:.3f}"
    unit_id = (
        f"C0__{model_id}__{factor}__{map_name}"
        f"__noise_{int(selection['noise_seed'])}__base_{base_id}"
    )
    metadata = {
        "bridge_kind": "historical_c0_exact_unit_replay",
        "coverage": str(selection["coverage"]),
        "current_checkpoint_sha256": checkpoint_sha256,
        "current_counterfactual_map": map_name,
        "current_factor_or_part_id": factor,
        "current_model_id": model_id,
        "current_protocol": protocol,
        "current_sample_or_pair_id": base_id,
        "full_dynamic_rng_draw_preserved": True,
        "full_dynamic_rng_shape": [4096, 3072],
        "historical_dominant": _dominant(
            float(historical["E"]),
            float(historical["C"]),
            float(historical["F"]),
        ),
        "historical_gate": gate,
        "historical_orientation": int(np.sign(endpoint_d)) if gate else 0,
        "identity_match": True,
        "quantity_provenance": {
            "endpoint_d": "independently sealed endpoint sample table",
            "endpoint_scores": "independently sealed endpoint sample table; retained in metadata",
            "stage_r": "regenerated by exact historical model/channel/noise primitives",
            "historical_M": "absolute value of independently sealed endpoint_d",
            "historical_E_C_F_Abs": (
                "regenerated by historical endpoint-aligned decomposition and "
                "direct alpha-grid trapezoid"
            ),
            "aggregate_audit": (
                "independent sealed sample_auc_selected row over two maps and three noise seeds"
            ),
        },
        "sealed_aggregate_audit": dict(sealed_audit),
        "sealed_endpoint_score_arithmetic_residual": float(
            endpoint_row["factual_probability"]
            - endpoint_row["counterfactual_probability"]
            - endpoint_d
        ),
        "sealed_endpoint_score_minus": float(endpoint_row["counterfactual_probability"]),
        "sealed_endpoint_score_plus": float(endpoint_row["factual_probability"]),
        "selected_position": position,
        "source_endpoint_samples": str(endpoint_path.resolve()),
        "source_sealed_summary": str(sealed_summary_path.resolve()),
    }
    common = {
        "experiment_family": "controlled_c0_protocol_robust",
        "reference_run": reference_run,
        "unit_id": unit_id,
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_or_pair_id": base_id,
        "factor_or_part_id": factor,
        "counterfactual_map": map_name,
        "protocol": protocol,
        "protocol_seed": int(selection["noise_seed"]),
        "endpoint_epsilon": C0_ENDPOINT_EPSILON,
        "endpoint_score_plus": np.nan,
        "endpoint_score_minus": np.nan,
        "endpoint_d": endpoint_d,
        **{f"historical_{name}": float(value) for name, value in historical.items()},
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }
    return [
        {
            **common,
            "stage_index": stage_index,
            "stage_t": float(stage_t),
            "quadrature_weight": float(weights[stage_index]),
            "stage_score_plus": np.nan,
            "stage_score_minus": np.nan,
            "stage_r": float(response[stage_index]),
        }
        for stage_index, stage_t in enumerate(alpha)
    ]


def _source_row(
    root: Path,
    task: str,
    architecture: str,
    seed: int,
    wall_map: int,
    object_map: int,
    pair_id: int,
) -> tuple[pd.Series, dict[str, Any], Path]:
    job_id = f"eval__{task}__{architecture}__seed_{seed}__wall_{wall_map}"
    job = root / "jobs" / "evaluation" / job_id
    samples_path = _regular_file(job / "samples.parquet", "C2 samples")
    receipt = _read_receipt(job / "receipt.json")
    samples = pd.read_parquet(samples_path)
    required = {
        "model_id",
        "task",
        "architecture",
        "seed",
        "pair_id",
        "base_id",
        "direction",
        "wall_map",
        "object_map",
        "endpoint_fact_id",
        "endpoint_cf_id",
        "swap_fact_id",
        "swap_cf_id",
        "q_endpoint_fact",
        "q_endpoint_cf",
        "q_swap_fact",
        "q_swap_cf",
        "endpoint_delta",
        "swap_delta",
        "correct_E",
        "correct_C",
        "correct_F",
        "correct_Abs",
        "swap_E",
        "swap_C",
        "swap_F",
        "swap_Abs",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"C2 samples lack columns {missing}: {samples_path}")
    selected = samples.loc[
        (samples["object_map"].astype(int) == object_map)
        & (samples["pair_id"].astype(int) == pair_id)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"C2 identity is not unique: object_map={object_map}, "
            f"pair_id={pair_id}, rows={len(selected)}, source={samples_path}"
        )
    row = selected.iloc[0]
    identities = {
        "job_id": job_id,
        "model_id": str(row["model_id"]),
        "task": task,
        "architecture": architecture,
        "seed": seed,
    }
    for key, expected in identities.items():
        if str(receipt.get(key)) != str(expected):
            raise ValueError(
                f"C2 receipt/sample mismatch for {key}: {receipt.get(key)!r} != {expected!r}"
            )
    if int(row["wall_map"]) != wall_map:
        raise ValueError("C2 sample wall-map identity changed")
    if not receipt.get("completed") or not receipt.get("map_semantics_valid"):
        raise ValueError(f"C2 receipt is not completed and semantic-valid: {job}")
    if len(str(receipt.get("checkpoint_sha256", ""))) != 64:
        raise ValueError(f"C2 receipt lacks checkpoint SHA-256: {job}")
    return row, receipt, samples_path


def _trajectory_rows(
    row: pd.Series,
    receipt: dict[str, Any],
    samples_path: Path,
    reference_run: str,
) -> list[dict[str, Any]]:
    ep_plus, ep_minus = float(row["q_endpoint_fact"]), float(row["q_endpoint_cf"])
    sw_plus, sw_minus = float(row["q_swap_fact"]), float(row["q_swap_cf"])
    endpoint_d, swap_d = float(row["endpoint_delta"]), float(row["swap_delta"])
    if not np.isclose(ep_plus - ep_minus, endpoint_d, atol=1e-10, rtol=0.0):
        raise ValueError(f"C2 endpoint score identity failed: {samples_path}")
    if not np.isclose(sw_plus - sw_minus, swap_d, atol=1e-10, rtol=0.0):
        raise ValueError(f"C2 swap score identity failed: {samples_path}")
    eta = C2_MIXTURE_ETA
    historical = {
        "M": abs(endpoint_d),
        **{
            name: (1.0 - eta) * float(row[f"correct_{name}"]) + eta * float(row[f"swap_{name}"])
            for name in ("E", "C", "F", "Abs")
        },
    }
    if not np.isclose(
        historical["Abs"],
        historical["E"] + historical["C"] + historical["F"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"C2 component identity failed: {samples_path}")
    model_id = str(row["model_id"])
    pair_id = int(row["pair_id"])
    wall_map, object_map = int(row["wall_map"]), int(row["object_map"])
    checkpoint = str(receipt["checkpoint_sha256"])
    cf_map = f"wall_{wall_map}__object_{object_map}"
    protocol = "context_swap_eta_0.5"
    factor = "object_color"
    gate = abs(endpoint_d) >= C2_ENDPOINT_EPSILON
    metadata = {
        "bridge_kind": "sealed_analytic_context_mixture",
        "current_checkpoint_sha256": checkpoint,
        "current_counterfactual_map": cf_map,
        "current_factor_or_part_id": factor,
        "current_model_id": model_id,
        "current_protocol": protocol,
        "current_sample_or_pair_id": str(pair_id),
        "historical_dominant": _dominant(historical["E"], historical["C"], historical["F"]),
        "historical_gate": gate,
        "historical_orientation": int(np.sign(endpoint_d)) if gate else 0,
        "identity_match": True,
        "mixture_eta": eta,
        "pair_identity": {
            key: int(row[key])
            for key in (
                "base_id",
                "direction",
                "endpoint_fact_id",
                "endpoint_cf_id",
                "swap_fact_id",
                "swap_cf_id",
            )
        },
        "source_receipt": str(samples_path.with_name("receipt.json").resolve()),
        "source_samples": str(samples_path.resolve()),
        "trajectory_interpretation": ["correct_context", "swapped_context"],
    }
    common: dict[str, Any] = {
        "experiment_family": "controlled_c2_context_swap",
        "reference_run": reference_run,
        "unit_id": (f"C2__{model_id}__wall_{wall_map}__object_{object_map}__pair_{pair_id}"),
        "model_id": model_id,
        "checkpoint_sha256": checkpoint,
        "sample_or_pair_id": str(pair_id),
        "factor_or_part_id": factor,
        "counterfactual_map": cf_map,
        "protocol": protocol,
        "protocol_seed": int(row["seed"]),
        "quadrature_weight": 0.5,
        "endpoint_epsilon": C2_ENDPOINT_EPSILON,
        "endpoint_score_plus": ep_plus,
        "endpoint_score_minus": ep_minus,
        "endpoint_d": endpoint_d,
        **{f"historical_{name}": value for name, value in historical.items()},
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }
    return [
        {
            **common,
            "stage_index": 0,
            "stage_t": 0.0,
            "stage_score_plus": ep_plus,
            "stage_score_minus": ep_minus,
            "stage_r": endpoint_d,
        },
        {
            **common,
            "stage_index": 1,
            "stage_t": 1.0,
            "stage_score_plus": sw_plus,
            "stage_score_minus": sw_minus,
            "stage_r": swap_d,
        },
    ]


def _historical_c1_modules(historical_root: Path) -> tuple[Any, Any]:
    import datetime as datetime_module
    import importlib.util
    import types

    root = historical_root.expanduser().resolve()
    source = root / "src"
    if root.is_symlink() or not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"historical repository is missing or unsafe: {root}")
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError(f"historical source tree is missing or unsafe: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    if not hasattr(datetime_module, "UTC"):
        datetime_module.UTC = (  # type: ignore[attr-defined]
            datetime_module.timezone.utc  # noqa: UP017 - Python 3.10 compatibility
        )
    if importlib.util.find_spec("h5py") is None:
        h5py_stub = types.ModuleType("h5py")

        def unavailable_h5py(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("h5py is unavailable in this verification environment")

        h5py_stub.File = unavailable_h5py  # type: ignore[attr-defined]
        sys.modules["h5py"] = h5py_stub
    from cmr.endpoint_behavior_v1 import phase_b_core, phase_b_runtime

    return phase_b_core, phase_b_runtime


def _c1_source_row(
    results_root: Path,
    selection: Mapping[str, Any],
) -> tuple[pd.Series, dict[str, Any], Path, Path]:
    job_id = str(selection["job_id"])
    job = results_root / "jobs" / "pass2" / job_id
    receipt_path = _regular_file(job / "receipt.json", "C1 receipt")
    samples_path = _regular_file(job / "per_sample.parquet", "C1 per-sample table")
    receipt = _read_receipt(receipt_path)
    if receipt.get("passed") is not True:
        raise ValueError(f"C1 receipt is not passed: {receipt_path}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"C1 receipt lacks output identities: {receipt_path}")
    sealed = outputs.get("per_sample.parquet")
    if not isinstance(sealed, Mapping):
        raise ValueError(f"C1 receipt lacks per-sample identity: {receipt_path}")
    if sealed.get("sha256") != sha256_file(samples_path):
        raise ValueError(f"C1 per-sample SHA-256 mismatch: {samples_path}")

    frame = pd.read_parquet(samples_path)
    required = {
        "model_id",
        "module",
        "factor",
        "evaluation_pass",
        "checkpoint_sha256",
        "geometry",
        "noise_seed",
        "counterfactual_map",
        "sample_id",
        "counterfactual_id",
        "endpoint_factual_score",
        "endpoint_counterfactual_score",
        "endpoint_delta",
        "endpoint_strength",
        "endpoint_active",
        "endpoint_null",
        "E",
        "C",
        "F",
        "Abs",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"C1 per-sample table lacks columns {missing}: {samples_path}")
    selected = frame.loc[
        frame["geometry"].eq("cmmr")
        & frame["noise_seed"].astype(int).eq(int(selection["noise_seed"]))
        & frame["counterfactual_map"].astype(int).eq(int(selection["map_index"]))
        & frame["sample_id"].astype(int).eq(int(selection["sample_id"]))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"C1 registered identity is not unique: {dict(selection)}, rows={len(selected)}"
        )
    row = selected.iloc[0]
    expected_state = str(selection["expected_state"])
    active = bool(row["endpoint_active"])
    null = bool(row["endpoint_null"])
    mixed = float(row["E"]) > 1.0e-5 and float(row["C"]) > 1.0e-5
    state_matches = {
        "active": active,
        "null": null,
        "mixed": active and mixed,
    }
    if expected_state not in state_matches or not state_matches[expected_state]:
        raise ValueError(
            f"C1 registered state {expected_state!r} changed for {job_id}, "
            f"sample {selection['sample_id']}"
        )
    if active == null:
        raise ValueError(f"C1 endpoint gate flags are inconsistent: {samples_path}")

    receipt_job = receipt.get("job")
    if not isinstance(receipt_job, Mapping):
        raise ValueError(f"C1 receipt lacks job identity: {receipt_path}")
    identities = {
        "model_id": row["model_id"],
        "module": row["module"],
        "factor": row["factor"],
        "evaluation_pass": row["evaluation_pass"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    for key, observed in identities.items():
        if str(receipt_job.get(key)) != str(observed):
            raise ValueError(f"C1 receipt/sample mismatch for {key}: {job_id}")
    return row, receipt, receipt_path, samples_path


def _c1_legacy_job(receipt_job: Mapping[str, Any]) -> dict[str, Any]:
    factory = receipt_job.get("factory_metadata")
    if not isinstance(factory, Mapping):
        raise ValueError("C1 receipt lacks factory metadata")
    return {
        **dict(factory),
        "model_id": str(receipt_job["model_id"]),
        "module": str(receipt_job["module"]),
        "factor": str(receipt_job["factor"]),
        "checkpoint_path": str(receipt_job["checkpoint"]),
        "checkpoint_sha256": str(receipt_job["checkpoint_sha256"]),
        "architecture": str(factory["architecture"]),
        "trajectory_id": factory.get("trajectory_id"),
    }


def _c1_sample_position(
    prepared: Any,
    row: pd.Series,
    map_index: int,
) -> int:
    factual_ids = np.asarray(prepared.factual_ids, dtype=np.int64)
    positions = np.flatnonzero(factual_ids == int(row["sample_id"]))
    if positions.size != 1:
        raise ValueError(f"C1 sample ID is absent or repeated: {int(row['sample_id'])}")
    position = int(positions[0])
    maps = prepared.counterfactual_maps
    if map_index < 0 or map_index >= len(maps):
        raise ValueError(f"C1 map index is outside prepared maps: {map_index}")
    if int(maps[map_index][position]) != int(row["counterfactual_id"]):
        raise ValueError("C1 prepared counterfactual identity changed")
    return position


def _replay_c1_scores(
    *,
    core: Any,
    config: Mapping[str, Any],
    receipt_job: Mapping[str, Any],
    legacy_job: Mapping[str, Any],
    prepared: Any,
    model: Any,
    geometry: Any,
    position: int,
    map_index: int,
    noise_seed: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    section = core._pass_config(config, "pass2")
    alpha = core._alpha_grid(section)
    registered_alpha = np.asarray(receipt_job["alpha_grid"], dtype=np.float64)
    if not np.array_equal(alpha, registered_alpha):
        raise ValueError("C1 receipt/config alpha grid changed")
    if noise_seed not in core._noise_seeds(section):
        raise ValueError(f"C1 noise seed is not registered: {noise_seed}")
    specs = core._geometry_specs(config, "pass2", section)
    cmmr = next((spec for spec in specs if spec.name == "cmmr"), None)
    if cmmr is None or not cmmr.primary:
        raise ValueError("C1 historical core lacks the primary CMMR geometry")

    factual_ids = np.asarray(prepared.factual_ids, dtype=np.int64)
    counterfactual_ids = np.asarray(
        prepared.counterfactual_maps[map_index],
        dtype=np.int64,
    )
    batch_size = core._batch_size(config, legacy_job)
    image_path = _regular_file(
        Path(str(receipt_job["processed_images"])),
        "C1 processed images",
    )
    transition_count = alpha.size - 1
    alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))
    selected_factual = None
    selected_counterfactual = None
    selected_standard = None
    batch_start = -1
    batch_stop = -1
    for start in range(0, factual_ids.size, batch_size):
        stop = min(factual_ids.size, start + batch_size)
        factual = core.load_images_tensor(
            image_path,
            factual_ids[start:stop],
            device=device,
        ).float()
        counterfactual = core.load_images_tensor(
            image_path,
            counterfactual_ids[start:stop],
            device=device,
        ).float()
        standard = torch.randn(
            (transition_count, *factual.shape),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        if start <= position < stop:
            local = position - start
            selected_factual = factual[local : local + 1].clone()
            selected_counterfactual = counterfactual[local : local + 1].clone()
            selected_standard = standard[:, local : local + 1].clone()
            batch_start, batch_stop = start, stop
            del factual, counterfactual, standard
            break
        del factual, counterfactual, standard
    if selected_factual is None or selected_counterfactual is None or selected_standard is None:
        raise AssertionError("C1 selected sample was not reached in replay batches")

    center_factual = core.prepare_mean(geometry.mean, selected_factual)
    center_counterfactual = core.prepare_mean(geometry.mean, selected_counterfactual)
    increments = core._transformed_increments(
        selected_standard,
        cmmr,
        geometry.covariance,
    )
    factual_path = core.reverse_markov_path(
        selected_factual,
        center_factual,
        alpha_tensor,
        increments,
    )
    counterfactual_path = core.reverse_markov_path(
        selected_counterfactual,
        center_counterfactual,
        alpha_tensor,
        increments,
    )
    if not torch.equal(factual_path[:, 0], counterfactual_path[:, 0]):
        raise AssertionError("C1 replay lost the shared alpha=0 state")
    q_factual = core._score_path(
        model,
        factual_path,
        device=device,
        batch_size=batch_size,
    )[0].astype(np.float64)
    q_counterfactual = core._score_path(
        model,
        counterfactual_path,
        device=device,
        batch_size=batch_size,
    )[0].astype(np.float64)
    audit = {
        "alpha0_score_abs_error": abs(float(q_factual[0] - q_counterfactual[0])),
        "batch_size": batch_size,
        "drawn_batch_start": batch_start,
        "drawn_batch_stop": batch_stop,
        "full_batch_rng_draw_preserved": True,
        "path_endpoint_score_minus": float(q_counterfactual[-1]),
        "path_endpoint_score_plus": float(q_factual[-1]),
        "selected_position": position,
    }
    del (
        selected_factual,
        selected_counterfactual,
        selected_standard,
        increments,
        factual_path,
        counterfactual_path,
    )
    return q_factual, q_counterfactual, audit


def export_c0(
    historical_root: str | Path,
    results_root: str | Path,
    config_path: str | Path,
    source_config_path: str | Path,
    output: str | Path,
    *,
    selection_manifest: str | Path | None = None,
    diagnostic_audit_output: str | Path | None = None,
    device: str = "cuda:0",
    reference_run: str = f"C0:{C0_REFERENCE_SHA256}",
) -> dict[str, Any]:
    """Replay eight fixed C0 candidates and qualify them against sealed aggregates."""

    import torch

    started = time.perf_counter()
    history = Path(historical_root).expanduser().resolve()
    results = Path(results_root).expanduser().resolve()
    if results.is_symlink() or not results.is_dir():
        raise FileNotFoundError(f"C0 result root is missing or unsafe: {results}")
    config_source = _regular_file(
        Path(config_path).expanduser().resolve(),
        "C0 effective config snapshot",
    )
    authoritative_config = _regular_file(
        Path(source_config_path).expanduser().resolve(),
        "C0 authoritative source config",
    )
    assets = _c0_registered_assets(results, config_source, authoritative_config)
    (
        main_common,
        main_sweep,
        model_adapter,
        v11_channels,
        geometry_class,
        historical_runtime,
    ) = _historical_c0_modules(history, assets["package_path"])
    config = main_common.load_main_config(config_source)
    if main_common.main_output_root(config) != results:
        raise ValueError("C0 effective config does not resolve to the sealed result root")
    epsilon = float(config["response"]["endpoint_sign_epsilon"])
    if epsilon != C0_ENDPOINT_EPSILON:
        raise ValueError("C0 endpoint epsilon changed")
    if tuple(map(int, config["repeats"]["noise_seeds"])) != C0_NOISE_SEEDS:
        raise ValueError("C0 registered noise seeds changed")
    alpha = np.asarray(main_common.alpha_grid(config), dtype=np.float64)
    points = [
        point
        for point in main_common.protocol_points(config)
        if point.family == C0_PROTOCOL_FAMILY
        and np.isclose(float(point.value), C0_PROTOCOL_VALUE, atol=0.0, rtol=0.0)
    ]
    if len(points) != 1:
        raise ValueError("C0 linear lambda=0 protocol point is not unique")
    point = points[0]

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C0 exact replay requires one CUDA device")
    torch.cuda.set_device(target)
    torch.cuda.reset_peak_memory_stats(target)

    delivery_inputs = assets["delivery_inputs"]
    manifest_path = _regular_file(
        results / str(config["models"]["manifest"]),
        "C0 model manifest",
    )
    _delivery_expected_sha(delivery_inputs, manifest_path, "model manifest")
    manifest_frame = pd.read_csv(manifest_path)
    endpoint_frame = _c0_endpoint_rows(assets["endpoint_path"])
    configured_models = {str(model.model_id): model for model in main_common.main_models(config)}
    selected_model_ids = list(
        dict.fromkeys(str(selection["model_id"]) for selection in C0_CANDIDATES)
    )
    missing_models = sorted(set(selected_model_ids) - set(configured_models))
    if missing_models:
        raise ValueError(f"C0 selected models are outside the config: {missing_models}")

    processed_path = _regular_file(
        main_common.resolve_path(config, str(config["frozen_inputs"]["processed_images"])),
        "C0 processed images",
    )
    geometry_path = _regular_file(
        main_common.resolve_path(config, str(config["frozen_inputs"]["geometry"])),
        "C0 covariance geometry",
    )
    geometry = geometry_class.from_geometry_archive(geometry_path)
    if int(geometry.ambient_dimension) != 3 * 32 * 32:
        raise ValueError("C0 covariance geometry is not 32x32 RGB")
    information = main_sweep._load_information_maps(results)
    if not np.array_equal(alpha, information.alphas):
        raise ValueError("C0 config and sealed information-map alpha grids disagree")
    protocol_index = information.protocol_index(point)
    common_inverse = information.alpha_of_c[protocol_index]
    common_grid = information.c_grid
    full_window = tuple(map(float, config["integration"]["full_window"]))

    dynamic_ids_by_task: dict[str, np.ndarray] = {}
    endpoint_ids_by_task: dict[str, np.ndarray] = {}
    position_by_task_base: dict[tuple[str, int], int] = {}
    for task in dict.fromkeys(str(selection["task"]) for selection in C0_CANDIDATES):
        dynamic_path = _regular_file(
            results / "data" / f"{task}_dynamic_base_ids.npy",
            "C0 dynamic ID lock",
        )
        endpoint_ids_path = _regular_file(
            results / "data" / f"{task}_endpoint_base_ids.npy",
            "C0 endpoint ID lock",
        )
        _delivery_expected_sha(delivery_inputs, dynamic_path, f"{task} dynamic ID lock")
        _delivery_expected_sha(
            delivery_inputs,
            endpoint_ids_path,
            f"{task} endpoint ID lock",
        )
        dynamic_ids = np.load(dynamic_path, allow_pickle=False).astype(
            np.int64,
            copy=False,
        )
        endpoint_ids = np.load(endpoint_ids_path, allow_pickle=False).astype(
            np.int64,
            copy=False,
        )
        expected_dynamic = int(config["data"]["dynamic_n_per_task"])
        expected_endpoint = int(config["data"]["endpoint_n_per_task"])
        if dynamic_ids.size != expected_dynamic or endpoint_ids.size != expected_endpoint:
            raise ValueError(f"C0 registered lock size changed for task {task}")
        selected_bases = sorted(
            {
                int(selection["base_id"])
                for selection in C0_CANDIDATES
                if str(selection["task"]) == task
            }
        )
        positions: list[int] = []
        for base_id in selected_bases:
            matches = np.flatnonzero(dynamic_ids == base_id)
            if matches.size != 1:
                raise ValueError(f"C0 base ID is absent or repeated: task={task}, base={base_id}")
            position = int(matches[0])
            position_by_task_base[(task, base_id)] = position
            positions.append(position)
        dynamic_ids_by_task[task] = dynamic_ids
        endpoint_ids_by_task[task] = endpoint_ids

    replay_noise_prefix: dict[int, torch.Tensor] = {}
    dynamic_count = int(config["data"]["dynamic_n_per_task"])
    maximum_historical_batch = max(int(value) for value in config["runtime"]["batch_size"].values())
    noise_prefix_count = min(dynamic_count, maximum_historical_batch)
    noise_shape = (dynamic_count, int(geometry.ambient_dimension))
    for noise_seed in C0_NOISE_SEEDS:
        standard = main_sweep._standard_normal(noise_shape, noise_seed, target)
        isotropic_standard = main_sweep._standard_normal(
            noise_shape,
            noise_seed + 1_000_003,
            target,
        )
        eta_covariance, eta_isotropic = v11_channels.linear_base_noises_torch(
            standard,
            isotropic_standard,
            geometry.covariance,
        )
        noise = main_sweep._noise_for_point(
            point,
            standard=standard,
            eta_covariance=eta_covariance,
            eta_isotropic=eta_isotropic,
            covariance=geometry.covariance,
        )
        replay_noise_prefix[noise_seed] = noise[:noise_prefix_count].clone()
        del standard, isotropic_standard, eta_covariance, eta_isotropic, noise
    torch.cuda.empty_cache()

    raw_responses: dict[tuple[str, int], np.ndarray] = {}
    scorer_attempts: dict[str, list[int]] = {}
    aggregate_audit_rows: list[dict[str, Any]] = []
    aggregate_audits: list[dict[str, Any]] = []
    candidate_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    candidate_sources: dict[tuple[str, int], dict[str, Any]] = {}
    loaded = None
    try:
        for model_id in selected_model_ids:
            model_selections = [
                selection
                for selection in C0_CANDIDATES
                if str(selection["model_id"]) == model_id
            ]
            task = str(model_selections[0]["task"])
            factor = str(model_selections[0]["factor"])
            if any(
                str(selection["task"]) != task or str(selection["factor"]) != factor
                for selection in model_selections
            ):
                raise ValueError(f"C0 model selection does not share task/factor: {model_id}")
            model_spec = configured_models[model_id]
            manifest_rows = manifest_frame.loc[manifest_frame["model_id"].astype(str).eq(model_id)]
            if len(manifest_rows) != 1:
                raise ValueError(f"C0 model manifest identity is not unique: {model_id}")
            manifest = manifest_rows.iloc[0]
            expected_manifest = {
                "task_name": task,
                "architecture": model_selections[0]["architecture"],
                "seed": model_selections[0]["model_seed"],
            }
            for key, value in expected_manifest.items():
                if str(manifest[key]) != str(value):
                    raise ValueError(f"C0 model manifest changed for {key}: {model_id}")

            checkpoint_path = _regular_file(
                Path(str(manifest["checkpoint_path"])),
                "C0 checkpoint",
            )
            checkpoint_sha = str(manifest["checkpoint_sha256"])
            if sha256_file(checkpoint_path) != checkpoint_sha:
                raise ValueError(f"C0 checkpoint SHA-256 changed: {checkpoint_path}")
            probability_cache_path = _regular_file(
                Path(str(manifest["probability_cache_path"])),
                "C0 probability cache",
            )
            if sha256_file(probability_cache_path) != str(manifest["probability_cache_sha256"]):
                raise ValueError(f"C0 probability cache SHA-256 changed: {model_id}")

            job = results / "jobs" / model_id
            model_audit_path = _regular_file(job / "audit.json", "C0 model audit")
            summary_path = _regular_file(
                job / "sample_auc_selected.parquet",
                "C0 sealed sample summary",
            )
            _delivery_expected_sha(delivery_inputs, model_audit_path, "model audit")
            summary_sha = _delivery_expected_sha(
                delivery_inputs,
                summary_path,
                "sealed sample summary",
            )
            model_audit = _read_receipt(model_audit_path)
            if (
                model_audit.get("completed") is not True
                or str(model_audit.get("model_id")) != model_id
                or str(model_audit.get("checkpoint_sha256")) != checkpoint_sha
            ):
                raise ValueError(f"C0 model audit identity is not complete: {model_id}")
            configured_batch_size = int(
                config["runtime"]["batch_size"][str(model_spec.architecture)]
            )
            historical_batch_size = int(model_audit.get("final_batch_size", 0))
            if (
                int(model_audit.get("dynamic_images", 0)) != dynamic_count
                or int(model_audit.get("configured_batch_size", 0)) != configured_batch_size
                or historical_batch_size < 1
            ):
                raise ValueError(f"C0 sealed forward layout changed: {model_id}")

            base_ids = [int(selection["base_id"]) for selection in model_selections]
            sealed_summary = pd.read_parquet(
                summary_path,
                columns=[
                    "model_id",
                    "base_id",
                    "factor",
                    "family",
                    "protocol_value",
                    "protocol_name",
                    "n_repeats_averaged",
                    "endpoint_abs",
                    "auc_abs_info",
                    "auc_align_info",
                    "auc_opp_info",
                    "auc_null_info",
                ],
                filters=[("base_id", "in", base_ids), ("factor", "==", factor)],
                use_threads=False,
            )
            sealed_summary = sealed_summary.loc[
                sealed_summary["family"].astype(str).eq(C0_PROTOCOL_FAMILY)
                & np.isclose(
                    sealed_summary["protocol_value"].astype(float),
                    C0_PROTOCOL_VALUE,
                    atol=0.0,
                    rtol=0.0,
                )
            ]

            loaded = model_adapter.load_validated_checkpoint(
                checkpoint_path,
                task=str(model_spec.task),
                architecture=str(model_spec.architecture),
                seed=int(model_spec.seed),
                device=target,
            )
            if loaded.checkpoint_sha256 != checkpoint_sha:
                raise ValueError(f"C0 checkpoint identity changed during load: {model_id}")

            dynamic_ids = dynamic_ids_by_task[task]
            positions = np.asarray(
                [position_by_task_base[(task, base_id)] for base_id in base_ids],
                dtype=np.int64,
            )
            variants, _endpoint_abs = main_sweep._load_variants(
                config,
                results,
                model_spec,
                manifest,
                endpoint_ids_by_task[task],
                dynamic_ids,
            )
            if int(model_audit.get("variant_count", 0)) != len(variants):
                raise ValueError(f"C0 sealed variant count changed: {model_id}")
            stack_size = len(variants) + 1
            replay_count = _c0_replay_prefix_count(
                dynamic_count=dynamic_count,
                stack_size=stack_size,
                historical_batch_size=historical_batch_size,
                selected_positions=positions,
            )
            if replay_count > noise_prefix_count:
                raise AssertionError("C0 retained noise prefix is too short")
            map_variant_indices: dict[str, int] = {}
            for map_seed in ("20260882", "20260883"):
                matches = [
                    index
                    for index, variant in enumerate(variants)
                    if str(variant.factor) == factor and str(variant.cf_map_seed) == map_seed
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "C0 historical variants do not uniquely bind "
                        f"{model_id}/{factor}/{map_seed}"
                    )
                map_variant_indices[map_seed] = matches[0]

            endpoint_deltas = np.stack(
                [variant.dynamic_endpoint_delta[:replay_count] for variant in variants],
                axis=1,
            ).astype(np.float64, copy=False)
            endpoint_rows_for_model: dict[tuple[int, str], pd.Series] = {}
            for selection in model_selections:
                local = base_ids.index(int(selection["base_id"]))
                for map_seed, variant_index in map_variant_indices.items():
                    endpoint_row = _c0_endpoint_row(
                        endpoint_frame,
                        selection,
                        map_seed,
                        check_expected_state=(map_seed == str(selection["cf_map_seed"])),
                    )
                    endpoint_rows_for_model[(int(selection["base_id"]), map_seed)] = endpoint_row
                    if int(endpoint_row["counterfactual_id"]) != int(
                        variants[variant_index].counterfactual_ids[positions[local]]
                    ):
                        raise ValueError(f"C0 counterfactual identity changed: {dict(selection)}")
                    position = int(positions[local])
                    historical_delta = float(endpoint_deltas[position, variant_index])
                    sealed_delta = float(endpoint_row["delta_endpoint"])
                    if not np.isclose(
                        historical_delta,
                        sealed_delta,
                        atol=1.0e-12,
                        rtol=0.0,
                    ):
                        raise ValueError(f"C0 endpoint cache/table disagreement: {dict(selection)}")
                    endpoint_deltas[position, variant_index] = sealed_delta

            factual = model_adapter.processed_images_to_nchw(
                processed_path,
                dynamic_ids[:replay_count],
                device=target,
            )
            tensors = [factual]
            tensors.extend(
                model_adapter.processed_images_to_nchw(
                    processed_path,
                    variant.counterfactual_ids[:replay_count],
                    device=target,
                )
                for variant in variants
            )
            clean_stack = torch.stack(tensors, dim=1)
            del tensors, factual
            scorer = main_sweep.BatchBackoff(historical_batch_size, [])
            aggregate_sums = {
                base_id: {name: 0.0 for name in ("Abs", "E", "C", "F")} for base_id in base_ids
            }
            repeat_count = 0
            for noise_seed in C0_NOISE_SEEDS:
                noise = replay_noise_prefix[noise_seed][:replay_count]
                delta = main_sweep._delta_curves(
                    loaded.model,
                    clean_stack,
                    endpoint_deltas,
                    noise,
                    geometry.mean,
                    alpha,
                    scorer,
                    target,
                )
                if scorer.selected != historical_batch_size:
                    raise RuntimeError(
                        "C0 exact replay cannot change the sealed forward batch size: "
                        f"model={model_id}, sealed={historical_batch_size}, "
                        f"observed={scorer.selected}"
                    )
                if float(np.max(np.abs(delta[:, :, 0]), initial=0.0)) != 0.0:
                    raise ValueError(f"C0 alpha=0 response is not exactly zero: {model_id}")
                for map_seed, variant_index in map_variant_indices.items():
                    components = main_sweep._component_matrices(
                        endpoint_deltas[:, variant_index],
                        delta[variant_index],
                        epsilon,
                    )
                    aucs = main_sweep._auc_components(
                        alpha,
                        components,
                        common_inverse,
                        common_grid,
                        full_window,
                    )
                    for local, base_id in enumerate(base_ids):
                        position = int(positions[local])
                        aggregate_sums[base_id]["Abs"] += float(aucs["abs"][position])
                        aggregate_sums[base_id]["E"] += float(aucs["align"][position])
                        aggregate_sums[base_id]["C"] += float(aucs["opp"][position])
                        aggregate_sums[base_id]["F"] += float(aucs["null"][position])
                        selection = model_selections[local]
                        if noise_seed == int(selection["noise_seed"]) and map_seed == str(
                            selection["cf_map_seed"]
                        ):
                            raw_responses[(model_id, base_id)] = np.asarray(
                                delta[variant_index, position],
                                dtype=np.float64,
                            ).copy()
                    repeat_count += 1
                del delta, noise
            if repeat_count != 6:
                raise AssertionError("C0 sealed-summary replay did not cover six repeats")

            for local, selection in enumerate(model_selections):
                base_id = int(selection["base_id"])
                summary_rows = sealed_summary.loc[
                    sealed_summary["model_id"].astype(str).eq(model_id)
                    & sealed_summary["base_id"].astype(int).eq(base_id)
                    & sealed_summary["factor"].astype(str).eq(factor)
                ]
                if len(summary_rows) != 1:
                    raise ValueError(f"C0 sealed summary identity is not unique: {dict(selection)}")
                sealed = summary_rows.iloc[0]
                if int(sealed["n_repeats_averaged"]) != 6:
                    raise ValueError(f"C0 sealed summary repeat count changed: {dict(selection)}")
                sealed_endpoint_abs = float(sealed["endpoint_abs"])
                replay_endpoint_abs = float(
                    np.mean(
                        [
                            abs(
                                float(
                                    endpoint_rows_for_model[(base_id, map_seed)]["delta_endpoint"]
                                )
                            )
                            for map_seed in map_variant_indices
                        ],
                        dtype=np.float64,
                    )
                )
                recomputed = {
                    "endpoint_abs": replay_endpoint_abs,
                    **{
                        name: aggregate_sums[base_id][component] / repeat_count
                        for name, component in (
                            ("auc_abs_info", "Abs"),
                            ("auc_align_info", "E"),
                            ("auc_opp_info", "C"),
                            ("auc_null_info", "F"),
                        )
                    },
                }
                sealed_values = {
                    "endpoint_abs": sealed_endpoint_abs,
                    "auc_abs_info": float(sealed["auc_abs_info"]),
                    "auc_align_info": float(sealed["auc_align_info"]),
                    "auc_opp_info": float(sealed["auc_opp_info"]),
                    "auc_null_info": float(sealed["auc_null_info"]),
                }
                errors = {
                    name: abs(recomputed[name] - sealed_values[name]) for name in sealed_values
                }
                maximum_error = max(errors.values())
                passed = maximum_error <= C0_SEALED_AUDIT_ATOL
                audit = {
                    "comparison_scope": "two counterfactual maps x three noise seeds",
                    "forward_layout": {
                        "dynamic_images": dynamic_count,
                        "stack_size": stack_size,
                        "historical_flat_state_count": dynamic_count * stack_size,
                        "historical_batch_size": historical_batch_size,
                        "retained_dynamic_prefix": replay_count,
                        "retained_flat_state_count": replay_count * stack_size,
                        "selected_dynamic_position": int(positions[local]),
                    },
                    "maximum_absolute_error": maximum_error,
                    "absolute_errors": errors,
                    "passed": passed,
                    "recomputed": recomputed,
                    "sealed": sealed_values,
                    "tolerance": C0_SEALED_AUDIT_ATOL,
                }
                aggregate_audits.append(
                    {
                        "model_id": model_id,
                        "base_id": base_id,
                        "factor": factor,
                        "protocol": str(sealed["protocol_name"]),
                        "audit": audit,
                    }
                )
                aggregate_audit_rows.append(
                    {
                        "model_id": model_id,
                        "base_id": base_id,
                        "factor": factor,
                        "protocol": str(sealed["protocol_name"]),
                        "n_repeats": repeat_count,
                        **{f"sealed_{name}": value for name, value in sealed_values.items()},
                        **{f"recomputed_{name}": value for name, value in recomputed.items()},
                        **{f"abs_error_{name}": value for name, value in errors.items()},
                        "maximum_absolute_error": maximum_error,
                        "passed": passed,
                    }
                )
                if (model_id, base_id) not in raw_responses:
                    raise AssertionError(f"C0 raw trajectory was not replayed: {dict(selection)}")
                endpoint_row = endpoint_rows_for_model[(base_id, str(selection["cf_map_seed"]))]
                historical = _c0_direct_summary(
                    main_sweep,
                    float(endpoint_row["delta_endpoint"]),
                    raw_responses[(model_id, base_id)],
                    alpha,
                    epsilon,
                )
                rows_for_unit = _c0_trajectory_rows(
                    selection=selection,
                    endpoint_row=endpoint_row,
                    response=raw_responses[(model_id, base_id)],
                    alpha=alpha,
                    historical=historical,
                    checkpoint_sha256=checkpoint_sha,
                    endpoint_path=assets["endpoint_path"],
                    sealed_summary_path=summary_path,
                    sealed_audit=audit,
                    position=int(positions[local]),
                    reference_run=reference_run,
                )
                map_path = _regular_file(
                    main_sweep._map_path(
                        results,
                        task,
                        factor,
                        str(selection["cf_map_seed"]),
                    ),
                    "C0 counterfactual map",
                )
                map_sha = _delivery_expected_sha(
                    delivery_inputs,
                    map_path,
                    "counterfactual map",
                )
                unit_id = str(rows_for_unit[0]["unit_id"])
                identity = _c0_candidate_identity(selection)
                candidate_rows[identity] = rows_for_unit
                candidate_sources[identity] = {
                    **dict(selection),
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha,
                    "counterfactual_id": int(endpoint_row["counterfactual_id"]),
                    "counterfactual_map_path": str(map_path.resolve()),
                    "counterfactual_map_sha256": map_sha,
                    "endpoint_samples": str(assets["endpoint_path"].resolve()),
                    "endpoint_samples_sha256": assets["endpoint_sha256"],
                    "position": int(positions[local]),
                    "sealed_aggregate_audit": audit,
                    "sealed_summary": str(summary_path.resolve()),
                    "sealed_summary_sha256": summary_sha,
                    "unit_id": unit_id,
                }
                print(
                    json.dumps(
                        {
                            "event": "c0_candidate_qualification",
                            "candidate": {
                                "model_id": model_id,
                                "base_id": base_id,
                            },
                            "candidates_evaluated": len(aggregate_audits),
                            "candidate_count": len(C0_CANDIDATES),
                            "candidate_qualified": passed,
                            "maximum_absolute_error": maximum_error,
                            "qualification_tolerance": C0_SEALED_AUDIT_ATOL,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            scorer_attempts[model_id] = list(map(int, scorer.attempts))
            del clean_stack, loaded
            loaded = None
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if loaded is not None:
            del loaded
        replay_noise_prefix.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_seconds = time.perf_counter() - started
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(target))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(target))
    aggregate_audit = pd.DataFrame(aggregate_audit_rows).sort_values(
        ["model_id", "base_id"],
        kind="stable",
    )
    qualified, excluded = _c0_partition_candidate_audits(
        C0_CANDIDATES,
        aggregate_audits,
    )
    qualified_identities = [_c0_candidate_identity(item) for item in qualified]
    all_identities = {_c0_candidate_identity(item) for item in C0_CANDIDATES}
    if set(candidate_rows) != all_identities or set(candidate_sources) != all_identities:
        raise AssertionError("C0 replay did not materialize every fixed candidate")
    rows = [
        row
        for identity in qualified_identities
        for row in candidate_rows[identity]
    ]
    sources = [candidate_sources[identity] for identity in qualified_identities]
    coverage = _c0_qualified_coverage(qualified, rows)
    if coverage["passed"] and excluded:
        qualification_status = "STRICT_AGGREGATE_QUALIFIED_WITH_EXCLUSIONS"
    elif coverage["passed"]:
        qualification_status = "STRICT_AGGREGATE_QUALIFIED"
    else:
        qualification_status = "FAIL_COVERAGE"
    diagnostic_path = (
        Path(diagnostic_audit_output).expanduser().resolve()
        if diagnostic_audit_output is not None
        else Path(output).expanduser().resolve().with_name(
            f"{Path(output).stem}_candidate_qualification.json"
        )
    )
    if diagnostic_path.is_relative_to(history):
        raise ValueError("C0 diagnostic audit output cannot modify the historical repository")
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_payload = {
        "artifact_type": "c0_fixed_candidate_qualification_diagnostic",
        "acceptance_effect": (
            "strict per-candidate qualification; exclusions are never replaced; "
            "formal output requires the qualified subset to retain C0 coverage"
        ),
        "completed_all_candidates": len(aggregate_audits) == len(C0_CANDIDATES),
        "candidate_count": len(C0_CANDIDATES),
        "selected_count": len(qualified),
        "excluded_count": len(excluded),
        "status": qualification_status,
        "tolerance": C0_SEALED_AUDIT_ATOL,
        "maximum_absolute_error": float(
            aggregate_audit["maximum_absolute_error"].max()
        ),
        "coverage": coverage,
        "selected": qualified,
        "excluded": excluded,
        "runtime": {
            "device": str(target),
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_allocated_bytes": peak_allocated_bytes,
            "peak_cuda_memory_reserved_bytes": peak_reserved_bytes,
            "scorer_batch_attempts": scorer_attempts,
        },
        "historical_package": str(assets["package_path"].resolve()),
        "historical_package_sha256": assets["package_sha256"],
        "historical_runtime": historical_runtime,
        "audits": aggregate_audits,
    }
    temporary_diagnostic = diagnostic_path.with_suffix(diagnostic_path.suffix + ".tmp")
    temporary_diagnostic.write_text(
        json.dumps(diagnostic_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_diagnostic.replace(diagnostic_path)
    diagnostic_sha256 = sha256_file(diagnostic_path)
    print(
        json.dumps(
            {
                "event": "c0_candidate_qualification_summary",
                "qualification": f"{len(qualified)}/{len(C0_CANDIDATES)}",
                "candidate_count": len(C0_CANDIDATES),
                "selected_count": len(qualified),
                "excluded_count": len(excluded),
                "coverage_gate_satisfied": bool(coverage["passed"]),
                "status": qualification_status,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not coverage["passed"]:
        raise ValueError(
            "C0 qualified candidates failed the predeclared coverage gate: "
            f"{json.dumps(coverage, sort_keys=True)}"
        )
    if not rows or not sources:
        raise AssertionError("C0 qualification produced no formal trajectory rows")

    active_count = int(coverage["observed"]["active_count"])
    null_count = int(coverage["observed"]["null_count"])
    mixed_count = int(coverage["observed"]["mixed_unit_count"])

    frame = pd.DataFrame(rows, columns=NEUTRAL_COLUMNS)
    destination = write_trajectory_record(frame, output)
    manifest_path_out = (
        Path(selection_manifest)
        if selection_manifest is not None
        else destination.with_name(f"{destination.stem}_selection.json")
    )
    manifest_path_out.parent.mkdir(parents=True, exist_ok=True)
    sealed_audit_path = manifest_path_out.with_name(
        f"{manifest_path_out.stem}_sealed_aggregate_audit.csv"
    )
    aggregate_audit.to_csv(sealed_audit_path, index=False)
    manifest = {
        "schema_version": 1,
        "experiment_family": "controlled_c0_protocol_robust",
        "reference_run": reference_run,
        "historical_repository": str(history),
        "historical_effective_config": str(config_source),
        "historical_effective_config_sha256": sha256_file(config_source),
        "historical_authoritative_source_config": str(authoritative_config),
        "historical_authoritative_source_config_sha256": sha256_file(authoritative_config),
        "historical_delivery_receipt": str(assets["delivery_path"].resolve()),
        "historical_delivery_receipt_sha256": sha256_file(assets["delivery_path"]),
        "historical_package": str(assets["package_path"].resolve()),
        "historical_package_sha256": assets["package_sha256"],
        "historical_runtime": historical_runtime,
        "quantity_provenance": {
            "independently_sealed": [
                "checkpoint/sample/map/noise/config identities",
                "endpoint factual and counterfactual probabilities",
                "endpoint delta and gate",
                "six-repeat sample_auc_selected aggregate summaries",
            ],
            "regenerated": [
                "raw stage response r(alpha)",
                "per-unit direct-alpha M/E/C/F/Abs used by current-core comparison",
            ],
            "independent_aggregate_audit": (
                "all eight fixed candidates were replayed over both maps and all three noise seeds "
                "and compared to separately sealed sample_auc_selected common-information AUC rows"
            ),
        },
        "selection_algorithm": (
            "eight fixed predeclared C0 candidates; qualify only candidates whose sealed "
            "six-repeat common-information maximum absolute error is <=5e-4; exclude failures "
            "without replacement; publish only if the qualified subset retains the registered "
            "architecture, active/null, mixed-E/C, and counterfactual-map coverage"
        ),
        "runtime": {
            "device": str(target),
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_allocated_bytes": peak_allocated_bytes,
            "peak_cuda_memory_reserved_bytes": peak_reserved_bytes,
            "scorer_batch_attempts": scorer_attempts,
        },
        "sealed_aggregate_audit": {
            "path": str(sealed_audit_path.resolve()),
            "sha256": sha256_file(sealed_audit_path),
            "row_count": int(len(aggregate_audit)),
            "pass_count": int(aggregate_audit["passed"].astype(bool).sum()),
            "failure_count": int((~aggregate_audit["passed"].astype(bool)).sum()),
            "qualified_count": int(aggregate_audit["passed"].astype(bool).sum()),
            "excluded_count": int((~aggregate_audit["passed"].astype(bool)).sum()),
            "maximum_absolute_error": float(aggregate_audit["maximum_absolute_error"].max()),
            "tolerance": C0_SEALED_AUDIT_ATOL,
        },
        "candidate_qualification": _c0_candidate_qualification_record(
            qualified=qualified,
            excluded=excluded,
            coverage=coverage,
            status=qualification_status,
            diagnostic_path=diagnostic_path,
            diagnostic_sha256=diagnostic_sha256,
        ),
        "sources": sources,
        "trajectory_record": str(destination.resolve()),
        "trajectory_record_sha256": sha256_file(destination),
        "unit_count": int(frame["unit_id"].nunique()),
        "row_count": int(len(frame)),
        "active_unit_count": active_count,
        "null_unit_count": null_count,
        "mixed_unit_count": mixed_count,
        "units": sorted(frame["unit_id"].unique().tolist()),
    }
    manifest_path_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "trajectory_record": destination,
        "selection_manifest": manifest_path_out,
        "sealed_aggregate_audit": sealed_audit_path,
        "candidate_qualification_diagnostic": diagnostic_path,
        "candidate_count": len(C0_CANDIDATES),
        "selected_count": len(qualified),
        "excluded_count": len(excluded),
        "qualification_status": qualification_status,
        "unit_count": manifest["unit_count"],
        "row_count": manifest["row_count"],
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_allocated_bytes": peak_allocated_bytes,
        "peak_cuda_memory_reserved_bytes": peak_reserved_bytes,
        "sealed_aggregate_maximum_absolute_error": manifest["sealed_aggregate_audit"][
            "maximum_absolute_error"
        ],
    }


def export_c1(
    historical_root: str | Path,
    results_root: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    selection_manifest: str | Path | None = None,
    device: str = "cuda:0",
    reference_run: str = f"C1:{C1_REFERENCE_SHA256}",
) -> dict[str, Any]:
    """Replay ten exact C1 CMMR sample trajectories with historical primitives."""

    import torch

    started = time.perf_counter()
    history = Path(historical_root).expanduser().resolve()
    results = Path(results_root).expanduser().resolve()
    config_source = _regular_file(
        Path(config_path).expanduser().resolve(),
        "C1 historical config",
    )
    if results.is_symlink() or not results.is_dir():
        raise FileNotFoundError(f"C1 result root is missing or unsafe: {results}")
    core, runtime = _historical_c1_modules(history)
    config = runtime.load_phase_b_config(config_source)
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C1 exact replay requires one CUDA device")
    torch.cuda.set_device(target)
    torch.cuda.reset_peak_memory_stats(target)
    torch.set_float32_matmul_precision(
        str(config.get("hardware", {}).get("matmul_precision", "high"))
    )

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    model = None
    loaded_checkpoint: str | None = None
    geometry = None
    geometry_path: Path | None = None
    try:
        for selection in C1_SELECTION:
            row, receipt, receipt_path, samples_path = _c1_source_row(
                results,
                selection,
            )
            receipt_job = receipt["job"]
            legacy_job = _c1_legacy_job(receipt_job)
            checkpoint = str(receipt_job["checkpoint_sha256"])
            if checkpoint != loaded_checkpoint:
                if model is not None:
                    del model
                    model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                checkpoint_path = _regular_file(
                    Path(str(receipt_job["checkpoint"])),
                    "C1 checkpoint",
                )
                model, _payload, observed = core._load_fp32_checkpoint(
                    checkpoint_path,
                    job=legacy_job,
                    device=target,
                )
                if observed != checkpoint:
                    raise ValueError("C1 checkpoint identity changed during loading")
                loaded_checkpoint = checkpoint

            local_geometry_path = _regular_file(
                Path(str(receipt_job["geometry"])),
                "C1 covariance geometry",
            )
            if geometry is None:
                geometry = core.load_geometry(
                    local_geometry_path,
                    include_empirical=False,
                )
                geometry_path = local_geometry_path
                if geometry.covariance.ambient_dimension != 3 * 32 * 32:
                    raise ValueError("C1 covariance geometry is not 32x32 RGB")
            elif local_geometry_path != geometry_path:
                raise ValueError("C1 selected units do not share one geometry")

            prepared = core.load_prepared_samples(
                config,
                pass_name="pass2",
                factor=str(receipt_job["factor"]),
                sample_prefix=str(receipt_job["sample_prefix"]),
            )
            position = _c1_sample_position(
                prepared,
                row,
                int(selection["map_index"]),
            )
            q_plus, q_minus, replay_audit = _replay_c1_scores(
                core=core,
                config=config,
                receipt_job=receipt_job,
                legacy_job=legacy_job,
                prepared=prepared,
                model=model,
                geometry=geometry,
                position=position,
                map_index=int(selection["map_index"]),
                noise_seed=int(selection["noise_seed"]),
                device=target,
            )
            endpoint_plus = float(row["endpoint_factual_score"])
            endpoint_minus = float(row["endpoint_counterfactual_score"])
            endpoint_d = float(row["endpoint_delta"])
            source_score_delta = float(
                row["endpoint_factual_score"] - row["endpoint_counterfactual_score"]
            )
            if not np.isclose(
                source_score_delta,
                endpoint_d,
                atol=1.0e-10,
                rtol=0.0,
            ):
                raise ValueError(f"C1 sealed endpoint identity failed: {samples_path}")
            replay_audit["path_vs_sealed_endpoint_plus_abs_error"] = abs(
                float(q_plus[-1]) - endpoint_plus
            )
            replay_audit["path_vs_sealed_endpoint_minus_abs_error"] = abs(
                float(q_minus[-1]) - endpoint_minus
            )
            response = q_plus - q_minus
            response[-1] = endpoint_d
            alpha = np.asarray(receipt_job["alpha_grid"], dtype=np.float64)
            weights = trapezoid_weights(alpha)
            historical = {
                "M": abs(endpoint_d),
                **{name: float(row[name]) for name in ("E", "C", "F", "Abs")},
            }
            if not np.isclose(
                historical["Abs"],
                historical["E"] + historical["C"] + historical["F"],
                atol=1.0e-10,
                rtol=0.0,
            ):
                raise ValueError(f"C1 sealed component identity failed: {samples_path}")
            gate = bool(row["endpoint_active"])
            orientation = int(np.sign(endpoint_d)) if gate else 0
            model_id = str(row["model_id"])
            factor = str(row["factor"])
            map_name = f"map_{int(selection['map_index'])}"
            protocol = "cmmr"
            sample_id = str(int(row["sample_id"]))
            unit_id = (
                f"C1__{model_id}__{factor}__{map_name}"
                f"__noise_{int(selection['noise_seed'])}__sample_{sample_id}"
            )
            metadata = {
                "bridge_kind": "historical_cmmr_exact_replay",
                "coverage": str(selection["coverage"]),
                "current_checkpoint_sha256": checkpoint,
                "current_counterfactual_map": map_name,
                "current_factor_or_part_id": factor,
                "current_model_id": model_id,
                "current_protocol": protocol,
                "current_sample_or_pair_id": sample_id,
                "historical_dominant": _dominant(
                    historical["E"],
                    historical["C"],
                    historical["F"],
                ),
                "historical_gate": gate,
                "historical_orientation": orientation,
                "identity_match": True,
                "replay_audit": replay_audit,
                "endpoint_arithmetic_provenance": (
                    "authoritative sealed float32 endpoint_delta; raw q+/q- "
                    "retained here because binary64 subtraction can differ by an FP32 ULP"
                ),
                "sealed_endpoint_score_arithmetic_residual": (
                    endpoint_plus - endpoint_minus - endpoint_d
                ),
                "sealed_endpoint_score_minus": endpoint_minus,
                "sealed_endpoint_score_plus": endpoint_plus,
                "sealed_endpoint_scores_used_at_alpha_1": True,
                "source_receipt": str(receipt_path.resolve()),
                "source_samples": str(samples_path.resolve()),
            }
            common = {
                "experiment_family": "controlled_c1_endpoint_behavior",
                "reference_run": reference_run,
                "unit_id": unit_id,
                "model_id": model_id,
                "checkpoint_sha256": checkpoint,
                "sample_or_pair_id": sample_id,
                "factor_or_part_id": factor,
                "counterfactual_map": map_name,
                "protocol": protocol,
                "protocol_seed": int(selection["noise_seed"]),
                "endpoint_epsilon": C1_ENDPOINT_EPSILON,
                "endpoint_score_plus": np.nan,
                "endpoint_score_minus": np.nan,
                "endpoint_d": endpoint_d,
                **{f"historical_{name}": value for name, value in historical.items()},
                "metadata_json": json.dumps(
                    metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            for stage_index, stage_t in enumerate(alpha):
                is_endpoint = stage_index == alpha.size - 1
                rows.append(
                    {
                        **common,
                        "stage_index": stage_index,
                        "stage_t": float(stage_t),
                        "quadrature_weight": float(weights[stage_index]),
                        "stage_score_plus": (np.nan if is_endpoint else float(q_plus[stage_index])),
                        "stage_score_minus": (
                            np.nan if is_endpoint else float(q_minus[stage_index])
                        ),
                        "stage_r": float(response[stage_index]),
                    }
                )
            sources.append(
                {
                    **dict(selection),
                    "checkpoint": str(receipt_job["checkpoint"]),
                    "checkpoint_sha256": checkpoint,
                    "counterfactual_id": int(row["counterfactual_id"]),
                    "position": position,
                    "receipt": str(receipt_path.resolve()),
                    "receipt_sha256": sha256_file(receipt_path),
                    "samples": str(samples_path.resolve()),
                    "samples_sha256": sha256_file(samples_path),
                    "unit_id": unit_id,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "c1_replay_unit_complete",
                        "unit_id": unit_id,
                        "units_complete": len(sources),
                        "units_total": len(C1_SELECTION),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_seconds = time.perf_counter() - started
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(target))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(target))
    frame = pd.DataFrame(rows, columns=NEUTRAL_COLUMNS)
    destination = write_trajectory_record(frame, output)
    manifest_path = (
        Path(selection_manifest)
        if selection_manifest is not None
        else destination.with_name(f"{destination.stem}_selection.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment_family": "controlled_c1_endpoint_behavior",
        "reference_run": reference_run,
        "historical_repository": str(history),
        "historical_config": str(config_source),
        "historical_config_sha256": sha256_file(config_source),
        "selection_algorithm": "fixed registered exact CMMR cells and sample IDs",
        "runtime": {
            "device": str(target),
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_allocated_bytes": peak_allocated_bytes,
            "peak_cuda_memory_reserved_bytes": peak_reserved_bytes,
        },
        "sources": sources,
        "trajectory_record": str(destination.resolve()),
        "trajectory_record_sha256": sha256_file(destination),
        "unit_count": int(frame["unit_id"].nunique()),
        "row_count": int(len(frame)),
        "units": sorted(frame["unit_id"].unique().tolist()),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "trajectory_record": destination,
        "selection_manifest": manifest_path,
        "unit_count": manifest["unit_count"],
        "row_count": manifest["row_count"],
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_allocated_bytes": peak_allocated_bytes,
        "peak_cuda_memory_reserved_bytes": peak_reserved_bytes,
    }


def export_c2(
    root: str | Path,
    output: str | Path,
    *,
    selection_manifest: str | Path | None = None,
    seed: int = 7101,
    object_map: int = 1,
    pair_id: int = 0,
    reference_run: str = f"C2:{C2_REFERENCE_SHA256}",
) -> dict[str, Any]:
    """Export 12 exact C2 cells: three tasks, two architectures, two wall maps."""

    source_root = Path(root).expanduser().resolve()
    if source_root.is_symlink() or not source_root.is_dir():
        raise FileNotFoundError(f"C2 result root is missing or unsafe: {source_root}")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for task in C2_TASKS:
        for architecture in C2_ARCHITECTURES:
            for wall_map in C2_WALL_MAPS:
                row, receipt, samples_path = _source_row(
                    source_root, task, architecture, seed, wall_map, object_map, pair_id
                )
                rows.extend(_trajectory_rows(row, receipt, samples_path, reference_run))
                receipt_path = samples_path.with_name("receipt.json")
                sources.append(
                    {
                        "job_id": str(receipt["job_id"]),
                        "model_id": str(row["model_id"]),
                        "object_map": object_map,
                        "pair_id": pair_id,
                        "receipt": str(receipt_path.resolve()),
                        "receipt_sha256": sha256_file(receipt_path),
                        "samples": str(samples_path.resolve()),
                        "samples_sha256": sha256_file(samples_path),
                        "wall_map": wall_map,
                    }
                )
    frame = pd.DataFrame(rows, columns=NEUTRAL_COLUMNS)
    destination = write_trajectory_record(frame, output)
    manifest_path = (
        Path(selection_manifest)
        if selection_manifest is not None
        else destination.with_name(f"{destination.stem}_selection.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment_family": "controlled_c2_context_swap",
        "reference_run": reference_run,
        "selection": {
            "architectures": list(C2_ARCHITECTURES),
            "eta": C2_MIXTURE_ETA,
            "object_map": object_map,
            "pair_id": pair_id,
            "seed": seed,
            "tasks": list(C2_TASKS),
            "wall_maps": list(C2_WALL_MAPS),
        },
        "selection_algorithm": "fixed registered cell cross-product",
        "source_root": str(source_root),
        "sources": sources,
        "trajectory_record": str(destination.resolve()),
        "trajectory_record_sha256": sha256_file(destination),
        "unit_count": int(frame["unit_id"].nunique()),
        "row_count": int(len(frame)),
        "units": sorted(frame["unit_id"].unique().tolist()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "trajectory_record": destination,
        "selection_manifest": manifest_path,
        "unit_count": manifest["unit_count"],
        "row_count": manifest["row_count"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="family", required=True)
    c0 = sub.add_parser("c0", help="replay exact C0 protocol-robust trajectories")
    c0.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    c0.add_argument("--results-root", type=Path, default=DEFAULT_C0_RESULTS_ROOT)
    c0.add_argument("--config", type=Path, default=DEFAULT_C0_CONFIG)
    c0.add_argument("--source-config", type=Path, default=DEFAULT_C0_SOURCE_CONFIG)
    c0.add_argument("--output", type=Path, required=True)
    c0.add_argument("--selection-manifest", type=Path)
    c0.add_argument("--diagnostic-audit-output", type=Path)
    c0.add_argument("--device", default="cuda:0")
    c1 = sub.add_parser("c1", help="replay registered C1 CMMR trajectories")
    c1.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    c1.add_argument("--results-root", type=Path, default=DEFAULT_C1_RESULTS_ROOT)
    c1.add_argument("--config", type=Path, default=DEFAULT_C1_CONFIG)
    c1.add_argument("--output", type=Path, required=True)
    c1.add_argument("--selection-manifest", type=Path)
    c1.add_argument("--device", default="cuda:0")
    c2 = sub.add_parser("c2", help="export sealed C2 context-swap responses")
    c2.add_argument("--root", type=Path, default=DEFAULT_C2_ROOT)
    c2.add_argument("--output", type=Path, required=True)
    c2.add_argument("--selection-manifest", type=Path)
    c2.add_argument("--seed", type=int, default=7101)
    c2.add_argument("--object-map", type=int, choices=(1, 2), default=1)
    c2.add_argument("--pair-id", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.family == "c0":
        result = export_c0(
            args.historical_root,
            args.results_root,
            args.config,
            args.source_config,
            args.output,
            selection_manifest=args.selection_manifest,
            diagnostic_audit_output=args.diagnostic_audit_output,
            device=args.device,
        )
    elif args.family == "c1":
        result = export_c1(
            args.historical_root,
            args.results_root,
            args.config,
            args.output,
            selection_manifest=args.selection_manifest,
            device=args.device,
        )
    else:
        result = export_c2(
            args.root,
            args.output,
            selection_manifest=args.selection_manifest,
            seed=args.seed,
            object_map=args.object_map,
            pair_id=args.pair_id,
        )
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
