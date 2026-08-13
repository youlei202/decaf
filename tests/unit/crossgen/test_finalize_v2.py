from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.crossgen import finalize_v2 as finalizer
from tools.crossgen.finalize_v2 import (
    C0_DIAGNOSTIC_RELATIVE,
    C0_EXCLUSION_REASON,
    C0_EXPECTED_CANDIDATES,
    C0_EXPECTED_EXCLUSION_ERRORS,
    C0_QUALIFICATION_STATUS,
    C0_QUALIFICATION_TOLERANCE,
    C0_RUNTIME_ATTRIBUTION_RELATIVE,
    FinalizationError,
    aggregate_unit_comparisons,
    capture_repository_provenance,
    collect_evidence,
    determine_overall_verdict,
    sha256_file,
    validate_c0_candidate_qualification,
    validate_c0_runtime_attribution,
    write_deterministic_zip,
)


def _comparison(unit: str, *, tier: str = "A", boundary: bool = False) -> pd.DataFrame:
    row: dict[str, object] = {
        "unit_id": unit,
        "boundary": boundary,
        "tier": tier,
        "hard_mismatch": tier == "FAIL",
        "gate_match": True,
        "orientation_match": True,
        "dominant_match": True,
        "identity_match": True,
    }
    for name in ("M", "E", "C", "F", "Abs"):
        row[f"abs_error_{name}"] = 0.0 if tier == "A" else 0.001
        row[f"signed_error_{name}"] = 0.0
        row[f"historical_{name}"] = 0.1
    return pd.DataFrame([row])


def _synthetic_attribution_e2e() -> pd.DataFrame:
    bridges = list(finalizer.ATTRIBUTION_BRIDGES.values())
    methods = ("decaf_3", "decaf_5", "decaf_9")
    rows = []
    for index in range(1476):
        dataset, model = bridges[index % len(bridges)]
        method = methods[(index // len(bridges)) % len(methods)]
        image_id = f"image_{index:04d}"
        factor = f"part_{index:04d}"
        reference = (
            "locked_gaussian_blur_k31_sigma12_raw_rgb"
            if dataset == "funnybirds"
            else "normalized_zero"
        )
        counterfactual = (
            "gaussian_blur_k31_sigma12"
            if dataset == "funnybirds"
            else "normalized_zero_4x4_patch_deletion"
        )
        row: dict[str, object] = {
            "unit_id": (
                f"attribution::{dataset}::{model}::{method}::{image_id}::{factor}"
            ),
            "dataset": dataset,
            "model": model,
            "method": method,
            "image_id": image_id,
            "factor_or_part_id": factor,
            "historical_checkpoint_sha256": f"{index % 6 + 1:064x}",
            "current_checkpoint_sha256": f"{index % 6 + 1:064x}",
            "historical_target": 0,
            "current_target": 0,
            "historical_counterfactual_map": counterfactual,
            "current_counterfactual_map": counterfactual,
            "historical_reference": reference,
            "current_reference": reference,
            "historical_intervention_operator": "endpoint_part_deletion",
            "current_intervention_operator": "endpoint_part_deletion",
            "historical_endpoint_d": 0.1,
            "current_endpoint_d": 0.1,
            "current_input_domain": (
                "raw_rgb_float_0_1"
                if dataset == "funnybirds"
                else "fixed_shape_model_input"
            ),
            "current_preprocess_inside_forward": dataset == "funnybirds",
            "checkpoint_match": True,
            "target_match": True,
            "counterfactual_map_match": True,
            "reference_match": True,
            "intervention_operator_match": True,
            "identity_match": True,
            "historical_gate": True,
            "current_gate": True,
            "historical_orientation": 1,
            "current_orientation": 1,
            "boundary": False,
            "gate_match": True,
            "orientation_match": True,
            "historical_dominant": "E",
            "current_dominant": "E",
            "dominant_match": True,
            "tier_a_pass": True,
            "tier_b_pass": True,
            "tier": "A",
            "hard_mismatch": False,
        }
        for name, value in {
            "score": 0.1,
            "M": 0.1,
            "E": 0.1,
            "C": 0.0,
            "F": 0.0,
            "Abs": 0.1,
        }.items():
            row[f"historical_{name}"] = value
            row[f"current_{name}"] = value
            row[f"signed_error_{name}"] = 0.0
            row[f"abs_error_{name}"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_imagenet9_e2e(
    monkeypatch: pytest.MonkeyPatch,
) -> pd.DataFrame:
    pair_ids = [
        f"pair_{index:02d}__{'same_next' if index % 2 == 0 else 'same_rand'}"
        for index in range(16)
    ]
    classes = {pair_id: index % 4 for index, pair_id in enumerate(pair_ids)}
    units = {
        (model_id, pair_id, reveal_path)
        for model_id in finalizer.IMAGENET9_E2E_MODEL_BINDINGS
        for pair_id in pair_ids
        for reveal_path in finalizer.IMAGENET9_REVEAL_PATHS
    }
    monkeypatch.setattr(
        finalizer,
        "_imagenet9_identity_contract",
        lambda _root: (units, classes),
    )
    monkeypatch.setattr(
        finalizer,
        "_validate_imagenet9_stage_evidence",
        lambda _root, _frame, _units, _classes: {"stage_count": 1296},
    )
    rows = []
    for model_id, pair_id, reveal_path in sorted(units):
        pair_type = pair_id.rsplit("__", 1)[-1]
        row: dict[str, object] = {
            "model_id": model_id,
            "historical_model_id": finalizer.IMAGENET9_E2E_MODEL_BINDINGS[model_id][
                "historical_model_id"
            ],
            "pair_id": pair_id,
            "reveal_path": reveal_path,
            "pair_type": pair_type,
            "pair_type_current_label": pair_type,
            "pair_type_historical_label": pair_type,
            "true_in9_class": classes[pair_id],
            "epsilon": 0.02,
            "current_endpoint_d": 0.1,
            "historical_endpoint_d": 0.1,
            "current_gate": True,
            "historical_gate": True,
            "current_orientation": 1,
            "historical_orientation": 1,
            "current_dominant": "E",
            "historical_dominant": "E",
            "checkpoint_identity_match": True,
            "sample_identity_match": True,
            "identity_match": True,
            "boundary": False,
            "gate_match": True,
            "orientation_match": True,
            "dominant_match": True,
            "tier_a_pass": True,
            "tier_b_pass": True,
            "tier": "A",
            "hard_mismatch": False,
        }
        for name, value in {"M": 0.1, "E": 0.1, "C": 0.0, "F": 0.0, "Abs": 0.1}.items():
            row[f"current_{name}"] = value
            row[f"historical_{name}"] = value
            row[f"abs_error_{name}"] = 0.0
            row[f"signed_error_{name}"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _c0_evidence(root: Path) -> pd.DataFrame:
    selected_identities = sorted(set(C0_EXPECTED_CANDIDATES) - set(C0_EXPECTED_EXCLUSION_ERRORS))
    excluded_identities = sorted(C0_EXPECTED_EXCLUSION_ERRORS)

    def candidate(identity: tuple[str, int]) -> dict[str, object]:
        qualified = identity in selected_identities
        maximum_error = 1.0e-6 if qualified else C0_EXPECTED_EXCLUSION_ERRORS[identity]
        absolute_errors = {
            "endpoint_abs": 0.0,
            "auc_abs_info": maximum_error,
            "auc_align_info": 0.0,
            "auc_opp_info": 0.0,
            "auc_null_info": 0.0,
        }
        record: dict[str, object] = {
            "model_id": identity[0],
            "base_id": identity[1],
            **C0_EXPECTED_CANDIDATES[identity],
            "protocol": "lambda=0.000",
            "qualified": qualified,
            "maximum_absolute_error": maximum_error,
            "qualification_tolerance": C0_QUALIFICATION_TOLERANCE,
            "audit": {
                "absolute_errors": absolute_errors,
                "comparison_scope": "two counterfactual maps x three noise seeds",
                "forward_layout": {
                    "dynamic_images": 4096,
                    "historical_batch_size": 1024,
                    "historical_flat_state_count": 49152,
                    "retained_dynamic_prefix": 86,
                    "retained_flat_state_count": 1032,
                    "selected_dynamic_position": 1,
                    "stack_size": 12,
                },
                "maximum_absolute_error": maximum_error,
                "passed": qualified,
                "recomputed": {
                    "endpoint_abs": 0.5,
                    "auc_abs_info": 0.1,
                    "auc_align_info": 0.05,
                    "auc_opp_info": 0.04,
                    "auc_null_info": 0.01,
                },
                "sealed": {
                    "endpoint_abs": 0.5,
                    "auc_abs_info": 0.1 + maximum_error,
                    "auc_align_info": 0.05,
                    "auc_opp_info": 0.04,
                    "auc_null_info": 0.01,
                },
                "tolerance": C0_QUALIFICATION_TOLERANCE,
            },
        }
        if not qualified:
            record["reason_code"] = C0_EXCLUSION_REASON
            record["reason"] = (
                f"{C0_EXCLUSION_REASON}: excluded_without_replacement; sealed six-repeat "
                f"common-information maximum_absolute_error={maximum_error:.17g} exceeds "
                f"qualification_tolerance={C0_QUALIFICATION_TOLERANCE:.17g}"
            )
        return record

    selected = [candidate(identity) for identity in selected_identities]
    excluded = [candidate(identity) for identity in excluded_identities]
    units = [f"C0__{identity[0]}__base_{identity[1]}" for identity in selected_identities]
    coverage: dict[str, object] = {
        "passed": True,
        "checks": {
            "two_registered_architectures": True,
            "at_least_two_active": True,
            "at_least_two_null": True,
            "at_least_one_mixed_E_C": True,
            "both_registered_counterfactual_maps": True,
        },
        "observed": {
            "architectures": ["resnet18", "small_vit"],
            "active_count": 4,
            "null_count": 2,
            "mixed_unit_count": 1,
            "mixed_units": [units[0]],
            "counterfactual_maps": ["20260882", "20260883"],
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
    diagnostic_path = root / C0_DIAGNOSTIC_RELATIVE
    diagnostic: dict[str, object] = {
        "artifact_type": "c0_fixed_candidate_qualification_diagnostic",
        "completed_all_candidates": True,
        "candidate_count": 8,
        "selected_count": 6,
        "excluded_count": 2,
        "status": C0_QUALIFICATION_STATUS,
        "tolerance": C0_QUALIFICATION_TOLERANCE,
        "maximum_absolute_error": max(C0_EXPECTED_EXCLUSION_ERRORS.values()),
        "coverage": coverage,
        "selected": selected,
        "excluded": excluded,
        "audits": [
            {
                "model_id": record["model_id"],
                "base_id": record["base_id"],
                "factor": record["factor"],
                "protocol": record["protocol"],
                "audit": record["audit"],
            }
            for record in (*selected, *excluded)
        ],
    }
    _write_json(diagnostic_path, diagnostic)
    diagnostic_digest = sha256_file(diagnostic_path)
    audit_path = root / "manifests/controlled_c0_selection_sealed_aggregate_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_id": record["model_id"],
                "base_id": record["base_id"],
                "factor": record["factor"],
                "protocol": record["protocol"],
                "maximum_absolute_error": record["maximum_absolute_error"],
                "passed": record["qualified"],
            }
            for record in (*selected, *excluded)
        ]
    ).to_csv(audit_path, index=False)
    qualification: dict[str, object] = {
        "candidate_count": 8,
        "selected_count": 6,
        "excluded_count": 2,
        "status": C0_QUALIFICATION_STATUS,
        "qualification_fraction": "6/8",
        "tolerance": C0_QUALIFICATION_TOLERANCE,
        "selected": selected,
        "excluded": excluded,
        "coverage": coverage,
        "diagnostic": {
            "path": str(diagnostic_path.resolve()),
            "sha256": diagnostic_digest,
        },
        "diagnostic_path": str(diagnostic_path.resolve()),
        "diagnostic_sha256": diagnostic_digest,
    }
    sources = [
        {
            "model_id": identity[0],
            "base_id": identity[1],
            "unit_id": unit,
            "sealed_aggregate_audit": record["audit"],
        }
        for identity, unit, record in zip(selected_identities, units, selected, strict=True)
    ]
    manifest: dict[str, object] = {
        "unit_count": 6,
        "active_unit_count": 4,
        "null_unit_count": 2,
        "mixed_unit_count": 1,
        "units": units,
        "sources": sources,
        "candidate_qualification": qualification,
        "sealed_aggregate_audit": {
            "path": str(audit_path.resolve()),
            "sha256": sha256_file(audit_path),
            "row_count": 8,
            "pass_count": 6,
            "failure_count": 2,
            "qualified_count": 6,
            "excluded_count": 2,
            "maximum_absolute_error": max(C0_EXPECTED_EXCLUSION_ERRORS.values()),
            "tolerance": C0_QUALIFICATION_TOLERANCE,
        },
    }
    _write_json(root / "manifests/controlled_c0_selection.json", manifest)
    return pd.concat([_comparison(unit) for unit in units], ignore_index=True)


def _c0_runtime_attribution(root: Path, diagnostic_sha256: str) -> dict[str, object]:
    failed_units = [
        {
            "model_id": identity[0],
            "base_id": identity[1],
            "maximum_absolute_error": maximum_error,
        }
        for identity, maximum_error in sorted(C0_EXPECTED_EXCLUSION_ERRORS.items())
    ]
    return {
        "schema_version": 1,
        "artifact_type": "decaf_cross_generation_controlled_c0_runtime_attribution",
        "selection_audit": {
            "unit_count": 8,
            "strict_pass_count": 6,
            "strict_failure_count": 2,
            "tolerance": C0_QUALIFICATION_TOLERANCE,
            "maximum_absolute_error": max(C0_EXPECTED_EXCLUSION_ERRORS.values()),
            "failed_units": failed_units,
        },
        "recommended_disposition": {
            "code": C0_EXCLUSION_REASON,
            "strict_equivalence": False,
            "action": (
                "Retain all eight predeclared units, report six strict passes and two failures, "
                "preserve exact errors, and do not loosen tolerance or remove selections."
            ),
        },
        "remaining_inference": {
            "hypothesis": (
                "MIG versus full-B200 and/or cuDNN algorithm selection are plausible contributors."
            ),
            "proven": False,
            "reason_not_proven": (
                "The historical driver and cuDNN kernel selections were not captured; "
                "kernel selections were not captured by a complete runtime lock."
            ),
        },
        "historical_runtime": {
            "confirmed": {"gpu_topology": "4 x NVIDIA B200 MIG 1g.23gb"},
            "unrecorded": ["exact kernel path on historical MIG instances"],
        },
        "input_evidence": [
            {
                "path": str((root / C0_DIAGNOSTIC_RELATIVE).resolve()),
                "sha256": diagnostic_sha256,
            }
        ],
    }


def _imagenet9_source_evidence(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object]]:
    prefix = finalizer.IMAGENET9_PACKAGE_PREFIX
    payloads = {
        relative: f"# {relative}\n".encode() for relative in finalizer.IMAGENET9_SOURCE_RELATIVES
    }
    for index in range(381 - len(payloads)):
        payloads[f"metadata/filler_{index:03d}.txt"] = f"{index}\n".encode()
    members = [
        {
            "path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in sorted(payloads.items())
    ]
    package_manifest = {
        "schema_version": 1,
        "namespace": prefix,
        "lightweight": True,
        "source_layout": "code/cmr",
        "recorded_member_count": 381,
        "members": members,
    }
    manifest_bytes = json.dumps(package_manifest, sort_keys=True).encode()
    package = root / "historical.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(finalizer.IMAGENET9_MANIFEST_MEMBER, manifest_bytes)
        for relative, payload in payloads.items():
            archive.writestr(f"{prefix}/{relative}", payload)
    package_digest = sha256_file(package)
    monkeypatch.setattr(finalizer, "IMAGENET9_HISTORICAL_PACKAGE_SHA256", package_digest)
    source_members = {
        relative: {
            "archive_member": f"{prefix}/{relative}",
            "bytes": len(payloads[relative]),
            "sha256": hashlib.sha256(payloads[relative]).hexdigest(),
        }
        for relative in finalizer.IMAGENET9_SOURCE_RELATIVES
    }
    binding: dict[str, object] = {
        "path": str(package.resolve()),
        "sha256": package_digest,
        "manifest_member": finalizer.IMAGENET9_MANIFEST_MEMBER,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "recorded_member_count": 381,
        "zip_import_root": f"{package.resolve()}/{prefix}/code",
        "source_authority": "sha256-verified lightweight ZIP; not historical Git HEAD",
        "source_members": source_members,
    }
    patch_path = root / "manifests/imagenet9_historical_patch_orders.json"
    _write_json(patch_path, {"historical_source_binding": binding})
    bridge_path = root / "provenance/imagenet9_bridge.json"
    _write_json(
        bridge_path,
        {
            "historical_source_binding": binding,
            "patch_order_manifest": str(patch_path.resolve()),
            "patch_order_manifest_sha256": sha256_file(patch_path),
        },
    )
    output = root / "trajectories/imagenet9_current_e2e_scans.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"current scan")
    _write_json(
        root / "provenance/imagenet9_current_e2e.json",
        {
            "output": str(output.resolve()),
            "output_sha256": sha256_file(output),
            "patch_order_manifest": str(patch_path.resolve()),
            "patch_order_manifest_sha256": sha256_file(patch_path),
            "patch_order_injection": True,
        },
    )
    return package, binding


def _attribution_chain_evidence(root: Path) -> None:
    sources: list[dict[str, str]] = []
    for key, (dataset, model_id) in finalizer.ATTRIBUTION_BRIDGES.items():
        historical = root / f"trajectories/attribution__{key}.parquet"
        historical.parent.mkdir(parents=True, exist_ok=True)
        historical.write_bytes(f"historical:{key}".encode())
        sources.append({"path": str(historical.resolve()), "sha256": sha256_file(historical)})
        _write_json(
            root / f"manifests/attribution__{key}_selection.json",
            {
                "schema_version": 1,
                "experiment_family": "attribution",
                "dataset": dataset,
                "model_id": model_id,
                "trajectory_record": str(historical.resolve()),
                "trajectory_record_sha256": sha256_file(historical),
            },
        )
        current = root / f"trajectories/attribution_current__{key}.parquet"
        current.write_bytes(f"current:{key}".encode())
        _write_json(
            root / f"provenance/attribution_current__{key}.json",
            {
                "schema_version": 1,
                "experiment_family": "attribution",
                "dataset": dataset,
                "model_id": model_id,
                "output": str(current.resolve()),
                "output_sha256": sha256_file(current),
            },
        )
    aggregate = root / "trajectories/attribution.parquet"
    aggregate.write_bytes(b"aggregate")
    _write_json(
        root / "manifests/attribution_aggregate.json",
        {
            "schema_version": 1,
            "experiment_family": "attribution",
            "output": str(aggregate.resolve()),
            "output_sha256": sha256_file(aggregate),
            "source_records": sources,
        },
    )


def _covertype_source_evidence(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    required = set(finalizer.COVERTYPE_REQUIRED_MODULES) | {"io"}
    module_names = sorted(required) + [f"extra_{index:02d}" for index in range(26 - len(required))]
    namespace_payloads = {
        f"{finalizer.COVERTYPE_NAMESPACE_PREFIX}{name}.py": f"# {name}\n".encode()
        for name in module_names
    }
    other_payloads = {
        f"metadata/file_{index:03d}.txt": f"{index}\n".encode()
        for index in range(110 - len(namespace_payloads))
    }
    payloads = {**namespace_payloads, **other_payloads}
    files = [
        {
            "path": member,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for member, payload in sorted(payloads.items())
    ]
    manifest_bytes = json.dumps(
        {
            "schema_version": 1,
            "namespace": "decaf_covertype_v1",
            "lightweight": True,
            "files": files,
        },
        sort_keys=True,
    ).encode()
    package = root / "covertype.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(finalizer.COVERTYPE_MANIFEST_MEMBER, manifest_bytes)
        for member, payload in payloads.items():
            archive.writestr(member, payload)
    monkeypatch.setattr(finalizer, "COVERTYPE_HISTORICAL_PACKAGE_SHA256", sha256_file(package))
    monkeypatch.setattr(
        finalizer,
        "COVERTYPE_HISTORICAL_MANIFEST_SHA256",
        hashlib.sha256(manifest_bytes).hexdigest(),
    )

    import_root = root / "provenance/historical_sources/covertype"
    namespace = import_root / "cmr/decaf_covertype_v1"
    namespace_members: dict[str, dict[str, object]] = {}
    for member, payload in namespace_payloads.items():
        relative = Path(member).relative_to(finalizer.COVERTYPE_ARCHIVE_SOURCE_PREFIX)
        output = import_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        namespace_members[member] = {
            "archive_member": member,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    shim = import_root / "cmr/__init__.py"
    shim_payload = b"# verification-only parent shim\n" + b" " * 41 + b"\n"
    assert len(shim_payload) == 74
    shim.write_bytes(shim_payload)
    monkeypatch.setattr(
        finalizer, "COVERTYPE_PARENT_SHIM_SHA256", hashlib.sha256(shim_payload).hexdigest()
    )
    loaded = {}
    for name in finalizer.COVERTYPE_LOADED_MODULES:
        module = (
            "cmr.decaf_covertype_v1"
            if name == "__init__"
            else f"cmr.decaf_covertype_v1.{name}"
        )
        filename = "__init__.py" if name == "__init__" else f"{name}.py"
        loaded[module] = str(namespace / filename)
    binding = {
        "authority_kind": "sha256_verified_lightweight_zip",
        "git_head_role": "context_only_untracked",
        "path": str(package.resolve()),
        "sha256": sha256_file(package),
        "manifest_member": finalizer.COVERTYPE_MANIFEST_MEMBER,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "archive_source_prefix": finalizer.COVERTYPE_ARCHIVE_SOURCE_PREFIX,
        "archive_inventory_verified": True,
        "namespace_member_count": 26,
        "namespace_members": namespace_members,
        "required_modules": sorted(finalizer.COVERTYPE_REQUIRED_MODULES),
        "import_root": str(import_root.resolve()),
        "materialized_namespace": str(namespace.resolve()),
        "materialized_member_count": 26,
        "parent_package_shim": {
            "path": str(shim.resolve()),
            "bytes": 74,
            "sha256": hashlib.sha256(shim_payload).hexdigest(),
            "role": "verification_only_import_isolation",
            "historical_source": False,
        },
        "parent_package_origin": str(shim.resolve()),
        "origin_verified": True,
        "loaded_module_origins": dict(sorted(loaded.items())),
    }
    selection: dict[str, object] = {
        "schema_version": 1,
        "family": "covertype",
        "historical_repository_head_role": "context_only_untracked",
        "historical_source_binding": binding,
        "selection_key": "sha256(namespace|sealed_test_source_index)",
        "selection_uses_model_outputs": False,
    }
    selection["selection_sha256"] = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(root / "manifests/covertype_selection.json", selection)


def _dino_evidence(root: Path) -> None:
    run_root = root / "runs/dinov2_g"
    checkpoint_files = []
    for index in range(2):
        checkpoint = root / f"assets/checkpoint_{index}.bin"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint {index}".encode())
        checkpoint_files.append(checkpoint)
    data_file = root / "assets/data.parquet"
    data_file.write_bytes(b"data")
    config_sha256 = "1" * 64
    contract_sha256 = "2" * 64
    jobs = []
    members = {}
    for index in range(16):
        scope = finalizer.DINO_SCOPES[index // 8]
        member_id = f"member-{index:02d}"
        output_relative = f"raw/members/{member_id}.parquet"
        receipt_relative = f"receipts/members/{member_id}.json"
        output = run_root / output_relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"output {index}".encode())
        job = {
            "member_id": member_id,
            "scope": scope,
            "model_id": "dinov2_vit_g_14",
            "output_path": output_relative,
            "receipt_path": receipt_relative,
            "config_sha256": config_sha256,
            "plan_contract_sha256": contract_sha256,
            "job_sha256": f"{index + 3:064x}",
        }
        jobs.append(job)
        members[member_id] = {"status": "completed", "optional": False}
        _write_json(
            run_root / receipt_relative,
            {
                "schema_version": 1,
                "kind": "member",
                "status": "completed",
                "error": None,
                "optional": False,
                "member_id": member_id,
                "details": {
                    "output_path": output_relative,
                    "output_sha256": sha256_file(output),
                    "scope": scope,
                    "model_id": "dinov2_vit_g_14",
                    "job_sha256": job["job_sha256"],
                    "config_sha256": config_sha256,
                    "plan_contract_sha256": contract_sha256,
                    "checkpoint_binding_manifest_sha256": "PENDING",
                    "data_binding_manifest_sha256": "PENDING",
                },
            },
        )
    plan = {
        "schema_version": 1,
        "experiment": "attribution",
        "profile": "large-model-smoke",
        "scope_names": list(finalizer.DINO_SCOPES),
        "member_count": 16,
        "expected_member_count": 16,
        "endpoint_m_stage": "analyze",
        "config_sha256": config_sha256,
        "plan_contract_sha256": contract_sha256,
        "audit": {"passed": True, "checked_members": 16, "errors": []},
        "members": jobs,
    }
    _write_json(run_root / "manifests/plan.json", plan)
    _write_json(
        run_root / "manifests/checkpoints.json",
        {
            "schema_version": 1,
            "resolved": True,
            "execution_claimed": False,
            "config_sha256": config_sha256,
            "plan_contract_sha256": contract_sha256,
            "items": [
                {
                    "model_id": "dinov2_vit_g_14",
                    "checkpoints": [
                        {"resolved_path": str(path.resolve()), "bytes_sha256": sha256_file(path)}
                        for path in checkpoint_files
                    ],
                }
            ],
        },
    )
    _write_json(
        run_root / "manifests/data.json",
        {
            "schema_version": 1,
            "resolved": True,
            "execution_claimed": False,
            "config_sha256": config_sha256,
            "plan_contract_sha256": contract_sha256,
            "items": [
                {
                    "scope": scope,
                    "dataset": "imagenet1k_idsds",
                    "images": 8,
                    "resolved_path": str(data_file.resolve()),
                    "bytes_sha256": sha256_file(data_file),
                    "expected_sha256": sha256_file(data_file),
                }
                for scope in finalizer.DINO_SCOPES
            ],
        },
    )
    checkpoint_digest = sha256_file(run_root / "manifests/checkpoints.json")
    data_digest = sha256_file(run_root / "manifests/data.json")
    for job in jobs:
        receipt_path = run_root / str(job["receipt_path"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["details"]["checkpoint_binding_manifest_sha256"] = checkpoint_digest
        receipt["details"]["data_binding_manifest_sha256"] = data_digest
        _write_json(receipt_path, receipt)
    _write_json(
        run_root / "receipts/compute.json",
        {
            "schema_version": 1,
            "stage": "compute",
            "status": "completed",
            "details": {
                "backend": "gpu",
                "scheduler": "single_gpu_dynamic_queue",
                "device": 0,
                "member_count": 16,
                "completed_members": 16,
                "resumed_members": 0,
                "failed_members": 0,
            },
        },
    )
    _write_json(
        run_root / "receipts/compute_members.json",
        {
            "schema_version": 1,
            "kind": "global",
            "all_processes_exited": True,
            "member_count": 16,
            "members": members,
            "details": {
                "backend": "gpu",
                "scheduler": "single_gpu_dynamic_queue",
                "visible_device": "cuda:0",
                "exclusive_member_concurrency": 1,
                "duplicate_execution": False,
                "failures": {},
                "member_count": 16,
                "plan_contract_sha256": contract_sha256,
                "config_sha256": config_sha256,
                "checkpoint_binding_manifest_sha256": checkpoint_digest,
                "data_binding_manifest_sha256": data_digest,
            },
        },
    )
    _write_json(
        run_root / "run.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": "dinov2_g",
            "experiment": "attribution",
            "profile": "large-model-smoke",
            "completed_stages": ["prepare", "compute", "analyze", "paper"],
        },
    )
    _write_json(
        root / "B200_VERIFICATION_STATUS.json",
        {
            "schema_version": 1,
            "status": "passed",
            "repository": {
                "commit": finalizer.DINO_REPOSITORY_COMMIT,
                "tree": finalizer.DINO_REPOSITORY_TREE,
                "tracked_worktree_clean": True,
            },
            "machine": {"gpu": {"name": "NVIDIA B200"}},
            "acceptance_gates": {gate: True for gate in finalizer.DINO_REQUIRED_GATES},
            "checkpoint_fingerprints": {"status": "PASS"},
            "final_audits": {"repository_audit": "PASS"},
            "representative_shards": {
                "dinov2_g": {
                    "status": "PASS",
                    "scope": "real_cuda_single_b200_shard",
                    "member_count": 16,
                }
            },
        },
    )


def _attribution_source_manifests(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for key, (dataset, model_id) in finalizer.ATTRIBUTION_BRIDGES.items():
        path = root / f"manifests/attribution__{key}_selection.json"
        authority = {"authority_kind": "a0" if dataset == "funnybirds" else "a2"}
        _write_json(
            path,
            {
                "schema_version": 1,
                "dataset": dataset,
                "model_id": model_id,
                "historical_source_binding": authority,
            },
        )
        paths[key] = path
        digests[key] = sha256_file(path)
    monkeypatch.setattr(finalizer, "ATTRIBUTION_SELECTION_SHA256", digests)
    monkeypatch.setattr(
        finalizer,
        "_attribution_a0_source_binding",
        lambda binding: {
            "authority_kind": binding["authority_kind"],
            "authority_sha256": "a" * 64,
            "origin_verified": True,
        },
    )
    monkeypatch.setattr(
        finalizer,
        "_attribution_a2_source_binding",
        lambda _root, binding: {
            "authority_kind": binding["authority_kind"],
            "authority_sha256": "b" * 64,
            "origin_verified": True,
        },
    )
    return paths


def test_aggregate_is_exactly_unit_weighted() -> None:
    first = pd.concat([_comparison("a"), _comparison("b", tier="B")])
    second = pd.concat([_comparison(f"c{index}") for index in range(8)])

    result = aggregate_unit_comparisons([first, second], label="synthetic")

    assert result["unit_count"] == 10
    assert result["tier_a_fraction"] == 0.9
    assert result["tier_b_fraction"] == 0.1
    assert result["tier_a_or_b_fraction"] == 1.0


def test_aggregate_rejects_duplicate_units() -> None:
    with pytest.raises(FinalizationError, match="duplicate"):
        aggregate_unit_comparisons([_comparison("same"), _comparison("same")], label="synthetic")


def test_aggregate_rejects_missing_semantics() -> None:
    frame = _comparison("a").drop(columns="identity_match")
    with pytest.raises(FinalizationError, match="identity_match"):
        aggregate_unit_comparisons([frame], label="synthetic")


@pytest.mark.parametrize(
    ("column", "forged"),
    (("tier", "B"), ("hard_mismatch", True), ("gate_match", False), ("identity_match", False)),
)
def test_attribution_e2e_rejects_forged_row_flags(column: str, forged: object) -> None:
    frame = _synthetic_attribution_e2e()
    frame.loc[0, column] = forged

    with pytest.raises(FinalizationError, match="independent row recomputation"):
        finalizer._validate_attribution_e2e_rows(frame)


@pytest.mark.parametrize(
    ("column", "forged"),
    (("tier", "B"), ("hard_mismatch", True), ("gate_match", False), ("identity_match", False)),
)
def test_imagenet9_e2e_rejects_forged_row_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    forged: object,
) -> None:
    frame = _synthetic_imagenet9_e2e(monkeypatch)
    frame.loc[0, column] = forged

    with pytest.raises(FinalizationError, match="independent row recomputation"):
        finalizer._validate_imagenet9_e2e_rows(tmp_path, frame)


def test_c0_candidate_qualification_requires_exact_six_of_eight(tmp_path: Path) -> None:
    frame = _c0_evidence(tmp_path)

    result = validate_c0_candidate_qualification(tmp_path, frame)

    assert result["qualification_fraction"] == "6/8"
    assert result["formal_unit_count"] == 6
    assert result["coverage"]["passed"] is True
    assert {
        (record["model_id"], record["base_id"], record["maximum_absolute_error"])
        for record in result["excluded"]
    } == {
        (identity[0], identity[1], error)
        for identity, error in C0_EXPECTED_EXCLUSION_ERRORS.items()
    }


def test_c0_candidate_qualification_rejects_eight_of_eight_claim(tmp_path: Path) -> None:
    frame = _c0_evidence(tmp_path)
    manifest_path = tmp_path / "manifests/controlled_c0_selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_qualification"]["status"] = "STRICT_AGGREGATE_QUALIFIED"
    manifest["candidate_qualification"]["qualification_fraction"] = "8/8"
    _write_json(manifest_path, manifest)

    with pytest.raises(FinalizationError, match="not an 8/8 pass"):
        validate_c0_candidate_qualification(tmp_path, frame)


def test_c0_candidate_qualification_rejects_changed_exclusion_error(tmp_path: Path) -> None:
    frame = _c0_evidence(tmp_path)
    manifest_path = tmp_path / "manifests/controlled_c0_selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_qualification"]["excluded"][0]["maximum_absolute_error"] = 0.002
    _write_json(manifest_path, manifest)

    with pytest.raises(FinalizationError, match="maximum_absolute_error"):
        validate_c0_candidate_qualification(tmp_path, frame)


def test_c0_candidate_qualification_rejects_diagnostic_sha_mismatch(tmp_path: Path) -> None:
    frame = _c0_evidence(tmp_path)
    diagnostic = tmp_path / C0_DIAGNOSTIC_RELATIVE
    diagnostic.write_text(diagnostic.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(FinalizationError, match="SHA-256 binding"):
        validate_c0_candidate_qualification(tmp_path, frame)


def test_c0_runtime_attribution_is_bound_and_non_causal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _c0_evidence(tmp_path)
    qualification = validate_c0_candidate_qualification(tmp_path, frame)
    attribution_path = tmp_path / C0_RUNTIME_ATTRIBUTION_RELATIVE
    _write_json(
        attribution_path,
        _c0_runtime_attribution(tmp_path, qualification["diagnostic_sha256"]),
    )
    monkeypatch.setattr(finalizer, "C0_RUNTIME_ATTRIBUTION_SHA256", sha256_file(attribution_path))

    result = validate_c0_runtime_attribution(tmp_path, qualification)

    assert result["remaining_inference"]["proven"] is False
    assert result["historical_runtime"]["kernel_runtime_metadata_locked"] is False


def test_imagenet9_source_binding_validates_sealed_zip_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, _binding = _imagenet9_source_evidence(tmp_path, monkeypatch)

    result = finalizer.validate_imagenet9_historical_source_binding(tmp_path)

    assert result["package_sha256"] == sha256_file(package)
    assert set(result["required_source_members"]) == finalizer.IMAGENET9_SOURCE_RELATIVES
    assert result["source_authority"].endswith("not historical Git HEAD")


def test_imagenet9_source_binding_rejects_mixed_patch_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _imagenet9_source_evidence(tmp_path, monkeypatch)
    patch_path = tmp_path / "manifests/imagenet9_historical_patch_orders.json"
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    patch["historical_source_binding"]["source_authority"] = "historical Git HEAD"
    _write_json(patch_path, patch)

    with pytest.raises(FinalizationError, match="bind different historical sources"):
        finalizer.validate_imagenet9_historical_source_binding(tmp_path)


def test_covertype_source_binding_validates_zip_materialization_and_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _covertype_source_evidence(tmp_path, monkeypatch)

    result = finalizer.validate_covertype_historical_source_binding(tmp_path)

    assert result["manifest_file_count"] == 110
    assert result["archive_file_count"] == 111
    assert result["namespace_member_count"] == 26
    assert result["materialized_member_count"] == 26
    assert len(result["loaded_module_origins"]) == 9
    assert result["origin_verified"] is True


def test_covertype_source_binding_rejects_materialized_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _covertype_source_evidence(tmp_path, monkeypatch)
    source = tmp_path / "provenance/historical_sources/covertype/cmr/decaf_covertype_v1/data.py"
    source.write_text("changed\n", encoding="utf-8")

    with pytest.raises(FinalizationError, match="materialized Covertype source"):
        finalizer.validate_covertype_historical_source_binding(tmp_path)


def test_dino_evidence_validates_real_b200_receipt_chain(tmp_path: Path) -> None:
    _dino_evidence(tmp_path)

    result = finalizer._dino_evidence(tmp_path)

    assert result["gpu_name"] == "NVIDIA B200"
    assert result["member_count"] == 16
    assert result["scope"] == "real_cuda_single_b200_shard"
    assert set(result["mandatory_gates"]) == finalizer.DINO_REQUIRED_GATES
    assert len(result["member_artifacts"]) == 16
    assert set(result["bound_artifacts"]) == {
        "verification_status",
        "run",
        "plan",
        "checkpoints",
        "data",
        "compute",
        "compute_members",
    }


def test_dino_evidence_rejects_missing_mandatory_gate(tmp_path: Path) -> None:
    _dino_evidence(tmp_path)
    status_path = tmp_path / "B200_VERIFICATION_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["acceptance_gates"]["repository_audit"] = False
    _write_json(status_path, status)

    with pytest.raises(FinalizationError, match="failed closed"):
        finalizer._dino_evidence(tmp_path)


def test_dino_snapshot_packages_all_seven_bound_receipts(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    package_root = tmp_path / "package"
    _dino_evidence(evidence_root)
    dino = finalizer._dino_evidence(evidence_root)
    status = {"families": {"dinov2-g": dino}}

    finalizer._snapshot_prior_dino_evidence(package_root, status)

    packaged = status["families"]["dinov2-g"]["packaged_evidence"]
    assert len(packaged) == 7
    assert {record["kind"] for record in packaged} == set(dino["bound_artifacts"])
    assert all(Path(record["path"]).is_file() for record in packaged)


def test_artifact_binding_rejects_stale_trajectory_hash(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.parquet"
    trajectory.write_bytes(b"current trajectory")
    summary = {
        "trajectory_record": str(trajectory.resolve()),
        "trajectory_record_sha256": "0" * 64,
    }

    with pytest.raises(FinalizationError, match="differs from the actual artifact"):
        finalizer._validate_artifact_binding(
            summary,
            trajectory,
            path_key="trajectory_record",
            sha_key="trajectory_record_sha256",
            label="synthetic core trajectory",
        )


def test_recomputed_core_comparison_rejects_mixed_csv() -> None:
    recomputed = _comparison("unit")
    actual = recomputed.copy()
    actual.loc[0, "abs_error_M"] = 0.25

    with pytest.raises(FinalizationError, match="current-core recomputation"):
        finalizer._validate_recomputed_core_comparison(
            actual,
            recomputed,
            label="synthetic core CSV",
        )


def test_attribution_artifact_chain_validates_all_six_bridges(tmp_path: Path) -> None:
    _attribution_chain_evidence(tmp_path)

    result = finalizer.validate_attribution_artifact_chain(tmp_path)

    assert result["source_count"] == 6
    assert set(result["legacy_sources"]) == set(finalizer.ATTRIBUTION_BRIDGES)
    assert set(result["current_outputs"]) == set(finalizer.ATTRIBUTION_BRIDGES)
    assert result["aggregate_output_sha256"] == sha256_file(
        tmp_path / "trajectories/attribution.parquet"
    )


def test_attribution_artifact_chain_rejects_stale_current_receipt(tmp_path: Path) -> None:
    _attribution_chain_evidence(tmp_path)
    key = next(iter(finalizer.ATTRIBUTION_BRIDGES))
    receipt_path = tmp_path / f"provenance/attribution_current__{key}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(FinalizationError, match="actual artifact"):
        finalizer.validate_attribution_artifact_chain(tmp_path)


def test_attribution_artifact_chain_rejects_mixed_legacy_selection(tmp_path: Path) -> None:
    _attribution_chain_evidence(tmp_path)
    key = next(iter(finalizer.ATTRIBUTION_BRIDGES))
    selection_path = tmp_path / f"manifests/attribution__{key}_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["trajectory_record_sha256"] = "0" * 64
    _write_json(selection_path, selection)

    with pytest.raises(FinalizationError, match="actual artifact"):
        finalizer.validate_attribution_artifact_chain(tmp_path)


def test_attribution_source_bindings_require_exact_three_of_three_authorities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _attribution_source_manifests(tmp_path, monkeypatch)

    result = finalizer.validate_attribution_historical_source_bindings(tmp_path)

    assert result["a0_funnybirds"]["authority_equal_3_of_3"] is True
    assert result["a0_funnybirds"]["selection_count"] == 3
    assert result["a2_imagenet1k_idsds"]["authority_equal_3_of_3"] is True
    assert result["a2_imagenet1k_idsds"]["selection_count"] == 3
    assert len(result["selection_manifests"]) == 6


def test_attribution_source_bindings_reject_one_mixed_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _attribution_source_manifests(tmp_path, monkeypatch)
    key = "funnybirds__funnybirds_vgg16"
    manifest = json.loads(paths[key].read_text(encoding="utf-8"))
    manifest["historical_source_binding"]["mixed"] = True
    _write_json(paths[key], manifest)
    digests = {
        name: sha256_file(path)
        for name, path in paths.items()
    }
    monkeypatch.setattr(finalizer, "ATTRIBUTION_SELECTION_SHA256", digests)

    with pytest.raises(FinalizationError, match="exact 3/3 authority"):
        finalizer.validate_attribution_historical_source_bindings(tmp_path)


def test_historical_source_snapshots_preserve_exact_bytes_and_reject_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "v2"
    external = tmp_path / "external"
    external.mkdir()
    a0_payloads = {
        "deployment": b'{"deployment":true}\n',
        "plan": b'{"job":1}\n{"job":2}\n',
        "receipt": b'{"receipt":true}\n',
    }
    a0_paths = {}
    for name, payload in a0_payloads.items():
        path = external / name
        path.write_bytes(payload)
        a0_paths[name] = path
    monkeypatch.setattr(
        finalizer,
        "ATTRIBUTION_A0_DEPLOYMENT_SHA256",
        hashlib.sha256(a0_payloads["deployment"]).hexdigest(),
    )
    monkeypatch.setattr(
        finalizer,
        "ATTRIBUTION_A0_PLAN_SHA256",
        hashlib.sha256(a0_payloads["plan"]).hexdigest(),
    )
    monkeypatch.setattr(
        finalizer,
        "ATTRIBUTION_A0_PLAN_RECEIPT_SHA256",
        hashlib.sha256(a0_payloads["receipt"]).hexdigest(),
    )

    a2_manifest = b'{"raw":"a2"}\n'
    covertype_manifest = b'{"raw":"covertype"}\n'
    packages = {}
    for name, member, payload in (
        ("a2", finalizer.ATTRIBUTION_A2_MANIFEST_MEMBER, a2_manifest),
        ("covertype", finalizer.COVERTYPE_MANIFEST_MEMBER, covertype_manifest),
    ):
        package = external / f"{name}.zip"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr(member, payload)
        packages[name] = package
    monkeypatch.setattr(finalizer, "ATTRIBUTION_A2_PACKAGE_SHA256", sha256_file(packages["a2"]))
    monkeypatch.setattr(
        finalizer,
        "ATTRIBUTION_A2_MANIFEST_SHA256",
        hashlib.sha256(a2_manifest).hexdigest(),
    )
    monkeypatch.setattr(
        finalizer,
        "COVERTYPE_HISTORICAL_PACKAGE_SHA256",
        sha256_file(packages["covertype"]),
    )
    monkeypatch.setattr(
        finalizer,
        "COVERTYPE_HISTORICAL_MANIFEST_SHA256",
        hashlib.sha256(covertype_manifest).hexdigest(),
    )
    attribution = {
        "a0_funnybirds": {
            "deployment_receipt_path": str(a0_paths["deployment"].resolve()),
            "deployment_receipt_sha256": finalizer.ATTRIBUTION_A0_DEPLOYMENT_SHA256,
            "formal_plan_path": str(a0_paths["plan"].resolve()),
            "formal_plan_sha256": finalizer.ATTRIBUTION_A0_PLAN_SHA256,
            "formal_plan_receipt_path": str(a0_paths["receipt"].resolve()),
            "formal_plan_receipt_sha256": finalizer.ATTRIBUTION_A0_PLAN_RECEIPT_SHA256,
        },
        "a2_imagenet1k_idsds": {
            "package_path": str(packages["a2"].resolve()),
            "package_sha256": finalizer.ATTRIBUTION_A2_PACKAGE_SHA256,
            "manifest_member": finalizer.ATTRIBUTION_A2_MANIFEST_MEMBER,
            "manifest_sha256": finalizer.ATTRIBUTION_A2_MANIFEST_SHA256,
        },
    }
    covertype = {
        "package_path": str(packages["covertype"].resolve()),
        "package_sha256": finalizer.COVERTYPE_HISTORICAL_PACKAGE_SHA256,
        "manifest_member": finalizer.COVERTYPE_MANIFEST_MEMBER,
        "manifest_sha256": finalizer.COVERTYPE_HISTORICAL_MANIFEST_SHA256,
    }

    dry = finalizer.snapshot_historical_source_provenance(
        root, attribution, covertype, write_outputs=False
    )
    assert dry["all_materialized"] is False
    assert dry["attribution_a0"]["formal_plan"]["source_path"] == str(
        a0_paths["plan"].resolve()
    )
    assert dry["attribution_a0"]["formal_plan"]["package_member"].endswith(
        "provenance/historical_source_snapshots/attribution_a0/formal_jobs.jsonl"
    )
    assert dry["attribution_a0"]["formal_plan"]["package_member_verified"] is False
    assert not (root / finalizer.HISTORICAL_SNAPSHOT_ROOT).exists()

    written = finalizer.snapshot_historical_source_provenance(
        root, attribution, covertype, write_outputs=True
    )
    assert written["all_materialized"] is True
    assert written["attribution_a0"]["formal_plan"]["package_member_verified"] is True
    assert written["attribution_a2"]["package_manifest"]["source_path"] == str(
        packages["a2"].resolve()
    )
    assert written["attribution_a2"]["package_manifest"]["source_archive_member"] == (
        finalizer.ATTRIBUTION_A2_MANIFEST_MEMBER
    )
    assert Path(written["attribution_a0"]["formal_plan"]["path"]).read_bytes() == a0_payloads[
        "plan"
    ]
    assert Path(written["attribution_a2"]["package_manifest"]["path"]).read_bytes() == a2_manifest
    assert (
        Path(written["covertype"]["package_manifest"]["path"]).read_bytes()
        == covertype_manifest
    )

    a0_paths["deployment"].write_bytes(b"drift\n")
    with pytest.raises(FinalizationError, match="SHA-256 changed"):
        finalizer.snapshot_historical_source_provenance(
            root, attribution, covertype, write_outputs=False
        )


def test_attribution_spearman_rejects_forged_hard_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = [
        (dataset, model, method, f"image_{image:02d}")
        for dataset, model in finalizer.ATTRIBUTION_BRIDGES.values()
        for method in ("decaf_3", "decaf_5", "decaf_9")
        for image in range(8)
    ]
    source_rows = []
    for index, key in enumerate(keys):
        repeats = 11 if index < 36 else 10
        historical = 0.5
        current = historical - (0.002941176470588225 if index == 0 else 0.0)
        for _ in range(repeats):
            source_rows.append(
                dict(
                    zip(
                        finalizer.ATTRIBUTION_SPEARMAN_IDENTITY_COLUMNS,
                        key,
                        strict=True,
                    )
                )
                | {
                    "historical_spearman": historical,
                    "current_spearman": current,
                }
            )
    attribution = pd.DataFrame(source_rows)
    rows = []
    for key, group in attribution.groupby(
        list(finalizer.ATTRIBUTION_SPEARMAN_IDENTITY_COLUMNS), sort=True
    ):
        historical = float(group["historical_spearman"].iloc[0])
        current = float(group["current_spearman"].iloc[0])
        signed = current - historical
        absolute = abs(signed)
        tier_a = bool(
            finalizer.np.isclose(
                current,
                historical,
                atol=finalizer.core_comparison.TIER_A_ATOL,
                rtol=finalizer.core_comparison.TIER_A_RTOL,
            )
        )
        tier_b = absolute <= finalizer.core_comparison.TIER_B_ABS
        rows.append(
            dict(
                zip(finalizer.ATTRIBUTION_SPEARMAN_IDENTITY_COLUMNS, key, strict=True)
            )
            | {
                "historical_spearman": historical,
                "current_spearman": current,
                "signed_error": signed,
                "absolute_error": absolute,
                "tier_a_pass": tier_a,
                "tier_b_pass": tier_b,
                "tier": "A" if tier_a else ("B" if tier_b else "FAIL"),
                "hard_mismatch": False,
            }
        )
    spearman = pd.DataFrame(rows)
    path = tmp_path / "comparisons/attribution_spearman.csv"
    path.parent.mkdir(parents=True)
    spearman.to_csv(path, index=False)
    errors = spearman["absolute_error"].to_numpy(dtype=float)
    summary = {
        "image_method_count": 144,
        "spearman_comparison": str(path.resolve()),
        "spearman_comparison_sha256": sha256_file(path),
        "spearman": {
            "median_absolute_error": float(finalizer.np.median(errors)),
            "p95_absolute_error": float(finalizer.np.percentile(errors, 95)),
            "maximum_absolute_error": float(finalizer.np.max(errors)),
            "mean_signed_error": float(spearman["signed_error"].mean()),
            "tier_a_fraction": 1.0,
            "tier_b_fraction": 0.0,
            "tier_a_or_b_fraction": 1.0,
            "hard_mismatch_fraction": 0.0,
        },
    }
    reconstructed = attribution.loc[
        :,
        [
            *finalizer.ATTRIBUTION_SPEARMAN_IDENTITY_COLUMNS,
            "historical_spearman",
            "current_spearman",
        ],
    ].drop_duplicates(list(finalizer.ATTRIBUTION_SPEARMAN_IDENTITY_COLUMNS))
    monkeypatch.setattr(
        finalizer,
        "_reconstruct_attribution_spearman",
        lambda *_args: (reconstructed, {"status": "TEST_RAW_RECONSTRUCTION"}),
    )
    result = finalizer._validate_attribution_spearman(
        tmp_path, attribution, summary, {"test": True}
    )
    assert result["unit_count"] == 144
    assert result["maximum_absolute_error"] == pytest.approx(0.002941176470588225)

    spearman.loc[0, "hard_mismatch"] = True
    spearman.to_csv(path, index=False)
    summary["spearman_comparison_sha256"] = sha256_file(path)
    with pytest.raises(FinalizationError, match="independent row recomputation"):
        finalizer._validate_attribution_spearman(
            tmp_path, attribution, summary, {"test": True}
        )


def test_attribution_reference_receipt_uses_analysis_bound_invocation(
    tmp_path: Path,
) -> None:
    replay_root = tmp_path / "verification/replay"
    current_root = replay_root / "family_replays/invocation-current/attribution"
    current_receipt = current_root / "receipts/attribution_reference_inputs.json"
    stale_receipt = (
        replay_root
        / "family_replays/invocation-stale/attribution/receipts/"
        "attribution_reference_inputs.json"
    )
    _write_json(current_receipt, {"schema_version": 1, "invocation": "current"})
    _write_json(stale_receipt, {"schema_version": 1, "invocation": "stale"})
    family_receipt = replay_root / "family_replays/family_replay_receipt.json"
    _write_json(
        family_receipt,
        {
            "families": [
                {
                    "family": "attribution",
                    "path": "family_replays/invocation-current/attribution",
                    "status": "completed",
                    "analysis": {"source_mode": "sealed_reference_replay"},
                    "contract": {"status": "passed"},
                }
            ]
        },
    )
    analysis_path = tmp_path / "verification/analysis_replay.json"
    _write_json(
        analysis_path,
        {"family_replay_receipt_sha256": sha256_file(family_receipt)},
    )

    selected, receipt = finalizer._attribution_reference_receipt(tmp_path)

    assert selected == current_receipt.resolve()
    assert receipt["invocation"] == "current"

    _write_json(analysis_path, {"family_replay_receipt_sha256": "0" * 64})
    with pytest.raises(FinalizationError, match="does not bind"):
        finalizer._attribution_reference_receipt(tmp_path)


def test_funnybirds_current_spearman_rejects_mutated_heldout_vector() -> None:
    from types import SimpleNamespace

    row = SimpleNamespace(
        patch_scores=np.asarray([1.0, 2.0, 3.0, 4.0]),
        heldout_background_texture_effects=np.asarray([1.0, 2.0, 4.0, 3.0]),
        heldout_telea_dilate3_effects=np.asarray([4.0, 1.0, 2.0, 3.0]),
        spearman_background_texture=0.8,
        spearman_telea_dilate3=-0.2,
        spearman=0.3,
        quality_aggregation="equal_mean_within_image",
    )
    assert finalizer._current_member_spearman(row, funnybirds=True) == pytest.approx(0.3)

    row.heldout_background_texture_effects = np.asarray([4.0, 3.0, 2.0, 1.0])
    with pytest.raises(FinalizationError, match="held-out raw-vector recomputation"):
        finalizer._current_member_spearman(row, funnybirds=True)


def test_current_decaf_vector_binding_rejects_synchronized_derived_forgery() -> None:
    raw = np.asarray([0.1, 0.2, 0.3, 0.4])
    finalizer._require_attribution_vector_binding(raw.copy(), raw, label="current_M")
    forged = raw.copy()
    forged[2] += 0.01
    with pytest.raises(FinalizationError, match="bound raw output vector"):
        finalizer._require_attribution_vector_binding(forged, raw, label="current_M")


def test_deterministic_zip_order_metadata_and_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    (root / "comparisons").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "comparisons/z.csv").write_text("z\n", encoding="utf-8")
    (root / "comparisons/a.json").write_text("{}\n", encoding="utf-8")
    (root / "diagnostics/logs").mkdir(parents=True)
    (root / C0_DIAGNOSTIC_RELATIVE).write_text("{}\n", encoding="utf-8")
    (root / C0_RUNTIME_ATTRIBUTION_RELATIVE).write_text("{}\n", encoding="utf-8")
    (root / "diagnostics/unrelated.json").write_text("{}\n", encoding="utf-8")
    (root / "diagnostics/logs/private.json").write_text("{}\n", encoding="utf-8")
    (root / "logs/private.log").write_text("secret\n", encoding="utf-8")
    attribution_source = (
        root / "provenance/historical_sources/attribution_idsds/cmr/source.py"
    )
    covertype_source = root / "provenance/historical_sources/covertype/cmr/source.py"
    unrelated_python = root / "provenance/unrelated.py"
    nested_forbidden = (
        root / "provenance/historical_sources/covertype/checkpoints/private.py"
    )
    excluded_zip = root / "provenance/historical_source_snapshots/source.zip"
    symlinked_source = root / "provenance/historical_sources/covertype/cmr/link.py"
    for path in (
        attribution_source,
        covertype_source,
        unrelated_python,
        nested_forbidden,
        excluded_zip,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")
    symlinked_source.symlink_to(covertype_source)
    verification = root / "verification"
    verification.mkdir()
    for name in (
        "analysis_replay.json",
        "headline_assertions.json",
        "cpu_verification.json",
    ):
        (verification / name).write_text("{}\n", encoding="utf-8")
    (verification / "paper_artifact_diff.csv").write_text("key,value\n", encoding="utf-8")
    receipt_root = verification / "paper_outputs/receipts"
    receipt_root.mkdir(parents=True)
    for name in ("canonical_receipt.json", "family_replay_receipt.json", "replay_receipt.json"):
        (receipt_root / name).write_text("{}\n", encoding="utf-8")
    forbidden_outputs = (
        verification / "paper_outputs/canonical/result.csv",
        verification / "paper_outputs/full/result.csv",
        verification / "replay/run/paper_outputs/canonical/result.csv",
        verification / "replay/run/paper_outputs/full/result.csv",
    )
    for path in forbidden_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("forbidden\n", encoding="utf-8")
    (root / "CROSS_GENERATION_EQUIVALENCE_REPORT_V2.md").write_text("report\n", encoding="utf-8")
    (root / "CROSS_GENERATION_EQUIVALENCE_STATUS_V2.json").write_text("{}\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    one = write_deterministic_zip(root, first)
    two = write_deterministic_zip(root, second)

    assert one["sha256"] == two["sha256"]
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert all("logs" not in name for name in archive.namelist())
        assert any(
            name.endswith("CROSS_GENERATION_EQUIVALENCE_REPORT_V2.md")
            for name in archive.namelist()
        )
        assert any(
            name.endswith("CROSS_GENERATION_EQUIVALENCE_STATUS_V2.json")
            for name in archive.namelist()
        )
        assert any(name.endswith(C0_DIAGNOSTIC_RELATIVE.as_posix()) for name in archive.namelist())
        assert any(
            name.endswith(C0_RUNTIME_ATTRIBUTION_RELATIVE.as_posix()) for name in archive.namelist()
        )
        assert all("diagnostics/unrelated.json" not in name for name in archive.namelist())
        assert all("diagnostics/logs" not in name for name in archive.namelist())
        verification_members = [
            name.split("/verification/", 1)[1]
            for name in archive.namelist()
            if "/verification/" in name
        ]
        assert set(verification_members) == {
            "analysis_replay.json",
            "headline_assertions.json",
            "cpu_verification.json",
            "paper_artifact_diff.csv",
            "paper_outputs/receipts/canonical_receipt.json",
            "paper_outputs/receipts/family_replay_receipt.json",
            "paper_outputs/receipts/replay_receipt.json",
        }
        assert all(not name.startswith("replay/") for name in verification_members)
        assert all("paper_outputs/canonical" not in name for name in verification_members)
        assert all("paper_outputs/full" not in name for name in verification_members)
        names = set(archive.namelist())
        assert any(
            name.endswith("historical_sources/attribution_idsds/cmr/source.py")
            for name in names
        )
        assert any(name.endswith("historical_sources/covertype/cmr/source.py") for name in names)
        assert all(not name.endswith("provenance/unrelated.py") for name in names)
        assert all("/checkpoints/" not in name for name in names)
        assert all(not name.endswith("source.zip") for name in names)
        assert all(not name.endswith("cmr/link.py") for name in names)
    assert first.with_name(f"{first.name}.sha256").read_text().startswith(one["sha256"])


def test_report_labels_core_and_e2e_statistics_without_conflation() -> None:
    core = {
        "unit_count": 1,
        "tier_a_fraction": 1.0,
        "tier_b_fraction": 0.0,
        "hard_mismatch_fraction": 0.0,
        "gate_agreement": 1.0,
        "orientation_agreement": 1.0,
        "dominant_mechanism_agreement": 1.0,
    }
    e2e = {"unit_count": 2, "tier_a_or_b_fraction": 1.0, "hard_mismatch_fraction": 0.0}
    variables = {
        name: {
            "median_absolute_error": 0.0,
            "p95_absolute_error": 0.0,
            "maximum_absolute_error": 0.0,
            "agreement": 1.0,
        }
        for name in ("d", *finalizer.SUMMARY_NAMES, "feature-vector agreement")
    }
    variables.update(
        {
            "gate": {"agreement": 1.0},
            "orientation": {"agreement": 1.0},
            "dominant mechanism": {"agreement": 1.0},
        }
    )
    variables["feature-vector agreement"]["reconstruction"] = {
        "historical_funnybirds": {"member_sha256": "a0-heldout"}
    }
    families = {
        name: {
            "status": "PASS_CORE_AND_E2E",
            "current_core": core,
            "current_e2e": e2e,
        }
        for name in ("controlled", "imagenet9", "attribution", "covertype")
    }
    families["controlled"].update(
        {
            "current_e2e": None,
            "c0_candidate_qualification": {
                "excluded": [],
                "manifest_sha256": "a",
                "diagnostic_sha256": "b",
                "sealed_aggregate_audit_sha256": "c",
            },
            "c0_runtime_attribution": {"sha256": "d"},
        }
    )
    families["imagenet9"]["historical_source_binding"] = {"package_sha256": "e"}
    families["attribution"]["historical_source_bindings"] = {
        "a0_funnybirds": {
            "source_python_file_count": 381,
            "source_tree_sha256": "a0-tree",
        },
        "a2_imagenet1k_idsds": {
            "package_sha256": "a2-package",
            "manifest_member_count": 2606,
            "archive_member_count": 2607,
            "namespace_member_count": 19,
        },
        "provenance_snapshots": {
            "attribution_a0": {
                "deployment_receipt": {"package_member": "snapshot/deployment.json"},
                "formal_plan": {"package_member": "snapshot/formal.jsonl"},
                "formal_plan_receipt": {"package_member": "snapshot/formal.receipt.json"},
            },
            "attribution_a2": {
                "package_manifest": {
                    "package_member": "snapshot/a2.json",
                    "sha256": "a2-manifest",
                }
            },
        },
    }
    families["covertype"]["historical_source_binding"] = {
        "provenance_snapshot": {
            "package_member": "snapshot/covertype.json",
            "sha256": "cover",
        }
    }
    families["dinov2-g"] = {
        "status": "PASS_CORE_WITH_METADATA_LIMIT",
        "current_core": None,
        "current_e2e": None,
    }
    report = finalizer._render_report(
        {
            "overall_verdict": "PASS_FOR_PAPER_REPRODUCTION",
            "families": families,
            "current_core_agreement": core,
            "current_e2e_agreement": e2e,
            "variable_statistics": variables,
            "remaining_scoped_gaps": ["bounded gap"],
            "scientific_mismatches": {
                "unique_hard_mismatch_count": 0,
                "hard_rows": [],
                "same_units_in_core_and_row_level_e2e": True,
                "shared_unit_ids": [],
                "current_core_only_unit_ids": [],
                "current_e2e_only_unit_ids": [],
                "current_e2e": {
                    "covertype_exact_estimator_test_units": {"unit_count": 480000}
                },
            },
            "scientific_mismatches_found": False,
            "paper_replay": {"assertions_passed": 27, "assertions_total": 27},
            "repository": {
                "commit": "f" * 40,
                "head_tree": "1" * 40,
                "index_tree": "2" * 40,
                "index_matches_head": True,
                "index_snapshot_sha256": "5" * 64,
                "working_tree_snapshot_sha256": "3" * 64,
            },
            "package": {"path": "/tmp/package.zip", "sha256": "4" * 64},
        }
    )

    assert "| Core Tier A | Core Tier B | Core hard mismatch |" in report
    assert "### Current-core variables and semantics" in report
    assert "### Current end-to-end variables" in report
    assert "Endpoint `d` and feature-vector agreement" in report
    assert "`M`, `E`, `C`, `F`, and `Abs`" in report


def test_paper_verdict_is_not_downgraded_by_scoped_optional_gaps() -> None:
    families = {
        "controlled": {"status": "PASS_CORE"},
        "imagenet9": {"status": "PASS_CORE_AND_E2E"},
        "attribution": {"status": "PASS_CORE_AND_E2E"},
        "covertype": {"status": "PASS_CORE_AND_E2E"},
    }
    paper = {"status": "PASS", "assertions_passed": 27, "assertions_total": 27}

    assert determine_overall_verdict(families, paper) == "PASS_FOR_PAPER_REPRODUCTION"


def test_repository_provenance_includes_untracked_diff_and_content_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True
    )
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
    source = repository / "tools/crossgen/new.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 7\n", encoding="utf-8")
    root = tmp_path / "v2"

    observed_git_calls: list[tuple[tuple[str, ...], str | None]] = []
    real_run = subprocess.run

    def observing_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[object]:
        command = args[0] if args else kwargs.get("args")
        if isinstance(command, list) and command and command[0] == "git":
            environment = kwargs.get("env")
            observed_git_calls.append(
                (
                    tuple(map(str, command)),
                    environment.get("GIT_OPTIONAL_LOCKS")
                    if isinstance(environment, dict)
                    else None,
                )
            )
        return real_run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(finalizer.subprocess, "run", observing_run)
    provenance = capture_repository_provenance(root, repository)

    assert "tools/crossgen/new.py" in provenance["untracked_paths"]
    assert provenance["source_hashes"]["tools/crossgen/new.py"]
    patch = (root / "provenance/repository_diff.patch").read_text(encoding="utf-8")
    assert "tools/crossgen/new.py" in patch
    assert "VALUE = 7" in patch
    manifest = json.loads((root / "provenance/repository_working_tree_manifest.json").read_text())
    assert any(row["path"] == "tools/crossgen/new.py" for row in manifest["entries"])
    assert observed_git_calls
    assert all(value == "0" for _, value in observed_git_calls)
    assert all("write-tree" not in command for command, _ in observed_git_calls)
    assert provenance["index_identity_kind"] == "git_ls_files_stage_z_sha256"
    assert provenance["index_matches_head"] is True

    dry_root = tmp_path / "dry-v2"
    dry = capture_repository_provenance(dry_root, repository, write_outputs=False)
    assert dry["working_tree_snapshot_sha256"] == provenance["working_tree_snapshot_sha256"]
    assert not dry_root.exists()


def test_collect_evidence_fails_closed_on_missing_mandatory_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v2"
    root.mkdir()
    with pytest.raises(FinalizationError, match="Controlled C0"):
        collect_evidence(root, tmp_path / "repo", tmp_path / "replay", tmp_path / "b200")
