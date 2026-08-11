from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from decaf.core.manifests import atomic_write_json, build_file_manifest
from decaf.experiments.common import RunContext, load_profile
from decaf.experiments.controlled.analyze import (
    controlled_reference_complete,
    controlled_reference_paths,
)
from decaf.experiments.controlled.cli import build_plan, compute_handler
from decaf.experiments.controlled.evaluate import (
    build_members,
    configuration_sha256,
    member_contract_sha256,
    validate_materialized_member_bundle,
)


def _write_materialized_bundle(root: Path, config: dict[str, object]) -> None:
    members = build_members(config)
    records = []
    for member in members:
        output = root / member.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "member_id": member.member_id,
                    "phase": member.phase,
                    "status": "completed",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "member_id": member.member_id,
                "output": member.output,
                "size": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
    atomic_write_json(
        root / "manifests" / "members.json",
        {
            "schema_version": 1,
            "kind": "controlled_members",
            "producer_execution_class": "accelerator",
            "configuration_sha256": configuration_sha256(config),
            "member_contract_sha256": member_contract_sha256(members),
            "members": records,
        },
    )

    analysis_files = []
    for relative in controlled_reference_paths(prefix="analysis"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{relative}".encode())
        analysis_files.append(path)
    analysis_manifest = build_file_manifest(analysis_files, root=root)
    analysis_manifest["kind"] = "controlled_analysis"
    atomic_write_json(root / "manifests" / "analysis.json", analysis_manifest)


def test_materialized_member_manifest_fails_closed_on_tampering(tmp_path: Path) -> None:
    config = load_profile("controlled", "smoke")
    source = tmp_path / "accelerator"
    source.mkdir()
    _write_materialized_bundle(source, config)
    members = build_members(config)
    bundle = validate_materialized_member_bundle(
        source,
        members,
        config_sha256=configuration_sha256(config),
    )
    assert len(bundle.artifacts) == 11

    first = bundle.artifacts[members[0].member_id].source
    first.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte identity mismatch"):
        validate_materialized_member_bundle(
            source,
            members,
            config_sha256=configuration_sha256(config),
        )


def test_paper_compute_ingests_complete_materialized_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_profile("controlled", "smoke")
    config["profile"] = "paper"
    config["execution"] = {
        "backend": "materialized_accelerator_outputs",
        "materialized_root_environment": "DECAF_CONTROLLED_GPU_OUTPUT_ROOT",
        "member_manifest": "manifests/members.json",
        "analysis_manifest": "manifests/analysis.json",
        "analysis_root": "analysis",
    }
    source = tmp_path / "accelerator"
    source.mkdir()
    _write_materialized_bundle(source, config)
    monkeypatch.setenv("DECAF_CONTROLLED_GPU_OUTPUT_ROOT", str(source))

    context = RunContext.create(
        experiment="controlled",
        profile="paper",
        stage="compute",
        output=tmp_path / "run",
        config=config,
        workers=1,
        resume=False,
    )
    atomic_write_json(context.path / "manifests" / "plan.json", build_plan(config))
    context.record_stage("prepare", "completed", started_at="fixture")
    result = compute_handler(context)

    assert result["members"] == 11
    assert result["analysis_inputs"] == 14
    assert result["byte_identity_verified"] is True
    assert result["gpu_execution_performed_here"] is False
    assert len(list((context.path / "receipts" / "members").glob("*.json"))) == 11
    assert controlled_reference_complete(context.path / "paper_data" / "reference")
