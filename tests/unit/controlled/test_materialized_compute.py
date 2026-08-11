from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from decaf.core.manifests import atomic_write_json, build_file_manifest, sha256_file
from decaf.experiments.common import RunContext, load_profile
from decaf.experiments.controlled.analyze import (
    controlled_reference_complete,
    controlled_reference_paths,
)
from decaf.experiments.controlled.cli import build_plan, compute_handler
from decaf.experiments.controlled.evaluate import (
    build_members,
    checkpoint_bindings_from_manifest,
    member_checkpoint_contract,
    member_spec_sha256,
    prepared_run_bindings,
    receipt_reusable,
    validate_checkpoint_binding_universe,
    validate_materialized_member_bundle,
    write_jobs_manifest,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _prepare_fixture_context(tmp_path: Path, config: dict[str, object]) -> RunContext:
    context = RunContext.create(
        experiment="controlled",
        profile="paper",
        stage="compute",
        output=tmp_path / "run",
        config=config,
        workers=1,
        resume=False,
    )
    members = build_members(config)
    atomic_write_json(context.path / "manifests" / "plan.json", build_plan(config))
    write_jobs_manifest(context.path / "manifests" / "jobs.jsonl", members)
    atomic_write_json(
        context.path / "manifests" / "data.json",
        {"schema_version": 1, "items": [{"id": "fixture", "sha256": _digest("data")}]},
    )

    checkpoints: dict[str, dict[str, object]] = {}
    probability_caches: dict[str, dict[str, object]] = {}
    for member in members:
        model_id = str(member.metadata["model_id"])
        if member.phase == "c0_evaluate":
            checkpoints.setdefault(
                model_id,
                {
                    "model_id": model_id,
                    "logical_path": f"checkpoints/c0/{model_id}.pt",
                    "bytes": 101,
                    "sha256": _digest(f"checkpoint:{model_id}"),
                },
            )
            probability_caches.setdefault(
                model_id,
                {
                    "model_id": model_id,
                    "logical_path": f"probability_caches/c0/{model_id}.npy",
                    "bytes": 103,
                    "sha256": _digest(f"cache:{model_id}"),
                },
            )
        elif member.phase in {"c1_train", "c2_train"}:
            family = "c1" if member.phase == "c1_train" else "c2"
            for output in member.metadata["checkpoint_outputs"]:
                checkpoint_id = Path(str(output)).stem
                checkpoints[checkpoint_id] = {
                    "model_id": checkpoint_id,
                    "logical_path": f"checkpoints/{family}/{checkpoint_id}.pt",
                    "bytes": 107,
                    "sha256": _digest(f"checkpoint:{checkpoint_id}"),
                    "producer_member_id": member.member_id,
                }
    atomic_write_json(
        context.path / "manifests" / "checkpoints.json",
        {
            "schema_version": 1,
            "items": [
                {
                    "id": "fixture",
                    "checkpoints": list(checkpoints.values()),
                    "probability_caches": list(probability_caches.values()),
                }
            ],
        },
    )
    context.record_stage("prepare", "completed", started_at="fixture")
    return context


def _write_materialized_bundle(
    root: Path,
    context: RunContext,
    *,
    identity_only_member: str | None = None,
) -> None:
    config = context.config
    members = build_members(config)
    run_bindings = prepared_run_bindings(context, members)
    checkpoint_bindings = checkpoint_bindings_from_manifest(
        context.path / "manifests" / "checkpoints.json"
    )
    records = []
    artifact_digests: dict[str, str] = {}
    for member in members:
        output = root / member.output
        output.parent.mkdir(parents=True, exist_ok=True)
        if member.member_id == identity_only_member:
            document = {
                "schema_version": 1,
                "member_id": member.member_id,
                "phase": member.phase,
                "status": "completed",
            }
        else:
            checkpoint_inputs, cache_inputs, produced = member_checkpoint_contract(
                member, checkpoint_bindings
            )
            document = {
                "schema_version": 2,
                "kind": "controlled_member_result",
                "member_id": member.member_id,
                "phase": member.phase,
                "status": "completed",
                "member_spec_sha256": member_spec_sha256(member),
                "run_bindings": run_bindings,
                "dependencies": list(member.dependencies),
                "dependency_artifacts": {
                    dependency: artifact_digests[dependency]
                    for dependency in member.dependencies
                },
                "input_bindings": {
                    "data_manifest_sha256": run_bindings["data_manifest_sha256"],
                    "checkpoint_manifest_sha256": run_bindings[
                        "checkpoint_manifest_sha256"
                    ],
                    "checkpoints": checkpoint_inputs,
                    "probability_caches": cache_inputs,
                },
                "produced_checkpoints": produced,
                "result": {
                    "schema": f"{member.phase}_summary_v1",
                    "record_count": 1,
                    "metrics": {"completed_units": 1.0},
                },
            }
        output.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        digest = sha256_file(output)
        artifact_digests[member.member_id] = digest
        records.append(
            {
                "member_id": member.member_id,
                "output": member.output,
                "size": output.stat().st_size,
                "sha256": digest,
            }
        )
    atomic_write_json(
        root / "manifests" / "members.json",
        {
            "schema_version": 2,
            "kind": "controlled_members",
            "producer_execution_class": "accelerator",
            "run_bindings": run_bindings,
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
    analysis_manifest.update(
        {
            "schema_version": 2,
            "kind": "controlled_analysis",
            "producer_execution_class": "accelerator",
            "run_bindings": run_bindings,
            "member_manifest_sha256": sha256_file(root / "manifests" / "members.json"),
        }
    )
    atomic_write_json(root / "manifests" / "analysis.json", analysis_manifest)


def _paper_fixture_config() -> dict[str, object]:
    config = load_profile("controlled", "smoke")
    config["profile"] = "paper"
    config["execution"] = {
        "backend": "materialized_accelerator_outputs",
        "materialized_root_environment": "DECAF_CONTROLLED_GPU_OUTPUT_ROOT",
        "member_manifest": "manifests/members.json",
        "analysis_manifest": "manifests/analysis.json",
        "analysis_root": "analysis",
    }
    return config


def test_materialized_member_manifest_fails_closed_on_tampering(tmp_path: Path) -> None:
    config = _paper_fixture_config()
    context = _prepare_fixture_context(tmp_path, config)
    source = tmp_path / "accelerator"
    source.mkdir()
    _write_materialized_bundle(source, context)
    members = build_members(config)
    run_bindings = prepared_run_bindings(context, members)
    checkpoint_bindings = checkpoint_bindings_from_manifest(
        context.path / "manifests" / "checkpoints.json"
    )
    bundle = validate_materialized_member_bundle(
        source,
        members,
        run_bindings=run_bindings,
        checkpoint_bindings=checkpoint_bindings,
    )
    assert len(bundle.artifacts) == 11

    first = bundle.artifacts[members[0].member_id].source
    first.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte identity mismatch"):
        validate_materialized_member_bundle(
            source,
            members,
            run_bindings=run_bindings,
            checkpoint_bindings=checkpoint_bindings,
        )


def test_materialized_member_manifest_rejects_identity_only_documents(tmp_path: Path) -> None:
    config = _paper_fixture_config()
    context = _prepare_fixture_context(tmp_path, config)
    members = build_members(config)
    source = tmp_path / "accelerator"
    source.mkdir()
    _write_materialized_bundle(source, context, identity_only_member=members[0].member_id)

    with pytest.raises(ValueError, match="identity or run binding mismatch"):
        validate_materialized_member_bundle(
            source,
            members,
            run_bindings=prepared_run_bindings(context, members),
            checkpoint_bindings=checkpoint_bindings_from_manifest(
                context.path / "manifests" / "checkpoints.json"
            ),
        )


def test_checkpoint_binding_universe_rejects_registered_extras(tmp_path: Path) -> None:
    config = _paper_fixture_config()
    context = _prepare_fixture_context(tmp_path, config)
    members = build_members(config)
    bindings = checkpoint_bindings_from_manifest(
        context.path / "manifests" / "checkpoints.json"
    )
    bindings["checkpoints"]["extra"] = {
        "model_id": "extra",
        "logical_path": "checkpoints/c1/extra.pt",
        "bytes": 1,
        "sha256": _digest("extra"),
        "producer_member_id": "c1_train__extra",
    }

    with pytest.raises(ValueError, match="checkpoint binding universe"):
        validate_checkpoint_binding_universe(members, bindings)


def test_prepared_run_bindings_rejects_jobs_not_matching_plan(tmp_path: Path) -> None:
    config = _paper_fixture_config()
    context = _prepare_fixture_context(tmp_path, config)
    jobs = context.path / "manifests" / "jobs.jsonl"
    jobs.write_text(jobs.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="jobs manifest does not match"):
        prepared_run_bindings(context, build_members(config))


def test_paper_compute_ingests_complete_materialized_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _paper_fixture_config()
    context = _prepare_fixture_context(tmp_path, config)
    source = tmp_path / "accelerator"
    source.mkdir()
    _write_materialized_bundle(source, context)
    monkeypatch.setenv("DECAF_CONTROLLED_GPU_OUTPUT_ROOT", str(source))

    result = compute_handler(context)

    assert result["members"] == 11
    assert result["analysis_inputs"] == 14
    assert result["byte_identity_verified"] is True
    assert result["gpu_execution_performed_here"] is False
    assert len(list((context.path / "receipts" / "members").glob("*.json"))) == 11
    assert controlled_reference_complete(context.path / "paper_data" / "reference")
    first_member = build_members(config)[0]
    assert receipt_reusable(
        context,
        first_member,
        run_bindings=prepared_run_bindings(context, build_members(config)),
    )
    receipt_path = context.path / "receipts" / "members" / f"{first_member.member_id}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["details"]["run_bindings"]["jobs_manifest_sha256"] = "0" * 64
    atomic_write_json(receipt_path, receipt)
    assert not receipt_reusable(
        context,
        first_member,
        run_bindings=prepared_run_bindings(context, build_members(config)),
    )


def test_paper_compute_rejects_analysis_from_a_different_member_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _paper_fixture_config()
    context = _prepare_fixture_context(tmp_path, config)
    source = tmp_path / "accelerator"
    source.mkdir()
    _write_materialized_bundle(source, context)
    manifest_path = source / "manifests" / "analysis.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["member_manifest_sha256"] = "0" * 64
    atomic_write_json(manifest_path, manifest)
    monkeypatch.setenv("DECAF_CONTROLLED_GPU_OUTPUT_ROOT", str(source))

    with pytest.raises(ValueError, match="member-manifest fingerprint mismatch"):
        compute_handler(context)
