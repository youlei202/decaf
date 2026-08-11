"""Controlled static member expansion and receipt-driven execution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from decaf.core.manifests import atomic_write_json, sha256_file
from decaf.core.receipts import (
    finalize_global_receipt,
    load_member_receipt,
    write_member_receipt,
)
from decaf.experiments.common import RunContext, atomic_text
from decaf.experiments.controlled.models import SHA256_PATTERN, expected_base_models
from decaf.experiments.controlled.protocols import (
    analytic_context_mixture,
    decompose_score_trajectory,
)
from decaf.experiments.controlled.train import (
    c1_checkpoint_producers,
    c1_factory_training_jobs,
    c2_training_jobs,
    selected_c1_checkpoints,
)


@dataclass(frozen=True, slots=True)
class PlanMember:
    """One deterministic unit in the controlled schedule contract."""

    member_id: str
    phase: str
    resource: str
    seed: int
    output: str
    dependencies: tuple[str, ...] = ()
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "phase": self.phase,
            "required": self.required,
            "resource": self.resource,
            "seed": self.seed,
            "dependencies": list(self.dependencies),
            "output": self.output,
            **dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MaterializedMemberArtifact:
    """One hash-verified accelerator artifact registered for ingestion."""

    member_id: str
    output: str
    source: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class MaterializedMemberBundle:
    """Validated materialized accelerator-member bundle."""

    root: Path
    manifest: Path
    producer_execution_class: str
    artifacts: Mapping[str, MaterializedMemberArtifact]


def _safe_token(value: Any) -> str:
    token = str(value).strip().replace("/", "_").replace(" ", "_")
    if (
        not token
        or token in {".", ".."}
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in token
        )
    ):
        raise ValueError(f"unsafe member token: {value!r}")
    return token


def build_members(config: Mapping[str, Any]) -> tuple[PlanMember, ...]:
    """Expand C0, C1, and C2 into sorted, path-unique member jobs."""

    members: list[PlanMember] = []
    base = config["base"]
    base_models = expected_base_models(
        tuple(map(str, base["tasks"])),
        tuple(map(str, base["architectures"])),
        tuple(map(int, base["seeds"])),
    )
    factors = tuple(map(str, base["factors"]))
    for record in base_models:
        for factor in factors:
            identifier = _safe_token(f"c0__{record.model_id}__{factor}")
            members.append(
                PlanMember(
                    member_id=identifier,
                    phase="c0_evaluate",
                    resource="accelerator",
                    seed=record.seed,
                    output=f"raw/c0/{record.model_id}/{factor}.json",
                    metadata={
                        "family": "C0",
                        "model_id": record.model_id,
                        "task": record.task,
                        "architecture": record.architecture,
                        "factor": factor,
                        "no_retraining": True,
                    },
                )
            )

    endpoint = config["endpoint_behavior"]
    c1_training = c1_factory_training_jobs(endpoint)
    c1_producers = c1_checkpoint_producers(endpoint)
    for job in c1_training:
        members.append(
            PlanMember(
                member_id=job.member_id,
                phase="c1_train",
                resource="accelerator",
                seed=job.seed,
                output=f"raw/c1/training/{job.model_id}.json",
                metadata={
                    "family": "C1",
                    "model_id": job.model_id,
                    "module": job.module,
                    "task": job.task,
                    "architecture": job.architecture,
                    "checkpoint_outputs": list(job.outputs),
                },
            )
        )
    passes = endpoint["passes"]
    for checkpoint in selected_c1_checkpoints(endpoint):
        for pass_name in sorted(passes):
            for factor in checkpoint["factors"]:
                identifier = _safe_token(f"c1__{pass_name}__{checkpoint['model_id']}__{factor}")
                members.append(
                    PlanMember(
                        member_id=identifier,
                        phase="c1_measure",
                        resource="accelerator",
                        seed=int(checkpoint["seed"]),
                        output=(f"raw/c1/{pass_name}/{checkpoint['model_id']}/{factor}.json"),
                        dependencies=(c1_producers[str(checkpoint["model_id"])],),
                        metadata={
                            "family": "C1",
                            "pass": pass_name,
                            "model_id": checkpoint["model_id"],
                            "module": checkpoint["module"],
                            "task": checkpoint["task"],
                            "architecture": checkpoint["architecture"],
                            "factor": factor,
                        },
                    )
                )

    contradiction = config["contradiction"]
    training = c2_training_jobs(contradiction)
    for job in training:
        members.append(
            PlanMember(
                member_id=job.member_id,
                phase="c2_train",
                resource="accelerator",
                seed=job.seed,
                output=f"raw/c2/training/{job.model_id}.json",
                metadata={
                    "family": "C2",
                    "model_id": job.model_id,
                    "task": job.task,
                    "architecture": job.architecture,
                    "checkpoint_outputs": list(job.outputs),
                },
            )
        )
        evaluation_id = _safe_token(f"c2_eval__{job.model_id}")
        members.append(
            PlanMember(
                member_id=evaluation_id,
                phase="c2_evaluate",
                resource="accelerator",
                seed=job.seed,
                output=f"raw/c2/evaluation/{job.model_id}.json",
                dependencies=(job.member_id,),
                metadata={
                    "family": "C2",
                    "model_id": job.model_id,
                    "task": job.task,
                    "architecture": job.architecture,
                    "wall_maps": list(map(int, contradiction["wall_maps"])),
                    "epsilon_grid": list(map(float, contradiction["epsilon_grid"])),
                },
            )
        )

    phase_order = {
        "c0_evaluate": 0,
        "c1_train": 1,
        "c1_measure": 2,
        "c2_train": 3,
        "c2_evaluate": 4,
    }
    members.sort(key=lambda member: (phase_order[member.phase], member.member_id))
    identifiers = [member.member_id for member in members]
    outputs = [member.output for member in members]
    receipts = [f"receipts/members/{member.member_id}.json" for member in members]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("controlled plan produced duplicate member IDs")
    if len(outputs) != len(set(outputs)):
        raise ValueError("controlled plan produced duplicate output paths")
    if len(receipts) != len(set(receipts)):
        raise ValueError("controlled plan produced duplicate receipt paths")
    known = set(identifiers)
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    for member in members:
        if not set(member.dependencies).issubset(known):
            raise ValueError(f"member {member.member_id} has an unknown dependency")
        if any(
            positions[dependency] >= positions[member.member_id]
            for dependency in member.dependencies
        ):
            raise ValueError(f"member {member.member_id} is not topologically ordered")
    return tuple(members)


def plan_counts(config: Mapping[str, Any], members: Sequence[PlanMember]) -> dict[str, int]:
    """Return scientific cardinalities separately from scheduler member count."""

    checkpoints = selected_c1_checkpoints(config["endpoint_behavior"])
    module_counts = {
        module: sum(row["module"] == module for row in checkpoints) for module in ("E", "C", "F")
    }
    counts = {
        "base_models": len(
            {member.metadata.get("model_id") for member in members if member.phase == "c0_evaluate"}
        ),
        "base_model_factor_units": sum(member.phase == "c0_evaluate" for member in members),
        "evidence_checkpoints": module_counts["E"],
        "causal_checkpoints": module_counts["C"],
        "fragility_checkpoints": module_counts["F"],
        "endpoint_behavior_checkpoints": len(checkpoints),
        "endpoint_behavior_training_jobs": sum(member.phase == "c1_train" for member in members),
        "endpoint_behavior_units_per_pass": sum(
            member.phase == "c1_measure" and member.metadata.get("pass") == "pass2"
            for member in members
        ),
        "endpoint_behavior_units_total": sum(member.phase == "c1_measure" for member in members),
        "contradiction_models": sum(member.phase == "c2_train" for member in members),
        "contradiction_evaluation_units": sum(member.phase == "c2_evaluate" for member in members),
        "scheduled_members": len(members),
    }
    return {key: int(value) for key, value in counts.items()}


def configuration_sha256(config: Mapping[str, Any]) -> str:
    """Fingerprint the portable scientific configuration, excluding its path."""

    payload = {key: value for key, value in config.items() if key != "_source"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def member_contract_sha256(members: Sequence[PlanMember]) -> str:
    """Fingerprint every scheduled member, dependency, and declared output."""

    encoded = json.dumps(
        [member.as_dict() for member in members],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_jobs_manifest(path: str | Path, members: Sequence[PlanMember]) -> Path:
    """Atomically persist the sorted JSONL schedule contract."""

    destination = Path(path)
    text = "".join(
        json.dumps(member.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        for member in members
    )
    atomic_text(destination, text)
    return destination


def resolve_materialized_output_root(
    config: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the externally produced accelerator bundle from a named variable."""

    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise ValueError("controlled execution config must be a mapping")
    variable = str(
        execution.get("materialized_root_environment", "DECAF_CONTROLLED_GPU_OUTPUT_ROOT")
    )
    env = os.environ if environment is None else environment
    raw_root = env.get(variable)
    if not raw_root:
        raise RuntimeError(
            f"{variable} is required for paper compute; it must identify a "
            "hash-registered materialized accelerator bundle"
        )
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"materialized Controlled output root is missing: {root}")
    return root


def _contained_file(root: Path, relative_value: str, *, label: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or not relative.parts:
        raise ValueError(f"{label} path must be relative")
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes the materialized root") from error
    if not candidate.is_file():
        raise ValueError(f"{label} path is not a regular file: {relative_value}")
    return candidate


def validate_materialized_member_bundle(
    root: str | Path,
    members: Sequence[PlanMember],
    *,
    config_sha256: str,
    manifest_relative: str = "manifests/members.json",
) -> MaterializedMemberBundle:
    """Validate exact member/path/size/hash coverage for accelerator outputs.

    The manifest's execution class is producer-declared.  This CPU-side loader
    verifies byte identity and member closure only; it deliberately does not
    claim that accelerator inference was independently rerun here.
    """

    bundle_root = Path(root).resolve(strict=True)
    manifest_path = _contained_file(bundle_root, manifest_relative, label="member manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("materialized member manifest must be an object")
    if payload.get("schema_version") != 1 or payload.get("kind") != "controlled_members":
        raise ValueError("materialized member manifest has an unsupported schema")
    producer_execution_class = str(payload.get("producer_execution_class", ""))
    if producer_execution_class != "accelerator":
        raise ValueError("materialized members must declare the accelerator execution class")
    if payload.get("configuration_sha256") != config_sha256:
        raise ValueError("materialized member configuration fingerprint mismatch")
    expected_contract = member_contract_sha256(members)
    if payload.get("member_contract_sha256") != expected_contract:
        raise ValueError("materialized member-plan fingerprint mismatch")
    raw_records = payload.get("members")
    if not isinstance(raw_records, list):
        raise ValueError("materialized member manifest members must be a list")

    expected = {member.member_id: member for member in members}
    artifacts: dict[str, MaterializedMemberArtifact] = {}
    registered_outputs: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("materialized member record must be an object")
        member_id = str(raw_record.get("member_id", ""))
        if member_id not in expected or member_id in artifacts:
            raise ValueError(f"unexpected or duplicate materialized member: {member_id!r}")
        member = expected[member_id]
        output = str(raw_record.get("output", ""))
        if output != member.output or output in registered_outputs:
            raise ValueError(f"materialized output path mismatch for {member_id}")
        registered_outputs.add(output)
        digest = str(raw_record.get("sha256", "")).lower()
        if not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"materialized output has an invalid SHA256: {member_id}")
        try:
            size = int(raw_record.get("size"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"materialized output has an invalid size: {member_id}") from error
        if size < 1:
            raise ValueError(f"materialized output is empty: {member_id}")
        source = _contained_file(bundle_root, output, label=f"member {member_id}")
        if source.stat().st_size != size or sha256_file(source) != digest:
            raise ValueError(f"materialized output byte identity mismatch: {member_id}")
        document = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(f"materialized output is not an object: {member_id}")
        identity = (
            document.get("schema_version") == 1
            and document.get("member_id") == member_id
            and document.get("phase") == member.phase
            and document.get("status") == "completed"
        )
        if not identity:
            raise ValueError(f"materialized output identity mismatch: {member_id}")
        artifacts[member_id] = MaterializedMemberArtifact(
            member_id=member_id,
            output=output,
            source=source,
            size=size,
            sha256=digest,
        )
    if set(artifacts) != set(expected):
        missing = sorted(set(expected) - set(artifacts))
        raise ValueError(f"materialized member coverage is incomplete: {missing[:5]}")
    return MaterializedMemberBundle(
        root=bundle_root,
        manifest=manifest_path,
        producer_execution_class=producer_execution_class,
        artifacts=artifacts,
    )


def _atomic_copy_verified(source: Path, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise ValueError(f"materialized copy SHA256 mismatch: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def materialized_member_executor(bundle: MaterializedMemberBundle) -> MemberExecutor:
    """Create an executor that ingests prevalidated accelerator artifacts."""

    def executor(context: RunContext, member: PlanMember) -> Sequence[Path]:
        artifact = bundle.artifacts[member.member_id]
        destination = context.path / member.output
        return (_atomic_copy_verified(artifact.source, destination, artifact.sha256),)

    return executor


def _artifact_record(context: RunContext, artifact: Path) -> dict[str, Any]:
    resolved = artifact.resolve(strict=True)
    try:
        relative = resolved.relative_to(context.path)
    except ValueError as error:
        raise ValueError("member artifact is outside the run directory") from error
    return {
        "path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _member_receipt_path(context: RunContext, member: PlanMember) -> Path:
    return context.path / "receipts" / "members" / f"{member.member_id}.json"


def receipt_reusable(context: RunContext, member: PlanMember) -> bool:
    """Return true only when a completed receipt still matches every artifact."""

    path = _member_receipt_path(context, member)
    if not path.is_file():
        return False
    try:
        receipt = load_member_receipt(path)
    except (OSError, ValueError, TypeError):
        return False
    if receipt.get("status") != "completed":
        return False
    artifacts = receipt.get("details", {}).get("artifacts", [])
    if not artifacts:
        return False
    for artifact in artifacts:
        target = context.path / str(artifact.get("path", ""))
        if not target.is_file() or target.stat().st_size != int(artifact.get("bytes", -1)):
            return False
        if sha256_file(target) != artifact.get("sha256"):
            return False
    return True


MemberExecutor = Callable[[RunContext, PlanMember], Sequence[Path]]


def execute_members(
    context: RunContext,
    members: Sequence[PlanMember],
    executor: MemberExecutor,
) -> dict[str, Any]:
    """Execute members with atomic terminal receipts and receipt-driven resume."""

    receipts: dict[str, Mapping[str, Any]] = {}
    completed = 0
    reused = 0
    for member in members:
        receipt_path = _member_receipt_path(context, member)
        unresolved = [
            dependency
            for dependency in member.dependencies
            if receipts.get(dependency, {}).get("status") != "completed"
        ]
        if unresolved:
            raise RuntimeError(
                f"member {member.member_id} has incomplete dependencies: {unresolved}"
            )
        if context.resume and receipt_reusable(context, member):
            receipts[member.member_id] = load_member_receipt(receipt_path)
            completed += 1
            reused += 1
            continue
        write_member_receipt(
            receipt_path,
            member.member_id,
            "running",
            details={"phase": member.phase, "output": member.output},
        )
        try:
            artifacts = tuple(executor(context, member))
            if not artifacts:
                raise RuntimeError("member executor returned no artifacts")
            records = [_artifact_record(context, artifact) for artifact in artifacts]
            write_member_receipt(
                receipt_path,
                member.member_id,
                "completed",
                details={"phase": member.phase, "artifacts": records},
            )
            completed += 1
        except Exception as error:
            write_member_receipt(
                receipt_path,
                member.member_id,
                "failed",
                details={"phase": member.phase},
                error=f"{type(error).__name__}: {error}",
            )
            raise
        receipts[member.member_id] = load_member_receipt(receipt_path)
    global_path = context.path / "receipts" / "compute_members.json"
    finalize_global_receipt(
        global_path,
        context.path.name,
        receipts,
        expected_members=[member.member_id for member in members],
        details={"completed": completed, "reused": reused},
    )
    return {"members": len(members), "completed": completed, "reused": reused}


def smoke_executor(context: RunContext, member: PlanMember) -> Sequence[Path]:
    """Run a tiny score-oracle workload without pretending to be GPU verification."""

    output = context.path / member.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if member.phase in {"c1_train", "c2_train"}:
        payload = {
            "schema_version": 1,
            "kind": "cpu_oracle_training_placeholder",
            "model_id": member.metadata["model_id"],
            "family": member.metadata["family"],
            "seed": member.seed,
            "checkpoint_outputs": member.metadata.get("checkpoint_outputs", []),
            "gpu_verification": "pending",
        }
    elif member.phase == "c2_evaluate":
        endpoint = np.asarray([1.0, 1.0, -1.0, -1.0])
        swapped = endpoint if member.metadata["task"] == "direct" else -endpoint
        curves = analytic_context_mixture(
            endpoint,
            swapped,
            member.metadata["epsilon_grid"],
        )
        payload = {
            "schema_version": 1,
            "kind": "cpu_score_oracle",
            "model_id": member.metadata["model_id"],
            "task": member.metadata["task"],
            "epsilon_grid": list(map(float, member.metadata["epsilon_grid"])),
            "metrics": {name: values.tolist() for name, values in curves.items()},
            "gpu_verification": "pending",
        }
    else:
        module = str(member.metadata.get("module", ""))
        factor = str(member.metadata.get("factor", ""))
        if module == "C":
            response, endpoint = np.asarray([0.0, -0.5, -1.0]), 1.0
        elif module == "F" or (
            member.phase == "c0_evaluate" and factor != member.metadata.get("task")
        ):
            response, endpoint = np.asarray([0.0, 0.4, 0.2]), 0.0
        else:
            response, endpoint = np.asarray([0.0, 0.5, 1.0]), 1.0
        scores = decompose_score_trajectory((0.0, 0.5, 1.0), response, endpoint=endpoint)
        payload = {
            "schema_version": 1,
            "kind": "cpu_score_oracle",
            "model_id": member.metadata["model_id"],
            "family": member.metadata["family"],
            "module": module,
            "factor": factor,
            "metrics": {
                name: float(np.asarray(scores[name])) for name in ("M", "E", "C", "F", "Abs", "Net")
            },
            "numeric_audit": scores["numeric_audit"],
            "gpu_verification": "pending",
        }
    atomic_write_json(output, payload)
    return (output,)


__all__ = [
    "MaterializedMemberArtifact",
    "MaterializedMemberBundle",
    "MemberExecutor",
    "PlanMember",
    "build_members",
    "configuration_sha256",
    "execute_members",
    "materialized_member_executor",
    "member_contract_sha256",
    "plan_counts",
    "receipt_reusable",
    "resolve_materialized_output_root",
    "smoke_executor",
    "validate_materialized_member_bundle",
    "write_jobs_manifest",
]
