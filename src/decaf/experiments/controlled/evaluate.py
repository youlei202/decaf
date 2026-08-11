"""Controlled static member expansion and receipt-driven execution."""

from __future__ import annotations

import json
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
from decaf.experiments.controlled.models import expected_base_models
from decaf.experiments.controlled.protocols import (
    analytic_context_mixture,
    decompose_score_trajectory,
)
from decaf.experiments.controlled.train import (
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

    phase_order = {"c0_evaluate": 0, "c1_measure": 1, "c2_train": 2, "c2_evaluate": 3}
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
    for member in members:
        if not set(member.dependencies).issubset(known):
            raise ValueError(f"member {member.member_id} has an unknown dependency")
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


def write_jobs_manifest(path: str | Path, members: Sequence[PlanMember]) -> Path:
    """Atomically persist the sorted JSONL schedule contract."""

    destination = Path(path)
    text = "".join(
        json.dumps(member.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        for member in members
    )
    atomic_text(destination, text)
    return destination


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
    if member.phase == "c2_train":
        payload = {
            "schema_version": 1,
            "kind": "cpu_oracle_training_placeholder",
            "model_id": member.metadata["model_id"],
            "seed": member.seed,
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
    "MemberExecutor",
    "PlanMember",
    "build_members",
    "execute_members",
    "plan_counts",
    "receipt_reusable",
    "smoke_executor",
    "write_jobs_manifest",
]
