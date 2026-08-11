"""Training-plan construction for the controlled family.

C0 is intentionally absent: that benchmark was registered as a no-retraining
evaluation of 30 pre-existing Experiment-3 models.  C1/C2 training is exposed
as deterministic jobs and delegated to an accelerator backend by the CLI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decaf.experiments.controlled.models import expected_contradiction_models


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """One model-factory job with deterministic selected outputs."""

    member_id: str
    family: str
    model_id: str
    module: str
    task: str
    architecture: str
    seed: int
    outputs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "family": self.family,
            "model_id": self.model_id,
            "module": self.module,
            "task": self.task,
            "architecture": self.architecture,
            "seed": self.seed,
            "outputs": list(self.outputs),
        }


def _token_probability(value: float) -> str:
    return f"p{int(round(float(value) * 100)):03d}"


def selected_c1_checkpoints(section: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expand the exact 52/18/18 C1 selected-checkpoint registry."""

    selected: list[dict[str, Any]] = []
    evidence = section["evidence_selection"]
    for architecture in sorted(evidence):
        by_correlation = evidence[architecture]
        for raw_probability in sorted(by_correlation, key=float):
            probability = float(raw_probability)
            by_seed = by_correlation[raw_probability]
            for raw_seed in sorted(by_seed, key=int):
                seed = int(raw_seed)
                trajectory_id = f"e__{_token_probability(probability)}__{architecture}__seed_{seed}"
                for epoch in sorted(map(int, by_seed[raw_seed])):
                    selected.append(
                        {
                            "model_id": f"{trajectory_id}__epoch_{epoch:03d}",
                            "module": "E",
                            "variant": "trajectory",
                            "task": "object_shape",
                            "architecture": architecture,
                            "seed": seed,
                            "p_train": probability,
                            "epoch": epoch,
                            "trajectory_id": trajectory_id,
                            "factors": ("object_shape", "wall_color"),
                        }
                    )

    architectures = tuple(map(str, section["architectures"]))
    modules = (
        (
            "C",
            "object_color",
            tuple(map(str, section["causal_variants"])),
            tuple(map(int, section["causal_seeds"])),
            ("object_color",),
        ),
        (
            "F",
            "object_shape",
            tuple(map(str, section["fragility_variants"])),
            tuple(map(int, section["fragility_seeds"])),
            ("floor_color", "object_shape"),
        ),
    )
    for module, task, variants, seeds, factors in modules:
        for variant in variants:
            for architecture in architectures:
                for seed in seeds:
                    prefix = module.lower()
                    selected.append(
                        {
                            "model_id": f"{prefix}__{variant}__{architecture}__seed_{seed}",
                            "module": module,
                            "variant": variant,
                            "task": task,
                            "architecture": architecture,
                            "seed": seed,
                            "epoch": int(section.get("terminal_epoch", 20)),
                            "trajectory_id": "",
                            "factors": factors,
                        }
                    )
    selected.sort(key=lambda row: str(row["model_id"]))
    counts: dict[str, int] = {}
    for row in selected:
        module = str(row["module"])
        counts[module] = counts.get(module, 0) + 1
    expected = {str(key): int(value) for key, value in section["expected_counts"].items()}
    if counts != expected or len({str(row["model_id"]) for row in selected}) != len(selected):
        raise ValueError(f"C1 checkpoint expansion changed: expected {expected}, found {counts}")
    return tuple(selected)


def c1_factory_training_jobs(section: Mapping[str, Any]) -> tuple[TrainingJob, ...]:
    """Build factory jobs whose outputs cover every selected C1 checkpoint.

    Evidence checkpoints are snapshots from a smaller set of training
    trajectories, whereas the causal and fragility checkpoints each have their
    own factory job.  The paper profile therefore expands to 44 jobs producing
    the registered 88 checkpoints; smoke profiles use the same rule at a much
    smaller cardinality.
    """

    selected = selected_c1_checkpoints(section)
    jobs: list[TrainingJob] = []
    trajectories: dict[str, list[str]] = {}
    for row in selected:
        if row["module"] == "E":
            trajectories.setdefault(str(row["trajectory_id"]), []).append(str(row["model_id"]))
            continue
        identifier = str(row["model_id"])
        jobs.append(
            TrainingJob(
                member_id=f"c1_train__{identifier}",
                family="C1",
                model_id=identifier,
                module=str(row["module"]),
                task=str(row["task"]),
                architecture=str(row["architecture"]),
                seed=int(row["seed"]),
                outputs=(f"checkpoints/c1/{identifier}.pt",),
            )
        )
    by_id = {str(row["trajectory_id"]): row for row in selected if row["module"] == "E"}
    for trajectory_id, checkpoints in sorted(trajectories.items()):
        row = by_id[trajectory_id]
        jobs.append(
            TrainingJob(
                member_id=f"c1_train__{trajectory_id}",
                family="C1",
                model_id=trajectory_id,
                module="E",
                task="object_shape",
                architecture=str(row["architecture"]),
                seed=int(row["seed"]),
                outputs=tuple(
                    f"checkpoints/c1/{checkpoint}.pt" for checkpoint in sorted(checkpoints)
                ),
            )
        )
    jobs.sort(key=lambda job: job.member_id)
    expected_jobs = section.get("expected_training_jobs")
    if expected_jobs is not None and len(jobs) != int(expected_jobs):
        raise AssertionError(
            "C1 factory plan changed: "
            f"expected {int(expected_jobs)} training jobs, found {len(jobs)}"
        )
    expected_outputs = {f"checkpoints/c1/{row['model_id']}.pt" for row in selected}
    actual_outputs = {output for job in jobs for output in job.outputs}
    if actual_outputs != expected_outputs or sum(len(job.outputs) for job in jobs) != len(
        expected_outputs
    ):
        raise AssertionError("C1 factory jobs do not uniquely cover selected checkpoints")
    return tuple(jobs)


def c1_checkpoint_producers(section: Mapping[str, Any]) -> dict[str, str]:
    """Map each selected C1 checkpoint ID to its unique factory member."""

    producers: dict[str, str] = {}
    for job in c1_factory_training_jobs(section):
        for output in job.outputs:
            checkpoint_id = Path(output).stem
            if checkpoint_id in producers:
                raise AssertionError(f"duplicate C1 checkpoint producer: {checkpoint_id}")
            producers[checkpoint_id] = job.member_id
    expected = {str(row["model_id"]) for row in selected_c1_checkpoints(section)}
    if set(producers) != expected:
        raise AssertionError("C1 checkpoint-producer mapping is incomplete")
    return producers


def c2_training_jobs(section: Mapping[str, Any]) -> tuple[TrainingJob, ...]:
    """Build the registered 30-model context-swap training grid."""

    records = expected_contradiction_models(
        tuple(map(str, section["tasks"])),
        tuple(map(str, section["architectures"])),
        tuple(map(int, section["seeds"])),
    )
    jobs = tuple(
        TrainingJob(
            member_id=f"c2_train__{record.model_id}",
            family="C2",
            model_id=record.model_id,
            module="context_swap",
            task=record.task,
            architecture=record.architecture,
            seed=record.seed,
            outputs=(f"checkpoints/c2/{record.model_id}.pt",),
        )
        for record in records
    )
    expected = int(section.get("expected_models", len(jobs)))
    if len(jobs) != expected:
        raise ValueError(f"C2 training grid must contain {expected} models")
    return jobs


def assert_c0_has_no_training_jobs(jobs: Sequence[TrainingJob]) -> None:
    """Fail if a future refactor accidentally schedules C0 retraining."""

    if any(job.family == "C0" for job in jobs):
        raise AssertionError("C0 is a sealed no-retraining experiment")


__all__ = [
    "TrainingJob",
    "assert_c0_has_no_training_jobs",
    "c1_checkpoint_producers",
    "c1_factory_training_jobs",
    "c2_training_jobs",
    "selected_c1_checkpoints",
]
