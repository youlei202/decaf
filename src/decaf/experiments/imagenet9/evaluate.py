"""Static planning and score-oracle evaluation for the ImageNet-9 protocols.

The CPU path deliberately evaluates model responses rather than loading restricted
images or claiming GPU inference.  A paper run may ingest sealed response paths
produced by the jobs in :func:`build_formal_plan`; the smoke profile creates a tiny,
deterministic score oracle that exercises the same DECAF implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.experiments.common import RunContext, atomic_json, atomic_text
from decaf.experiments.imagenet9.baselines import baseline_plan, method_model_compatible
from decaf.experiments.imagenet9.data import (
    resolve_dataset_root,
    smoke_pair_frame,
    validate_split_fingerprints,
)
from decaf.experiments.imagenet9.models import deep_model_registry, model_registry
from decaf.experiments.imagenet9.pairs import load_pair_manifest
from decaf.experiments.imagenet9.reveal import decompose_score_path
from decaf.paper.reference import sha256_file

RESPONSE_COLUMNS = {
    "pair_id",
    "model_id",
    "reveal_path",
    "stage_index",
    "alpha",
    "response",
}
BASELINE_COLUMNS = {"pair_id", "pair_type", "model_id", "method_id", "score"}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jobs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = model_registry(config)
    deep_records = deep_model_registry(config, records)
    methods = baseline_plan(list(map(str, config["baselines"]["methods"])))
    paths = list(map(str, config["experiment_grid"]["reveal_paths"]))
    shard_size = int(config["data"]["shard_size"])
    score_shards = math.ceil(int(config["data"]["score_pairs"]) / shard_size)
    deep_shards = math.ceil(int(config["data"]["deep_pairs"]) / shard_size)
    config_sha256 = _canonical_sha256(config)
    jobs: list[dict[str, Any]] = []
    for record in records:
        if record.source != "experiment":
            continue
        job_id = f"train__{record.model_id}"
        jobs.append(
            {
                "job_id": job_id,
                "kind": "finetune",
                "model_id": record.model_id,
                "training_regime": record.training_regime,
                "seed": record.seed,
                "checkpoint_key": record.checkpoint_key,
                "config_sha256": config_sha256,
                "dataset_split_sha256": str(config["data"]["paired_manifest_sha256"]),
                "output": f"artifacts/checkpoints/{record.checkpoint_key}",
                "receipt": f"receipts/members/{job_id}.json",
                "depends_on": [],
                "dependency_outputs": [],
            }
        )
    for record in records:
        dependency = f"train__{record.model_id}" if record.source == "experiment" else None
        for reveal_path in paths:
            for shard in range(score_shards):
                job_id = f"scan__{record.model_id}__{reveal_path}__shard_{shard:03d}"
                jobs.append(
                    {
                        "job_id": job_id,
                        "kind": "decaf_scan",
                        "model_id": record.model_id,
                        "reveal_path": reveal_path,
                        "shard": shard,
                        "shard_unit": "source_pair_rows",
                        "source_row_start": shard * shard_size,
                        "source_row_stop": min(
                            (shard + 1) * shard_size,
                            int(config["data"]["score_pairs"]),
                        ),
                        "checkpoint_key": record.checkpoint_key,
                        "config_sha256": config_sha256,
                        "dataset_split_sha256": str(config["data"]["score_split_sha256"]),
                        "depends_on": [dependency] if dependency else [],
                        "dependency_outputs": (
                            [f"artifacts/checkpoints/{record.checkpoint_key}"] if dependency else []
                        ),
                        "output": f"raw/scans/{job_id}.parquet",
                        "receipt": f"receipts/members/{job_id}.json",
                    }
                )
    for record in deep_records:
        dependency = f"train__{record.model_id}" if record.source == "experiment" else None
        for method in methods:
            for shard in range(deep_shards):
                job_id = f"baseline__{record.model_id}__{method['method_id']}__shard_{shard:03d}"
                jobs.append(
                    {
                        "job_id": job_id,
                        "kind": "saliency_baseline",
                        "model_id": record.model_id,
                        "method_id": method["method_id"],
                        "shard": shard,
                        "shard_unit": "source_pair_rows",
                        "source_row_start": shard * shard_size,
                        "source_row_stop": min(
                            (shard + 1) * shard_size,
                            int(config["data"]["deep_pairs"]),
                        ),
                        "checkpoint_key": record.checkpoint_key,
                        "config_sha256": config_sha256,
                        "dataset_split_sha256": str(config["data"]["deep_split_sha256"]),
                        "depends_on": [dependency] if dependency else [],
                        "dependency_outputs": (
                            [f"artifacts/checkpoints/{record.checkpoint_key}"] if dependency else []
                        ),
                        "output": f"raw/baselines/{job_id}.parquet",
                        "receipt": f"receipts/members/{job_id}.json",
                    }
                )
    return jobs


def build_formal_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build and validate the complete, non-executing GPU experiment plan."""

    records = model_registry(config)
    deep_records = deep_model_registry(config, records)
    jobs = _jobs(config)
    outputs = [str(job["output"]) for job in jobs]
    receipts = [str(job["receipt"]) for job in jobs]
    off_the_shelf = sum(record.source != "experiment" for record in records)
    fine_tuned = sum(record.source == "experiment" for record in records)
    seeds = sorted({record.seed for record in records if record.seed is not None})
    shard_size = int(config["data"]["shard_size"])
    score_shards = math.ceil(int(config["data"]["score_pairs"]) / shard_size)
    deep_shards = math.ceil(int(config["data"]["deep_pairs"]) / shard_size)
    methods = baseline_plan(list(map(str, config["baselines"]["methods"])))
    training_jobs = [job for job in jobs if job["kind"] == "finetune"]
    scan_jobs = [job for job in jobs if job["kind"] == "decaf_scan"]
    baseline_jobs = [job for job in jobs if job["kind"] == "saliency_baseline"]
    expected_score_shards = set(range(score_shards))
    expected_deep_shards = set(range(deep_shards))
    scan_shard_coverage = all(
        {
            int(job["shard"])
            for job in scan_jobs
            if job["model_id"] == record.model_id and job["reveal_path"] == reveal_path
        }
        == expected_score_shards
        for record in records
        for reveal_path in config["experiment_grid"]["reveal_paths"]
    )
    baseline_shard_coverage = all(
        {
            int(job["shard"])
            for job in baseline_jobs
            if job["model_id"] == record.model_id and job["method_id"] == method["method_id"]
        }
        == expected_deep_shards
        for record in deep_records
        for method in methods
    )
    fine_tuned_records = [record for record in records if record.source == "experiment"]
    training_by_model = {str(job["model_id"]): job for job in training_jobs}
    training_by_id = {str(job["job_id"]): job for job in training_jobs}
    checkpoint_coverage = (
        len(training_jobs) == len(fine_tuned_records)
        and all(
            record.model_id in training_by_model
            and training_by_model[record.model_id]["checkpoint_key"] == record.checkpoint_key
            for record in fine_tuned_records
        )
        and len({record.checkpoint_key for record in records}) == len(records)
    )
    dependency_artifact_coverage = all(
        len(job.get("depends_on", ())) == len(job.get("dependency_outputs", ()))
        and all(
            dependency in training_by_id and training_by_id[dependency]["output"] == output
            for dependency, output in zip(
                job.get("depends_on", ()),
                job.get("dependency_outputs", ()),
                strict=True,
            )
        )
        for job in jobs
    )
    compatibility = [
        {
            "model_id": record.model_id,
            "method_id": str(method["method_id"]),
            "compatible": method_model_compatible(str(method["method_id"]), gradient_access=True),
        }
        for record in deep_records
        for method in methods
    ]
    hashes = {
        "paired_manifest_sha256": str(config["data"]["paired_manifest_sha256"]),
        "score_split_sha256": str(config["data"]["score_split_sha256"]),
        "deep_split_sha256": str(config["data"]["deep_split_sha256"]),
    }
    hashes_registered = all(
        value == "smoke-fixture" or len(value) == 64 for value in hashes.values()
    )
    profile = str(config.get("profile", "custom"))
    paper_expected = profile == "paper"
    assertions = {
        "expected_model_count": len(records) == (72 if paper_expected else 1),
        "expected_off_the_shelf_count": off_the_shelf == (24 if paper_expected else 1),
        "expected_fine_tuned_count": fine_tuned == (48 if paper_expected else 0),
        "expected_deep_model_count": len(deep_records) == (32 if paper_expected else 1),
        "expected_deep_pair_count": int(config["data"]["deep_pairs"])
        == (768 if paper_expected else 4),
        "expected_seed_count": len(seeds) == (2 if paper_expected else 0),
        "expected_shard_count": score_shards >= 1 and deep_shards >= 1,
        "exact_scan_shard_coverage": scan_shard_coverage,
        "exact_baseline_shard_coverage": baseline_shard_coverage,
        "method_model_compatibility": bool(compatibility)
        and all(item["compatible"] for item in compatibility),
        "checkpoint_coverage": checkpoint_coverage,
        "dependency_artifact_coverage": dependency_artifact_coverage,
        "dataset_split_hashes": hashes_registered,
        "unique_output_paths": len(outputs) == len(set(outputs)),
        "unique_receipt_paths": len(receipts) == len(set(receipts)),
        "historical_protocol_registered": not paper_expected
        or all(key in config for key in ("training", "corruptions", "statistics"))
        and list(config["experiment_grid"].get("epsilon_sensitivity", ())) == [0.01, 0.02, 0.05],
    }
    if not all(assertions.values()):
        failed = sorted(name for name, passed in assertions.items() if not passed)
        raise ValueError(f"invalid ImageNet-9 formal plan: {failed}")
    return {
        "schema_version": 1,
        "experiment": "imagenet9",
        "profile": profile,
        "execution_class": "gpu-required-for-model-inference",
        "gpu_execution_verified": False,
        "counts": {
            "models": len(records),
            "off_the_shelf_models": off_the_shelf,
            "fine_tuned_models": fine_tuned,
            "deep_benchmark_models": len(deep_records),
            "deep_pairs": int(config["data"]["deep_pairs"]),
            "score_pairs": int(config["data"]["score_pairs"]),
            "expanded_deep_pairs": int(config["data"]["deep_pairs"])
            * len(config["experiment_grid"]["pair_types"]),
            "expanded_score_pairs": int(config["data"]["score_pairs"])
            * len(config["experiment_grid"]["pair_types"]),
            "seeds": len(seeds),
            "score_shards": score_shards,
            "deep_shards": deep_shards,
            "methods": len(methods),
            "training_jobs": len(training_jobs),
            "scan_jobs": len(scan_jobs),
            "baseline_jobs": len(baseline_jobs),
            "jobs": len(jobs),
        },
        "dataset_split_hashes": hashes,
        "checkpoint_registry": [record.as_dict() for record in records],
        "method_registry": methods,
        "method_model_compatibility": compatibility,
        "worker_contract": {
            "status": "external_gpu_worker_required",
            "execution_verified": False,
            "job_source": "manifests/jobs.jsonl",
            "response_schema": sorted(RESPONSE_COLUMNS),
            "baseline_schema": sorted(BASELINE_COLUMNS),
            "receipt_required_fields": [
                "schema_version",
                "job_id",
                "job_sha256",
                "status",
                "output",
                "output_sha256",
                "row_count",
                "dependency_artifacts",
            ],
            "note": (
                "Static planning and CPU replay are verified here; CUDA training, "
                "scan, and saliency jobs remain pending by design."
            ),
        },
        "jobs": jobs,
        "assertions": assertions,
    }


def validate_response_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate long-form score trajectories before scientific evaluation."""

    missing = sorted(RESPONSE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"response paths are missing columns: {missing}")
    normalized = frame.copy()
    for column in ("pair_id", "model_id", "reveal_path"):
        normalized[column] = normalized[column].astype(str)
    normalized["stage_index"] = normalized["stage_index"].astype(int)
    normalized["alpha"] = normalized["alpha"].astype(float)
    normalized["response"] = normalized["response"].astype(float)
    key = ["pair_id", "model_id", "reveal_path", "stage_index"]
    if normalized.duplicated(key).any():
        raise ValueError("response paths contain duplicate trajectory stages")
    for _, group in normalized.groupby(key[:-1], sort=False):
        ordered = group.sort_values("stage_index", kind="stable")
        alpha = ordered["alpha"].to_numpy(dtype=np.float64)
        if len(alpha) < 2 or alpha[0] != 0.0 or alpha[-1] != 1.0 or np.any(np.diff(alpha) <= 0):
            raise ValueError("each response path must strictly span alpha zero to one")
    return normalized.sort_values(key, kind="stable").reset_index(drop=True)


def evaluate_response_frame(frame: pd.DataFrame, *, epsilon: float) -> pd.DataFrame:
    """Compute DECAF scores for each model/pair/reveal trajectory."""

    validated = validate_response_frame(frame)
    rows: list[dict[str, Any]] = []
    group_columns = ["pair_id", "model_id", "reveal_path"]
    for identifiers, group in validated.groupby(group_columns, sort=True):
        ordered = group.sort_values("stage_index", kind="stable")
        scores = decompose_score_path(
            ordered["alpha"].to_numpy(dtype=np.float64),
            ordered["response"].to_numpy(dtype=np.float64),
            epsilon=epsilon,
        )
        if not bool(scores["numeric_audit"]["passed"]):
            raise RuntimeError(f"DECAF conservation failed for trajectory {identifiers}")
        row: dict[str, Any] = dict(zip(group_columns, identifiers, strict=True))
        for name in ("M", "E", "C", "F", "Abs", "Net", "endpoint_delta"):
            row[name] = float(np.asarray(scores[name]))
        row["endpoint_active"] = bool(np.asarray(scores["endpoint_active"]))
        row["predicted_component"] = max(("E", "C", "F"), key=lambda name: row[name])
        for optional in ("pair_type", "expected_component"):
            if optional in ordered:
                values = ordered[optional].dropna().astype(str).unique()
                if len(values) == 1:
                    row[optional] = values[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns, kind="stable").reset_index(drop=True)


def _smoke_responses(config: Mapping[str, Any]) -> pd.DataFrame:
    pairs = smoke_pair_frame()
    model_id = model_registry(config)[0].model_id
    alpha = list(map(float, config["experiment_grid"]["alpha"]))
    templates = (
        ("E", [0.0, 0.4, 0.8]),
        ("C", [0.0, 0.35, -0.6]),
        ("F", [0.0, 0.4, 0.01]),
        ("E", [0.0, -0.2, 0.5]),
    )
    if len(alpha) != 3:
        raise ValueError("the smoke score oracle expects exactly three alpha stages")
    rows: list[dict[str, Any]] = []
    for pair, (expected, template) in zip(pairs.to_dict("records"), templates, strict=True):
        for reveal_path in map(str, config["experiment_grid"]["reveal_paths"]):
            response = np.asarray(template, dtype=np.float64)
            if reveal_path.startswith("patch"):
                response = response.copy()
                response[1] *= 1.25
            for stage_index, (position, value) in enumerate(zip(alpha, response, strict=True)):
                rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "pair_type": pair["pair_type"],
                        "model_id": model_id,
                        "reveal_path": reveal_path,
                        "expected_component": expected,
                        "stage_index": stage_index,
                        "alpha": position,
                        "response": float(value),
                    }
                )
    return pd.DataFrame(rows)


def prepare(context: RunContext) -> dict[str, Any]:
    """Materialize public manifests without copying any restricted imagery."""

    records = model_registry(context.config)
    if context.profile == "smoke":
        pairs = smoke_pair_frame()
        atomic_text(context.path / "manifests" / "pairs.csv", pairs.to_csv(index=False))
        data_manifest = {
            "schema_version": 1,
            "status": "synthetic_score_oracle",
            "restricted_images_copied": False,
            "pair_count": len(pairs),
        }
        checkpoints = {
            "schema_version": 1,
            "status": "synthetic_score_oracle",
            "items": [{"model_id": "smoke_score_oracle", "requires_checkpoint": False}],
        }
    else:
        root = resolve_dataset_root(context.config)
        fingerprints = validate_split_fingerprints(root, context.config)
        score_path = root / "manifests" / "score_split.parquet"
        deep_path = root / "manifests" / "deep_split.parquet"
        score_pairs = load_pair_manifest(
            score_path,
            expected_rows=int(context.config["data"]["score_pairs"]),
            dataset_root=root,
        )
        deep_pairs = load_pair_manifest(
            deep_path,
            expected_rows=int(context.config["data"]["deep_pairs"]),
            dataset_root=root,
        )
        atomic_text(
            context.path / "manifests" / "score_pairs.csv",
            score_pairs.to_csv(index=False),
        )
        atomic_text(
            context.path / "manifests" / "deep_pairs.csv",
            deep_pairs.to_csv(index=False),
        )
        data_manifest = {
            "schema_version": 1,
            "status": "verified_external",
            "dataset_root_environment": context.config["data"]["root_environment"],
            "score_manifest": str(score_path.relative_to(root)),
            "deep_manifest": str(deep_path.relative_to(root)),
            "score_source_rows": int(context.config["data"]["score_pairs"]),
            "deep_source_rows": int(context.config["data"]["deep_pairs"]),
            "score_expanded_pairs": len(score_pairs),
            "deep_expanded_pairs": len(deep_pairs),
            "shard_unit": "source_pair_rows",
            "fingerprints": fingerprints,
            "restricted_images_copied": False,
        }
        checkpoint_variable = str(context.config["checkpoints"]["root_environment"])
        checkpoint_root = os.environ.get(checkpoint_variable)
        checkpoint_items = []
        for record in records:
            item = record.as_dict()
            if record.source == "experiment" and checkpoint_root:
                checkpoint = Path(checkpoint_root) / record.checkpoint_key
                item["exists"] = checkpoint.is_file()
                item["sha256"] = sha256_file(checkpoint) if checkpoint.is_file() else None
            else:
                item["exists"] = None
                item["sha256"] = None
            checkpoint_items.append(item)
        checkpoints = {
            "schema_version": 1,
            "root_environment": checkpoint_variable,
            "root_configured": bool(checkpoint_root),
            "items": checkpoint_items,
        }
    atomic_json(context.path / "manifests" / "data.json", data_manifest)
    atomic_json(context.path / "manifests" / "checkpoints.json", checkpoints)
    plan = build_formal_plan(context.config)
    atomic_json(context.path / "manifests" / "plan.json", plan)
    for job in plan["jobs"]:
        context.append_job(job)
    return {"models": len(records), "planned_jobs": plan["counts"]["jobs"]}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.part{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _dependency_artifacts(context: RunContext, job: Mapping[str, Any]) -> list[dict[str, str]]:
    dependencies = list(map(str, job.get("depends_on", ())))
    outputs = list(map(str, job.get("dependency_outputs", ())))
    if len(dependencies) != len(outputs):
        raise ValueError(f"dependency output mapping is incomplete: {job['job_id']}")
    artifacts: list[dict[str, str]] = []
    for dependency, relative in zip(dependencies, outputs, strict=True):
        path = context.path / relative
        if not path.is_file():
            raise FileNotFoundError(f"dependency artifact is missing: {job['job_id']}:{relative}")
        artifacts.append(
            {
                "job_id": dependency,
                "output": relative,
                "output_sha256": sha256_file(path),
            }
        )
    return artifacts


def _write_member_receipt(
    context: RunContext,
    job: Mapping[str, Any],
    output: Path,
    row_count: int,
) -> None:
    atomic_json(
        context.path / str(job["receipt"]),
        {
            "schema_version": 1,
            "job_id": str(job["job_id"]),
            "job_sha256": _canonical_sha256(job),
            "status": "completed",
            "output": str(job["output"]),
            "output_sha256": sha256_file(output),
            "row_count": row_count,
            "dependency_artifacts": _dependency_artifacts(context, job),
            "gpu_inference_executed": False,
        },
    )


def _validate_member_receipt(
    context: RunContext,
    job: Mapping[str, Any],
    output: Path,
    row_count: int,
) -> None:
    receipt_path = context.path / str(job["receipt"])
    if not receipt_path.is_file():
        raise FileNotFoundError(f"GPU worker receipt is missing: {job['receipt']}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "job_id": str(job["job_id"]),
        "job_sha256": _canonical_sha256(job),
        "status": "completed",
        "output": str(job["output"]),
        "output_sha256": sha256_file(output),
        "row_count": row_count,
        "dependency_artifacts": _dependency_artifacts(context, job),
    }
    failures = [key for key, value in expected.items() if receipt.get(key) != value]
    if failures:
        raise ValueError(f"GPU worker receipt differs from its planned output: {failures}")


def _materialize_smoke_members(
    context: RunContext,
    plan: Mapping[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    pairs = smoke_pair_frame()
    baseline_frames: list[pd.DataFrame] = []
    for job in plan["jobs"]:
        if job["kind"] == "decaf_scan":
            member = frame[
                (frame["model_id"] == job["model_id"])
                & (frame["reveal_path"] == job["reveal_path"])
            ].reset_index(drop=True)
        elif job["kind"] == "saliency_baseline":
            member = pd.DataFrame(
                {
                    "pair_id": pairs["pair_id"].astype(str),
                    "pair_type": pairs["pair_type"].astype(str),
                    "model_id": str(job["model_id"]),
                    "method_id": str(job["method_id"]),
                    "score": np.linspace(0.0, 1.0, len(pairs), dtype=np.float64),
                }
            )
            baseline_frames.append(member)
        else:
            raise RuntimeError(f"unexpected smoke job kind: {job['kind']}")
        output = context.path / str(job["output"])
        _atomic_parquet(output, member)
        _write_member_receipt(context, job, output, len(member))
    return pd.concat(baseline_frames, ignore_index=True)


def _expected_member_pairs(
    context: RunContext,
    manifest_name: str,
    job: Mapping[str, Any],
) -> pd.DataFrame:
    manifest_path = context.path / "manifests" / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared pair manifest is missing: manifests/{manifest_name}")
    pairs = pd.read_csv(manifest_path)
    required = {"pair_id", "pair_type", "source_pair_id", "source_row_index"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"prepared pair manifest is missing columns: {missing}")
    start = int(job["source_row_start"])
    stop = int(job["source_row_stop"])
    selected = pairs[
        (pairs["source_row_index"].astype(int) >= start)
        & (pairs["source_row_index"].astype(int) < stop)
    ].copy()
    if selected["source_pair_id"].nunique() != stop - start:
        raise ValueError(f"planned source-row shard is incomplete: {job['job_id']}")
    return selected


def _load_materialized_members(
    context: RunContext,
    plan: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    response_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    completed_jobs: set[str] = set()
    for job in plan["jobs"]:
        dependencies = set(map(str, job.get("depends_on", ())))
        if not dependencies <= completed_jobs:
            raise ValueError(f"planned dependencies are incomplete: {job['job_id']}")
        output = context.path / str(job["output"])
        if not output.is_file():
            raise FileNotFoundError(
                f"planned GPU output is missing: {job['output']}; "
                "execute the external worker contract first"
            )
        if job["kind"] == "finetune":
            _validate_member_receipt(context, job, output, 1)
            completed_jobs.add(str(job["job_id"]))
            continue
        member = pd.read_parquet(output)
        if job["kind"] == "decaf_scan":
            member = validate_response_frame(member)
            if "pair_type" not in member:
                raise ValueError(f"formal scan output has no pair_type: {job['job_id']}")
            if set(member["model_id"]) != {job["model_id"]}:
                raise ValueError(f"scan output has the wrong model: {job['job_id']}")
            if set(member["reveal_path"]) != {job["reveal_path"]}:
                raise ValueError(f"scan output has the wrong reveal path: {job['job_id']}")
            expected = _expected_member_pairs(context, "score_pairs.csv", job)
            expected_ids = set(expected["pair_id"].astype(str))
            if set(member["pair_id"].astype(str)) != expected_ids:
                raise ValueError(f"scan output has the wrong pair support: {job['job_id']}")
            expected_types = expected.set_index("pair_id")["pair_type"].astype(str).to_dict()
            observed_types = member.groupby("pair_id")["pair_type"].first().astype(str).to_dict()
            if observed_types != expected_types:
                raise ValueError(f"scan output has the wrong pair labels: {job['job_id']}")
            configured_alpha = np.asarray(
                context.config["experiment_grid"]["alpha"], dtype=np.float64
            )
            expected_rows = len(expected_ids) * len(configured_alpha)
            if len(member) != expected_rows:
                raise ValueError(f"scan output has the wrong row count: {job['job_id']}")
            configured_stages = np.arange(len(configured_alpha), dtype=np.int64)
            for pair_id, trajectory in member.groupby("pair_id", sort=False):
                ordered = trajectory.sort_values("stage_index", kind="stable")
                observed_stages = ordered["stage_index"].to_numpy(dtype=np.int64)
                observed_alpha = ordered["alpha"].to_numpy(dtype=np.float64)
                if not np.array_equal(observed_stages, configured_stages):
                    raise ValueError(
                        "scan output differs from the configured stage grid: "
                        f"{job['job_id']}:{pair_id}"
                    )
                if not np.array_equal(observed_alpha, configured_alpha):
                    raise ValueError(
                        "scan output differs from the configured alpha grid: "
                        f"{job['job_id']}:{pair_id}"
                    )
            _validate_member_receipt(context, job, output, expected_rows)
            response_frames.append(member)
        else:
            missing = sorted(BASELINE_COLUMNS - set(member.columns))
            if missing:
                raise ValueError(f"baseline output is missing columns: {missing}")
            if set(member["model_id"].astype(str)) != {job["model_id"]}:
                raise ValueError(f"baseline output has the wrong model: {job['job_id']}")
            if set(member["method_id"].astype(str)) != {job["method_id"]}:
                raise ValueError(f"baseline output has the wrong method: {job['job_id']}")
            expected = _expected_member_pairs(context, "deep_pairs.csv", job)
            expected_ids = set(expected["pair_id"].astype(str))
            if set(member["pair_id"].astype(str)) != expected_ids:
                raise ValueError(f"baseline output has the wrong pair support: {job['job_id']}")
            if len(member) != len(expected_ids):
                raise ValueError(f"baseline output has the wrong row count: {job['job_id']}")
            expected_types = expected.set_index("pair_id")["pair_type"].astype(str).to_dict()
            observed_types = member.set_index("pair_id")["pair_type"].astype(str).to_dict()
            if observed_types != expected_types:
                raise ValueError(f"baseline output has the wrong pair labels: {job['job_id']}")
            _validate_member_receipt(context, job, output, len(expected_ids))
            baseline_frames.append(member)
        completed_jobs.add(str(job["job_id"]))
    if not response_frames or not baseline_frames:
        raise RuntimeError("the planned scan and saliency outputs must both be materialized")
    return (
        pd.concat(response_frames, ignore_index=True),
        pd.concat(baseline_frames, ignore_index=True),
    )


def compute(context: RunContext) -> dict[str, Any]:
    """Evaluate a smoke oracle or sealed GPU-produced response trajectories."""

    plan = build_formal_plan(context.config)
    if context.profile == "smoke":
        frame = _smoke_responses(context.config)
        baseline_scores = _materialize_smoke_members(context, plan, frame)
    else:
        frame, baseline_scores = _load_materialized_members(context, plan)
    frame = validate_response_frame(frame)
    atomic_text(context.path / "raw" / "response_paths.csv", frame.to_csv(index=False))
    scores = evaluate_response_frame(
        frame,
        epsilon=float(context.config["experiment_grid"].get("epsilon", 0.02)),
    )
    atomic_text(context.path / "metrics" / "decaf_scores.csv", scores.to_csv(index=False))
    atomic_text(
        context.path / "metrics" / "baseline_scores.csv",
        baseline_scores.to_csv(index=False),
    )
    return {
        "trajectory_count": len(scores),
        "baseline_rows": len(baseline_scores),
        "completed_member_receipts": len(plan["jobs"]),
        "source": "synthetic_score_oracle"
        if context.profile == "smoke"
        else "sealed_gpu_responses",
        "gpu_inference_executed": False,
    }


__all__ = [
    "RESPONSE_COLUMNS",
    "build_formal_plan",
    "compute",
    "evaluate_response_frame",
    "prepare",
    "validate_response_frame",
]
