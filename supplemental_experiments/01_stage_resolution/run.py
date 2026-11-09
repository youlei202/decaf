#!/usr/bin/env python3
"""Real ImageNet-9 inference and a separate, cache-only grid-resolution audit."""
# ruff: noqa: E402 -- standalone entrypoint adds the repository src directory.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from stage_analysis import nested_indices, sample_diagnostics, summarize

from decaf.experiments.imagenet9.pairs import WIDE_PAIR_COLUMNS, load_pair_manifest


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temp = Path(handle.name)
    os.replace(temp, path)


def configuration(path):
    source = Path(path).resolve()
    config = json.loads(source.read_text())
    for key in ("dataset_root", "pair_manifest"):
        config[key] = str((source.parent / config[key]).resolve())
    if config.get("mapping"):
        config["mapping"] = str((source.parent / config["mapping"]).resolve())
    defaults = {
        "stage_counts": [3, 5, 9, 17, 33, 65],
        "epsilon": 0.02,
        "paths": ["blend", "patch_A"],
        "pair_types": ["same_rand", "same_next"],
        "selection_seed": 20261109,
        "max_source_pairs": None,
        "split": None,
        "image_size": 224,
        "blur_sigma": 8.0,
        "patch_grid": [8, 8],
        "patch_seed": 7101,
        "pair_shard_size": 32,
        "pair_batch_size": 4,
        "query_batch_size": 32,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    if config["epsilon"] != 0.02:
        raise ValueError("this stage-resolution audit fixes epsilon=0.02")
    config["stage_counts"] = sorted(map(int, config["stage_counts"]))
    nested_indices(max(config["stage_counts"]), config["stage_counts"])
    if not config["paths"] or set(config["paths"]) - {"blend", "patch_A", "patch_B"}:
        raise ValueError("paths must be blend, patch_A, or patch_B")
    if len(set(config["paths"])) != len(config["paths"]):
        raise ValueError("paths must not contain duplicates")
    if not config["pair_types"] or set(config["pair_types"]) - {"same_rand", "same_next"}:
        raise ValueError("invalid pair_types")
    for key in ("pair_shard_size", "pair_batch_size", "query_batch_size", "image_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not config.get("models"):
        raise ValueError("at least one real local model checkpoint is required")
    for model in config["models"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", model["model_id"]):
            raise ValueError("model_id must be a portable filename component")
        model["checkpoint"] = str((source.parent / model["checkpoint"]).resolve())
        model.setdefault("kind", "pretrained")
        model.setdefault("output_classes", 1000)
        if model["kind"] not in {"pretrained", "fine_tuned"}:
            raise ValueError("model kind must be pretrained or fine_tuned")
        if model["output_classes"] not in {9, 1000}:
            raise ValueError("output_classes must be 9 or 1000")
    ids = [m["model_id"] for m in config["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError("model IDs must be unique")
    return config


def selected_pairs(config):
    """Select by source, preserving both paired interventions and class balance."""
    path = Path(config["pair_manifest"])
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    is_wide = WIDE_PAIR_COLUMNS <= set(raw.columns)
    if config["split"] is not None:
        if "split" not in raw:
            raise ValueError("requested split is absent from pair manifest")
        wanted = set(raw.loc[raw.split == config["split"], "pair_id"].astype(str))
    else:
        wanted = None
    pairs = load_pair_manifest(path, dataset_root=Path(config["dataset_root"]))
    if "source_pair_id" not in pairs:
        pairs["source_pair_id"] = pairs.pair_id
    if wanted is not None:
        # Wide rows identify source foregrounds; typed rows already identify pairs.
        # A typed manifest may have a distinct source_pair_id and must be filtered
        # by its own pair_id rather than comparing unlike identifiers.
        split_key = "source_pair_id" if is_wide else "pair_id"
        pairs = pairs[pairs[split_key].isin(wanted)]
    pairs = pairs[pairs.pair_type.isin(config["pair_types"])].copy()
    if pairs.empty or not pairs.class_id.between(0, 8).all():
        raise ValueError("no legal ImageNet-9 pairs selected")
    limit = config["max_source_pairs"]
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("max_source_pairs must be a positive integer or null")
        sources = pairs[["source_pair_id", "class_id"]].drop_duplicates()
        if sources.source_pair_id.duplicated().any():
            raise ValueError("source pairs have inconsistent class IDs")
        sources["key"] = sources.source_pair_id.map(lambda x: digest([config["selection_seed"], x]))
        sources = sources.sort_values(["class_id", "key"])
        sources["class_depth"] = sources.groupby("class_id").cumcount()
        sources = sources.sort_values(["class_depth", "class_id"]).head(limit)
        pairs = pairs[pairs.source_pair_id.isin(sources.source_pair_id)]
    return pairs.sort_values("pair_id").reset_index(drop=True)


def build_plan(config):
    pairs = selected_pairs(config)
    root = Path(config["dataset_root"])
    image_files = sorted(set(pairs.original_path) | set(pairs.counterfactual_path))
    image_hashes = {}
    for relative in image_files:
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("image escapes dataset_root")
        image_hashes[relative] = file_hash(path)
    models = []
    for model in config["models"]:
        actual = file_hash(model["checkpoint"])
        if model.get("checkpoint_sha256") and actual != model["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {model['model_id']}")
        models.append({**model, "checkpoint_sha256": actual})
    mapping_hash = None
    if any(m["output_classes"] == 1000 for m in models):
        if not config.get("mapping"):
            raise ValueError("1000-way models require the official ImageNet-9 mapping JSON")
        mapping_hash = file_hash(config["mapping"])
    code_files = [
        HERE / "run.py",
        HERE / "stage_analysis.py",
        *[
            REPO / "src/decaf" / name
            for name in (
                "core/decomposition.py",
                "core/quadrature.py",
                "core/trajectories.py",
                "experiments/imagenet9/gpu_models.py",
                "experiments/imagenet9/gpu_methods.py",
                "experiments/imagenet9/pairs.py",
            )
        ],
    ]
    tasks = []
    for model in models:
        for path in config["paths"]:
            for start in range(0, len(pairs), config["pair_shard_size"]):
                stop = min(start + config["pair_shard_size"], len(pairs))
                task_id = f"{model['model_id']}__{path}__{start:06d}_{stop:06d}"
                tasks.append(
                    {
                        "task_id": task_id,
                        "model_id": model["model_id"],
                        "path": path,
                        "start": start,
                        "stop": stop,
                    }
                )
    plan = {
        "schema_version": 1,
        "experiment": "stage_resolution",
        "config": config,
        "models": models,
        "pairs": pairs.to_dict("records"),
        "tasks": tasks,
        "config_sha256": digest(config),
        "pair_manifest_sha256": file_hash(config["pair_manifest"]),
        "image_sha256": image_hashes,
        "data_sha256": digest(image_hashes),
        "mapping_sha256": mapping_hash,
        "code_sha256": {str(p.relative_to(REPO)): file_hash(p) for p in code_files},
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def prepare(config_path, output):
    plan = build_plan(configuration(config_path))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("output manifest differs: use a new output directory")
    else:
        atomic_json(manifest_path, plan)
    return plan


def verified_cache(output, plan, task):
    path = output / "cache" / f"{task['task_id']}.npz"
    receipt_path = path.with_suffix(".json")
    if not path.exists() or not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("plan_sha256") != plan["plan_sha256"] or receipt.get("task") != task:
        raise ValueError(f"cache receipt contract mismatch: {path}")
    if receipt.get("cache_sha256") != file_hash(path):
        raise ValueError(f"cache hash mismatch: {path}; remove damaged shard before resuming")
    return path


def canonical_state_dict(state, architecture):
    """Handle the legacy DenseNet key spelling used by official torchvision weights."""
    if not architecture.startswith("densenet"):
        return state
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.([12]\.(?:weight|bias|running_mean|running_var))$"
    )
    result = {}
    for key, value in state.items():
        match = pattern.match(key)
        key = match.group(1) + match.group(2) if match else key
        if key in result:
            raise ValueError("duplicate checkpoint key after legacy DenseNet normalization")
        result[key] = value
    return result


def model_loader(model, config, device):
    import torch
    import torchvision

    from decaf.experiments.imagenet9.gpu_models import (
        ModelAsset,
        load_model,
        load_official_mapping,
        probability_model,
    )

    if model["kind"] == "fine_tuned":
        asset = ModelAsset(
            model_id=model["model_id"],
            kind="fine_tuned",
            architecture=model["architecture"],
            architecture_family=model.get("architecture_family", model["architecture"]),
            output_classes=model["output_classes"],
            checkpoint=Path(model["checkpoint"]),
            checkpoint_sha256=model["checkpoint_sha256"],
            checkpoint_bytes=Path(model["checkpoint"]).stat().st_size,
        )
        base, _ = load_model(asset, device=device)
    else:
        constructor = getattr(torchvision.models, model["architecture"])
        base = constructor(weights=None, num_classes=model["output_classes"])
        payload = torch.load(model["checkpoint"], map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        state = canonical_state_dict(state, model["architecture"])
        base.load_state_dict(state, strict=True)
    mapping = (
        load_official_mapping(config["mapping"], torch) if model["output_classes"] == 1000 else None
    )
    return (
        probability_model(base, mapping_matrix=mapping, output_classes=model["output_classes"])
        .to(device=device, dtype=torch.float32)
        .eval()
    )


def infer(plan, output, *, device, worker_index=0, workers=1, model_id=None):
    import torch
    import torchvision

    from decaf.experiments.imagenet9.gpu_methods import reveal_sequence
    from decaf.experiments.imagenet9.gpu_models import preprocess_paths

    if workers < 1 or not 0 <= worker_index < workers:
        raise ValueError("require 0 <= worker-index < workers")
    config = plan["config"]
    if model_id and model_id not in {m["model_id"] for m in plan["models"]}:
        raise ValueError("unknown model-id")
    torch.manual_seed(config["selection_seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    grid = np.linspace(0, 1, max(config["stage_counts"]))
    all_pairs = pd.DataFrame(plan["pairs"])
    tasks = [
        t
        for i, t in enumerate(plan["tasks"])
        if i % workers == worker_index and (model_id is None or t["model_id"] == model_id)
    ]
    current_id, model = None, None
    for task in tasks:
        if verified_cache(output, plan, task):
            print(json.dumps({"status": "cached", "task": task["task_id"]}), flush=True)
            continue
        if current_id != task["model_id"]:
            del model
            model = model_loader(
                next(m for m in plan["models"] if m["model_id"] == task["model_id"]), config, device
            )
            current_id = task["model_id"]
        started = time.monotonic()
        pairs = all_pairs.iloc[task["start"] : task["stop"]]
        scores = [[], []]
        with torch.inference_mode():
            for start in range(0, len(pairs), config["pair_batch_size"]):
                batch = pairs.iloc[start : start + config["pair_batch_size"]]
                plus, minus = [
                    torch.from_numpy(
                        preprocess_paths(
                            batch[col].tolist(),
                            dataset_root=Path(config["dataset_root"]),
                            size=config["image_size"],
                        )
                    ).to(device)
                    for col in ("original_path", "counterfactual_path")
                ]
                branches = reveal_sequence(
                    plus,
                    minus,
                    pair_ids=batch.pair_id.tolist(),
                    path=task["path"],
                    alpha=grid,
                    blur_sigma=config["blur_sigma"],
                    patch_grid=tuple(config["patch_grid"]),
                    patch_seed=config["patch_seed"],
                )
                targets = torch.tensor(batch.class_id.to_numpy(), device=device).repeat(len(grid))
                for branch_id, branch in enumerate(branches):
                    flat = branch.flatten(0, 1)
                    pieces = []
                    for offset in range(0, len(flat), config["query_batch_size"]):
                        probabilities = model(flat[offset : offset + config["query_batch_size"]])
                        target = targets[offset : offset + config["query_batch_size"]]
                        pieces.append(
                            probabilities.gather(1, target[:, None]).squeeze(1).cpu().numpy()
                        )
                    scores[branch_id].append(
                        np.concatenate(pieces).reshape(len(grid), len(batch)).T
                    )
        q_plus, q_minus = [np.concatenate(branch, axis=0).astype(np.float64) for branch in scores]
        if not np.isfinite(q_plus).all() or not np.isfinite(q_minus).all():
            raise FloatingPointError("non-finite measured model scores")
        path = output / "cache" / f"{task['task_id']}.npz"
        atomic_npz(
            path,
            grid=grid,
            q_plus=q_plus,
            q_minus=q_minus,
            pair_ids=pairs.pair_id.to_numpy(dtype=str),
        )
        receipt = {
            "task": task,
            "plan_sha256": plan["plan_sha256"],
            "cache_sha256": file_hash(path),
            "elapsed_seconds": time.monotonic() - started,
            "execution": "real_model_inference",
            "device": device,
            "hostname": platform.node(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "python": platform.python_version(),
            "model_input_evaluations": 2 * len(pairs) * len(grid),
            "precision": "float32",
        }
        atomic_json(path.with_suffix(".json"), receipt)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "task": task["task_id"],
                    "seconds": receipt["elapsed_seconds"],
                }
            ),
            flush=True,
        )


def analyze(output):
    plan = json.loads((output / "manifest.json").read_text())
    config = plan["config"]
    frames, missing, receipts = [], [], []
    for task in plan["tasks"]:
        path = verified_cache(output, plan, task)
        if path is None:
            missing.append(task["task_id"])
            continue
        pairs = pd.DataFrame(plan["pairs"][task["start"] : task["stop"]])
        with np.load(path, allow_pickle=False) as data:
            expected = np.linspace(0, 1, max(config["stage_counts"]))
            if not np.array_equal(data["grid"], expected):
                raise ValueError("cache grid differs from manifest")
            if not np.array_equal(data["pair_ids"], pairs.pair_id.to_numpy(dtype=str)):
                raise ValueError("cache pair support differs from manifest")
            frame = sample_diagnostics(
                data["q_plus"] - data["q_minus"],
                data["pair_ids"],
                config["stage_counts"],
                epsilon=config["epsilon"],
            )
        frame["model_id"], frame["path"] = task["model_id"], task["path"]
        frame = frame.merge(
            pairs[["pair_id", "pair_type", "source_pair_id", "class_id"]],
            on="pair_id",
            validate="many_to_one",
        )
        frames.append(frame)
        receipts.append(json.loads(path.with_suffix(".json").read_text()))
    if missing:
        atomic_json(output / "analysis_status.json", {"complete": False, "missing_tasks": missing})
        raise RuntimeError(
            f"{len(missing)}/{len(plan['tasks'])} cache shards missing; run infer/resume first"
        )
    samples = pd.concat(frames, ignore_index=True)
    identity = ["model_id", "path", "pair_id", "stages"]
    if samples.duplicated(identity).any():
        raise ValueError("duplicate sample-grid identities")
    models, ranks = summarize(samples)
    result_dir = output / "analysis"
    result_dir.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(result_dir / "sample_metrics.parquet", index=False)
    models.to_csv(result_dir / "model_metrics.csv", index=False)
    ranks.to_csv(result_dir / "model_rank_stability.csv", index=False)
    status = {
        "complete": True,
        "plan_sha256": plan["plan_sha256"],
        "analysis_code_sha256": file_hash(HERE / "stage_analysis.py"),
        "models": len(plan["models"]),
        "typed_pairs": len(plan["pairs"]),
        "source_pairs": len({p["source_pair_id"] for p in plan["pairs"]}),
        "tasks": len(plan["tasks"]),
        "stage_counts": config["stage_counts"],
        "epsilon": config["epsilon"],
        "model_input_evaluations": sum(r["model_input_evaluations"] for r in receipts),
        "conservation_max_residual": float(samples.conservation_residual.max()),
        "interpretation": (
            "Empirical agreement with finest sampled grid; "
            "not proof of continuous-path convergence."
        ),
        "undefined_metrics": (
            "Zero-reference relative errors and constant/under-3-model ranks remain NaN."
        ),
    }
    atomic_json(output / "analysis_status.json", status)
    print(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "infer", "analyze"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-id")
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(args.output)
    else:
        if args.config is None:
            parser.error("--config is required for plan and infer")
        plan = prepare(args.config, args.output)
        if args.command == "plan":
            print(
                json.dumps(
                    {
                        "plan_sha256": plan["plan_sha256"],
                        "tasks": len(plan["tasks"]),
                        "typed_pairs": len(plan["pairs"]),
                        "models": len(plan["models"]),
                    }
                )
            )
        else:
            infer(
                plan,
                args.output,
                device=args.device,
                worker_index=args.worker_index,
                workers=args.workers,
                model_id=args.model_id,
            )


if __name__ == "__main__":
    main()
