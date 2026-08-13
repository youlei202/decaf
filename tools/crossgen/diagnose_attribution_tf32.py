"""One-unit cuDNN TF32 attribution diagnostic for the cross-generation audit.

This intentionally fixed diagnostic is not a benchmark runner. It evaluates
one registered IDSDS trajectory twice while changing only cuDNN TF32, then
records the regenerated stage scores beside the immutable historical row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tools.crossgen.schema import sha256_file

LEGACY_SOURCE = Path(
    "/work/Users/leiyo/GitHub/covariance-matched-markov-revelation/src"
)
IDSDS_SOURCE_ROOT = Path(
    "/work/Users/leiyo/decaf_idsds_funnybirds_v1_data/official/idsds"
)
CHECKPOINT_ROOT = Path(
    "/work/Users/leiyo/decaf_idsds_funnybirds_v1_data/official/idsds_checkpoints"
)
IDSDS_MANIFEST = Path(
    "/work/Users/leiyo/decaf_idsds_funnybirds_v1_results/manifests/"
    "imagenet_idsds_10k.parquet"
)
SEALED_RESULTS = Path(
    "/work/Users/leiyo/decaf_idsds_funnybirds_v1_results/imagenet/"
    "per_image_idsds.parquet"
)
MODEL_ID = "resnet50"
IMAGE_ID = "ILSVRC2012_val_00000076_n02791270"
TARGET = 424
METHOD = "decaf_5"
SCHEDULE = "DECAF-5"
PATCH_INDEX = 12
SEED = 0
EPSILON = 0.02


def _vector(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _historical_row() -> dict[str, Any]:
    frame = pd.read_parquet(SEALED_RESULTS)
    selected = frame.loc[
        (frame["dataset"].astype(str) == "imagenet")
        & (frame["scope"].astype(str) == "science")
        & (frame["model"].astype(str) == MODEL_ID)
        & (frame["method"].astype(str) == METHOD)
        & (frame["image_id"].astype(str) == IMAGE_ID)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"sealed identity is not unique: rows={len(selected)}")
    row = selected.iloc[0]
    if int(row["label"]) != TARGET:
        raise RuntimeError(f"sealed target changed: {row['label']} != {TARGET}")
    vectors = {
        name: _vector(row[f"decaf_{name}"])
        for name in ("M", "E", "signed_E", "C", "F", "Abs")
    }
    vectors["effects"] = _vector(row["effects"])
    return {
        "patch": {name: values[PATCH_INDEX] for name, values in vectors.items()},
        "vectors": vectors,
        "member_path": str(row["member_path"]),
        "deletion_target_sha256": str(row["deletion_target_sha256"]),
        "source": str(SEALED_RESULTS.resolve()),
        "source_sha256": sha256_file(SEALED_RESULTS),
    }


def _input_image() -> torch.Tensor:
    from cmr.decaf_idsds_funnybirds_v1.data import ImageNetParquetDataset

    frame = pd.read_parquet(IDSDS_MANIFEST)
    selected = frame.loc[frame["image_id"].astype(str) == IMAGE_ID]
    if len(selected) != 1:
        raise RuntimeError(f"manifest identity is not unique: rows={len(selected)}")
    if int(selected.iloc[0]["label"]) != TARGET:
        raise RuntimeError(
            f"manifest target changed: {selected.iloc[0]['label']} != {TARGET}"
        )
    item = ImageNetParquetDataset(selected, model_id=MODEL_ID)[0]
    if str(item["image_id"]) != IMAGE_ID or int(item["label"]) != TARGET:
        raise RuntimeError("loaded sample identity changed")
    return item["image"].unsqueeze(0)


def _run_once(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    allow_cudnn_tf32: bool,
) -> dict[str, Any]:
    from cmr.decaf_idsds_funnybirds_v1.attribution import compute_decaf_attribution

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = bool(allow_cudnn_tf32)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = compute_decaf_attribution(
        model,
        image,
        torch.tensor([TARGET], device=image.device),
        schedule=SCHEDULE,
        epsilon=EPSILON,
        internal_batch_size=17,
    )
    torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - started
    metadata = result.metadata
    vectors = {
        name: _vector(metadata[name])
        for name in ("M", "E", "signed_E", "C", "F", "Abs", "endpoint_delta")
    }
    q_plus = np.asarray(_vector(metadata["q_plus"]), dtype=np.float64)
    q_minus = np.asarray(
        metadata["q_minus"].detach().cpu().numpy(), dtype=np.float64
    )[0, PATCH_INDEX]
    return {
        "allow_cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "runtime_seconds": runtime_seconds,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "patch": {name: values[PATCH_INDEX] for name, values in vectors.items()},
        "vectors": vectors,
        "stage_t": _vector(metadata["schedule"]),
        "stage_q_plus": q_plus.tolist(),
        "stage_q_minus_patch": q_minus.tolist(),
        "stage_r_patch": (q_plus - q_minus).tolist(),
    }


def run(output: str | Path) -> Path:
    if str(LEGACY_SOURCE) not in sys.path:
        sys.path.insert(0, str(LEGACY_SOURCE))
    from cmr.decaf_idsds_funnybirds_v1.models import load_idsds_model_adapter

    if not torch.cuda.is_available():
        raise RuntimeError("the fixed TF32 diagnostic requires CUDA")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    image = _input_image().to("cuda:0")
    model = load_idsds_model_adapter(
        MODEL_ID,
        device="cuda:0",
        precision="fp32",
        source_root=IDSDS_SOURCE_ROOT,
        checkpoint_root=CHECKPOINT_ROOT,
    ).eval()
    checkpoint = Path(model.model.idsds_checkpoint_path).resolve()
    runs = {
        "cudnn_tf32_on": _run_once(model, image, allow_cudnn_tf32=True),
        "cudnn_tf32_off": _run_once(model, image, allow_cudnn_tf32=False),
    }
    payload = {
        "schema_version": 1,
        "purpose": "single-unit causal cuDNN TF32 attribution diagnostic",
        "seed": SEED,
        "model_id": MODEL_ID,
        "image_id": IMAGE_ID,
        "target": TARGET,
        "method": METHOD,
        "schedule": SCHEDULE,
        "patch_index_zero_based": PATCH_INDEX,
        "endpoint_epsilon": EPSILON,
        "device": str(torch.cuda.get_device_name(0)),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "model_precision": "fp32",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_contract_sha256": str(model.model.idsds_checkpoint_sha256),
        "input_manifest": str(IDSDS_MANIFEST.resolve()),
        "input_manifest_sha256": sha256_file(IDSDS_MANIFEST),
        "historical": _historical_row(),
        "runs": runs,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = run(args.output)
    print(destination.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
