"""Strict offline single-GPU executor for attribution verification shards.

This module owns the complete runtime path used on the verification node.  It
does not import the historical repository, invoke an historical command, use a
network model registry, or download a missing asset.  Every source tree,
checkpoint, dataset manifest, and prepared-data root is supplied by an
environment variable and checked before deserialization.

Torch and torchvision are imported lazily so static planning and the default
CPU oracle remain usable in the repository's lightweight Python environment.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.core.manifests import sha256_file
from decaf.experiments.attribution.endpoint import row_spearman
from decaf.experiments.attribution.methods import decaf_trajectory
from decaf.experiments.attribution.plan import (
    DELETION_TARGET_METHOD,
    FUNNYBIRDS_DELETION_TARGET_METHOD,
)
from decaf.experiments.common import RunContext, atomic_json

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IDSDS_REVISION = "8e842009423f14ac790b352b1f86846cc381415c"
FUNNYBIRDS_REVISION = "91b4b4628ffa962148144ee6bb5af5f022cac2f8"
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
IDSDS_MANIFEST_SHA256 = "3f6f9bad1c631f3eb95e8e2ae2fb171dd86470deaed7f3c93259feea952c0e79"
COMMON_SUPPORT_SHA256 = "b400ab055e7d47f25ef1eba6201e6f34f8b611a61db53433dc8dbe61f8d95034"
FUNNYBIRDS_MANIFEST_SHA256 = "bc4d1c647fd0f5ab6611bacfa5a558e15b246916cee037ca80cc6b056d890f2c"
PARTIMAGENET_MANIFEST_SHA256 = "d1198f5a06bd4ef9656473a047fe4e01ddabf2a76f5868dfa7ee6579ae710657"

# Frozen method-level random streams from the registered IDSDS/FunnyBirds
# comparison.  These are intentionally independent of the bootstrap seed and
# of image/shard ordering: the formal methods share one bank at each input
# resolution.
UNIFORM_BASELINE_SEED = 8212
RISE_MASK_SEED = 8213
KERNEL_SHAP_SEED = 8214
GRADIENT_SHAP_SEED = 8215
SMOOTHGRAD_SEED = 8216
ATTRIBUTION_MONTE_CARLO_SAMPLES = 16
ATTRIBUTION_PERTURBATION_BUDGET = 512
RAW_RGB_INPUT_DOMAIN = "raw_rgb_float_0_1"
MODEL_INPUT_DOMAIN = "fixed_shape_model_input"
FUNNYBIRDS_SUPPLEMENT_INPUT_DOMAIN = "fixed_shape_official_funnybirds_model_input"
FUNNYBIRDS_SUPPLEMENT_METHODS = frozenset({"ig_u_32", "rise_u_512"})


def _stable_method_seed(member_id: str, image_id: str) -> int:
    """Match the frozen worker's per-job, per-image stochastic seed contract."""

    digest = hashlib.sha256(f"{member_id}\0{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """One exact checkpoint byte identity and its mandatory environment key."""

    checkpoint_id: str
    environment: str
    sha256: str
    filename: str


@dataclass(frozen=True, slots=True)
class CheckpointAsset:
    """One validated local checkpoint."""

    checkpoint_id: str
    path: Path
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class OfflineAssets:
    """Validated paths needed by one runtime operation."""

    checkpoints: Mapping[str, CheckpointAsset]
    manifests: Mapping[str, Path]
    source_roots: Mapping[str, Path]
    prepared_root: Path | None
    common_support: Path | None


@dataclass(frozen=True, slots=True)
class AttributionSample:
    """One real image and its aligned feature/intervention tensors on CPU."""

    dataset: str
    image_id: str
    target: int
    image: Any
    masks: Any
    reference: Any
    interventions: Mapping[str, Any]
    part_names: tuple[str, ...]
    raw_height: int
    raw_width: int


@dataclass(frozen=True, slots=True)
class PreparedSample:
    """One per-image runtime sample in its registered attribution domain.

    IDSDS and DINO samples use fixed-shape model inputs.  Ordinary FunnyBirds
    and PartImageNet science keeps the raw image geometry so perturbations are
    formed in RGB ``[0,1]`` before the model adapter performs its transform.
    """

    dataset: str
    image_id: str
    target: int
    image: Any
    masks: Any
    reference: Any
    interventions: Mapping[str, Any]
    part_names: tuple[str, ...]
    raw_height: int
    raw_width: int


CHECKPOINT_SPECS: dict[str, CheckpointSpec] = {
    spec.checkpoint_id: spec
    for spec in (
        CheckpointSpec(
            "idsds_resnet50",
            "DECAF_CHECKPOINT_IDSDS_RESNET50",
            "dc4b6f9424ca154e5fa27aa5f574e4d7d94e2c969979c66ed978d4ef9eb799b4",
            "resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar",
        ),
        CheckpointSpec(
            "idsds_vgg16",
            "DECAF_CHECKPOINT_IDSDS_VGG16",
            "ec5aad9340d467f6375f784336e3a083e4ee50abc53ecdc8209754d4841f78a4",
            "vgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar",
        ),
        CheckpointSpec(
            "idsds_vit_base_patch16_224",
            "DECAF_CHECKPOINT_IDSDS_VIT_B16",
            "858fb793a1debb2e03254545e2a57f7533ea9a07a1d2445706afec27e3985033",
            "vit_base_patch16_224_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar",
        ),
        CheckpointSpec(
            "funnybirds_resnet",
            "DECAF_CHECKPOINT_FUNNYBIRDS_RESNET50",
            "88f0fcf517ab5aa318325db7ffef46d65b52f0b8e18c325f5280d97f631ad139",
            "resnet50_final_0_checkpoint_best.pth.tar",
        ),
        CheckpointSpec(
            "funnybirds_vgg",
            "DECAF_CHECKPOINT_FUNNYBIRDS_VGG16",
            "56ca79a65e0163bddd2c0f16bba5775ff3005e5b1a1a1ca53df5054b29bd1366",
            "vgg16_final_1_checkpoint_best.pth.tar",
        ),
        CheckpointSpec(
            "funnybirds_vit",
            "DECAF_CHECKPOINT_FUNNYBIRDS_VIT_B16",
            "3d639524730445bba226abf39e907001331fe440a0f4111e1169b6e144172008",
            "vit_base_patch16_224_final_1_checkpoint_best.pth.tar",
        ),
        CheckpointSpec(
            "torchvision_resnet50",
            "DECAF_CHECKPOINT_TORCHVISION_RESNET50",
            "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
            "resnet50-11ad3fa6.pth",
        ),
        CheckpointSpec(
            "dinov2_vitg14_backbone",
            "DECAF_CHECKPOINT_DINOV2_VITG14_BACKBONE",
            "baf8467e50af277596bbbafa06887c177ee899ab46033649c383577d7e9309d3",
            "dinov2_vitg14_pretrain.pth",
        ),
        CheckpointSpec(
            "dinov2_vitg14_linear_head",
            "DECAF_CHECKPOINT_DINOV2_VITG14_HEAD",
            "ab61850e248839f9e242d7a5a1284feb3eace2dd109fa28f530a75d1bd17942a",
            "dinov2_vitg14_linear4_head.pth",
        ),
    )
}

MODEL_CHECKPOINTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("resnet50", "imagenet1k_idsds"): ("idsds_resnet50",),
    ("vgg16", "imagenet1k_idsds"): ("idsds_vgg16",),
    ("vit_base_patch16_224", "imagenet1k_idsds"): ("idsds_vit_base_patch16_224",),
    ("funnybirds_resnet50", "funnybirds"): ("funnybirds_resnet",),
    ("funnybirds_vgg16", "funnybirds"): ("funnybirds_vgg",),
    ("funnybirds_vit_b_16", "funnybirds"): ("funnybirds_vit",),
    ("resnet50", "partimagenet"): ("torchvision_resnet50",),
    ("dinov2_vit_g_14", "imagenet1k_idsds"): (
        "dinov2_vitg14_backbone",
        "dinov2_vitg14_linear_head",
    ),
}

MANIFEST_SPECS = {
    "imagenet1k_idsds": (
        "DECAF_IDSDS_MANIFEST",
        IDSDS_MANIFEST_SHA256,
    ),
    "funnybirds": (
        "DECAF_FUNNYBIRDS_MANIFEST",
        FUNNYBIRDS_MANIFEST_SHA256,
    ),
    "partimagenet": (
        "DECAF_PARTIMAGENET_MANIFEST",
        PARTIMAGENET_MANIFEST_SHA256,
    ),
}

SOURCE_SPECS = {
    "idsds": ("DECAF_IDSDS_SOURCE_ROOT", IDSDS_REVISION),
    "funnybirds": ("DECAF_FUNNYBIRDS_SOURCE_ROOT", FUNNYBIRDS_REVISION),
    "dinov2": ("DECAF_DINOV2_SOURCE_ROOT", DINOV2_REVISION),
}

FINGERPRINT_CASES = (
    ("funnybirds_resnet50", "funnybirds"),
    ("funnybirds_vgg16", "funnybirds"),
    ("funnybirds_vit_b_16", "funnybirds"),
    ("resnet50", "imagenet1k_idsds"),
    ("vgg16", "imagenet1k_idsds"),
    ("vit_base_patch16_224", "imagenet1k_idsds"),
    ("dinov2_vit_g_14", "imagenet1k_idsds"),
)

_SOURCE_MODULES: dict[tuple[str, str], types.ModuleType] = {}
_SOURCE_REVISIONS: set[tuple[str, str]] = set()
_CHECKPOINT_CACHE: dict[tuple[str, str], CheckpointAsset] = {}
_ACTIVE_MODEL_KEY: tuple[str, str, str, str] | None = None
_ACTIVE_MODEL: Any | None = None
_SAMPLE_CACHE: dict[tuple[str, str, int, str], tuple[AttributionSample, ...]] = {}
_RAW_RGB_ADAPTER_TYPE: type[Any] | None = None


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised by CPU-only audit
        raise RuntimeError(
            "real attribution verification requires the configured GPU Python with torch"
        ) from error
    return torch


@contextmanager
def _strict_fp32_backends() -> Any:
    """Disable TF32 for one attribution member and restore process state."""

    torch = _torch()
    previous_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_cudnn = bool(torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def _required_path(
    environment: str,
    *,
    environ: Mapping[str, str] | None = None,
    directory: bool = False,
) -> Path:
    selected = os.environ if environ is None else environ
    value = selected.get(environment)
    if not value:
        raise RuntimeError(f"offline attribution verification requires ${environment}")
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if path.is_symlink() or not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"${environment} is not a safe {kind}: {path}")
    if not directory and path.stat().st_size <= 0:
        raise FileNotFoundError(f"${environment} is empty: {path}")
    return path


def _validate_file(path: Path, expected_sha256: str, *, label: str) -> str:
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError(f"invalid expected SHA256 for {label}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch: {path} ({observed} != {expected_sha256})")
    return observed


def resolve_checkpoint(
    checkpoint_id: str,
    environ: Mapping[str, str] | None = None,
) -> CheckpointAsset:
    """Resolve and hash one exact checkpoint; never search for a substitute."""

    try:
        spec = CHECKPOINT_SPECS[str(checkpoint_id)]
    except KeyError as error:
        raise KeyError(f"unknown attribution checkpoint: {checkpoint_id}") from error
    path = _required_path(spec.environment, environ=environ)
    if path.name != spec.filename:
        raise RuntimeError(
            f"${spec.environment} must name {spec.filename!r}, received {path.name!r}"
        )
    key = (spec.checkpoint_id, str(path))
    cached = _CHECKPOINT_CACHE.get(key)
    if cached is not None:
        if path.stat().st_size != cached.bytes:
            raise RuntimeError(f"checkpoint size changed after validation: {path}")
        return cached
    observed = _validate_file(path, spec.sha256, label=spec.checkpoint_id)
    asset = CheckpointAsset(spec.checkpoint_id, path, observed, path.stat().st_size)
    _CHECKPOINT_CACHE[key] = asset
    return asset


def _git_revision(path: Path) -> str:
    process = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"official source is not a readable Git checkout: {path}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def _resolve_source(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    environment, revision = SOURCE_SPECS[name]
    path = _required_path(environment, environ=environ, directory=True)
    key = (str(path), revision)
    if key not in _SOURCE_REVISIONS:
        observed = _git_revision(path)
        if observed != revision:
            raise RuntimeError(
                f"{name} source revision mismatch: {path} ({observed} != {revision})"
            )
        _SOURCE_REVISIONS.add(key)
    return path


def resolve_offline_assets(
    environ: Mapping[str, str] | None = None,
    *,
    checkpoint_ids: Sequence[str] | None = None,
    datasets: Sequence[str] = ("imagenet1k_idsds", "funnybirds"),
    sources: Sequence[str] = ("idsds", "funnybirds", "dinov2"),
    require_prepared: bool = True,
    require_common_support: bool = True,
) -> OfflineAssets:
    """Resolve a requested offline inventory, validating every known digest."""

    selected_checkpoints = (
        tuple(checkpoint for case in FINGERPRINT_CASES for checkpoint in MODEL_CHECKPOINTS[case])
        if checkpoint_ids is None
        else tuple(str(value) for value in checkpoint_ids)
    )
    checkpoint_assets = {
        checkpoint_id: resolve_checkpoint(checkpoint_id, environ)
        for checkpoint_id in dict.fromkeys(selected_checkpoints)
    }
    manifests: dict[str, Path] = {}
    for dataset in datasets:
        try:
            environment, expected = MANIFEST_SPECS[str(dataset)]
        except KeyError as error:
            raise KeyError(f"unknown attribution dataset manifest: {dataset}") from error
        path = _required_path(environment, environ=environ)
        _validate_file(path, expected, label=f"{dataset} manifest")
        manifests[str(dataset)] = path
    source_roots = {name: _resolve_source(str(name), environ=environ) for name in sources}
    prepared_root = (
        _required_path("DECAF_ATTRIBUTION_PREP_ROOT", environ=environ, directory=True)
        if require_prepared
        else None
    )
    common_support = None
    if require_common_support:
        common_support = _required_path("DECAF_IDSDS_COMMON_SUPPORT", environ=environ)
        _validate_file(
            common_support,
            COMMON_SUPPORT_SHA256,
            label="strict common-support manifest",
        )
    return OfflineAssets(
        checkpoints=checkpoint_assets,
        manifests=manifests,
        source_roots=source_roots,
        prepared_root=prepared_root,
        common_support=common_support,
    )


def load_checkpoint_state_dict(
    path: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Hash, CPU-deserialize, unwrap, and conservatively normalize a state dict."""

    torch = _torch()
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"checkpoint is missing or unsafe: {source}")
    _validate_file(source, str(expected_sha256).lower(), label="checkpoint")
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        payload = torch.load(source, mmap=True, **kwargs)
    except (TypeError, RuntimeError):
        payload = torch.load(source, **kwargs)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a mapping: {source}")
    state: Any = payload
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if not isinstance(state, Mapping) or not state:
        raise KeyError(f"checkpoint has no non-empty state dict: {source}")
    normalized = {str(key): value for key, value in state.items()}
    for _ in range(3):
        prefix = next(
            (
                candidate
                for candidate in ("module.model.", "module.", "model.")
                if normalized and all(key.startswith(candidate) for key in normalized)
            ),
            None,
        )
        if prefix is None:
            break
        normalized = {key[len(prefix) :]: value for key, value in normalized.items()}
    if not all(isinstance(value, torch.Tensor) for value in normalized.values()):
        raise TypeError(f"checkpoint state dict contains non-tensor values: {source}")
    return normalized


def _load_source_module(path: Path, module_name: str) -> types.ModuleType:
    key = (str(path), module_name)
    cached = _SOURCE_MODULES.get(key)
    if cached is not None:
        return cached
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"official model source is missing or unsafe: {path}")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot construct an import specification for {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    _SOURCE_MODULES[key] = module
    return module


def _load_vit_module(path: Path, package_name: str) -> types.ModuleType:
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(path.parent)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    return _load_source_module(path, f"{package_name}.ViT_new")


def _build_source_model(
    model_id: str,
    dataset: str,
    source_root: Path,
) -> Any:
    torch = _torch()
    if dataset == "funnybirds":
        lookup = {
            "funnybirds_resnet50": ("models/resnet.py", "resnet50", 50),
            "funnybirds_vgg16": ("models/vgg.py", "vgg16", 50),
            "funnybirds_vit_b_16": (
                "models/ViT/ViT_new.py",
                "vit_base_patch16_224",
                50,
            ),
        }
        namespace = "_decaf_official_funnybirds"
    else:
        lookup = {
            "resnet50": ("models/resnet.py", "resnet50", 1_000),
            "vgg16": ("models/vgg.py", "vgg16", 1_000),
            "vit_base_patch16_224": (
                "models/ViT/ViT_new.py",
                "vit_base_patch16_224",
                1_000,
            ),
        }
        namespace = "_decaf_official_idsds"
    try:
        relative, constructor_name, classes = lookup[model_id]
    except KeyError as error:
        raise KeyError(f"unsupported source model: {dataset}/{model_id}") from error
    source = source_root / relative
    module = (
        _load_vit_module(source, f"{namespace}_vit")
        if "ViT_new.py" in relative
        else _load_source_module(source, f"{namespace}_{model_id}")
    )
    constructor = getattr(module, constructor_name, None)
    if not callable(constructor):
        raise RuntimeError(f"official source lacks {constructor_name}: {source}")
    model = constructor(pretrained=False, num_classes=classes)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"official constructor did not return a torch module: {source}")
    return model


def _load_dinov2_model(source_root: Path, checkpoints: Sequence[CheckpointAsset]) -> Any:
    torch = _torch()
    if len(checkpoints) != 2:
        raise ValueError("DINOv2-g requires exactly one backbone and one linear head")
    model = torch.hub.load(
        str(source_root),
        "dinov2_vitg14_lc",
        source="local",
        pretrained=False,
        verbose=False,
    )
    if not hasattr(model, "backbone") or not hasattr(model, "linear_head"):
        raise TypeError("official DINOv2-g classifier has an unexpected structure")
    backbone, head = checkpoints
    model.backbone.load_state_dict(
        load_checkpoint_state_dict(backbone.path, backbone.sha256), strict=True
    )
    model.linear_head.load_state_dict(
        load_checkpoint_state_dict(head.path, head.sha256), strict=True
    )
    return model


def _precision(precision: str) -> Any:
    torch = _torch()
    values = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    try:
        return values[str(precision).lower()]
    except KeyError as error:
        raise ValueError("precision must be fp32, bf16, or fp16") from error


def _required_sources(model_id: str, dataset: str) -> tuple[str, ...]:
    if dataset == "funnybirds":
        return ("funnybirds",)
    if model_id == "dinov2_vit_g_14":
        return ("dinov2",)
    if dataset == "imagenet1k_idsds":
        return ("idsds",)
    return ()


def load_model(
    model_id: str,
    *,
    dataset: str,
    device: str = "cuda:0",
    precision: str = "fp32",
    assets: OfflineAssets | None = None,
) -> Any:
    """Load one registered model from exact local bytes, with no fallback."""

    torch = _torch()
    key = (str(model_id), str(dataset))
    try:
        checkpoint_ids = MODEL_CHECKPOINTS[key]
    except KeyError as error:
        raise KeyError(f"no offline model contract for {dataset}/{model_id}") from error
    selected_assets = assets or resolve_offline_assets(
        checkpoint_ids=checkpoint_ids,
        datasets=(),
        sources=_required_sources(model_id, dataset),
        require_prepared=False,
        require_common_support=False,
    )
    checkpoints = [
        selected_assets.checkpoints.get(checkpoint_id) or resolve_checkpoint(checkpoint_id)
        for checkpoint_id in checkpoint_ids
    ]
    if any(checkpoint is None for checkpoint in checkpoints):  # pragma: no cover
        raise AssertionError("checkpoint resolution unexpectedly returned None")
    if dataset == "funnybirds":
        source = selected_assets.source_roots.get("funnybirds") or _resolve_source("funnybirds")
        model = _build_source_model(model_id, dataset, source)
        model.load_state_dict(
            load_checkpoint_state_dict(checkpoints[0].path, checkpoints[0].sha256),
            strict=True,
        )
    elif model_id == "dinov2_vit_g_14":
        source = selected_assets.source_roots.get("dinov2") or _resolve_source("dinov2")
        model = _load_dinov2_model(source, checkpoints)
    elif dataset == "imagenet1k_idsds":
        source = selected_assets.source_roots.get("idsds") or _resolve_source("idsds")
        model = _build_source_model(model_id, dataset, source)
        model.load_state_dict(
            load_checkpoint_state_dict(checkpoints[0].path, checkpoints[0].sha256),
            strict=True,
        )
    elif key == ("resnet50", "partimagenet"):
        from torchvision import models as tv_models

        model = tv_models.resnet50(weights=None, progress=False)
        model.load_state_dict(
            load_checkpoint_state_dict(checkpoints[0].path, checkpoints[0].sha256),
            strict=True,
        )
    else:  # pragma: no cover - guarded by MODEL_CHECKPOINTS
        raise AssertionError(f"unsupported model loader: {key}")
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real attribution verification requires an available CUDA device")
    model.to(device=selected_device, dtype=_precision(precision))
    model.eval()
    model.decaf_model_id = model_id  # type: ignore[attr-defined]
    model.decaf_dataset = dataset  # type: ignore[attr-defined]
    model.decaf_precision = str(precision)  # type: ignore[attr-defined]
    model.decaf_checkpoint_assets = tuple(  # type: ignore[attr-defined]
        checkpoint.to_dict() for checkpoint in checkpoints
    )
    return model


def _active_model(
    model_id: str,
    dataset: str,
    device: str,
    precision: str,
) -> Any:
    global _ACTIVE_MODEL, _ACTIVE_MODEL_KEY
    torch = _torch()
    key = (model_id, dataset, device, precision)
    if _ACTIVE_MODEL_KEY == key and _ACTIVE_MODEL is not None:
        return _ACTIVE_MODEL
    if _ACTIVE_MODEL is not None:
        del _ACTIVE_MODEL
        _ACTIVE_MODEL = None
        _ACTIVE_MODEL_KEY = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _ACTIVE_MODEL = load_model(
        model_id,
        dataset=dataset,
        device=device,
        precision=precision,
    )
    _ACTIVE_MODEL_KEY = key
    return _ACTIVE_MODEL


def _pil_tensor(path: Path, *, expected_sha256: str | None = None) -> Any:
    torch = _torch()
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - GPU environment owns dependency
        raise RuntimeError("real image loading requires Pillow") from error
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"image is missing or unsafe: {path}")
    if expected_sha256:
        _validate_file(path, expected_sha256, label="prepared source image")
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True, order="C")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


def _decode_parquet_image(row: Any) -> Any:
    extracted = getattr(row, "extracted_path", None)
    if extracted is not None and not pd.isna(extracted) and str(extracted):
        return _pil_tensor(Path(str(extracted)).resolve())
    import pyarrow.parquet as pq

    shard = Path(str(row.source_shard)).resolve()
    if shard.is_symlink() or not shard.is_file():
        raise FileNotFoundError(f"ImageNet source shard is missing or unsafe: {shard}")
    parquet = pq.ParquetFile(shard)
    group_value = getattr(row, "row_group", None)
    offset_value = getattr(row, "row_in_group", None)
    if pd.isna(group_value) or pd.isna(offset_value):
        remaining = int(row.row_index)
        group = 0
        while group < parquet.metadata.num_row_groups:
            rows = int(parquet.metadata.row_group(group).num_rows)
            if remaining < rows:
                break
            remaining -= rows
            group += 1
        offset = remaining
    else:
        group, offset = int(group_value), int(offset_value)
    table = parquet.read_row_group(group, columns=["image", "label"], use_threads=False)
    if int(table.column("label")[offset].as_py()) != int(row.label):
        raise RuntimeError(f"ImageNet manifest label drifted for {row.image_id}")
    value = table.column("image")[offset].as_py()
    if not isinstance(value, Mapping):
        raise TypeError(f"ImageNet image struct is invalid for {row.image_id}")
    encoded = value.get("bytes")
    if not isinstance(encoded, (bytes, bytearray, memoryview)) or not encoded:
        raise ValueError(f"ImageNet image bytes are absent for {row.image_id}")
    torch = _torch()
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("real ImageNet loading requires Pillow") from error
    with Image.open(io.BytesIO(bytes(encoded))) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True, order="C")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


def _idsds_masks(height: int, width: int) -> Any:
    torch = _torch()
    if height % 4 or width % 4:
        raise ValueError("IDSDS model inputs must be divisible into a 4x4 patch grid")
    masks = torch.zeros((16, height, width), dtype=torch.float32)
    patch_height, patch_width = height // 4, width // 4
    for row in range(4):
        for column in range(4):
            masks[
                row * 4 + column,
                row * patch_height : (row + 1) * patch_height,
                column * patch_width : (column + 1) * patch_width,
            ] = 1.0
    if not torch.equal(masks.sum(0), torch.ones((height, width))):
        raise AssertionError("IDSDS masks do not form the registered row-major partition")
    return masks


def _support_ids(path: Path, dataset: str, model_id: str, count: int) -> tuple[str, ...]:
    support = pd.read_parquet(path)
    required = {"dataset", "model", "image_id", "correctly_classified", "included"}
    missing = sorted(required - set(support.columns))
    if missing:
        raise ValueError(f"strict common support is missing columns: {missing}")
    support_dataset = "imagenet" if dataset == "imagenet1k_idsds" else dataset
    selected = support.loc[
        support["dataset"].astype(str).eq(support_dataset)
        & support["model"].astype(str).eq(model_id)
        & support["correctly_classified"].astype(bool)
        & support["included"].astype(bool),
        "image_id",
    ].astype(str)
    identifiers = tuple(dict.fromkeys(selected))
    if len(identifiers) < count:
        raise RuntimeError(
            f"strict common support has only {len(identifiers)} rows for {dataset}/{model_id}"
        )
    return identifiers[:count]


def _prepared_rows(
    prepared_root: Path,
    dataset: str,
    *,
    subset: str,
) -> pd.DataFrame:
    frozen_path = prepared_root / "frozen_subset_manifest.parquet"
    _validate_file(
        frozen_path,
        PARTIMAGENET_MANIFEST_SHA256,
        label="prepared frozen-subset manifest",
    )
    inventory = pd.read_parquet(frozen_path)
    selected = inventory.loc[
        inventory["dataset"].astype(str).eq(dataset)
        & inventory["subset"].astype(str).eq(subset)
        & inventory["frozen"].astype(bool)
    ].sort_values("shard_id", kind="stable")
    if selected.empty:
        raise RuntimeError(f"prepared subset is empty: {dataset}/{subset}")
    frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        embedded = Path(str(row.shard_path)).resolve()
        try:
            embedded.relative_to(prepared_root)
        except ValueError as error:
            raise RuntimeError(
                f"prepared shard escapes $DECAF_ATTRIBUTION_PREP_ROOT: {embedded}"
            ) from error
        _validate_file(embedded, str(row.shard_sha256), label=str(row.shard_id))
        frame = pd.read_parquet(embedded)
        if len(frame) != int(row.image_count):
            raise RuntimeError(f"prepared shard row count drifted: {embedded}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    if result["image_id"].astype(str).duplicated().any():
        raise RuntimeError(f"prepared subset contains duplicate image IDs: {dataset}/{subset}")
    return result


def _prepared_sample(row: Any, dataset: str) -> AttributionSample:
    torch = _torch()
    image_path = Path(str(row.source_image_path)).resolve()
    image = _pil_tensor(image_path, expected_sha256=str(row.image_sha256))
    intervention_path = Path(str(row.intervention_path)).resolve()
    _validate_file(
        intervention_path,
        str(row.intervention_sha256),
        label=f"{dataset} prepared intervention",
    )
    with np.load(intervention_path, allow_pickle=False) as archive:
        required = {
            "part_masks",
            "telea",
            "background_texture",
            "names",
            "reference_blur",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"prepared intervention is missing arrays {sorted(missing)}")
        masks = torch.from_numpy(np.array(archive["part_masks"], copy=True)).float()
        reference = (
            torch.from_numpy(np.array(archive["reference_blur"], copy=True))
            .permute(2, 0, 1)
            .contiguous()
            .float()
        )
        interventions = {
            "telea": torch.from_numpy(np.array(archive["telea"], copy=True))
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .div_(255.0),
            "background_texture": torch.from_numpy(
                np.array(archive["background_texture"], copy=True)
            )
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .div_(255.0),
        }
        names = tuple(str(value) for value in archive["names"].tolist())
    height, width = map(int, image.shape[-2:])
    if (
        tuple(masks.shape[-2:]) != (height, width)
        or tuple(reference.shape[-2:]) != (height, width)
        or len(names) != int(masks.shape[0])
        or any(tuple(value.shape[-2:]) != (height, width) for value in interventions.values())
    ):
        raise ValueError(f"prepared image/mask/intervention alignment drifted: {row.image_id}")
    if not 1 <= len(names) <= 8:
        raise ValueError(f"prepared semantic group count lies outside [1,8]: {row.image_id}")
    return AttributionSample(
        dataset=dataset,
        image_id=str(row.image_id),
        target=int(row.target_class),
        image=image,
        masks=masks,
        reference=reference,
        interventions=interventions,
        part_names=names,
        raw_height=height,
        raw_width=width,
    )


def _idsds_candidates(
    manifest: Path,
    identifiers: Sequence[str] | None,
    *,
    limit: int | None = None,
) -> list[AttributionSample]:
    torch = _torch()
    frame = pd.read_parquet(manifest)
    required = {"image_id", "label", "source_shard", "row_index"}
    missing = sorted(required - set(frame.columns))
    if missing or frame["image_id"].astype(str).duplicated().any():
        raise ValueError(f"IDSDS manifest is invalid; missing={missing}")
    if identifiers is not None:
        order = {value: index for index, value in enumerate(identifiers)}
        frame = frame.loc[frame["image_id"].astype(str).isin(order)].copy()
        frame["_order"] = frame["image_id"].astype(str).map(order)
        frame = frame.sort_values("_order", kind="stable")
        if tuple(frame["image_id"].astype(str)) != tuple(identifiers):
            raise RuntimeError("fixed IDSDS image IDs are not exactly present in the manifest")
    elif limit is not None:
        frame = frame.head(int(limit))
    samples: list[AttributionSample] = []
    for row in frame.itertuples(index=False):
        raw = _decode_parquet_image(row)
        height, width = map(int, raw.shape[-2:])
        samples.append(
            AttributionSample(
                dataset="imagenet1k_idsds",
                image_id=str(row.image_id),
                target=int(row.label),
                image=raw,
                masks=torch.empty((0, height, width)),
                reference=torch.empty((0, height, width)),
                interventions={},
                part_names=tuple(f"patch_{index:02d}" for index in range(16)),
                raw_height=height,
                raw_width=width,
            )
        )
    return samples


def load_fixed_samples(
    dataset: str,
    model_id: str,
    *,
    count: int = 8,
    assets: OfflineAssets | None = None,
) -> list[AttributionSample]:
    """Load a deterministic real-image support or candidate set on CPU.

    The aligned FunnyBirds/IDSDS models use their already-frozen strict support.
    DINOv2-g and PartImageNet return deterministic prepared candidates; the
    caller then applies the actual model-correctness gate and persists the eight
    selected IDs in the new run directory.
    """

    if count != 8:
        raise ValueError("single-B200 verification fixes exactly eight images")
    checkpoint_ids = MODEL_CHECKPOINTS[(model_id, dataset)]
    selected_assets = assets or resolve_offline_assets(
        checkpoint_ids=checkpoint_ids,
        datasets=(dataset,),
        sources=_required_sources(model_id, dataset),
        require_prepared=dataset in {"funnybirds", "partimagenet"},
        require_common_support=(
            dataset == "funnybirds"
            or (dataset == "imagenet1k_idsds" and model_id != "dinov2_vit_g_14")
        ),
    )
    cache_token = "support" if selected_assets.common_support else "candidates"
    cache_key = (dataset, model_id, count, cache_token)
    cached = _SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    if dataset == "imagenet1k_idsds":
        identifiers = None
        if selected_assets.common_support is not None:
            identifiers = _support_ids(selected_assets.common_support, dataset, model_id, count)
        manifest = selected_assets.manifests[dataset]
        result = _idsds_candidates(
            manifest,
            identifiers,
            limit=None if identifiers is not None else 256,
        )
    elif dataset == "funnybirds":
        if selected_assets.prepared_root is None or selected_assets.common_support is None:
            raise RuntimeError("FunnyBirds requires prepared inputs and strict support")
        identifiers = _support_ids(selected_assets.common_support, dataset, model_id, count)
        rows = _prepared_rows(selected_assets.prepared_root, dataset, subset="funnybirds")
        by_id = {str(row.image_id): row for row in rows.itertuples(index=False)}
        missing = [value for value in identifiers if value not in by_id]
        if missing:
            raise RuntimeError(f"prepared FunnyBirds support is missing IDs: {missing}")
        result = [_prepared_sample(by_id[value], dataset) for value in identifiers]
    elif dataset == "partimagenet":
        if selected_assets.prepared_root is None:
            raise RuntimeError("PartImageNet requires $DECAF_ATTRIBUTION_PREP_ROOT")
        rows = _prepared_rows(selected_assets.prepared_root, dataset, subset="compute")
        result = [_prepared_sample(row, dataset) for row in rows.head(64).itertuples(index=False)]
    else:
        raise KeyError(f"unsupported attribution dataset: {dataset}")
    _SAMPLE_CACHE[cache_key] = tuple(result)
    return list(result)


def _resize_crop(
    value: Any,
    resize_size: int,
    *,
    mode: str,
    crop_size: int | None = None,
) -> Any:
    torch = _torch()
    function = torch.nn.functional
    crop = resize_size if crop_size is None else crop_size
    original_ndim = value.ndim
    batch = value.unsqueeze(0) if original_ndim == 3 else value
    height, width = map(int, batch.shape[-2:])
    if height <= width:
        resized_height = resize_size
        resized_width = max(resize_size, int(resize_size * width / height))
    else:
        resized_width = resize_size
        resized_height = max(resize_size, int(resize_size * height / width))
    kwargs: dict[str, Any] = {"size": (resized_height, resized_width), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs.update({"align_corners": False, "antialias": True})
    resized = function.interpolate(batch.float(), **kwargs)
    top, left = (resized_height - crop) // 2, (resized_width - crop) // 2
    result = resized[..., top : top + crop, left : left + crop]
    if tuple(result.shape[-2:]) != (crop, crop):
        raise RuntimeError("resize/center-crop produced an invalid output shape")
    return result[0] if original_ndim == 3 else result


def _normalize_imagenet(value: Any) -> Any:
    mean = value.new_tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = value.new_tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    batch = value.unsqueeze(0) if value.ndim == 3 else value
    normalized = (batch - mean) / std
    return normalized[0] if value.ndim == 3 else normalized


def _normalize_idsds(value: Any, model_id: str) -> Any:
    """Apply the model-specific normalization frozen by the IDSDS loader."""

    if model_id != "vit_base_patch16_224":
        return _normalize_imagenet(value)
    batch = value.unsqueeze(0) if value.ndim == 3 else value
    normalized = (batch - 0.5) / 0.5
    return normalized[0] if value.ndim == 3 else normalized


def _validate_raw_rgb(value: Any) -> Any:
    """Validate the public raw-RGB tensor contract without changing values."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim not in {3, 4}
        or int(value.shape[-3]) != 3
        or not value.is_floating_point()
    ):
        raise ValueError("raw RGB input must be floating [3,H,W] or [B,3,H,W]")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("raw RGB input contains NaN or Inf")
    tolerance = 1.0e-6
    if bool((detached < -tolerance).any()) or bool((detached > 1.0 + tolerance).any()):
        minimum = float(detached.amin())
        maximum = float(detached.amax())
        raise ValueError(f"raw RGB input must lie in [0,1], observed [{minimum}, {maximum}]")
    return value


def _preprocess_raw_model_inputs(value: Any, model_id: str, dataset: str) -> Any:
    """Map raw RGB to the canonical model view inside every science forward."""

    torch = _torch()
    images = _validate_raw_rgb(value)
    if dataset == "partimagenet":
        return _normalize_imagenet(_resize_crop(images, 224, mode="bilinear"))
    if dataset != "funnybirds":
        raise ValueError(f"raw-RGB preprocessing is not registered for {dataset}")
    if model_id == "funnybirds_vit_b_16":
        batch = images.unsqueeze(0) if images.ndim == 3 else images
        transformed = torch.nn.functional.interpolate(batch, size=(224, 224), mode="nearest")
        return transformed[0] if images.ndim == 3 else transformed
    if tuple(images.shape[-2:]) != (256, 256):
        raise ValueError(f"{model_id} requires the official raw 256x256 FunnyBirds tensor")
    return images


def _raw_rgb_adapter_type() -> type[Any]:
    """Create the torch adapter lazily so CPU-only planning needs no torch import."""

    global _RAW_RGB_ADAPTER_TYPE
    if _RAW_RGB_ADAPTER_TYPE is not None:
        return _RAW_RGB_ADAPTER_TYPE
    torch = _torch()

    class RawRGBModelAdapter(torch.nn.Module):
        """Expose raw RGB while retaining a bare fixed-input model view."""

        def __init__(self, model: Any, model_id: str, dataset: str) -> None:
            super().__init__()
            if dataset not in {"funnybirds", "partimagenet"}:
                raise ValueError("RawRGBModelAdapter supports FunnyBirds/PartImageNet")
            self.model = model
            self.decaf_model_id = str(model_id)
            self.decaf_dataset = str(dataset)
            self.decaf_input_domain = RAW_RGB_INPUT_DOMAIN
            self.decaf_model_input_domain = MODEL_INPUT_DOMAIN
            self.raw_forward_calls = 0
            self.preprocessed_forward_calls = 0

        def preprocess(self, images: Any) -> Any:
            return _preprocess_raw_model_inputs(images, self.decaf_model_id, self.decaf_dataset)

        def forward(self, images: Any, *args: Any, **kwargs: Any) -> Any:
            self.raw_forward_calls += 1
            return self.model(self.preprocess(images), *args, **kwargs)

        def forward_preprocessed(self, images: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                not isinstance(images, torch.Tensor)
                or images.ndim != 4
                or int(images.shape[1]) != 3
                or not images.is_floating_point()
                or not bool(torch.isfinite(images.detach()).all())
            ):
                raise ValueError("canonical model input must be finite floating NCHW RGB")
            size = (
                224
                if (
                    self.decaf_dataset == "partimagenet"
                    or self.decaf_model_id == "funnybirds_vit_b_16"
                )
                else 256
            )
            if tuple(images.shape[-2:]) != (size, size):
                raise ValueError(f"canonical input for {self.decaf_model_id} must be {size}x{size}")
            self.preprocessed_forward_calls += 1
            return self.model(images, *args, **kwargs)

    _RAW_RGB_ADAPTER_TYPE = RawRGBModelAdapter
    return RawRGBModelAdapter


def _raw_rgb_model(model: Any, model_id: str, dataset: str) -> Any:
    """Wrap a bare checkpoint model for raw-domain attribution science."""

    return _raw_rgb_adapter_type()(model, model_id, dataset).eval()


def _runtime_model(model: Any, model_id: str, dataset: str) -> Any:
    if dataset in {"funnybirds", "partimagenet"}:
        return _raw_rgb_model(model, model_id, dataset)
    return model


def _idsds_pil_preprocess(value: Any, model_id: str) -> Any:
    """Apply the frozen PIL Resize(256)/CenterCrop(224) validation path."""

    torch = _torch()
    try:
        from PIL import Image
        from torchvision import transforms
    except ImportError as error:  # pragma: no cover - readiness owns dependencies
        raise RuntimeError("IDSDS preprocessing requires Pillow and torchvision") from error
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("IDSDS source image must be a CHW RGB tensor")
    source = value.detach().cpu().float()
    scaled = source * 255.0
    rounded = scaled.round()
    if (
        not bool(torch.isfinite(source).all())
        or bool((source < 0.0).any())
        or bool((source > 1.0).any())
        or not bool(torch.allclose(scaled, rounded, atol=1.0e-4, rtol=0.0))
    ):
        raise ValueError("IDSDS source tensor is not an exact decoded uint8 RGB image")
    array = rounded.clamp_(0.0, 255.0).to(dtype=torch.uint8).permute(1, 2, 0).contiguous().numpy()
    image = Image.fromarray(array, mode="RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    return _normalize_idsds(transform(image), model_id).contiguous()


def preprocess_sample(
    sample: AttributionSample | PreparedSample,
    model_id: str,
    *,
    dataset: str,
) -> PreparedSample:
    """Build the canonical fixed-shape model view for one sample.

    Fingerprinting and correctness use this view with the bare checkpoint
    model.  It is also the frozen FunnyBirds supplement domain.
    """

    torch = _torch()
    if sample.dataset != dataset:
        raise ValueError("sample dataset does not match preprocessing dataset")
    if dataset == "imagenet1k_idsds":
        image = _idsds_pil_preprocess(sample.image, model_id)
        masks = _idsds_masks(224, 224)
        reference = torch.zeros_like(image)
        interventions: dict[str, Any] = {}
    elif dataset == "partimagenet":
        image = _normalize_imagenet(_resize_crop(sample.image, 224, mode="bilinear"))
        masks = _resize_crop(sample.masks, 224, mode="nearest")
        reference = _normalize_imagenet(_resize_crop(sample.reference, 224, mode="bilinear"))
        interventions = {
            name: _normalize_imagenet(_resize_crop(value, 224, mode="bilinear"))
            for name, value in sample.interventions.items()
        }
    elif dataset == "funnybirds":
        if model_id == "funnybirds_vit_b_16":
            image = torch.nn.functional.interpolate(
                sample.image.unsqueeze(0), size=(224, 224), mode="nearest"
            )[0]
            masks = torch.nn.functional.interpolate(
                sample.masks.unsqueeze(1), size=(224, 224), mode="nearest"
            )[:, 0]
            reference = torch.nn.functional.interpolate(
                sample.reference.unsqueeze(0), size=(224, 224), mode="nearest"
            )[0]
            interventions = {
                name: torch.nn.functional.interpolate(value, size=(224, 224), mode="nearest")
                for name, value in sample.interventions.items()
            }
        else:
            if tuple(sample.image.shape[-2:]) != (256, 256):
                raise ValueError("FunnyBirds CNN inputs must be the official 256x256 tensors")
            image = sample.image
            masks = sample.masks
            reference = sample.reference
            interventions = dict(sample.interventions)
    else:
        raise KeyError(f"unsupported attribution dataset: {dataset}")
    height, width = map(int, image.shape[-2:])
    if (
        tuple(masks.shape[-2:]) != (height, width)
        or tuple(reference.shape) != tuple(image.shape)
        or any(tuple(value.shape[-3:]) != tuple(image.shape) for value in interventions.values())
        or int(masks.shape[0]) != len(sample.part_names)
    ):
        raise ValueError(f"preprocessed sample alignment drifted: {sample.image_id}")
    values = (image, masks, reference, *interventions.values())
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError(f"preprocessed sample contains NaN or Inf: {sample.image_id}")
    return PreparedSample(
        dataset=dataset,
        image_id=sample.image_id,
        target=sample.target,
        image=image.contiguous(),
        masks=masks.contiguous(),
        reference=reference.contiguous(),
        interventions={key: value.contiguous() for key, value in interventions.items()},
        part_names=sample.part_names,
        raw_height=sample.raw_height,
        raw_width=sample.raw_width,
    )


def _prepare_runtime_sample(
    sample: AttributionSample,
    model_id: str,
    *,
    dataset: str,
) -> PreparedSample:
    """Keep raw-domain science raw; prepare IDSDS/DINO model inputs once."""

    if dataset not in {"funnybirds", "partimagenet"}:
        return preprocess_sample(sample, model_id, dataset=dataset)
    if sample.dataset != dataset:
        raise ValueError("sample dataset does not match runtime dataset")
    image = _validate_raw_rgb(sample.image)
    reference = _validate_raw_rgb(sample.reference)
    interventions = {
        str(name): _validate_raw_rgb(value) for name, value in sample.interventions.items()
    }
    masks = _validate_masks(sample.masks, image)
    height, width = map(int, image.shape[-2:])
    if (
        tuple(reference.shape) != tuple(image.shape)
        or any(tuple(value.shape[-3:]) != tuple(image.shape) for value in interventions.values())
        or int(masks.shape[0]) != len(sample.part_names)
        or (height, width) != (sample.raw_height, sample.raw_width)
    ):
        raise ValueError(f"raw runtime sample alignment drifted: {sample.image_id}")
    # Do not resize, stack, or otherwise coalesce different source images here.
    return PreparedSample(
        dataset=dataset,
        image_id=sample.image_id,
        target=sample.target,
        image=image.contiguous(),
        masks=masks.contiguous(),
        reference=reference.contiguous(),
        interventions={key: value.contiguous() for key, value in interventions.items()},
        part_names=sample.part_names,
        raw_height=sample.raw_height,
        raw_width=sample.raw_width,
    )


def canonical_tensor_fingerprint(tensor: Any) -> dict[str, Any]:
    """Hash a canonical CPU-contiguous tensor including dtype and shape."""

    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("canonical tensor fingerprint requires a torch tensor")
    value = tensor.detach().cpu().contiguous()
    array = value.numpy()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    payload = memoryview(array).cast("B")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(payload)
    return {
        "sha256": digest.hexdigest(),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "byte_order": "little" if sys.byteorder == "little" else "big",
        "layout": "contiguous_c_order",
        "contiguous": True,
        "contiguous_bytes": int(array.nbytes),
    }


def _extract_logits(output: Any, *, classes: int) -> Any:
    torch = _torch()
    if isinstance(output, Mapping):
        output = output.get("logits")
    elif isinstance(output, (tuple, list)):
        output = output[0] if output else None
    elif hasattr(output, "logits"):
        output = output.logits
    if (
        not isinstance(output, torch.Tensor)
        or output.ndim != 2
        or int(output.shape[1]) != classes
        or not bool(torch.isfinite(output).all())
    ):
        raise ValueError(f"model must return finite [B,{classes}] logits")
    return output


def _classes(dataset: str) -> int:
    return 50 if dataset == "funnybirds" else 1_000


def _target_scores(model: Any, images: Any, target: int, dataset: str) -> Any:
    torch = _torch()
    logits = _extract_logits(model(images), classes=_classes(dataset))
    labels = torch.full(
        (int(logits.shape[0]),), int(target), dtype=torch.long, device=logits.device
    )
    return logits.gather(1, labels[:, None]).squeeze(1)


def _uses_raw_rgb_adapter(model: Any) -> bool:
    return getattr(model, "decaf_input_domain", None) == RAW_RGB_INPUT_DOMAIN


def _correct(model: Any, sample: PreparedSample, device: str, precision: str) -> bool:
    torch = _torch()
    forward = model
    value = sample
    if _uses_raw_rgb_adapter(model):
        value = preprocess_sample(
            sample,
            str(model.decaf_model_id),
            dataset=sample.dataset,
        )
        forward = model.forward_preprocessed
    image = value.image.unsqueeze(0).to(device=device, dtype=_precision(precision))
    with torch.inference_mode():
        logits = _extract_logits(forward(image), classes=_classes(sample.dataset))
    return int(logits.argmax(1).item()) == int(sample.target)


def _selection_path(context: RunContext, dataset: str, model_id: str) -> Path:
    safe = f"{dataset}--{model_id}".replace("/", "_")
    return context.path / "manifests/fixed_samples" / f"{safe}.json"


def _selected_samples(
    context: RunContext,
    dataset: str,
    model_id: str,
    model: Any,
    *,
    device: str,
    precision: str,
) -> list[PreparedSample]:
    path = _selection_path(context, dataset, model_id)
    candidates = load_fixed_samples(dataset, model_id, count=8)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        identifiers = payload.get("image_ids")
        if (
            payload.get("schema_version") != 1
            or payload.get("dataset") != dataset
            or payload.get("model_id") != model_id
            or not isinstance(identifiers, list)
            or len(identifiers) != 8
            or len(set(map(str, identifiers))) != 8
        ):
            raise RuntimeError(f"fixed sample manifest drifted: {path}")
        by_id = {sample.image_id: sample for sample in candidates}
        if not set(map(str, identifiers)).issubset(by_id):
            # DINO/Part candidates are deliberately wider than eight; reload if
            # a future candidate cache was narrowed before this resumed run.
            candidates = load_fixed_samples(dataset, model_id, count=8)
            by_id = {sample.image_id: sample for sample in candidates}
        missing = [value for value in map(str, identifiers) if value not in by_id]
        if missing:
            raise RuntimeError(f"fixed sample IDs are absent from offline assets: {missing}")
        prepared = [
            _prepare_runtime_sample(by_id[str(value)], model_id, dataset=dataset)
            for value in identifiers
        ]
        if not all(_correct(model, sample, device, precision) for sample in prepared):
            raise RuntimeError("a resumed fixed sample is no longer correctly classified")
        return prepared
    prepared: list[PreparedSample] = []
    for candidate in candidates:
        value = _prepare_runtime_sample(candidate, model_id, dataset=dataset)
        if _correct(model, value, device, precision):
            prepared.append(value)
        if len(prepared) == 8:
            break
    if len(prepared) != 8:
        raise RuntimeError(
            f"only {len(prepared)} correctly classified candidates found for {dataset}/{model_id}"
        )
    atomic_json(
        path,
        {
            "schema_version": 1,
            "dataset": dataset,
            "model_id": model_id,
            "image_ids": [sample.image_id for sample in prepared],
            "targets": [sample.target for sample in prepared],
            "correctly_classified": True,
            "correctness_input_domain": MODEL_INPUT_DOMAIN,
            "runtime_default_input_domain": (
                RAW_RGB_INPUT_DOMAIN
                if dataset in {"funnybirds", "partimagenet"}
                else MODEL_INPUT_DOMAIN
            ),
            "cross_image_coalescing": False,
            "selection": "first_eight_in_frozen_candidate_order",
        },
    )
    return prepared


def _device_contract() -> tuple[str, str]:
    torch = _torch()
    devices = os.environ.get("DECAF_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    values = tuple(part.strip() for part in devices.split(",") if part.strip())
    if len(values) != 1:
        raise RuntimeError("single-B200 attribution verification requires exactly one device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "single-B200 attribution verification requires exactly one visible CUDA device"
        )
    device = "cuda:0"
    name = str(torch.cuda.get_device_name(torch.device(device)))
    if "B200" not in name and os.environ.get("DECAF_ALLOW_NON_B200_TEST") != "1":
        raise RuntimeError(f"expected NVIDIA B200, observed {name}")
    return device, name


def _validate_masks(masks: Any, image: Any) -> Any:
    torch = _torch()
    if (
        masks.ndim != 3
        or tuple(masks.shape[-2:]) != tuple(image.shape[-2:])
        or int(masks.shape[0]) < 1
        or int(masks.shape[0]) > 16
    ):
        raise ValueError("feature masks must be [K,H,W] with K in [1,16]")
    value = masks.to(device=image.device, dtype=image.dtype)
    if (
        not bool(torch.isfinite(value).all())
        or bool((value < 0).any())
        or bool((value > 1).any())
        or bool((value.flatten(1).sum(1) <= 0).any())
        or bool((value.sum(0) > 1.0 + 1.0e-6).any())
    ):
        raise ValueError("feature masks must be finite, non-empty, disjoint indicators")
    return value


def _aggregate_dense(dense: Any, masks: Any) -> Any:
    value = dense[0] if dense.ndim == 4 and int(dense.shape[0]) == 1 else dense
    if value.ndim == 3:
        value = value.sum(0)
    if value.ndim != 2:
        raise ValueError("dense attribution must reduce to one HxW map")
    return (masks * value.unsqueeze(0)).flatten(1).sum(1)


def _endpoint(model: Any, image: Any, baseline: Any, masks: Any, target: int, dataset: str) -> Any:
    torch = _torch()
    groups = _validate_masks(masks, image)
    x = image.unsqueeze(0)
    b = baseline.unsqueeze(0)
    removed = b + (1.0 - groups[:, None]) * (x - b)
    with torch.no_grad():
        factual = _target_scores(model, x, target, dataset)[0]
        counterfactual = _target_scores(model, removed, target, dataset)
    return factual - counterfactual


def _heldout_effects(
    model: Any,
    image: Any,
    variants: Any,
    target: int,
    dataset: str,
) -> Any:
    torch = _torch()
    with torch.no_grad():
        factual = _target_scores(model, image.unsqueeze(0), target, dataset)[0]
        values = _target_scores(model, variants, target, dataset)
    return factual - values


def _decaf(
    method_id: str,
    model: Any,
    image: Any,
    baseline: Any,
    masks: Any,
    target: int,
    dataset: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    groups = _validate_masks(masks, image)
    stages = int(method_id.rsplit("_", 1)[1])
    grid = torch.linspace(0.0, 1.0, stages, device=image.device, dtype=image.dtype)
    x, b = image.unsqueeze(0), baseline.unsqueeze(0)
    residual = x - b
    factual_images = b + grid[:, None, None, None] * residual
    with torch.no_grad():
        factual = _target_scores(model, factual_images, target, dataset).double()
        counterfactual: list[Any] = []
        keep = 1.0 - groups[:, None]
        for stage in grid:
            variants = b + stage * keep * residual
            counterfactual.append(_target_scores(model, variants, target, dataset).double())
    response = factual.unsqueeze(0) - torch.stack(counterfactual, dim=1)
    scores = decaf_trajectory(
        method_id,
        grid.detach().cpu().double().numpy(),
        response.detach().cpu().double().numpy(),
        axis=1,
    )
    if not scores["numeric_audit"]["passed"]:
        raise AssertionError("real-GPU DECAF trajectory failed its numeric audit")
    part_scores = torch.as_tensor(scores["E"], device=image.device, dtype=torch.float64)
    metadata = {
        "M": np.asarray(scores["M"], dtype=np.float64),
        "E": np.asarray(scores["E"], dtype=np.float64),
        "C": np.asarray(scores["C"], dtype=np.float64),
        "F": np.asarray(scores["F"], dtype=np.float64),
        "Abs": np.asarray(scores["Abs"], dtype=np.float64),
        "numeric_audit_passed": True,
        "forward_rows": stages * (1 + int(groups.shape[0])),
        "backward_calls": 0,
    }
    return part_scores, metadata


def _fixed_uniform_baseline(image: Any) -> Any:
    """Return the frozen, resolution-shared U[-1,1) IG-U baseline."""

    torch = _torch()
    rng = np.random.RandomState(UNIFORM_BASELINE_SEED)
    value = rng.uniform(-1.0, 1.0, size=tuple(image.shape)).astype(np.float32)
    return torch.from_numpy(value).to(device=image.device, dtype=image.dtype)


def _gauss_legendre_rule(steps: int, image: Any) -> tuple[Any, Any]:
    """Map NumPy's frozen Gauss-Legendre rule from [-1,1] onto [0,1]."""

    torch = _torch()
    count = int(steps)
    if count <= 0:
        raise ValueError("IG steps must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    alphas = torch.as_tensor(0.5 * (nodes + 1.0), device=image.device, dtype=image.dtype)
    quadrature_weights = torch.as_tensor(0.5 * weights, device=image.device, dtype=image.dtype)
    return alphas, quadrature_weights


def _integrated_gradients(
    model: Any,
    image: Any,
    baseline: Any,
    masks: Any,
    target: int,
    dataset: str,
    *,
    steps: int,
    quadrature: str = "gauss_legendre",
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    groups = _validate_masks(masks, image)
    x, b = image.unsqueeze(0), baseline.unsqueeze(0)
    if quadrature == "gauss_legendre":
        alphas, quadrature_weights = _gauss_legendre_rule(steps, image)
    elif quadrature == "endpoint_trapezoid":
        if int(steps) < 2:
            raise ValueError("trapezoid IG requires at least two steps")
        alphas = torch.linspace(0.0, 1.0, steps, device=image.device, dtype=image.dtype)
        quadrature_weights = None
    else:
        raise ValueError(f"unsupported IG quadrature: {quadrature}")
    gradients: list[Any] = []
    internal = 8
    for start in range(0, steps, internal):
        local = alphas[start : start + internal]
        points = (b + local[:, None, None, None] * (x - b)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = _target_scores(model, points, target, dataset).sum()
        gradients.extend(torch.autograd.grad(score, points)[0].detach().unbind(0))
    stacked = torch.stack(gradients, dim=0)
    if quadrature_weights is None:
        average = torch.trapezoid(stacked, x=alphas, dim=0)
        weight_sum = 1.0
    else:
        average = torch.einsum("s,schw->chw", quadrature_weights, stacked)
        weight_sum = float(quadrature_weights.double().sum().cpu())
    dense = (x - b) * average.unsqueeze(0)
    return _aggregate_dense(dense, groups), {
        "steps": steps,
        "quadrature": quadrature,
        "quadrature_weight_sum": weight_sum,
        "forward_rows": steps,
        "backward_calls": math.ceil(steps / internal),
    }


def _smoothgrad(
    model: Any,
    image: Any,
    masks: Any,
    target: int,
    dataset: str,
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    groups = _validate_masks(masks, image)
    if dataset == "imagenet1k_idsds":
        effective_seed = SMOOTHGRAD_SEED
        rng = np.random.RandomState(effective_seed)
        # The leading singleton is the frozen shared-image axis.  Constructing
        # this bank independently for each call therefore gives every image and
        # every shard the same sixteen normalized-space perturbations.
        noise_array = rng.standard_normal(
            size=(ATTRIBUTION_MONTE_CARLO_SAMPLES, 1, *tuple(image.shape))
        ).astype(np.float32)
        noise = torch.from_numpy(noise_array[:, 0]).to(device=image.device, dtype=image.dtype)
        noisy = image.unsqueeze(0) + 0.15 * noise
        random_bank = "shared_numpy_randomstate"
        clipping = "none"
    else:
        effective_seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(effective_seed)
        noise = torch.randn(
            (ATTRIBUTION_MONTE_CARLO_SAMPLES, *image.shape), generator=generator
        ).to(image)
        # FunnyBirds/PartImageNet perturb and clamp raw RGB.  The raw-model
        # adapter performs geometry and normalization inside the forward, so
        # no normalized-space values are clipped here.
        noisy = (image.unsqueeze(0) + 0.15 * noise).clamp(0.0, 1.0)
        random_bank = "per_image_torch_generator"
        clipping = "[0,1]"
    noisy = noisy.detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    value = _target_scores(model, noisy, target, dataset).sum()
    gradient = torch.autograd.grad(value, noisy)[0].mean(0, keepdim=True).detach()
    return _aggregate_dense(gradient, groups), {
        "samples": ATTRIBUTION_MONTE_CARLO_SAMPLES,
        "seed": effective_seed,
        "noise_std": 0.15,
        "noise_space": (
            MODEL_INPUT_DOMAIN if dataset == "imagenet1k_idsds" else RAW_RGB_INPUT_DOMAIN
        ),
        "random_bank": random_bank,
        "clipping": clipping,
        "forward_rows": ATTRIBUTION_MONTE_CARLO_SAMPLES,
        "backward_calls": 1,
    }


def _make_deeplift_compatible(model: Any) -> None:
    torch = _torch()
    if getattr(model, "_decaf_deeplift_compatible", False):
        return
    for module in tuple(model.modules()):
        if (
            hasattr(module, "relu")
            and hasattr(module, "conv1")
            and hasattr(module, "conv2")
            and callable(getattr(module, "forward", None))
        ):
            if hasattr(module, "conv3"):
                module.relu1 = torch.nn.ReLU(inplace=False)
                module.relu2 = torch.nn.ReLU(inplace=False)
                module.relu3 = torch.nn.ReLU(inplace=False)

                def bottleneck_forward(current: Any, value: Any) -> Any:
                    identity = value
                    output = current.relu1(current.bn1(current.conv1(value)))
                    output = current.relu2(current.bn2(current.conv2(output)))
                    output = current.bn3(current.conv3(output))
                    if current.downsample is not None:
                        identity = current.downsample(value)
                    return current.relu3(output + identity)

                module.forward = types.MethodType(bottleneck_forward, module)
            elif hasattr(module, "bn1") and hasattr(module, "bn2"):
                module.relu1 = torch.nn.ReLU(inplace=False)
                module.relu2 = torch.nn.ReLU(inplace=False)

                def basic_forward(current: Any, value: Any) -> Any:
                    identity = value
                    output = current.relu1(current.bn1(current.conv1(value)))
                    output = current.bn2(current.conv2(output))
                    if current.downsample is not None:
                        identity = current.downsample(value)
                    return current.relu2(output + identity)

                module.forward = types.MethodType(basic_forward, module)
            module.relu = torch.nn.Identity()
        elif isinstance(module, torch.nn.ReLU):
            module.inplace = False
    model._decaf_deeplift_compatible = True


def _deep_lift(
    model: Any,
    image: Any,
    baseline: Any,
    masks: Any,
    target: int,
) -> tuple[Any, dict[str, Any]]:
    try:
        from captum.attr import DeepLift
    except ImportError as error:  # pragma: no cover - readiness owns dependency audit
        raise RuntimeError("DeepLIFT verification requires captum") from error
    groups = _validate_masks(masks, image)
    _make_deeplift_compatible(model)
    dense = (
        DeepLift(model)
        .attribute(image.unsqueeze(0), baselines=baseline.unsqueeze(0), target=int(target))
        .detach()
    )
    return _aggregate_dense(dense, groups), {
        "forward_rows": 2,
        "backward_calls": 1,
    }


def _gradient_shap(
    model: Any,
    image: Any,
    baseline: Any,
    masks: Any,
    target: int,
    dataset: str,
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    groups = _validate_masks(masks, image)
    if dataset == "imagenet1k_idsds":
        effective_seed = GRADIENT_SHAP_SEED
        rng = np.random.RandomState(effective_seed)
        alphas = torch.from_numpy(
            rng.uniform(0.0, 1.0, size=ATTRIBUTION_MONTE_CARLO_SAMPLES).astype(np.float32)
        ).to(device=image.device, dtype=image.dtype)
        difference = image - baseline
        points = (
            (baseline.unsqueeze(0) + alphas[:, None, None, None] * difference.unsqueeze(0))
            .detach()
            .requires_grad_(True)
        )
        model.zero_grad(set_to_none=True)
        score = _target_scores(model, points, target, dataset).sum()
        gradients = torch.autograd.grad(score, points)[0].detach()
        dense = difference.unsqueeze(0) * gradients.mean(0, keepdim=True)
        return _aggregate_dense(dense, groups), {
            "samples": ATTRIBUTION_MONTE_CARLO_SAMPLES,
            "seed": effective_seed,
            "stdev": 0.0,
            "baseline": "normalized_space_zero",
            "random_bank": "shared_numpy_randomstate",
            "forward_rows": ATTRIBUTION_MONTE_CARLO_SAMPLES,
            "backward_calls": 1,
        }
    try:
        from captum.attr import GradientShap
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("GradientSHAP verification requires captum") from error
    baseline_distribution = baseline.unsqueeze(0).expand(2, -1, -1, -1).contiguous()
    fork_devices = [image.device.index or 0] if image.device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        dense = (
            GradientShap(model)
            .attribute(
                image.unsqueeze(0),
                baselines=baseline_distribution,
                target=int(target),
                n_samples=ATTRIBUTION_MONTE_CARLO_SAMPLES,
                stdevs=0.0,
            )
            .detach()
        )
    return _aggregate_dense(dense, groups), {
        "samples": ATTRIBUTION_MONTE_CARLO_SAMPLES,
        "seed": int(seed),
        "stdev": 0.0,
        "baseline": "two_identical_locked_references",
        "random_bank": "per_image_torch_generator",
        "forward_rows": ATTRIBUTION_MONTE_CARLO_SAMPLES,
        "backward_calls": ATTRIBUTION_MONTE_CARLO_SAMPLES,
    }


def _idsds_rise_masks(height: int, width: int, image: Any) -> Any:
    """Build the pinned NumPy/skimage RISE bank used by IDSDS."""

    torch = _torch()
    from skimage.transform import resize

    number = ATTRIBUTION_PERTURBATION_BUDGET
    grid_size = 8
    probability = 0.1
    rng = np.random.RandomState(RISE_MASK_SEED)
    grid = (rng.rand(number, grid_size, grid_size) < probability).astype(np.float32)
    cell_height = int(math.ceil(int(height) / grid_size))
    cell_width = int(math.ceil(int(width) / grid_size))
    up_height = (grid_size + 1) * cell_height
    up_width = (grid_size + 1) * cell_width
    masks = np.empty((number, 1, int(height), int(width)), dtype=np.float32)
    for index in range(number):
        shift_height = int(rng.randint(0, cell_height))
        shift_width = int(rng.randint(0, cell_width))
        upsampled = resize(
            grid[index],
            (up_height, up_width),
            order=1,
            mode="reflect",
            anti_aliasing=False,
        )
        masks[index, 0] = upsampled[
            shift_height : shift_height + int(height),
            shift_width : shift_width + int(width),
        ]
    return torch.from_numpy(masks).to(device=image.device, dtype=image.dtype)


def _native_rise_masks(
    height: int,
    width: int,
    image: Any,
    *,
    seed: int,
) -> Any:
    """Build the locked FunnyBirds/PartImageNet shifted-crop RISE bank."""

    torch = _torch()
    number = ATTRIBUTION_PERTURBATION_BUDGET
    grid_size = 7
    probability = 0.5
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    coarse = (
        torch.rand((number, 1, grid_size, grid_size), generator=generator) < probability
    ).float()
    cell_height = int(math.ceil(int(height) / grid_size))
    cell_width = int(math.ceil(int(width) / grid_size))
    upsampled = torch.nn.functional.interpolate(
        coarse,
        size=((grid_size + 1) * cell_height, (grid_size + 1) * cell_width),
        mode="bilinear",
        align_corners=False,
    )
    shifts_height = torch.randint(0, cell_height, (number,), generator=generator)
    shifts_width = torch.randint(0, cell_width, (number,), generator=generator)
    masks = torch.empty((number, 1, int(height), int(width)), dtype=torch.float32)
    for index, (shift_height, shift_width) in enumerate(
        zip(shifts_height.tolist(), shifts_width.tolist(), strict=True)
    ):
        masks[index] = upsampled[
            index,
            :,
            shift_height : shift_height + int(height),
            shift_width : shift_width + int(width),
        ]
    return masks.to(device=image.device, dtype=image.dtype)


def _target_probabilities(model: Any, images: Any, target: int, dataset: str) -> Any:
    torch = _torch()
    logits = _extract_logits(model(images), classes=_classes(dataset))
    labels = torch.full(
        (int(logits.shape[0]),), int(target), dtype=torch.long, device=logits.device
    )
    return logits.softmax(dim=1).gather(1, labels[:, None]).squeeze(1)


def _rise(
    model: Any,
    image: Any,
    baseline: Any,
    masks: Any,
    target: int,
    dataset: str,
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    groups = _validate_masks(masks, image)
    height, width = map(int, image.shape[-2:])
    if dataset == "imagenet1k_idsds":
        effective_seed = RISE_MASK_SEED
        grid_size = 8
        probability = 0.1
        random_masks = _idsds_rise_masks(height, width, image)
        score_function = _target_probabilities
        mask_generation = "pinned_idsds_skimage_resize_order1_reflect_random_shift_crop"
        random_bank = "shared_numpy_randomstate"
        score_space = "ground_truth_class_probability"
    else:
        effective_seed = int(seed)
        grid_size = 7
        probability = 0.5
        random_masks = _native_rise_masks(height, width, image, seed=effective_seed)
        score_function = _target_scores
        mask_generation = "native_torch_bilinear_random_shift_crop"
        random_bank = "per_image_torch_generator"
        score_space = "ground_truth_class_logit"
    variants = baseline.unsqueeze(0) + random_masks * (image.unsqueeze(0) - baseline.unsqueeze(0))
    values: list[Any] = []
    with torch.no_grad():
        for start in range(0, ATTRIBUTION_PERTURBATION_BUDGET, 64):
            values.append(score_function(model, variants[start : start + 64], target, dataset))
    scores = torch.cat(values)
    dense = (scores[:, None, None, None] * random_masks).mean(0, keepdim=True) / probability
    return _aggregate_dense(dense, groups), {
        "queries": ATTRIBUTION_PERTURBATION_BUDGET,
        "seed": effective_seed,
        "mask_grid": [grid_size, grid_size],
        "keep_probability": probability,
        "mask_generation": mask_generation,
        "random_bank": random_bank,
        "score_space": score_space,
        "forward_rows": ATTRIBUTION_PERTURBATION_BUDGET,
        "backward_calls": 0,
    }


def _coalitions(groups: int, queries: int, seed: int, device: Any, dtype: Any) -> Any:
    torch = _torch()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if (1 << groups) <= queries:
        values = torch.arange(1 << groups, dtype=torch.long)
        exhaustive = ((values[:, None] >> torch.arange(groups)) & 1).float()
        ordered = torch.cat((exhaustive[:1], exhaustive[-1:], exhaustive[1:-1]))
        repeats = queries - int(ordered.shape[0])
        if repeats:
            indices = torch.randint(0, int(ordered.shape[0]), (repeats,), generator=generator)
            ordered = torch.cat((ordered, ordered[indices]))
    else:
        middle = torch.randint(0, 2, (queries - 2, groups), generator=generator).float()
        ordered = torch.cat((torch.zeros((1, groups)), torch.ones((1, groups)), middle))
    return ordered.to(device=device, dtype=dtype)


def _kernel_shap_coalitions(
    groups: int,
    queries: int,
    seed: int,
    device: Any,
    dtype: Any,
) -> Any:
    """Return the endpoint-first, subset/complement-balanced frozen design."""

    torch = _torch()
    count = int(groups)
    budget = int(queries)
    if count <= 0 or budget < 2:
        raise ValueError("KernelSHAP requires positive groups and at least two queries")
    rng = np.random.RandomState(int(seed))
    rows: list[np.ndarray] = [
        np.zeros(count, dtype=np.float32),
        np.ones(count, dtype=np.float32),
    ]
    half = count // 2
    for size in range(1, half + 1):
        if len(rows) >= budget:
            break
        layer = list(combinations(range(count), size))
        if size < count - size:
            remaining = budget - len(rows)
            selected_count = min(len(layer), remaining // 2)
            selected_indices = (
                rng.choice(len(layer), size=selected_count, replace=False).tolist()
                if selected_count < len(layer)
                else list(range(len(layer)))
            )
            for index in selected_indices:
                coalition = np.zeros(count, dtype=np.float32)
                coalition[list(layer[index])] = 1.0
                rows.append(coalition)
                if len(rows) < budget:
                    rows.append(1.0 - coalition)
            if len(rows) < budget and selected_count < len(layer):
                unused = np.setdiff1d(
                    np.arange(len(layer)),
                    np.asarray(selected_indices),
                    assume_unique=False,
                )
                index = int(unused[int(rng.randint(0, len(unused)))])
                coalition = np.zeros(count, dtype=np.float32)
                coalition[list(layer[index])] = 1.0
                rows.append(coalition)
            if selected_count < len(layer) or remaining < 2 * len(layer):
                break
        else:
            remaining = budget - len(rows)
            selected_count = min(len(layer), remaining)
            selected_indices = rng.choice(len(layer), size=selected_count, replace=False).tolist()
            for index in selected_indices:
                coalition = np.zeros(count, dtype=np.float32)
                coalition[list(layer[index])] = 1.0
                rows.append(coalition)
    while len(rows) < budget:
        rows.append(rows[int(rng.randint(0, len(rows)))].copy())
    result = torch.from_numpy(np.stack(rows[:budget], axis=0))
    return result.to(device=device, dtype=dtype)


def _coalition_images(image: Any, baseline: Any, masks: Any, coalitions: Any) -> Any:
    torch = _torch()
    present = torch.einsum("nk,khw->nhw", coalitions, masks).clamp_(0.0, 1.0)
    foreground = masks.amax(0).clamp_(0.0, 1.0)
    context = baseline.unsqueeze(0) + (1.0 - foreground)[None, None] * (
        image.unsqueeze(0) - baseline.unsqueeze(0)
    )
    return context + present[:, None] * (image.unsqueeze(0) - baseline.unsqueeze(0))


def _coalition_values(
    model: Any,
    image: Any,
    baseline: Any,
    masks: Any,
    target: int,
    dataset: str,
    coalitions: Any,
) -> Any:
    torch = _torch()
    images = _coalition_images(image, baseline, masks, coalitions)
    values: list[Any] = []
    with torch.no_grad():
        for start in range(0, int(images.shape[0]), 64):
            values.append(_target_scores(model, images[start : start + 64], target, dataset))
    return torch.cat(values)


def _kernel_shap(
    model: Any,
    image: Any,
    baseline: Any,
    masks: Any,
    target: int,
    dataset: str,
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    groups = _validate_masks(masks, image)
    count = int(groups.shape[0])
    idsds_contract = dataset == "imagenet1k_idsds"
    effective_seed = KERNEL_SHAP_SEED if idsds_contract else int(seed)
    z = (
        _kernel_shap_coalitions(
            count,
            ATTRIBUTION_PERTURBATION_BUDGET,
            effective_seed,
            image.device,
            image.dtype,
        )
        if idsds_contract
        else _coalitions(
            count,
            ATTRIBUTION_PERTURBATION_BUDGET,
            effective_seed,
            image.device,
            image.dtype,
        )
    )
    y = _coalition_values(model, image, baseline, groups, target, dataset, z)
    sizes = z.sum(1).long()
    weights = torch.empty_like(y)
    endpoint = (sizes == 0) | (sizes == count)
    weights[endpoint] = 1.0e6
    for size in range(1, count):
        selected = sizes == size
        weights[selected] = (count - 1) / (math.comb(count, size) * size * (count - size))
    design = torch.cat(
        (
            torch.ones(
                (ATTRIBUTION_PERTURBATION_BUDGET, 1),
                device=z.device,
                dtype=z.dtype,
            ),
            z,
        ),
        dim=1,
    )
    root = weights.clamp_min(0).sqrt()[:, None]
    weighted_design = (design * root).double()
    weighted_y = (y * root[:, 0]).double()
    ridge = torch.eye(count + 1, device=z.device, dtype=torch.float64) * 1.0e-8
    ridge[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        weighted_design.T @ weighted_design + ridge,
        weighted_design.T @ weighted_y,
    )[1:]
    total_effect = y[1].double() - y[0].double()
    pre_correction_residual = total_effect - coefficients.sum()
    if idsds_contract:
        # The finite endpoint weight makes the unconstrained solve only
        # approximately complete.  The frozen IDSDS contract explicitly
        # enforces local accuracy after solving; the locked FunnyBirds/Part
        # worker preserves the raw ridge coefficients instead.
        coefficients = coefficients + pre_correction_residual / count
    local_accuracy_residual = coefficients.sum() - total_effect
    return coefficients, {
        "queries": ATTRIBUTION_PERTURBATION_BUDGET,
        "unique_coalitions": int(torch.unique(z, dim=0).shape[0]),
        "seed": effective_seed,
        "coalition_design": (
            "endpoint_first_subset_size_complement_balanced"
            if idsds_contract
            else "endpoint_first_exhaustive_then_torch_repeats"
        ),
        "local_accuracy_correction": idsds_contract,
        "reference_game": "deletion_game",
        "pre_correction_abs_residual": float(pre_correction_residual.abs().cpu()),
        "local_accuracy_abs_residual": float(local_accuracy_residual.abs().cpu()),
        "forward_rows": ATTRIBUTION_PERTURBATION_BUDGET,
        "backward_calls": 0,
    }


def _part_occlusion(
    model: Any,
    image: Any,
    baseline: Any,
    masks: Any,
    target: int,
    dataset: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    groups = _validate_masks(masks, image)
    count = int(groups.shape[0])
    coalitions = torch.ones((count, count), device=image.device, dtype=image.dtype)
    coalitions.fill_diagonal_(0.0)
    with torch.no_grad():
        factual = _target_scores(model, image.unsqueeze(0), target, dataset)[0]
    removed = _coalition_values(model, image, baseline, groups, target, dataset, coalitions)
    return factual - removed, {
        "forward_rows": count + 1,
        "backward_calls": 0,
    }


def _exact_part_shapley(
    model: Any,
    image: Any,
    baseline: Any,
    masks: Any,
    target: int,
    dataset: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    groups = _validate_masks(masks, image)
    count = int(groups.shape[0])
    if count > 8:
        raise ValueError("Exact Part-Shapley is permitted only for at most eight groups")
    query_count = 1 << count
    z = _coalitions(count, query_count, 0, image.device, image.dtype)
    values = _coalition_values(model, image, baseline, groups, target, dataset, z)
    index = (z.long() * (2 ** torch.arange(count, device=z.device))).sum(1)
    ordered = torch.empty_like(values)
    ordered[index] = values
    scores = torch.zeros(count, device=image.device, dtype=values.dtype)
    for feature in range(count):
        bit = 1 << feature
        for subset in range(1 << count):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            scores[feature] += weight * (ordered[subset | bit] - ordered[subset])
    return scores, {
        "queries": query_count,
        "groups": count,
        "forward_rows": query_count,
        "backward_calls": 0,
    }


def _method_runtime_view(
    method_id: str,
    model: Any,
    sample: PreparedSample,
) -> tuple[Any, PreparedSample, str, bool]:
    """Select the registered, method-specific input domain."""

    if sample.dataset == "funnybirds" and method_id in FUNNYBIRDS_SUPPLEMENT_METHODS:
        if _uses_raw_rgb_adapter(model):
            prepared = preprocess_sample(
                sample,
                str(model.decaf_model_id),
                dataset=sample.dataset,
            )
            return (
                model.model,
                prepared,
                FUNNYBIRDS_SUPPLEMENT_INPUT_DOMAIN,
                False,
            )
        return model, sample, FUNNYBIRDS_SUPPLEMENT_INPUT_DOMAIN, False
    if _uses_raw_rgb_adapter(model):
        return model, sample, RAW_RGB_INPUT_DOMAIN, True
    return model, sample, MODEL_INPUT_DOMAIN, False


def _method(
    method_id: str,
    model: Any,
    sample: PreparedSample,
    *,
    device: str,
    precision: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run only the requested attribution kernel, without quality targets."""

    science_model, science_sample, input_domain, preprocess_inside = _method_runtime_view(
        method_id, model, sample
    )
    dtype = _precision(precision)
    image = science_sample.image.to(device=device, dtype=dtype)
    masks = science_sample.masks.to(device=device, dtype=dtype)
    reference = science_sample.reference.to(device=device, dtype=dtype)
    if method_id.startswith("decaf_"):
        scores, metadata = _decaf(
            method_id,
            science_model,
            image,
            reference,
            masks,
            sample.target,
            sample.dataset,
        )
    elif method_id in {"ig_16", "ig_32", "ig_u_32"}:
        steps = int(method_id.split("_")[1]) if method_id != "ig_u_32" else 32
        baseline = reference
        if method_id == "ig_u_32":
            baseline = _fixed_uniform_baseline(image)
        scores, metadata = _integrated_gradients(
            science_model,
            image,
            baseline,
            masks,
            sample.target,
            sample.dataset,
            steps=steps,
            quadrature=(
                "gauss_legendre"
                if sample.dataset == "imagenet1k_idsds" or method_id == "ig_u_32"
                else "endpoint_trapezoid"
            ),
        )
        metadata["baseline"] = (
            "fixed_uniform_U[-1,1)"
            if method_id == "ig_u_32"
            else (
                "normalized_space_zero"
                if sample.dataset == "imagenet1k_idsds"
                else "locked_gaussian_blur_k31_sigma12"
            )
        )
        if method_id == "ig_u_32":
            metadata["baseline_seed"] = UNIFORM_BASELINE_SEED
            metadata["baseline_bank"] = "shared_numpy_randomstate"
    elif method_id == "smoothgrad_16":
        scores, metadata = _smoothgrad(
            science_model, image, masks, sample.target, sample.dataset, seed=seed
        )
    elif method_id == "deep_lift":
        scores, metadata = _deep_lift(science_model, image, reference, masks, sample.target)
    elif method_id == "gradient_shap":
        scores, metadata = _gradient_shap(
            science_model,
            image,
            reference,
            masks,
            sample.target,
            sample.dataset,
            seed=seed,
        )
    elif method_id == "rise_512":
        scores, metadata = _rise(
            science_model,
            image,
            reference,
            masks,
            sample.target,
            sample.dataset,
            seed=seed,
        )
    elif method_id == "kernel_shap_512":
        scores, metadata = _kernel_shap(
            science_model,
            image,
            reference,
            masks,
            sample.target,
            sample.dataset,
            seed=seed,
        )
    elif method_id == "part_occlusion":
        scores, metadata = _part_occlusion(
            science_model, image, reference, masks, sample.target, sample.dataset
        )
    elif method_id == "exact_part_shapley":
        scores, metadata = _exact_part_shapley(
            science_model, image, reference, masks, sample.target, sample.dataset
        )
    else:
        raise KeyError(f"single-B200 runtime does not implement method {method_id!r}")
    values = scores.detach().cpu().double().numpy().reshape(-1)
    if values.shape != (int(masks.shape[0]),) or not np.isfinite(values).all():
        raise RuntimeError(f"attribution shape/finite audit failed for {method_id}")
    metadata.setdefault("numeric_audit_passed", True)
    metadata.update(
        {
            "input_domain": input_domain,
            "perturbation_domain": input_domain,
            "masking_domain": input_domain,
            "baseline_domain": input_domain,
            "model_preprocess_inside_forward": preprocess_inside,
            "cross_image_coalescing": False,
            "preprocess_before_cross_image_coalescing": True,
        }
    )
    return values, metadata


def _target_frame(
    job: Mapping[str, Any],
    samples: Sequence[PreparedSample],
    model: Any,
    *,
    device: str,
    precision: str,
) -> pd.DataFrame:
    method_id = str(job["method_id"])
    input_domain = RAW_RGB_INPUT_DOMAIN if _uses_raw_rgb_adapter(model) else MODEL_INPUT_DOMAIN
    preprocess_inside = _uses_raw_rgb_adapter(model)
    rows: list[dict[str, Any]] = []
    for offset, sample in enumerate(samples):
        image = sample.image.to(device=device, dtype=_precision(precision))
        masks = sample.masks.to(device=device, dtype=_precision(precision))
        reference = sample.reference.to(device=device, dtype=_precision(precision))
        operator = "endpoint_part_deletion"
        if method_id in {DELETION_TARGET_METHOD, FUNNYBIRDS_DELETION_TARGET_METHOD}:
            effects = _endpoint(model, image, reference, masks, sample.target, sample.dataset)
        elif method_id == "__heldout_background_texture__":
            operator = "background_texture"
            variants = sample.interventions["background_texture"].to(
                device=device, dtype=_precision(precision)
            )
            effects = _heldout_effects(model, image, variants, sample.target, sample.dataset)
        elif method_id == "__heldout_telea_dilate3__":
            operator = "telea_dilate3"
            variants = sample.interventions["telea"].to(device=device, dtype=_precision(precision))
            effects = _heldout_effects(model, image, variants, sample.target, sample.dataset)
        else:
            raise KeyError(f"unknown attribution target member: {method_id}")
        values = effects.detach().cpu().double().numpy().reshape(-1)
        if len(values) != len(sample.part_names) or not np.isfinite(values).all():
            raise RuntimeError(f"held-out target schema drifted for {sample.image_id}")
        rows.append(
            {
                "image_index": int(job["image_start"]) + offset,
                "image_id": sample.image_id,
                "scope": job["scope"],
                "dataset": job["dataset"],
                "model": job["model_id"],
                "method": method_id,
                "target_class": sample.target,
                "target_effects": values,
                "part_names": np.asarray(sample.part_names, dtype=str),
                "intervention_operator": operator,
                "reference": (
                    "normalized_zero"
                    if sample.dataset == "imagenet1k_idsds"
                    else "locked_gaussian_blur_k31_sigma12_raw_rgb"
                ),
                "input_domain": input_domain,
                "perturbation_domain": input_domain,
                "masking_domain": input_domain,
                "baseline_domain": input_domain,
                "model_preprocess_inside_forward": preprocess_inside,
                "cross_image_coalescing": False,
                "preprocess_before_cross_image_coalescing": True,
                "heldout_schema_version": 1,
                "raw_image_height": sample.raw_height,
                "raw_image_width": sample.raw_width,
                "preprocess_before_coalescing": True,
                "mask_transform": "aligned_nearest_neighbor",
            }
        )
    return pd.DataFrame(rows)


def _quality_frame(
    job: Mapping[str, Any],
    samples: Sequence[PreparedSample],
    model: Any,
    *,
    device: str,
    precision: str,
) -> pd.DataFrame:
    method_id = str(job["method_id"])
    rows: list[dict[str, Any]] = []
    for offset, sample in enumerate(samples):
        seed = _stable_method_seed(str(job["member_id"]), sample.image_id)
        patch_scores, metadata = _method(
            method_id,
            model,
            sample,
            device=device,
            precision=precision,
            seed=seed,
        )
        if job.get("depends_on"):
            # The dependency binder replaces these placeholders with the exact
            # shared deletion/held-out targets before validation and writing.
            # Recomputing them here would contaminate both compute accounting
            # and the single-GPU schedule with unreported forward rows.
            endpoint = np.zeros_like(patch_scores)
            quality = 0.0
        else:
            image = sample.image.to(device=device, dtype=_precision(precision))
            masks = sample.masks.to(device=device, dtype=_precision(precision))
            reference = sample.reference.to(device=device, dtype=_precision(precision))
            endpoint = (
                _endpoint(
                    model,
                    image,
                    reference,
                    masks,
                    sample.target,
                    sample.dataset,
                )
                .detach()
                .cpu()
                .double()
                .numpy()
            )
            quality = float(row_spearman(patch_scores, endpoint)[0])
        rows.append(
            {
                "image_index": int(job["image_start"]) + offset,
                "image_id": sample.image_id,
                "scope": job["scope"],
                "dataset": job["dataset"],
                "model": job["model_id"],
                "method": method_id,
                "target_class": sample.target,
                "spearman": quality,
                "patch_scores": patch_scores,
                "endpoint_effects": endpoint,
                "quality_target_effects": endpoint.copy(),
                "decaf_M": np.abs(endpoint),
                "decaf_E": np.asarray(metadata.get("E", np.zeros_like(endpoint))),
                "decaf_C": np.asarray(metadata.get("C", np.zeros_like(endpoint))),
                "decaf_F": np.asarray(metadata.get("F", np.zeros_like(endpoint))),
                "decaf_Abs": np.asarray(metadata.get("Abs", np.abs(endpoint))),
                "finite_complete": True,
                "numeric_audit_passed": bool(metadata["numeric_audit_passed"]),
                "raw_image_height": sample.raw_height,
                "raw_image_width": sample.raw_width,
                "part_group_count": len(sample.part_names),
                "part_names": np.asarray(sample.part_names, dtype=str),
                "preprocess_before_coalescing": True,
                "input_domain": metadata["input_domain"],
                "perturbation_domain": metadata["perturbation_domain"],
                "masking_domain": metadata["masking_domain"],
                "baseline_domain": metadata["baseline_domain"],
                "model_preprocess_inside_forward": metadata["model_preprocess_inside_forward"],
                "cross_image_coalescing": False,
                "preprocess_before_cross_image_coalescing": True,
                "method_metadata_json": json.dumps(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key not in {"M", "E", "C", "F", "Abs"}
                    },
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def _timing_frame(
    job: Mapping[str, Any],
    samples: Sequence[PreparedSample],
    model: Any,
    *,
    device: str,
    precision: str,
) -> pd.DataFrame:
    torch = _torch()
    method_id = str(job["method_id"])
    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    forward_rows = 0
    backward_calls = 0
    input_domains: set[str] = set()
    preprocess_modes: set[bool] = set()
    for sample in samples:
        seed = _stable_method_seed(str(job["member_id"]), sample.image_id)
        _, metadata = _method(
            method_id,
            model,
            sample,
            device=device,
            precision=precision,
            seed=seed,
        )
        forward_rows += int(metadata.get("forward_rows", 0))
        backward_calls += int(metadata.get("backward_calls", 0))
        input_domains.add(str(metadata["input_domain"]))
        preprocess_modes.add(bool(metadata["model_preprocess_inside_forward"]))
    torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(torch.device(device)))
    count = len(samples)
    if len(input_domains) != 1 or len(preprocess_modes) != 1:
        raise RuntimeError("timed method changed its registered input-domain contract")
    return pd.DataFrame(
        [
            {
                "scope": job["scope"],
                "dataset": job["dataset"],
                "model": job["model_id"],
                "method": method_id,
                "repeat": int(job["repeat"]),
                "wall_seconds_per_image": elapsed / count,
                "peak_allocated_bytes": peak,
                "forward_rows_per_image": forward_rows / count,
                "backward_calls_per_image": backward_calls / count,
                "timed_images": count,
                "cuda_synchronized_before": True,
                "cuda_synchronized_after": True,
                "input_domain": next(iter(input_domains)),
                "model_preprocess_inside_forward": next(iter(preprocess_modes)),
                "cross_image_coalescing": False,
                "preprocess_before_cross_image_coalescing": True,
                "timing_comparison_scope": "compute_path_only_not_paper_timing",
            }
        ]
    )


def evaluate_member(job: Mapping[str, Any], context: RunContext) -> pd.DataFrame:
    """Execute one real member on the sole visible B200."""

    delay = float(os.environ.get("DECAF_RESUME_TEST_MEMBER_DELAY_SECONDS", "0"))
    if delay < 0.0 or delay > 60.0 or not math.isfinite(delay):
        raise ValueError("DECAF_RESUME_TEST_MEMBER_DELAY_SECONDS must lie in [0,60]")
    if delay:
        time.sleep(delay)
    with _strict_fp32_backends() as numeric_contract:
        device, device_name = _device_contract()
        execution = context.config.get("execution", {})
        if not isinstance(execution, Mapping):
            raise TypeError("execution configuration must be a mapping")
        precision = str(execution.get("precision", "fp32"))
        dataset = str(job["dataset"])
        model_id = str(job["model_id"])
        bare_model = _active_model(model_id, dataset, device, precision)
        model = _runtime_model(bare_model, model_id, dataset)
        samples = _selected_samples(
            context,
            dataset,
            model_id,
            model,
            device=device,
            precision=precision,
        )
        start, stop = int(job["image_start"]), int(job["image_stop"])
        selected = samples[start:stop]
        if len(selected) != int(job["image_count"]):
            raise RuntimeError("fixed sample selection does not cover the member image range")
        kind = str(job["kind"])
        if kind in {
            "shared_deletion_targets",
            "shared_part_deletion_targets",
            "shared_heldout_targets",
        }:
            frame = _target_frame(job, selected, model, device=device, precision=precision)
        elif kind in {"timing", "large_model_timing"}:
            frame = _timing_frame(job, selected, model, device=device, precision=precision)
        else:
            frame = _quality_frame(job, selected, model, device=device, precision=precision)
        torch = _torch()
        raw_checkpoint_assets = tuple(getattr(bare_model, "decaf_checkpoint_assets", ()))
        if not raw_checkpoint_assets:
            raise RuntimeError("real attribution model lacks checkpoint lineage")
        checkpoint_assets = tuple(
            {
                "checkpoint_id": str(asset["checkpoint_id"]),
                "sha256": str(asset["sha256"]),
                "bytes": int(asset["bytes"]),
            }
            for asset in raw_checkpoint_assets
        )
        checkpoint_json = json.dumps(
            checkpoint_assets, sort_keys=True, separators=(",", ":")
        )
        frame["runtime_device"] = device
        frame["runtime_device_name"] = device_name
        frame["runtime_precision"] = precision
        frame["runtime_torch_version"] = str(torch.__version__)
        frame["runtime_cuda_version"] = str(torch.version.cuda)
        frame["runtime_cuda_matmul_allow_tf32"] = numeric_contract[
            "cuda_matmul_allow_tf32"
        ]
        frame["runtime_cudnn_allow_tf32"] = numeric_contract["cudnn_allow_tf32"]
        frame["checkpoint_assets_json"] = checkpoint_json
        frame["checkpoint_assets_sha256"] = hashlib.sha256(
            checkpoint_json.encode("utf-8")
        ).hexdigest()
        return frame


def validate_checkpoint_fingerprint_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    verify_paths: bool = False,
) -> list[dict[str, Any]]:
    """Validate the exact seven-case verifier interchange schema."""

    normalized = [dict(row) for row in rows]
    expected = [(model_id, dataset) for model_id, dataset in FINGERPRINT_CASES]
    observed = [(str(row.get("model_id")), str(row.get("dataset"))) for row in normalized]
    if len(normalized) != 7 or observed != expected:
        raise ValueError(f"fingerprint cases drifted: {observed} != {expected}")
    for row in normalized:
        required = {
            "family",
            "case_id",
            "model_id",
            "checkpoints",
            "sample_ids",
            "preprocessed_tensor",
            "target_class",
            "logits",
            "probabilities",
            "precision",
            "device",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"fingerprint row is missing fields: {missing}")
        if row["family"] != "attribution" or row["precision"] != "fp32":
            raise ValueError("fingerprint family/precision drifted")
        if not str(row["device"]).startswith("cuda"):
            raise ValueError("fingerprint device must be CUDA")
        checkpoints = row["checkpoints"]
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError("fingerprint checkpoints must be a non-empty list")
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping):
                raise TypeError("fingerprint checkpoint rows must be mappings")
            path = Path(str(checkpoint.get("path", "")))
            digest = checkpoint.get("sha256")
            size = checkpoint.get("bytes")
            if (
                not path.is_absolute()
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
            ):
                raise ValueError("fingerprint checkpoint path/SHA256 is invalid")
            if not isinstance(size, int) or size <= 0:
                raise ValueError("fingerprint checkpoint byte count is invalid")
            if verify_paths and (
                not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest
            ):
                raise RuntimeError(f"fingerprint checkpoint bytes drifted: {path}")
        sample_ids = row["sample_ids"]
        if (
            not isinstance(sample_ids, list)
            or not sample_ids
            or any(not str(value) for value in sample_ids)
        ):
            raise ValueError("fingerprint sample_ids are invalid")
        tensor = row["preprocessed_tensor"]
        if not isinstance(tensor, Mapping):
            raise TypeError("preprocessed_tensor must be a mapping")
        if (
            not isinstance(tensor.get("sha256"), str)
            or not _SHA256.fullmatch(str(tensor["sha256"]))
            or tensor.get("dtype") != "torch.float32"
            or tensor.get("byte_order") not in {"little", "big"}
            or tensor.get("layout") != "contiguous_c_order"
            or not isinstance(tensor.get("shape"), list)
        ):
            raise ValueError("preprocessed tensor fingerprint is invalid")
        logits = np.asarray(row["logits"], dtype=np.float64)
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        if (
            logits.ndim != 2
            or logits.shape[0] != len(sample_ids)
            or logits.shape != probabilities.shape
            or logits.shape[1] not in {50, 1_000}
            or not np.isfinite(logits).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.allclose(probabilities.sum(1), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError("fingerprint logits/probabilities are invalid")
        target = row["target_class"]
        if not isinstance(target, int) or target < 0 or target >= logits.shape[1]:
            raise ValueError("fingerprint target_class is invalid")
    return normalized


def collect_checkpoint_fingerprints(device: str = "cuda:0") -> list[dict[str, Any]]:
    """Return exactly seven strict model/checkpoint/input/output fingerprints."""

    torch = _torch()
    visible_device, device_name = _device_contract()
    if str(device) not in {"cuda", "cuda:0", visible_device}:
        raise ValueError(f"fingerprinting is bound to {visible_device}, received {device}")
    rows: list[dict[str, Any]] = []
    for model_id, dataset in FINGERPRINT_CASES:
        checkpoint_ids = MODEL_CHECKPOINTS[(model_id, dataset)]
        assets = resolve_offline_assets(
            checkpoint_ids=checkpoint_ids,
            datasets=(dataset,),
            sources=_required_sources(model_id, dataset),
            require_prepared=dataset == "funnybirds",
            require_common_support=model_id != "dinov2_vit_g_14",
        )
        model = load_model(
            model_id,
            dataset=dataset,
            device=visible_device,
            precision="fp32",
            assets=assets,
        )
        candidates = load_fixed_samples(dataset, model_id, count=8, assets=assets)
        prepared: PreparedSample | None = None
        for candidate in candidates:
            value = preprocess_sample(candidate, model_id, dataset=dataset)
            if _correct(model, value, visible_device, "fp32"):
                prepared = value
                break
        if prepared is None:
            raise RuntimeError(f"no correctly classified fingerprint sample: {dataset}/{model_id}")
        tensor = prepared.image.unsqueeze(0).to(device=visible_device, dtype=torch.float32)
        torch.cuda.synchronize(torch.device(visible_device))
        with torch.inference_mode():
            logits = _extract_logits(model(tensor), classes=_classes(dataset))
            probabilities = logits.softmax(dim=1)
        if not torch.allclose(
            probabilities.sum(1),
            torch.ones(1, device=probabilities.device),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError("fingerprint probabilities are not normalized")
        torch.cuda.synchronize(torch.device(visible_device))
        checkpoints = [assets.checkpoints[value].to_dict() for value in checkpoint_ids]
        rows.append(
            {
                "schema_version": 1,
                "family": "attribution",
                "case_id": f"attribution/{dataset}/{model_id}",
                "dataset": dataset,
                "model_id": model_id,
                "checkpoint_identity": (
                    checkpoint_ids[0] if len(checkpoint_ids) == 1 else list(checkpoint_ids)
                ),
                "checkpoints": checkpoints,
                "sample_ids": [prepared.image_id],
                "preprocessed_tensor": canonical_tensor_fingerprint(tensor),
                "target_class": prepared.target,
                "logits": logits.detach().cpu().float().tolist(),
                "probabilities": probabilities.detach().cpu().float().tolist(),
                "precision": "fp32",
                "device": visible_device,
                "device_name": device_name,
                "cuda_synchronized": True,
                "offline": True,
                "fallback_used": False,
            }
        )
        del model, tensor, logits, probabilities
        gc.collect()
        torch.cuda.empty_cache()
    return validate_checkpoint_fingerprint_rows(rows, verify_paths=True)


__all__ = [
    "AttributionSample",
    "CHECKPOINT_SPECS",
    "CheckpointAsset",
    "CheckpointSpec",
    "FINGERPRINT_CASES",
    "OfflineAssets",
    "PreparedSample",
    "canonical_tensor_fingerprint",
    "collect_checkpoint_fingerprints",
    "evaluate_member",
    "load_checkpoint_state_dict",
    "load_fixed_samples",
    "load_model",
    "preprocess_sample",
    "resolve_checkpoint",
    "resolve_offline_assets",
    "validate_checkpoint_fingerprint_rows",
]
