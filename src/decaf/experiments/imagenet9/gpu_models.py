"""Offline model, probability, and preprocessing contracts for ImageNet-9 B200 smoke.

The module intentionally imports PyTorch, torchvision, and Pillow only inside
runtime functions.  Importing the normal CPU reproduction package therefore
does not acquire an optional GPU dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decaf.paper.reference import sha256_file

PAIRED_MANIFEST_SHA256 = "e12b57770a84a3dfceb36d4b11b66eda3e6140203fec2c46e31cd986ceb80a10"
MAPPING_SHA256 = "5b72ab44c804129744f5dff9372d8048c3b85da38d48385d73686ee0b0e7b638"
NUM_IMAGENET_CLASSES = 1000
NUM_IMAGENET9_CLASSES = 9


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """One exact offline checkpoint selected for the verification shard."""

    model_id: str
    kind: str
    architecture: str
    architecture_family: str
    output_classes: int
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_bytes: int

    def record(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": self.kind,
            "architecture": self.architecture,
            "architecture_family": self.architecture_family,
            "output_classes": self.output_classes,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
        }


@dataclass(frozen=True, slots=True)
class B200Assets:
    """Resolved real-data and checkpoint assets, all rooted by environment."""

    dataset_root: Path
    pair_manifest: Path
    mapping: Path
    weight_cache_root: Path
    checkpoint_root: Path
    models: tuple[ModelAsset, ...]


def _required_environment_path(
    environment: Mapping[str, str], variable: str, *, directory: bool
) -> Path:
    raw = environment.get(variable)
    if not raw:
        kind = "root" if directory else "file"
        raise RuntimeError(f"set {variable} to the required offline asset {kind}")
    path = Path(raw).expanduser().resolve()
    if directory and not path.is_dir():
        raise FileNotFoundError(f"{variable} does not identify a directory: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"{variable} does not identify a file: {path}")
    return path


def _rooted_file(
    environment: Mapping[str, str],
    *,
    direct_variable: str,
    root: Path,
    relative: str,
) -> Path:
    override = environment.get(direct_variable)
    path = Path(override).expanduser().resolve() if override else (root / relative).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        source = direct_variable if override else f"{root}/{relative}"
        raise FileNotFoundError(f"offline checkpoint is missing ({source}): {path}")
    return path


def resolve_b200_assets(
    config: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
) -> B200Assets:
    """Resolve and hash every B200 asset without downloading or substituting."""

    env = os.environ if environment is None else environment
    smoke = config.get("b200_smoke")
    if not isinstance(smoke, Mapping):
        raise KeyError("ImageNet-9 smoke config has no b200_smoke section")
    raw_variables = smoke.get("asset_environments")
    if not isinstance(raw_variables, Mapping):
        raise KeyError("b200_smoke.asset_environments must be a mapping")
    variables = {str(key): str(value) for key, value in raw_variables.items()}
    for key in (
        "dataset_root",
        "weight_cache_root",
        "checkpoint_root",
        "pretrained_resnet18",
        "finetuned_cnn",
        "finetuned_transformer",
        "official_mapping",
    ):
        if not variables.get(key):
            raise KeyError(f"b200_smoke.asset_environments is missing {key}")

    configured_data_root = _required_environment_path(
        env, variables["dataset_root"], directory=True
    )
    nested_data_root = configured_data_root / str(config["data"]["dataset_subdirectory"])
    dataset_root = (
        nested_data_root.resolve()
        if (nested_data_root / "manifests" / "paired_variants.parquet").is_file()
        else configured_data_root
    )
    pair_manifest = (dataset_root / "manifests" / "paired_variants.parquet").resolve()
    if not pair_manifest.is_file():
        raise FileNotFoundError(f"real paired_variants.parquet is missing: {pair_manifest}")
    pair_sha256 = sha256_file(pair_manifest)
    if pair_sha256 != PAIRED_MANIFEST_SHA256:
        raise ValueError(
            "ImageNet-9 paired manifest fingerprint mismatch: "
            f"{pair_sha256} != {PAIRED_MANIFEST_SHA256}"
        )

    weight_cache_root = _required_environment_path(
        env, variables["weight_cache_root"], directory=True
    )
    checkpoint_root = _required_environment_path(env, variables["checkpoint_root"], directory=True)
    mapping_override = env.get(variables["official_mapping"])
    if mapping_override:
        mapping = Path(mapping_override).expanduser().resolve()
    else:
        candidates = (
            dataset_root / "manifests" / "in_to_in9.json",
            dataset_root / "metadata" / "in_to_in9.json",
        )
        mapping = next((path.resolve() for path in candidates if path.is_file()), candidates[0])
    if not mapping.is_file():
        raise FileNotFoundError(
            f"official ImageNet-1000 to ImageNet-9 mapping is missing: {mapping}"
        )
    mapping_sha256 = sha256_file(mapping)
    if mapping_sha256 != MAPPING_SHA256:
        raise ValueError(
            "official ImageNet-9 mapping fingerprint mismatch: "
            f"{mapping_sha256} != {MAPPING_SHA256}"
        )

    raw_models = smoke.get("models")
    if not isinstance(raw_models, Sequence) or isinstance(raw_models, (str, bytes)):
        raise TypeError("b200_smoke.models must be a sequence")
    models: list[ModelAsset] = []
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise TypeError("each b200_smoke model must be a mapping")
        kind = str(raw["kind"])
        alias = str(raw["checkpoint_environment"])
        if alias not in variables:
            raise KeyError(f"unknown checkpoint environment alias: {alias}")
        root = weight_cache_root if kind == "off_the_shelf" else checkpoint_root
        checkpoint = _rooted_file(
            env,
            direct_variable=variables[alias],
            root=root,
            relative=str(raw["checkpoint_relative"]),
        )
        digest = sha256_file(checkpoint)
        registered_digest = raw.get("checkpoint_sha256")
        if registered_digest is not None and digest != str(registered_digest):
            raise ValueError(
                f"registered checkpoint fingerprint mismatch for {raw['model_id']}: {digest}"
            )
        registered_bytes = raw.get("checkpoint_bytes")
        if registered_bytes is not None and checkpoint.stat().st_size != int(registered_bytes):
            raise ValueError(
                "registered checkpoint byte count mismatch for "
                f"{raw['model_id']}: {checkpoint.stat().st_size}"
            )
        models.append(
            ModelAsset(
                model_id=str(raw["model_id"]),
                kind=kind,
                architecture=str(raw["architecture"]),
                architecture_family=str(raw["architecture_family"]),
                output_classes=int(raw["output_classes"]),
                checkpoint=checkpoint,
                checkpoint_sha256=digest,
                checkpoint_bytes=checkpoint.stat().st_size,
            )
        )
    if len(models) != 3:
        raise ValueError("ImageNet-9 B200 smoke requires exactly three models")
    coverage = {(model.kind, model.architecture_family) for model in models}
    required = {
        ("off_the_shelf", "cnn"),
        ("fine_tuned", "cnn"),
        ("fine_tuned", "transformer"),
    }
    if coverage != required or [model.output_classes for model in models].count(1000) != 1:
        raise ValueError(
            "B200 model inventory must be 1k off-the-shelf, 9-way CNN, 9-way transformer"
        )
    if any(model.output_classes not in {9, 1000} for model in models):
        raise ValueError("B200 models must declare either 9 or 1000 output classes")
    return B200Assets(
        dataset_root=dataset_root,
        pair_manifest=pair_manifest,
        mapping=mapping,
        weight_cache_root=weight_cache_root,
        checkpoint_root=checkpoint_root,
        models=tuple(models),
    )


def load_official_mapping(path: str | Path, torch: Any) -> Any:
    """Validate the complete official mapping and return a 9x1000 matrix."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("official ImageNet-9 mapping JSON must be an object")
    normalized: dict[int, int] = {}
    for key, value in raw.items():
        index = int(key)
        if index in normalized:
            raise ValueError("official mapping contains duplicate integer keys")
        normalized[index] = int(value)
    expected = set(range(NUM_IMAGENET_CLASSES))
    if set(normalized) != expected:
        raise ValueError("official mapping must contain every ImageNet index 0..999 exactly once")
    if any(value < -1 or value >= NUM_IMAGENET9_CLASSES for value in normalized.values()):
        raise ValueError("official mapping values must be -1 or a superclass in 0..8")
    if {value for value in normalized.values() if value >= 0} != set(range(NUM_IMAGENET9_CLASSES)):
        raise ValueError("official mapping does not cover all nine ImageNet-9 classes")
    matrix = torch.zeros((NUM_IMAGENET9_CLASSES, NUM_IMAGENET_CLASSES), dtype=torch.float32)
    for source, target in normalized.items():
        if target >= 0:
            matrix[target, source] = 1.0
    return matrix


def extract_logits(output: Any, torch: Any) -> Any:
    """Extract a rank-2 classification tensor from common model wrappers."""

    if torch.is_tensor(output):
        logits = output
    elif hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, Mapping) and "logits" in output:
        logits = output["logits"]
    elif isinstance(output, (tuple, list)) and output:
        logits = extract_logits(output[0], torch)
    else:
        raise TypeError("model output has no explicit classification logits")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("classification logits must have shape [batch, classes]")
    if int(logits.shape[-1]) not in {NUM_IMAGENET9_CLASSES, NUM_IMAGENET_CLASSES}:
        raise ValueError(f"expected 9 or 1000 logits, received {tuple(logits.shape)}")
    return logits


def to_imagenet9_probabilities(
    logits: Any,
    *,
    torch: Any,
    mapping_matrix: Any | None,
    expected_classes: int,
) -> Any:
    """Apply exactly one softmax, then the official 1000-to-9 mass mapping.

    Nine-way fine-tuned heads are softmaxed directly.  For ImageNet-1k heads,
    unmapped mass is deliberately omitted and the nine mapped components are
    not renormalized (and, crucially, never softmaxed a second time).
    """

    if logits.ndim != 2 or int(logits.shape[-1]) != int(expected_classes):
        raise ValueError(
            f"model declares {expected_classes} classes but emitted shape {tuple(logits.shape)}"
        )
    probabilities = torch.softmax(logits.float(), dim=-1)
    if expected_classes == NUM_IMAGENET9_CLASSES:
        result = probabilities
    elif expected_classes == NUM_IMAGENET_CLASSES:
        if mapping_matrix is None or tuple(mapping_matrix.shape) != (9, 1000):
            raise ValueError("a 9x1000 official mapping is required for ImageNet-1k logits")
        result = probabilities @ mapping_matrix.to(
            device=probabilities.device, dtype=probabilities.dtype
        ).transpose(0, 1)
    else:
        raise ValueError("expected_classes must equal 9 or 1000")
    detached = result.detach()
    if (
        not bool(torch.isfinite(detached).all())
        or bool((detached < -1.0e-7).any())
        or bool((detached > 1.0 + 1.0e-7).any())
    ):
        raise FloatingPointError("ImageNet-9 probability adapter produced invalid probabilities")
    return result


def _load_state_dict(payload: Any, *, fine_tuned: bool) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    if fine_tuned:
        state = payload.get("model_state_dict", payload.get("state_dict"))
        if not isinstance(state, Mapping):
            raise KeyError("fine-tuned checkpoint has no model_state_dict/state_dict")
        return state
    if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        return payload["state_dict"]
    return payload


def load_model(asset: ModelAsset, *, device: str = "cuda:0") -> tuple[Any, dict[str, Any]]:
    """Strictly load one torchvision model from existing local bytes only."""

    try:
        import torch
        import torchvision
        import torchvision.models as vision_models
    except ImportError as error:
        raise RuntimeError("ImageNet-9 B200 verification requires torch and torchvision") from error
    constructor = getattr(vision_models, asset.architecture, None)
    if constructor is None:
        raise ValueError(f"unknown torchvision architecture: {asset.architecture}")
    payload = torch.load(asset.checkpoint, map_location="cpu", weights_only=False)
    if asset.kind == "fine_tuned":
        if not isinstance(payload, Mapping) or not isinstance(payload.get("model_spec"), Mapping):
            raise KeyError(f"fine-tuned checkpoint has no model_spec: {asset.checkpoint}")
        specification = payload["model_spec"]
        architecture = str(specification.get("architecture", ""))
        classes = int(specification.get("num_classes", -1))
        if architecture != asset.architecture or classes != 9 or asset.output_classes != 9:
            raise ValueError(
                "fine-tuned checkpoint identity differs from the registered direct 9-way model: "
                f"{architecture}/{classes}"
            )
        model = constructor(weights=None, num_classes=9)
        state = _load_state_dict(payload, fine_tuned=True)
    else:
        if asset.architecture != "resnet18" or asset.output_classes != 1000:
            raise ValueError("off-the-shelf smoke slot must be ImageNet-1k ResNet-18")
        model = constructor(weights=None)
        state = _load_state_dict(payload, fine_tuned=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device=torch.device(device), dtype=torch.float32).eval()
    return model, {
        "strict": True,
        "architecture": asset.architecture,
        "output_classes": asset.output_classes,
        "checkpoint_sha256": asset.checkpoint_sha256,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
    }


def probability_model(
    model: Any,
    *,
    mapping_matrix: Any,
    output_classes: int,
) -> Any:
    """Wrap a logits model with normalization and the one-softmax adapter."""

    import torch

    class ProbabilityModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = model
            self.output_classes = int(output_classes)
            self.register_buffer(
                "mapping_matrix",
                mapping_matrix if self.output_classes == 1000 else torch.empty(0),
                persistent=True,
            )
            self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

        def normalized_logits(self, images: Any) -> Any:
            return extract_logits(self.base_model((images - self.mean) / self.std), torch)

        def forward(self, images: Any) -> Any:
            logits = self.normalized_logits(images)
            mapping = self.mapping_matrix if self.mapping_matrix.numel() else None
            return to_imagenet9_probabilities(
                logits,
                torch=torch,
                mapping_matrix=mapping,
                expected_classes=self.output_classes,
            )

    return ProbabilityModel().eval()


def _resolve_image(dataset_root: Path, relative: str | Path) -> Path:
    candidate = (dataset_root / Path(relative)).resolve(strict=True)
    try:
        candidate.relative_to(dataset_root.resolve())
    except ValueError as error:
        raise ValueError(f"image path escapes the dataset root: {relative}") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def preprocess_image(path: str | Path, *, dataset_root: Path, size: int = 224) -> np.ndarray:
    """Apply ImageNet evaluation geometry to one raw image before coalescing."""

    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError("Pillow is required for ImageNet-9 preprocessing") from error
    selected = _resolve_image(dataset_root, path)
    with Image.open(selected) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        short_side = round(int(size) * 256 / 224)
        width, height = image.size
        scale = short_side / min(width, height)
        resized_width = max(int(size), round(width * scale))
        resized_height = max(int(size), round(height * scale))
        image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        left = (resized_width - int(size)) // 2
        top = (resized_height - int(size)) // 2
        cropped = image.crop((left, top, left + int(size), top + int(size)))
        array = np.asarray(cropped, dtype=np.float32) / np.float32(255.0)
    if array.shape != (int(size), int(size), 3):
        raise ValueError(f"preprocessed image has the wrong shape: {array.shape}")
    return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)


def preprocess_paths(
    paths: Sequence[str | Path], *, dataset_root: Path, size: int = 224
) -> np.ndarray:
    """Preprocess variable raw sizes independently and stack only fixed tensors."""

    if not paths:
        raise ValueError("at least one image path is required")
    individual = [preprocess_image(path, dataset_root=dataset_root, size=size) for path in paths]
    return np.ascontiguousarray(np.stack(individual, axis=0), dtype=np.float32)


def canonical_tensor_identity(value: Any) -> dict[str, Any]:
    """Hash C-contiguous little-endian float32 tensor bytes."""

    array = np.ascontiguousarray(np.asarray(value), dtype=np.dtype("<f4"))
    return {
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        "dtype": "float32",
        "shape": list(array.shape),
        "bytes": int(array.nbytes),
        "byte_order": "little-endian",
        "layout": "C-contiguous",
    }


__all__ = [
    "B200Assets",
    "MAPPING_SHA256",
    "ModelAsset",
    "PAIRED_MANIFEST_SHA256",
    "canonical_tensor_identity",
    "extract_logits",
    "load_model",
    "load_official_mapping",
    "preprocess_image",
    "preprocess_paths",
    "probability_model",
    "resolve_b200_assets",
    "to_imagenet9_probabilities",
]
