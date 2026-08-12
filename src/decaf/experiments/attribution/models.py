"""Portable model contracts for the attribution experiments.

This module intentionally has no dependency on a tensor framework. GPU model
construction is delegated to a runtime adapter and imported only by the compute
stage, so planning and analysis remain usable on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One registered architecture and its public checkpoint identities."""

    model_id: str
    display_name: str
    architecture: str
    datasets: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    requires_gpu: bool = True


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("oracle_linear", "CPU score oracle", "linear", ("oracle",), (), False),
    ModelSpec(
        "resnet50",
        "ResNet-50",
        "cnn",
        ("imagenet1k_idsds", "partimagenet"),
        ("idsds_resnet50", "torchvision_resnet50"),
    ),
    ModelSpec(
        "vgg16",
        "VGG-16",
        "cnn",
        ("imagenet1k_idsds",),
        ("idsds_vgg16",),
    ),
    ModelSpec(
        "vit_base_patch16_224",
        "ViT-B/16",
        "transformer",
        ("imagenet1k_idsds",),
        ("idsds_vit_base_patch16_224",),
    ),
    ModelSpec(
        "funnybirds_resnet50",
        "FunnyBirds ResNet-50",
        "cnn",
        ("funnybirds",),
        ("funnybirds_resnet",),
    ),
    ModelSpec(
        "funnybirds_vgg16",
        "FunnyBirds VGG-16",
        "cnn",
        ("funnybirds",),
        ("funnybirds_vgg",),
    ),
    ModelSpec(
        "funnybirds_vit_b_16",
        "FunnyBirds ViT-B/16",
        "transformer",
        ("funnybirds",),
        ("funnybirds_vit",),
    ),
    ModelSpec(
        "convnext_large",
        "ConvNeXt-L",
        "cnn",
        ("partimagenet",),
        ("torchvision_convnext_large",),
    ),
    ModelSpec(
        "swin_b",
        "Swin-B",
        "transformer",
        ("partimagenet",),
        ("torchvision_swin_b",),
    ),
    ModelSpec(
        "dinov2_vit_l_14",
        "DINOv2 ViT-L/14",
        "transformer",
        ("partimagenet",),
        ("dinov2_vitl14_backbone", "dinov2_vitl14_linear_head"),
    ),
    ModelSpec(
        "dinov2_vit_g_14",
        "DINOv2 ViT-g/14",
        "transformer",
        ("imagenet1k_idsds", "partimagenet"),
        ("dinov2_vitg14_backbone", "dinov2_vitg14_linear_head"),
    ),
)

MODELS = {spec.model_id: spec for spec in MODEL_SPECS}

ALIGNED_ARCHITECTURES: tuple[tuple[str, str, str], ...] = (
    ("resnet", "resnet50", "funnybirds_resnet50"),
    ("vgg", "vgg16", "funnybirds_vgg16"),
    ("vit_b_16", "vit_base_patch16_224", "funnybirds_vit_b_16"),
)

IDSDS_MODELS = tuple(pair[1] for pair in ALIGNED_ARCHITECTURES)
FUNNYBIRDS_MODELS = tuple(pair[2] for pair in ALIGNED_ARCHITECTURES)
BOUNDARY_MODELS = ("resnet50", "convnext_large", "swin_b", "dinov2_vit_l_14")
LARGE_MODEL = "dinov2_vit_g_14"


def get_model(model_id: str) -> ModelSpec:
    """Resolve a registered model ID."""

    try:
        return MODELS[model_id]
    except KeyError as error:
        raise KeyError(f"unknown attribution model: {model_id}") from error


def supports_dataset(model_id: str, dataset: str) -> bool:
    """Return whether a model is registered for a dataset contract."""

    return dataset in get_model(model_id).datasets


def checkpoint_coverage(model_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Return checkpoint IDs, rejecting uncovered non-oracle models."""

    coverage = {model_id: get_model(model_id).checkpoint_ids for model_id in model_ids}
    missing = [
        model_id
        for model_id, checkpoints in coverage.items()
        if get_model(model_id).requires_gpu and not checkpoints
    ]
    if missing:
        raise ValueError(f"models have no checkpoint contract: {missing}")
    return coverage


__all__ = [
    "ALIGNED_ARCHITECTURES",
    "BOUNDARY_MODELS",
    "FUNNYBIRDS_MODELS",
    "IDSDS_MODELS",
    "LARGE_MODEL",
    "MODELS",
    "ModelSpec",
    "checkpoint_coverage",
    "get_model",
    "supports_dataset",
]
