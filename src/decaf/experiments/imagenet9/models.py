"""Portable model-zoo registry and lazy vision-model loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelRecord:
    """One immutable model slot in the ImageNet-9 plan."""

    model_id: str
    source: str
    architecture: str
    weights: str
    training_regime: str
    seed: int | None
    checkpoint_key: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_registry(config: Mapping[str, Any]) -> list[ModelRecord]:
    """Expand the registered pretrained and fine-tuned Cartesian products."""

    specification = config["models"]
    records: list[ModelRecord] = []
    for architecture, weights in specification.get("torchvision", ()):
        records.append(
            ModelRecord(
                model_id=f"tv_{architecture}",
                source="torchvision",
                architecture=str(architecture),
                weights=str(weights),
                training_regime="upstream_pretrained",
                seed=None,
                checkpoint_key=f"torchvision/{architecture}/{weights}",
            )
        )
    for architecture in specification.get("timm", ()):
        token = str(architecture).split(".", maxsplit=1)[0].replace("/", "_")
        records.append(
            ModelRecord(
                model_id=f"timm_{token}",
                source="timm",
                architecture=str(architecture),
                weights="pretrained",
                training_regime="upstream_pretrained",
                seed=None,
                checkpoint_key=f"timm/{architecture}",
            )
        )
    for architecture in specification.get("finetune_backbones", ()):
        for regime in specification.get("finetune_regimes", ()):
            for seed in specification.get("finetune_seeds", ()):
                model_id = f"ft_{architecture}_{regime}_s{int(seed)}"
                records.append(
                    ModelRecord(
                        model_id=model_id,
                        source="experiment",
                        architecture=str(architecture),
                        weights="generated",
                        training_regime=str(regime),
                        seed=int(seed),
                        checkpoint_key=f"{model_id}/best.pt",
                    )
                )
    identifiers = [record.model_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("ImageNet-9 model IDs are not unique")
    return records


def deep_model_registry(
    config: Mapping[str, Any],
    records: list[ModelRecord] | None = None,
) -> list[ModelRecord]:
    """Return the sealed 32-model deep-benchmark subset."""

    all_records = records or model_registry(config)
    pretrained = [record for record in all_records if record.source != "experiment"]
    indices = tuple(int(value) for value in config["models"]["deep_pretrained_indices"])
    try:
        selected = [pretrained[index] for index in indices]
    except IndexError as error:
        raise ValueError("deep pretrained index is outside the model registry") from error
    deep_backbones = set(map(str, config["models"].get("deep_finetuned_backbones", ())))
    selected_seed = int(config["models"].get("deep_selection_seed", 7101))
    selected.extend(
        record
        for record in all_records
        if record.source == "experiment"
        and record.architecture in deep_backbones
        and record.seed == selected_seed
    )
    identifiers = [record.model_id for record in selected]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("deep-benchmark model IDs are not unique")
    return selected


def load_registered_model(record: ModelRecord, checkpoint_root: Path | None = None) -> Any:
    """Load a registered model through optional GPU dependencies."""

    if record.source == "torchvision":
        try:
            import torchvision.models as vision_models
        except ImportError as error:
            raise RuntimeError(
                "torchvision is required for ImageNet-9 model inference; install the vision extra"
            ) from error
        constructor = getattr(vision_models, record.architecture, None)
        if constructor is None:
            raise ValueError(f"unknown torchvision architecture: {record.architecture}")
        weights_enum = vision_models.get_model_weights(record.architecture)
        return constructor(weights=getattr(weights_enum, record.weights))
    if record.source == "timm":
        try:
            import timm
        except ImportError as error:
            raise RuntimeError(
                "timm is required for ImageNet-9 model inference; install the vision extra"
            ) from error
        return timm.create_model(record.architecture, pretrained=True)
    if checkpoint_root is None:
        raise ValueError("checkpoint_root is required for a fine-tuned model")
    checkpoint = checkpoint_root / record.checkpoint_key
    if not checkpoint.is_file():
        raise FileNotFoundError(f"registered checkpoint does not exist: {record.checkpoint_key}")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for ImageNet-9 checkpoint loading; install the vision extra"
        ) from error
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint must contain a mapping: {record.checkpoint_key}")
    specification = payload.get("model_spec")
    if not isinstance(specification, Mapping):
        raise KeyError(f"checkpoint has no model_spec: {record.checkpoint_key}")
    architecture = str(specification.get("architecture", record.architecture))
    if architecture != record.architecture:
        raise ValueError(
            "checkpoint architecture differs from its registry entry: "
            f"{architecture} != {record.architecture}"
        )
    try:
        import torchvision.models as vision_models
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required for ImageNet-9 checkpoint loading; install the vision extra"
        ) from error
    constructor = getattr(vision_models, architecture, None)
    if constructor is None:
        raise ValueError(f"unknown fine-tuned architecture: {architecture}")
    model = constructor(weights=None, num_classes=int(specification.get("num_classes", 9)))
    state = payload.get("model_state_dict", payload.get("state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError(f"checkpoint has no model state dict: {record.checkpoint_key}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


__all__ = [
    "ModelRecord",
    "deep_model_registry",
    "load_registered_model",
    "model_registry",
]
