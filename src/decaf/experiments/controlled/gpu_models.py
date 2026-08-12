"""Self-contained loaders for the historical 32 x 32 controlled models.

PyTorch is intentionally imported lazily.  The repository's default smoke
profile is a NumPy score oracle and must remain importable in environments that
do not install the optional GPU stack.  These model definitions mirror the
historical ResNet-18 and Small-ViT state-dict contracts without importing the
historical repository at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARCHITECTURES = ("resnet18", "small_vit")


def normalize_architecture(value: Any) -> str:
    """Normalize the two registered historical architecture names."""

    name = str(value).strip().lower().replace("-", "_")
    aliases = {
        "resnet_18": "resnet18",
        "vit": "small_vit",
        "smallvit": "small_vit",
    }
    normalized = aliases.get(name, name)
    if normalized not in ARCHITECTURES:
        raise ValueError(f"unsupported controlled architecture {value!r}; expected {ARCHITECTURES}")
    return normalized


def require_single_cuda(device: str = "cuda:0") -> tuple[Any, Any, Any, dict[str, Any]]:
    """Import PyTorch and require the one visible device to be an NVIDIA B200."""

    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as error:  # pragma: no cover - exercised on the GPU node
        raise RuntimeError("DECAF_B200_VERIFY=1 requires a PyTorch GPU environment") from error

    selected = torch.device(device)
    if selected.type != "cuda" or selected.index not in {None, 0}:
        raise ValueError("controlled B200 verification requires device='cuda:0'")
    if not torch.cuda.is_available():
        raise RuntimeError("controlled B200 verification requires CUDA")
    count = int(torch.cuda.device_count())
    if count != 1:
        raise RuntimeError(
            f"controlled B200 verification requires exactly one visible CUDA device; got {count}"
        )
    name = str(torch.cuda.get_device_name(0))
    if "B200" not in name.upper():
        raise RuntimeError(f"controlled verification expected an NVIDIA B200; got {name!r}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(0)
    record = {
        "requested": "cuda:0",
        "resolved": str(selected),
        "name": name,
        "count_visible": count,
        "total_memory_bytes": int(properties.total_memory),
        "capability": [int(properties.major), int(properties.minor)],
        "precision": {
            "model": "float32",
            "matmul": "highest",
            "tf32": False,
        },
    }
    return torch, nn, functional, record


def _resnet18(torch: Any, nn: Any, configuration: Mapping[str, Any]) -> Any:
    """Build the torchvision-compatible CIFAR-stem ResNet-18 contract."""

    num_classes = int(configuration.get("num_classes", 2))
    in_channels = int(configuration.get("in_channels", 3))
    if num_classes < 2 or in_channels < 1:
        raise ValueError("invalid ResNet-18 class/channel configuration")

    def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> Any:
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

    def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Any:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Any | None = None,
        ) -> None:
            super().__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, value: Any) -> Any:
            identity = value
            output = self.relu(self.bn1(self.conv1(value)))
            output = self.bn2(self.conv2(output))
            if self.downsample is not None:
                identity = self.downsample(value)
            return self.relu(output + identity)

    class ResNet18For32x32(nn.Module):
        architecture_name = "resnet18"

        def __init__(self) -> None:
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
            self.layer1 = self._make_layer(64, 2)
            self.layer2 = self._make_layer(128, 2, stride=2)
            self.layer3 = self._make_layer(256, 2, stride=2)
            self.layer4 = self._make_layer(512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> Any:
            downsample = None
            if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                    nn.BatchNorm2d(planes * BasicBlock.expansion),
                )
            layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * BasicBlock.expansion
            layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
            return nn.Sequential(*layers)

        def forward(self, value: Any) -> Any:
            value = self.relu(self.bn1(self.conv1(value)))
            value = self.maxpool(value)
            value = self.layer1(value)
            value = self.layer2(value)
            value = self.layer3(value)
            value = self.layer4(value)
            value = self.avgpool(value)
            value = torch.flatten(value, 1)
            return self.fc(value)

    return ResNet18For32x32()


def _small_vit(torch: Any, nn: Any, configuration: Mapping[str, Any]) -> Any:
    """Build the exact historical pre-norm Small-ViT state-dict contract."""

    image_size = int(configuration.get("image_size", 32))
    patch_size = int(configuration.get("patch_size", 4))
    in_channels = int(configuration.get("in_channels", 3))
    embedding_dim = int(configuration.get("embedding_dim", 192))
    depth = int(configuration.get("depth", 6))
    heads = int(configuration.get("heads", 3))
    mlp_ratio = float(configuration.get("mlp_ratio", 4.0))
    dropout = float(configuration.get("dropout", 0.0))
    num_classes = int(configuration.get("num_classes", 2))
    if image_size != 32 or patch_size < 1 or image_size % patch_size:
        raise ValueError("Small-ViT requires a valid 32x32 patch configuration")
    if embedding_dim % heads or depth < 1 or heads < 1 or num_classes < 2:
        raise ValueError("invalid Small-ViT architecture configuration")

    class PatchEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_patches = (image_size // patch_size) ** 2
            self.projection = nn.Conv2d(
                in_channels,
                embedding_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )

        def forward(self, images: Any) -> Any:
            if images.ndim != 4 or tuple(images.shape[-2:]) != (image_size, image_size):
                raise ValueError(f"Small-ViT expected BCHW 32x32 input; got {tuple(images.shape)}")
            return self.projection(images).flatten(2).transpose(1, 2)

    class SmallVisionTransformer(nn.Module):
        architecture_name = "small_vit"

        def __init__(self) -> None:
            super().__init__()
            self.patch_embedding = PatchEmbedding()
            token_count = self.patch_embedding.num_patches + 1
            self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            self.position_embedding = nn.Parameter(torch.zeros(1, token_count, embedding_dim))
            self.embedding_dropout = nn.Dropout(dropout)
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                dim_feedforward=int(round(embedding_dim * mlp_ratio)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=depth,
                norm=nn.LayerNorm(embedding_dim),
                enable_nested_tensor=False,
            )
            self.head = nn.Linear(embedding_dim, num_classes)

        def forward(self, images: Any) -> Any:
            tokens = self.patch_embedding(images)
            class_token = self.class_token.expand(images.shape[0], -1, -1)
            tokens = torch.cat((class_token, tokens), dim=1)
            tokens = self.embedding_dropout(tokens + self.position_embedding)
            return self.head(self.encoder(tokens)[:, 0])

    return SmallVisionTransformer()


def build_model(
    architecture: str,
    configuration: Mapping[str, Any],
    *,
    torch: Any,
    nn: Any,
) -> Any:
    """Build one registered architecture without external source imports."""

    normalized = normalize_architecture(architecture)
    configured = configuration.get("architecture", configuration.get("name", normalized))
    if normalize_architecture(configured) != normalized:
        raise ValueError(
            f"checkpoint model configuration architecture differs: {configured!r} != {normalized!r}"
        )
    if normalized == "resnet18":
        return _resnet18(torch, nn, configuration)
    return _small_vit(torch, nn, configuration)


def _checkpoint_payload(torch: Any, path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"historical checkpoint payload is not a mapping: {path}")
    return payload


def _model_configuration(
    payload: Mapping[str, Any],
    architecture: str,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if override:
        configuration: Any = dict(override)
    else:
        configuration = payload.get("model_config")
        if configuration is None and isinstance(payload.get("config"), Mapping):
            configuration = payload["config"].get("model")
        if configuration is None:
            configuration = {"architecture": architecture, "num_classes": 2}
    if not isinstance(configuration, Mapping):
        raise KeyError("checkpoint has no reconstructable model configuration")
    result = dict(configuration)
    result.setdefault("architecture", architecture)
    return result


def _state_dict(
    payload: Mapping[str, Any],
    *,
    state_dict_key: str | None,
) -> Mapping[str, Any]:
    if state_dict_key:
        candidate = payload.get(state_dict_key)
        if not isinstance(candidate, Mapping):
            raise KeyError(f"checkpoint state_dict_key {state_dict_key!r} is missing")
        return candidate
    for key in ("model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    # A bare state dict is accepted only when every value is tensor-like.
    if payload and all(hasattr(value, "shape") for value in payload.values()):
        return payload
    raise KeyError("checkpoint has no model_state_dict/state_dict")


def _matching_state_dict(
    state: Mapping[str, Any],
    model: Any,
    *,
    strip_prefix: str | None,
) -> tuple[dict[str, Any], str | None]:
    expected = set(model.state_dict())
    prefixes: list[str | None] = [None]
    if strip_prefix:
        prefixes.append(strip_prefix)
    prefixes.extend(("module.", "_orig_mod.", "model."))
    matches: list[tuple[dict[str, Any], str | None]] = []
    seen: set[str | None] = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        if prefix is None:
            transformed = {str(key): value for key, value in state.items()}
        elif all(str(key).startswith(prefix) for key in state):
            transformed = {str(key)[len(prefix) :]: value for key, value in state.items()}
        else:
            continue
        if set(transformed) == expected:
            matches.append((transformed, prefix))
    if len(matches) != 1:
        observed = sorted(map(str, state))[:5]
        raise ValueError(
            "historical checkpoint keys do not identify exactly one strict prefix transform; "
            f"matches={len(matches)}, first_keys={observed}"
        )
    return matches[0]


def load_historical_model(
    path: Path,
    architecture: str,
    *,
    device: str,
    model_config: Mapping[str, Any] | None = None,
    state_dict_key: str | None = None,
    strip_prefix: str | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Strictly reconstruct and load one trusted historical checkpoint."""

    torch, nn, _functional, device_record = require_single_cuda(device)
    payload = _checkpoint_payload(torch, path)
    configuration = _model_configuration(payload, architecture, model_config)
    model = build_model(architecture, configuration, torch=torch, nn=nn)
    state, used_prefix = _matching_state_dict(
        _state_dict(payload, state_dict_key=state_dict_key),
        model,
        strip_prefix=strip_prefix,
    )
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device))
    model.float()
    model.eval()
    load_record = {
        "architecture": normalize_architecture(architecture),
        "model_config": configuration,
        "state_dict_key": state_dict_key
        or ("model_state_dict" if "model_state_dict" in payload else "state_dict"),
        "stripped_prefix": used_prefix,
        "strict": True,
    }
    return model, load_record, device_record


__all__ = [
    "ARCHITECTURES",
    "build_model",
    "load_historical_model",
    "normalize_architecture",
    "require_single_cuda",
]
