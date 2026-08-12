"""Self-contained reveal and attribution methods for the ImageNet-9 GPU shard."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def stable_seed(*parts: object) -> int:
    """Derive a process-hash-independent 63-bit seed."""

    material = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def gaussian_blur(images: Any, *, sigma: float) -> Any:
    """Apply a deterministic separable Gaussian blur to BCHW tensors."""

    import torch
    import torch.nn.functional as functional

    selected = float(sigma)
    if not math.isfinite(selected) or selected < 0:
        raise ValueError("blur sigma must be finite and non-negative")
    if selected == 0:
        return images.clone()
    radius = max(1, math.ceil(4.0 * selected))
    coordinates = torch.arange(
        -radius, radius + 1, device=images.device, dtype=images.dtype
    )
    kernel = torch.exp(-0.5 * (coordinates / selected).square())
    kernel /= kernel.sum()
    channels = int(images.shape[1])
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    mode = "reflect" if radius < min(images.shape[-2:]) else "replicate"
    result = functional.conv2d(
        functional.pad(images, (radius, radius, 0, 0), mode=mode),
        horizontal,
        groups=channels,
    )
    return functional.conv2d(
        functional.pad(result, (0, 0, radius, radius), mode=mode),
        vertical,
        groups=channels,
    )


def _patch_order(
    plus: Any,
    minus: Any,
    *,
    pair_id: str,
    label: str,
    seed: int,
    grid: tuple[int, int],
) -> tuple[int, ...]:
    import torch

    rows, columns = grid
    height, width = map(int, plus.shape[-2:])
    y_edges = [(index * height) // rows for index in range(rows + 1)]
    x_edges = [(index * width) // columns for index in range(columns + 1)]
    difference = (plus - minus).detach().double().square().sum(dim=0)
    energy_tensors = []
    for row in range(rows):
        for column in range(columns):
            energy_tensors.append(
                difference[
                    y_edges[row] : y_edges[row + 1],
                    x_edges[column] : x_edges[column + 1],
                ].sum()
            )
    # One host transfer per pair avoids synchronizing CUDA once per patch.
    energies = [float(value) for value in torch.stack(energy_tensors).cpu().tolist()]

    def tie_key(index: int) -> int:
        token = f"decaf-imagenet9|patch|{seed}|{label}|{pair_id}|{index}".encode()
        return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")

    return tuple(
        sorted(range(rows * columns), key=lambda index: (-energies[index], tie_key(index), index))
    )


def reveal_sequence(
    plus: Any,
    minus: Any,
    *,
    pair_ids: Sequence[str],
    path: str,
    alpha: Sequence[float],
    blur_sigma: float,
    patch_grid: tuple[int, int],
    patch_seed: int,
) -> tuple[Any, Any]:
    """Return stage-major paired states with one shared neutral and patch order."""

    import torch

    if plus.shape != minus.shape or plus.ndim != 4:
        raise ValueError("reveal endpoints must be equal-shape BCHW tensors")
    if len(pair_ids) != int(plus.shape[0]):
        raise ValueError("pair_ids must contain one value per endpoint pair")
    positions = tuple(float(value) for value in alpha)
    if len(positions) < 2 or positions[0] != 0.0 or positions[-1] != 1.0:
        raise ValueError("reveal alpha must span zero to one")
    if any(not math.isfinite(value) for value in positions) or any(
        right <= left for left, right in zip(positions, positions[1:], strict=False)
    ):
        raise ValueError("reveal alpha must be finite and strictly increasing")
    neutral = gaussian_blur((plus + minus) * 0.5, sigma=blur_sigma)
    plus_stages: list[Any] = []
    minus_stages: list[Any] = []
    selected = str(path)
    if selected == "blend":
        for position in positions:
            plus_stages.append(torch.lerp(neutral, plus, position))
            minus_stages.append(torch.lerp(neutral, minus, position))
    elif selected in {"patch_A", "patch_B"}:
        label = selected[-1]
        rows, columns = patch_grid
        if min(rows, columns) < 1:
            raise ValueError("patch grid dimensions must be positive")
        height, width = map(int, plus.shape[-2:])
        y_edges = [(index * height) // rows for index in range(rows + 1)]
        x_edges = [(index * width) // columns for index in range(columns + 1)]
        orders = [
            _patch_order(
                plus[index],
                minus[index],
                pair_id=str(pair_id),
                label=label,
                seed=int(patch_seed),
                grid=patch_grid,
            )
            for index, pair_id in enumerate(pair_ids)
        ]
        for position in positions:
            count = math.floor(rows * columns * position + 1.0e-12)
            mask = torch.zeros(
                (plus.shape[0], 1, height, width),
                device=plus.device,
                dtype=torch.bool,
            )
            for batch_index, order in enumerate(orders):
                for patch_index in order[:count]:
                    row, column = divmod(patch_index, columns)
                    mask[
                        batch_index,
                        :,
                        y_edges[row] : y_edges[row + 1],
                        x_edges[column] : x_edges[column + 1],
                    ] = True
            plus_stages.append(torch.where(mask, plus, neutral))
            minus_stages.append(torch.where(mask, minus, neutral))
    else:
        raise ValueError(f"unknown reveal path: {path}")
    return torch.stack(plus_stages, dim=0), torch.stack(minus_stages, dim=0)


def _target_scores(model: Any, images: Any, targets: Any) -> Any:
    probabilities = model(images)
    if probabilities.ndim != 2 or int(probabilities.shape[-1]) != 9:
        raise ValueError("probability model must return [batch, 9]")
    if int(probabilities.shape[0]) != int(targets.numel()):
        raise ValueError("target batch differs from probability batch")
    return probabilities.gather(1, targets.reshape(-1, 1)).squeeze(1)


def _gradients(model: Any, images: Any, targets: Any) -> Any:
    import torch

    points = images.detach().requires_grad_(True)
    scores = _target_scores(model, points, targets)
    return torch.autograd.grad(scores.sum(), points)[0]


def _input_x_gradient(model: Any, images: Any, targets: Any) -> Any:
    import torch

    with torch.enable_grad():
        return images * _gradients(model, images, targets)


def _integrated_gradients(
    model: Any,
    images: Any,
    targets: Any,
    *,
    steps: int,
    internal_batch_size: int,
) -> Any:
    import torch

    if steps != 16:
        raise ValueError("ImageNet-9 B200 IG must use exactly 16 steps")
    baseline = torch.zeros_like(images)
    difference = images - baseline
    accumulated = torch.zeros_like(images)
    positions = torch.arange(1, steps + 1, device=images.device, dtype=images.dtype) / steps
    max_steps = max(1, int(internal_batch_size) // int(images.shape[0]))
    with torch.enable_grad():
        for chunk in positions.split(max_steps):
            states = baseline.unsqueeze(0) + chunk[:, None, None, None, None] * difference
            flat = states.reshape(-1, *images.shape[1:]).detach().requires_grad_(True)
            repeated = targets.repeat(int(chunk.numel()))
            gradients = torch.autograd.grad(_target_scores(model, flat, repeated).sum(), flat)[0]
            accumulated += gradients.reshape(
                int(chunk.numel()), int(images.shape[0]), *images.shape[1:]
            ).sum(dim=0)
    return difference * accumulated / steps


def _shared_noise(
    *,
    samples: int,
    pairs: int,
    shape: Sequence[int],
    seed: int,
    device: Any,
    dtype: Any,
) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    base = torch.randn((samples, pairs, *shape), generator=generator, dtype=torch.float32)
    paired = torch.cat((base, base), dim=1)
    return paired.to(device=device, dtype=dtype)


def _smoothgrad(
    model: Any,
    images: Any,
    targets: Any,
    *,
    pairs: int,
    samples: int,
    noise_std: float,
    seed: int,
    internal_batch_size: int,
) -> Any:
    import torch

    if samples != 16:
        raise ValueError("ImageNet-9 B200 SmoothGrad must use exactly 16 samples")
    noises = _shared_noise(
        samples=samples,
        pairs=pairs,
        shape=images.shape[1:],
        seed=seed,
        device=images.device,
        dtype=images.dtype,
    )
    accumulated = torch.zeros_like(images)
    max_samples = max(1, int(internal_batch_size) // int(images.shape[0]))
    with torch.enable_grad():
        for chunk in noises.split(max_samples):
            states = (images.unsqueeze(0) + float(noise_std) * chunk).clamp(0.0, 1.0)
            flat = states.reshape(-1, *images.shape[1:]).detach().requires_grad_(True)
            repeated = targets.repeat(int(chunk.shape[0]))
            gradients = torch.autograd.grad(_target_scores(model, flat, repeated).sum(), flat)[0]
            accumulated += gradients.reshape(
                int(chunk.shape[0]), int(images.shape[0]), *images.shape[1:]
            ).sum(dim=0)
    return accumulated / samples


def _grid_edges(length: int, cells: int) -> list[int]:
    return [(index * length) // cells for index in range(cells + 1)]


def _occlusion(
    model: Any,
    images: Any,
    targets: Any,
    *,
    grid: tuple[int, int],
    internal_batch_size: int,
) -> Any:
    import torch

    rows, columns = grid
    if rows * columns != 49:
        raise ValueError("ImageNet-9 B200 occlusion must use a 7x7/49-query grid")
    height, width = map(int, images.shape[-2:])
    y_edges = _grid_edges(height, rows)
    x_edges = _grid_edges(width, columns)
    masks = []
    for row in range(rows):
        for column in range(columns):
            mask = torch.zeros((1, height, width), device=images.device, dtype=torch.bool)
            mask[:, y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]] = True
            masks.append(mask)
    stacked = torch.stack(masks, dim=0)
    attribution = torch.zeros(
        (images.shape[0], 1, height, width), device=images.device, dtype=images.dtype
    )
    max_queries = max(1, int(internal_batch_size) // int(images.shape[0]))
    with torch.no_grad():
        reference = _target_scores(model, images, targets)
        for chunk in stacked.split(max_queries):
            states = torch.where(chunk[:, None], torch.zeros_like(images)[None], images[None])
            flat = states.reshape(-1, *images.shape[1:])
            repeated = targets.repeat(int(chunk.shape[0]))
            scores = _target_scores(model, flat, repeated).reshape(
                int(chunk.shape[0]), int(images.shape[0])
            )
            drops = reference[None] - scores
            attribution += torch.einsum(
                "qb,qchw->bchw", drops, chunk.to(dtype=images.dtype)
            )
    return attribution


def _shared_rise_masks(
    *,
    count: int,
    pairs: int,
    grid: tuple[int, int],
    height: int,
    width: int,
    keep_probability: float,
    seed: int,
    device: Any,
    dtype: Any,
) -> Any:
    import torch
    import torch.nn.functional as functional

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    coarse = torch.rand(
        (count, pairs, 1, *grid),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    coarse = (coarse < float(keep_probability)).to(dtype=dtype)
    resized = functional.interpolate(
        coarse.reshape(count * pairs, 1, *grid),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(count, pairs, 1, height, width)
    return torch.cat((resized, resized), dim=1)


def _rise(
    model: Any,
    images: Any,
    targets: Any,
    *,
    pairs: int,
    masks: int,
    grid: tuple[int, int],
    keep_probability: float,
    seed: int,
    internal_batch_size: int,
) -> Any:
    import torch

    if masks != 256:
        raise ValueError("ImageNet-9 B200 RISE must use exactly 256 masks")
    height, width = map(int, images.shape[-2:])
    shared = _shared_rise_masks(
        count=masks,
        pairs=pairs,
        grid=grid,
        height=height,
        width=width,
        keep_probability=keep_probability,
        seed=seed,
        device=images.device,
        dtype=images.dtype,
    )
    attribution = torch.zeros(
        (images.shape[0], 1, height, width), device=images.device, dtype=images.dtype
    )
    max_queries = max(1, int(internal_batch_size) // int(images.shape[0]))
    with torch.no_grad():
        for chunk in shared.split(max_queries):
            states = images[None] * chunk
            flat = states.reshape(-1, *images.shape[1:])
            repeated = targets.repeat(int(chunk.shape[0]))
            scores = _target_scores(model, flat, repeated).reshape(
                int(chunk.shape[0]), int(images.shape[0])
            )
            attribution += torch.einsum("qb,qbchw->bchw", scores, chunk)
    return attribution / (masks * float(keep_probability))


def paired_saliency_scores(
    method: str,
    model: Any,
    plus: Any,
    minus: Any,
    targets: Any,
    *,
    pair_ids: Sequence[str],
    settings: Mapping[str, Any],
) -> np.ndarray:
    """Compute paired attributions with shared stochastic draws and scalar deltas."""

    import torch

    if plus.shape != minus.shape or int(plus.shape[0]) != len(pair_ids):
        raise ValueError("paired saliency endpoints or IDs differ")
    pairs = int(plus.shape[0])
    images = torch.cat((plus, minus), dim=0)
    repeated_targets = torch.cat((targets, targets), dim=0)
    seed = stable_seed(
        "imagenet9-b200-attribution",
        int(settings.get("selection_seed", 91701)),
        method,
        *pair_ids,
    )
    internal = int(settings["attribution_internal_batch_size"])
    selected = str(method)
    if selected == "input_x_gradient":
        attribution = _input_x_gradient(model, images, repeated_targets)
    elif selected == "integrated_gradients":
        attribution = _integrated_gradients(
            model,
            images,
            repeated_targets,
            steps=int(settings["integrated_gradients_steps"]),
            internal_batch_size=internal,
        )
    elif selected == "smoothgrad":
        attribution = _smoothgrad(
            model,
            images,
            repeated_targets,
            pairs=pairs,
            samples=int(settings["smoothgrad_samples"]),
            noise_std=float(settings["smoothgrad_noise_std"]),
            seed=seed,
            internal_batch_size=internal,
        )
    elif selected == "occlusion":
        attribution = _occlusion(
            model,
            images,
            repeated_targets,
            grid=tuple(map(int, settings["occlusion_grid"])),
            internal_batch_size=internal,
        )
    elif selected == "rise":
        attribution = _rise(
            model,
            images,
            repeated_targets,
            pairs=pairs,
            masks=int(settings["rise_masks"]),
            grid=tuple(map(int, settings["rise_grid"])),
            keep_probability=float(settings["rise_keep_probability"]),
            seed=seed,
            internal_batch_size=internal,
        )
    else:
        raise ValueError(f"unknown ImageNet-9 B200 method: {method}")
    spatial = attribution.sum(dim=1) if attribution.shape[1] > 1 else attribution[:, 0]
    score = (spatial[:pairs] - spatial[pairs:]).abs().mean(dim=(-2, -1))
    result = score.detach().float().cpu().numpy().astype(np.float64, copy=False)
    if result.shape != (pairs,) or not np.isfinite(result).all():
        raise FloatingPointError(f"paired saliency produced invalid scores: {method}")
    return result


__all__ = [
    "gaussian_blur",
    "paired_saliency_scores",
    "reveal_sequence",
    "stable_seed",
]
