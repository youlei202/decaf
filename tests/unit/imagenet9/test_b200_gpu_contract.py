"""CPU/mocked regressions for the gated ImageNet-9 single-B200 executor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from decaf.experiments.common import load_profile
from decaf.experiments.imagenet9.gpu_methods import reveal_sequence
from decaf.experiments.imagenet9.gpu_models import (
    canonical_tensor_identity,
    load_official_mapping,
    preprocess_paths,
    to_imagenet9_probabilities,
)
from decaf.experiments.imagenet9.gpu_runtime import (
    _paired_stage_probabilities,
    b200_enabled,
    validate_checkpoint_fingerprint_records,
)


def test_b200_gate_is_exact_and_default_smoke_stays_dormant() -> None:
    config = load_profile("imagenet9", "smoke")
    paper = load_profile("imagenet9", "paper")
    assert not b200_enabled(config, {})
    assert not b200_enabled(config, {"DECAF_B200_VERIFY": "true"})
    assert b200_enabled(config, {"DECAF_B200_VERIFY": "1"})
    assert not b200_enabled(paper, {"DECAF_B200_VERIFY": "1"})


def test_mapping_applies_one_softmax_without_mapped_mass_renormalization(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    mapping = {str(index): (-1 if index >= 9 else index) for index in range(1000)}
    path = tmp_path / "in_to_in9.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    matrix = load_official_mapping(path, torch)
    logits = torch.zeros((1, 1000), dtype=torch.float32)
    probabilities = to_imagenet9_probabilities(
        logits,
        torch=torch,
        mapping_matrix=matrix,
        expected_classes=1000,
    )

    assert tuple(probabilities.shape) == (1, 9)
    assert torch.allclose(probabilities, torch.full((1, 9), 0.001))
    assert float(probabilities.sum()) == pytest.approx(0.009)
    assert not torch.allclose(probabilities, torch.softmax(probabilities, dim=-1))

    direct_logits = torch.arange(9, dtype=torch.float32).reshape(1, 9)
    direct = to_imagenet9_probabilities(
        direct_logits,
        torch=torch,
        mapping_matrix=None,
        expected_classes=9,
    )
    assert torch.allclose(direct, torch.softmax(direct_logits, dim=-1))


def test_canonical_tensor_hash_binds_dtype_shape_and_bytes() -> None:
    first = canonical_tensor_identity(np.zeros((1, 3, 224, 224), dtype=np.float32))
    second = canonical_tensor_identity(np.ones((1, 3, 224, 224), dtype=np.float32))
    assert first["dtype"] == "float32"
    assert first["shape"] == [1, 3, 224, 224]
    assert first["byte_order"] == "little-endian"
    assert first["layout"] == "C-contiguous"
    assert first["sha256"] != second["sha256"]


def test_variable_raw_sizes_are_preprocessed_before_batching(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    first = tmp_path / "wide.png"
    second = tmp_path / "tall.png"
    image_module.new("RGB", (317, 191), color=(255, 0, 0)).save(first)
    image_module.new("RGB", (173, 349), color=(0, 255, 0)).save(second)

    batch = preprocess_paths([first.name, second.name], dataset_root=tmp_path, size=224)
    assert batch.shape == (2, 3, 224, 224)
    assert batch.dtype == np.float32
    assert batch.flags.c_contiguous


def test_reveal_paths_share_the_neutral_and_are_nested() -> None:
    torch = pytest.importorskip("torch")
    plus = torch.linspace(0.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    minus = 1.0 - plus
    alpha = (0.0, 0.25, 0.5, 1.0)
    blend_plus, blend_minus = reveal_sequence(
        plus,
        minus,
        pair_ids=["pair-0"],
        path="blend",
        alpha=alpha,
        blur_sigma=2.0,
        patch_grid=(4, 4),
        patch_seed=7101,
    )
    assert torch.equal(blend_plus[0], blend_minus[0])
    assert torch.equal(blend_plus[-1], plus)
    assert torch.equal(blend_minus[-1], minus)

    patch_plus, patch_minus = reveal_sequence(
        plus,
        minus,
        pair_ids=["pair-0"],
        path="patch_A",
        alpha=alpha,
        blur_sigma=2.0,
        patch_grid=(4, 4),
        patch_seed=7101,
    )
    neutral = patch_plus[0]
    revealed_quarter = (patch_plus[1] != neutral).any(dim=1)
    revealed_half = (patch_plus[2] != neutral).any(dim=1)
    assert torch.all(~revealed_quarter | revealed_half)
    assert torch.equal(patch_plus[0], patch_minus[0])
    assert torch.equal(patch_plus[-1], plus)
    assert torch.equal(patch_minus[-1], minus)


def test_shared_midpoint_is_forwarded_once_not_in_two_batch_lanes() -> None:
    torch = pytest.importorskip("torch")

    class LaneVariantProbabilityModel(torch.nn.Module):
        def forward(self, images: object) -> object:
            batch = images.shape[0]  # type: ignore[union-attr]
            lane = torch.arange(batch, dtype=torch.float32).reshape(-1, 1)
            logits = torch.cat((lane * 1.0e-4, torch.zeros((batch, 8))), dim=1)
            return torch.softmax(logits, dim=1)

    model = LaneVariantProbabilityModel()
    shared = torch.zeros((2, 3, 4, 4))
    plus = torch.stack((shared, torch.ones_like(shared)), dim=0)
    minus = torch.stack((shared, torch.full_like(shared, 2.0)), dim=0)

    duplicated = model(torch.cat((shared, shared), dim=0))
    assert not torch.equal(duplicated[:2], duplicated[2:])
    plus_probabilities, minus_probabilities = _paired_stage_probabilities(
        model,
        plus,
        minus,
        inference_batch_size=64,
        torch=torch,
    )

    assert torch.equal(plus_probabilities[0], minus_probabilities[0])
    assert torch.equal(
        plus_probabilities[0] - minus_probabilities[0],
        torch.zeros_like(plus_probabilities[0]),
    )


def _fingerprint_case(
    checkpoint: Path,
    *,
    case_id: str,
    kind: str,
    architecture_family: str,
    width: int,
) -> dict[str, object]:
    import hashlib

    probabilities = np.full((1, width), 1.0 / width, dtype=np.float64)
    return {
        "family": "imagenet9",
        "case_id": case_id,
        "model_id": case_id,
        "model_kind": kind,
        "architecture_family": architecture_family,
        "checkpoints": [
            {
                "path": str(checkpoint.resolve()),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "bytes": checkpoint.stat().st_size,
            }
        ],
        "sample_ids": ["pair:mixed_same"],
        "preprocessed_tensor": {
            "sha256": "a" * 64,
            "dtype": "float32",
            "shape": [1, 3, 224, 224],
            "byte_order": "little-endian",
            "layout": "C-contiguous",
        },
        "target_class": 0,
        "logits": np.zeros((1, width)).tolist(),
        "probabilities": probabilities.tolist(),
        "precision": "float32",
        "device": "cuda:0",
    }


def test_fingerprint_validator_requires_exact_three_case_coverage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"offline checkpoint")
    records = [
        _fingerprint_case(
            checkpoint,
            case_id="imagenet9_off_the_shelf",
            kind="off_the_shelf",
            architecture_family="cnn",
            width=1000,
        ),
        _fingerprint_case(
            checkpoint,
            case_id="imagenet9_finetuned_cnn",
            kind="fine_tuned",
            architecture_family="cnn",
            width=9,
        ),
        _fingerprint_case(
            checkpoint,
            case_id="imagenet9_finetuned_transformer",
            kind="fine_tuned",
            architecture_family="transformer",
            width=9,
        ),
    ]
    assert validate_checkpoint_fingerprint_records(records) == records
    with pytest.raises(ValueError, match="exactly three"):
        validate_checkpoint_fingerprint_records(records[:2])
    records[2]["architecture_family"] = "cnn"
    with pytest.raises(ValueError, match="cover"):
        validate_checkpoint_fingerprint_records(records)
