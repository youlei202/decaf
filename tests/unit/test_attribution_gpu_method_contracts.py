from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decaf.experiments.attribution.endpoint import append_endpoint_m, row_spearman
from decaf.experiments.attribution.evaluate import _operatorwise_heldout_spearman
from decaf.experiments.attribution.gpu_runtime import (
    FUNNYBIRDS_SUPPLEMENT_INPUT_DOMAIN,
    GRADIENT_SHAP_SEED,
    KERNEL_SHAP_SEED,
    MODEL_INPUT_DOMAIN,
    RAW_RGB_INPUT_DOMAIN,
    RISE_MASK_SEED,
    SMOOTHGRAD_SEED,
    UNIFORM_BASELINE_SEED,
    AttributionSample,
    PreparedSample,
    _coalitions,
    _correct,
    _fixed_uniform_baseline,
    _gauss_legendre_rule,
    _gradient_shap,
    _idsds_candidates,
    _idsds_pil_preprocess,
    _idsds_rise_masks,
    _kernel_shap,
    _kernel_shap_coalitions,
    _method,
    _native_rise_masks,
    _normalize_imagenet,
    _prepare_runtime_sample,
    _quality_frame,
    _raw_rgb_model,
    _resize_crop,
    _smoothgrad,
    _stable_method_seed,
    preprocess_sample,
)


def _sha256_tensor(value: object) -> str:
    array = value.detach().cpu().contiguous().numpy()  # type: ignore[attr-defined]
    return hashlib.sha256(array.tobytes()).hexdigest()


def _masks(torch: object, groups: int = 4) -> object:
    if groups == 16:
        return torch.eye(16, dtype=torch.float32).reshape(16, 4, 4)
    result = torch.zeros((4, 4, 4), dtype=torch.float32)
    result[0, :2, :2] = 1.0
    result[1, :2, 2:] = 1.0
    result[2, 2:, :2] = 1.0
    result[3, 2:, 2:] = 1.0
    return result


def _sample(torch: object, dataset: str = "imagenet1k_idsds") -> PreparedSample:
    return PreparedSample(
        dataset=dataset,
        image_id="golden-0",
        target=0,
        image=torch.linspace(-1.0, 1.0, 48, dtype=torch.float32).reshape(3, 4, 4),
        masks=_masks(torch),
        reference=torch.zeros((3, 4, 4), dtype=torch.float32),
        interventions={},
        part_names=("a", "b", "c", "d"),
        raw_height=4,
        raw_width=4,
    )


def _counting_model(torch: object, classes: int = 1_000) -> object:
    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_rows = 0

        def forward(self, value: object) -> object:
            self.forward_rows += int(value.shape[0])
            score = value.square().sum(dim=(1, 2, 3))
            zeros = torch.zeros(
                (int(value.shape[0]), classes - 1),
                device=value.device,
                dtype=value.dtype,
            )
            return torch.cat((score[:, None], zeros), dim=1)

    return CountingModel()


def _raw_sample(torch: object, dataset: str, *, height: int, width: int) -> AttributionSample:
    image = torch.linspace(0.0, 1.0, 3 * height * width).reshape(3, height, width)
    masks = torch.zeros((2, height, width), dtype=torch.float32)
    masks[0, : height // 2] = 1.0
    masks[1, height // 2 :] = 1.0
    return AttributionSample(
        dataset=dataset,
        image_id=f"{dataset}-raw",
        target=0,
        image=image,
        masks=masks,
        reference=torch.zeros_like(image),
        interventions={"telea": image.unsqueeze(0), "background_texture": image.unsqueeze(0)},
        part_names=("top", "bottom"),
        raw_height=height,
        raw_width=width,
    )


def _differentiable_classifier(torch: object, classes: int) -> object:
    class Classifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: list[object] = []

        def forward(self, value: object) -> object:
            self.inputs.append(value.detach().clone())
            score = value.square().mean(dim=(1, 2, 3)) + 1.0
            zeros = torch.zeros(
                (int(value.shape[0]), classes - 1),
                device=value.device,
                dtype=value.dtype,
            )
            return torch.cat((score[:, None], zeros), dim=1)

    return Classifier()


def test_raw_adapter_matches_canonical_part_and_funny_vit_model_views() -> None:
    torch = pytest.importorskip("torch")
    for dataset, model_id, shape, classes in (
        ("partimagenet", "resnet50", (7, 13), 1_000),
        ("funnybirds", "funnybirds_vit_b_16", (11, 17), 50),
    ):
        raw = _raw_sample(torch, dataset, height=shape[0], width=shape[1])
        runtime = _prepare_runtime_sample(raw, model_id, dataset=dataset)
        canonical = preprocess_sample(runtime, model_id, dataset=dataset)
        bare = _differentiable_classifier(torch, classes)
        adapter = _raw_rgb_model(bare, model_id, dataset)

        raw_output = adapter(runtime.image.unsqueeze(0))
        canonical_output = adapter.forward_preprocessed(canonical.image.unsqueeze(0))

        assert torch.equal(raw_output, canonical_output)
        assert tuple(runtime.image.shape[-2:]) == shape
        assert tuple(runtime.masks.shape[-2:]) == shape
        assert tuple(canonical.image.shape[-2:]) == (224, 224)


def test_part_smoothgrad_clamps_raw_rgb_before_model_normalization() -> None:
    torch = pytest.importorskip("torch")
    runtime = _prepare_runtime_sample(
        _raw_sample(torch, "partimagenet", height=5, width=9),
        "resnet50",
        dataset="partimagenet",
    )
    bare = _differentiable_classifier(torch, 1_000)
    adapter = _raw_rgb_model(bare, "resnet50", "partimagenet")
    captured: list[object] = []
    adapter.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )

    scores, metadata = _smoothgrad(
        adapter,
        runtime.image,
        runtime.masks,
        runtime.target,
        runtime.dataset,
        seed=17,
    )

    assert scores.shape == (2,)
    assert captured and float(captured[0].amin()) >= 0.0
    assert float(captured[0].amax()) <= 1.0
    expected = _normalize_imagenet(_resize_crop(captured[0], 224, mode="bilinear"))
    assert torch.equal(bare.inputs[0], expected)
    assert metadata["noise_space"] == RAW_RGB_INPUT_DOMAIN


def test_runtime_domain_metadata_and_canonical_correctness_are_explicit() -> None:
    torch = pytest.importorskip("torch")
    runtime = _prepare_runtime_sample(
        _raw_sample(torch, "partimagenet", height=6, width=10),
        "resnet50",
        dataset="partimagenet",
    )
    bare = _differentiable_classifier(torch, 1_000)
    adapter = _raw_rgb_model(bare, "resnet50", "partimagenet")

    assert _correct(adapter, runtime, "cpu", "fp32")
    assert adapter.raw_forward_calls == 0
    assert adapter.preprocessed_forward_calls == 1
    scores, metadata = _method("decaf_3", adapter, runtime, device="cpu", precision="fp32", seed=3)
    assert scores.shape == (2,)
    assert metadata["input_domain"] == RAW_RGB_INPUT_DOMAIN
    assert metadata["model_preprocess_inside_forward"] is True
    assert metadata["cross_image_coalescing"] is False
    assert metadata["preprocess_before_cross_image_coalescing"] is True
    assert adapter.raw_forward_calls > 0

    funny = _prepare_runtime_sample(
        _raw_sample(torch, "funnybirds", height=256, width=256),
        "funnybirds_resnet50",
        dataset="funnybirds",
    )
    funny_bare = _differentiable_classifier(torch, 50)
    funny_adapter = _raw_rgb_model(funny_bare, "funnybirds_resnet50", "funnybirds")
    _, supplement = _method("ig_u_32", funny_adapter, funny, device="cpu", precision="fp32", seed=9)
    assert supplement["input_domain"] == FUNNYBIRDS_SUPPLEMENT_INPUT_DOMAIN
    assert supplement["model_preprocess_inside_forward"] is False
    assert funny_adapter.raw_forward_calls == 0
    assert MODEL_INPUT_DOMAIN != RAW_RGB_INPUT_DOMAIN


def test_decaf_primary_scores_are_unsigned_e_for_negative_endpoints() -> None:
    torch = pytest.importorskip("torch")
    sample = _sample(torch)

    class SignedLinearModel(torch.nn.Module):
        def forward(self, value: object) -> object:
            score = value.sum(dim=(1, 2, 3))
            zeros = torch.zeros(
                (int(value.shape[0]), 999),
                device=value.device,
                dtype=value.dtype,
            )
            return torch.cat((score[:, None], zeros), dim=1)

    scores, metadata = _method(
        "decaf_3",
        SignedLinearModel(),
        sample,
        device="cpu",
        precision="fp32",
        seed=3,
    )
    endpoint = np.asarray(
        [(sample.image * mask).sum().item() for mask in sample.masks],
        dtype=np.float64,
    )
    unsigned_e = np.asarray(metadata["E"], dtype=np.float64)
    signed_e = np.sign(endpoint) * unsigned_e

    assert bool((endpoint < 0.0).any())
    assert bool((unsigned_e > 0.0).any())
    np.testing.assert_allclose(scores, unsigned_e, rtol=0.0, atol=0.0)
    assert bool((scores >= 0.0).all())
    assert not np.array_equal(scores, signed_e)


def test_idsds_random_banks_and_gauss_legendre_rule_match_frozen_goldens() -> None:
    torch = pytest.importorskip("torch")
    image = torch.empty((3, 9, 11), dtype=torch.float32)

    baseline = _fixed_uniform_baseline(image)
    idsds_masks = _idsds_rise_masks(9, 11, image)
    native_masks = _native_rise_masks(9, 11, image, seed=12_345)
    coalitions = _kernel_shap_coalitions(
        16, 512, KERNEL_SHAP_SEED, torch.device("cpu"), torch.float32
    )

    assert UNIFORM_BASELINE_SEED == 8212
    assert RISE_MASK_SEED == 8213
    assert _sha256_tensor(baseline) == (
        "35479dd012ae0c7f94e299db810bb22a196655375d5e43ee807316f835643ff6"
    )
    assert _sha256_tensor(idsds_masks) == (
        "265751115e67df0ee18edcfcf1667da50db4c1a7b1a1c6710829b52f84435a09"
    )
    assert _sha256_tensor(native_masks) == (
        "a24c41f3d83a5951dcede7b4704e4b68c1b2a871f081aa3e86c3acc3cfab0d4c"
    )
    assert _sha256_tensor(coalitions) == (
        "794b344437db3b1918ca6b99651444c2114d8d8e66033840fe3364aa0fa8a146"
    )
    assert torch.equal(coalitions[0], torch.zeros(16))
    assert torch.equal(coalitions[1], torch.ones(16))
    assert torch.equal(coalitions[2], 1.0 - coalitions[3])
    assert _stable_method_seed("job-0", "image-0") == 417_995_398

    nodes, weights = _gauss_legendre_rule(4, image)
    assert nodes.tolist() == pytest.approx(
        [0.06943184420297371, 0.33000947820757187, 0.6699905217924281, 0.9305681557970262],
        abs=1.0e-7,
    )
    assert weights.tolist() == pytest.approx(
        [0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269],
        abs=1.0e-7,
    )
    assert float(weights.sum()) == pytest.approx(1.0)


def test_real_idsds_manifest_sample_matches_frozen_pil_tensor_goldens() -> None:
    torch = pytest.importorskip("torch")
    manifest_value = os.environ.get("DECAF_IDSDS_MANIFEST")
    if not manifest_value:
        pytest.skip("DECAF_IDSDS_MANIFEST is required for the real-image golden")
    manifest = Path(manifest_value).resolve()
    assert manifest.is_file()
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        "3f6f9bad1c631f3eb95e8e2ae2fb171dd86470deaed7f3c93259feea952c0e79"
    )
    sample = _idsds_candidates(manifest, None, limit=1)[0]
    assert sample.image_id == "ILSVRC2012_val_00016018_n01440764"
    assert sample.target == 0
    assert tuple(sample.image.shape) == (3, 375, 500)

    cnn = _idsds_pil_preprocess(sample.image, "resnet50")
    vit = _idsds_pil_preprocess(sample.image, "vit_base_patch16_224")
    assert torch.isfinite(cnn).all() and torch.isfinite(vit).all()
    assert _sha256_tensor(cnn) == (
        "7954b97dcf3f094a012549d6fae323fb1df07f0b720129406008c0b3c10a8f71"
    )
    assert _sha256_tensor(vit) == (
        "67f2ac87c5a59a439308dd2fea24d342c51a4ba226069f40fe996be4b137757d"
    )


def test_idsds_gradient_random_banks_ignore_per_image_seed() -> None:
    torch = pytest.importorskip("torch")
    sample = _sample(torch)
    first_model = _counting_model(torch)
    second_model = _counting_model(torch)
    first_smooth, first_metadata = _smoothgrad(
        first_model,
        sample.image,
        sample.masks,
        sample.target,
        sample.dataset,
        seed=1,
    )
    second_smooth, second_metadata = _smoothgrad(
        second_model,
        sample.image,
        sample.masks,
        sample.target,
        sample.dataset,
        seed=999,
    )
    assert torch.equal(first_smooth, second_smooth)
    assert first_metadata["seed"] == second_metadata["seed"] == SMOOTHGRAD_SEED
    assert first_metadata["random_bank"] == "shared_numpy_randomstate"

    first_model = _counting_model(torch)
    second_model = _counting_model(torch)
    first_shap, first_metadata = _gradient_shap(
        first_model,
        sample.image,
        sample.reference,
        sample.masks,
        sample.target,
        sample.dataset,
        seed=1,
    )
    second_shap, second_metadata = _gradient_shap(
        second_model,
        sample.image,
        sample.reference,
        sample.masks,
        sample.target,
        sample.dataset,
        seed=999,
    )
    assert torch.equal(first_shap, second_shap)
    assert first_metadata["seed"] == second_metadata["seed"] == GRADIENT_SHAP_SEED
    assert first_metadata["random_bank"] == "shared_numpy_randomstate"


def test_method_and_dependency_quality_do_not_hide_endpoint_forward_rows() -> None:
    torch = pytest.importorskip("torch")
    sample = _sample(torch)
    model = _counting_model(torch)
    _scores, metadata = _method("ig_32", model, sample, device="cpu", precision="fp32", seed=7)
    assert model.forward_rows == metadata["forward_rows"] == 32
    assert metadata["quadrature"] == "gauss_legendre"

    native_sample = _sample(torch, dataset="funnybirds")
    native_model = _counting_model(torch, classes=50)
    _scores, native_metadata = _method(
        "ig_32",
        native_model,
        native_sample,
        device="cpu",
        precision="fp32",
        seed=7,
    )
    assert native_model.forward_rows == native_metadata["forward_rows"] == 32
    assert native_metadata["quadrature"] == "endpoint_trapezoid"

    supplement_model = _counting_model(torch, classes=50)
    _scores, supplement_metadata = _method(
        "ig_u_32",
        supplement_model,
        native_sample,
        device="cpu",
        precision="fp32",
        seed=7,
    )
    assert supplement_metadata["quadrature"] == "gauss_legendre"
    assert supplement_metadata["baseline_seed"] == UNIFORM_BASELINE_SEED

    model = _counting_model(torch)
    frame = _quality_frame(
        {
            "image_start": 0,
            "member_id": "idsds-quality",
            "scope": "smoke_idsds_primary",
            "dataset": "imagenet1k_idsds",
            "model_id": "resnet50",
            "method_id": "ig_32",
            "depends_on": [{"member_id": "shared-target"}],
        },
        [sample],
        model,
        device="cpu",
        precision="fp32",
    )
    assert model.forward_rows == 32
    assert frame.loc[0, "endpoint_effects"].tolist() == [0.0] * 4


def test_dependency_free_quality_persists_endpoint_magnitude() -> None:
    torch = pytest.importorskip("torch")
    sample = _sample(torch)
    model = _counting_model(torch)
    frame = _quality_frame(
        {
            "image_start": 0,
            "member_id": "dinov2-quality",
            "scope": "smoke_dinov2_g_quality",
            "dataset": "imagenet1k_idsds",
            "model_id": "dinov2_vit_g_14",
            "method_id": "ig_16",
            "depends_on": [],
        },
        [sample],
        model,
        device="cpu",
        precision="fp32",
    )

    endpoint = np.asarray(frame.loc[0, "endpoint_effects"], dtype=np.float64)
    endpoint_m = np.asarray(frame.loc[0, "decaf_M"], dtype=np.float64)
    assert np.array_equal(endpoint_m, np.abs(endpoint))


def test_kernelshap_budget_design_and_exact_local_accuracy() -> None:
    torch = pytest.importorskip("torch")
    model = _counting_model(torch)
    image = torch.linspace(0.0, 1.0, 48, dtype=torch.float32).reshape(3, 4, 4)
    baseline = torch.zeros_like(image)
    masks = _masks(torch, groups=16)

    scores, metadata = _kernel_shap(
        model,
        image,
        baseline,
        masks,
        0,
        "imagenet1k_idsds",
        seed=1,
    )
    total_effect = image.square().sum().double()
    assert model.forward_rows == metadata["queries"] == 512
    assert metadata["seed"] == KERNEL_SHAP_SEED
    assert metadata["coalition_design"] == ("endpoint_first_subset_size_complement_balanced")
    assert scores.sum() == pytest.approx(float(total_effect), abs=1.0e-10)
    assert metadata["local_accuracy_abs_residual"] <= 1.0e-12

    native_coalitions = _coalitions(4, 512, 12_345, torch.device("cpu"), torch.float32)
    assert _sha256_tensor(native_coalitions) == (
        "839dc11e6ff875cbaabca0a8ac2916891039041c2ec3d24e89dcc19bcebffb25"
    )
    assert native_coalitions[:16].int().tolist() == [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ]

    native_model = _counting_model(torch, classes=50)
    native_scores, native_metadata = _kernel_shap(
        native_model,
        image,
        baseline,
        _masks(torch),
        0,
        "funnybirds",
        seed=12_345,
    )
    assert native_scores.shape == (4,)
    assert native_model.forward_rows == native_metadata["queries"] == 512
    assert native_metadata["coalition_design"] == ("endpoint_first_exhaustive_then_torch_repeats")
    assert native_metadata["local_accuracy_correction"] is False


def test_heldout_quality_averages_operator_spearman_not_effect_vectors() -> None:
    patch = np.asarray([0.0, 1.0, 2.0, 3.0])
    background = np.asarray([0.0, 2.0, 3.0, 1.0])
    telea = np.asarray([2.0, 1.0, 0.0, 3.0])
    first, second, averaged = _operatorwise_heldout_spearman([patch], [background], [telea])
    vector_mean_score = float(row_spearman(patch, (background + telea) / 2.0)[0])
    assert first == pytest.approx([0.4])
    assert second == pytest.approx([0.2])
    assert averaged == pytest.approx([0.3])
    assert vector_mean_score == pytest.approx(0.9486832980505138)
    assert averaged[0] != pytest.approx(vector_mean_score)

    constant_patch = np.ones(4, dtype=np.float64)
    constant_effect = np.zeros(4, dtype=np.float64)
    safe_first, safe_second, safe_average = _operatorwise_heldout_spearman(
        [constant_patch], [constant_effect], [telea]
    )
    assert safe_first == [0.0]
    assert safe_second == [0.0]
    assert safe_average == [0.0]

    frame = pd.DataFrame(
        [
            {
                "scope": "funnybirds_primary",
                "dataset": "funnybirds",
                "model": "funnybirds_resnet50",
                "method": "decaf_5",
                "image_id": "bird-0",
                "endpoint_effects": np.asarray([-1.0, 2.0, -3.0, 4.0]),
                "quality_target_effects": (background + telea) / 2.0,
                "decaf_M": np.asarray([1.0, 2.0, 3.0, 4.0]),
                "patch_scores": patch,
                "spearman": averaged[0],
                "quality_aggregation": "equal_mean_of_operator_spearman",
                "heldout_background_texture_effects": background,
                "heldout_telea_dilate3_effects": telea,
                "heldout_background_texture_spearman": first[0],
                "heldout_telea_dilate3_spearman": second[0],
            }
        ]
    )
    combined, _audit = append_endpoint_m(frame)
    endpoint = combined.loc[combined["method"] == "endpoint_m"].iloc[0]
    expected = 0.5 * (
        float(row_spearman(frame.loc[0, "decaf_M"], background)[0])
        + float(row_spearman(frame.loc[0, "decaf_M"], telea)[0])
    )
    assert float(endpoint["spearman"]) == pytest.approx(expected)
