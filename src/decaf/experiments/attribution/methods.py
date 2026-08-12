"""Attribution method registry and the sole DECAF trajectory adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decaf.core.trajectories import trajectory_scores


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Scientific and resource contract for one attribution method."""

    method_id: str
    display_name: str
    access: str
    family: str
    stages: int | None = None
    analysis_only: bool = False


METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec("decaf_3", "DECAF-3", "forward_only", "decaf", 3),
    MethodSpec("decaf_5", "DECAF-5", "forward_only", "decaf", 5),
    MethodSpec("decaf_9", "DECAF-9", "forward_only", "decaf", 9),
    MethodSpec("endpoint_m", "Endpoint M", "forward_only", "endpoint", analysis_only=True),
    MethodSpec("input_x_gradient", "Input x Gradient", "gradient", "gradient"),
    MethodSpec("ig_16", "IG-16", "gradient", "integrated_gradients"),
    MethodSpec("ig_32", "IG-32", "gradient", "integrated_gradients"),
    MethodSpec("ig_u_32", "IG-U-32", "gradient", "integrated_gradients"),
    MethodSpec("deep_lift", "DeepLIFT", "internal_activation", "deep_lift"),
    MethodSpec("gradient_shap", "GradientSHAP", "gradient", "gradient_shap"),
    MethodSpec("smoothgrad_16", "SmoothGrad-16", "gradient", "smoothgrad"),
    MethodSpec("rise_512", "RISE-512", "forward_only", "rise"),
    MethodSpec("rise_u_512", "RISE-U-512", "forward_only", "rise"),
    MethodSpec("kernel_shap_512", "KernelSHAP-512", "forward_only", "kernel_shap"),
    MethodSpec("part_lime_1000", "Part-LIME-1000", "forward_only", "part_lime"),
    MethodSpec("part_occlusion", "Part Occlusion", "forward_only", "part_oracle"),
    MethodSpec(
        "exact_part_shapley",
        "Exact Part-Shapley",
        "forward_only",
        "part_oracle",
    ),
)

METHODS = {spec.method_id: spec for spec in METHOD_SPECS}

MAIN_METHODS = (
    "decaf_3",
    "decaf_5",
    "decaf_9",
    "input_x_gradient",
    "ig_16",
    "ig_32",
    "ig_u_32",
    "deep_lift",
    "gradient_shap",
    "smoothgrad_16",
    "rise_512",
    "rise_u_512",
    "kernel_shap_512",
)
FUNNYBIRDS_SUPPLEMENT_METHODS = ("ig_u_32", "rise_u_512")
FUNNYBIRDS_PRIMARY_METHODS = (
    "decaf_3",
    "decaf_5",
    "decaf_9",
    "deep_lift",
    "gradient_shap",
    "ig_16",
    "ig_32",
    "input_x_gradient",
    "kernel_shap_512",
    "rise_512",
    "smoothgrad_16",
)
FULL50K_METHODS = ("decaf_5", "ig_32", "ig_u_32")
LARGE_MODEL_METHODS = (
    "decaf_3",
    "decaf_5",
    "decaf_9",
    "deep_lift",
    "gradient_shap",
    "ig_16",
    "ig_32",
    "smoothgrad_16",
)
VERIFY_MAIN_METHODS = (
    "decaf_3",
    "decaf_5",
    "decaf_9",
    "deep_lift",
    "ig_32",
    "ig_u_32",
    "gradient_shap",
    "smoothgrad_16",
    "rise_512",
    "kernel_shap_512",
)
VERIFY_RESUME_METHODS = (
    "decaf_3",
    "ig_32",
    "rise_512",
    "kernel_shap_512",
)
VERIFY_LARGE_MODEL_METHODS = LARGE_MODEL_METHODS
VERIFY_BOUNDARY_METHODS = (
    "decaf_5",
    "ig_32",
    "part_occlusion",
    "kernel_shap_512",
    "exact_part_shapley",
)
BOUNDARY_METHODS = (
    "decaf_3",
    "decaf_5",
    "decaf_9",
    "deep_lift",
    "endpoint_m",
    "exact_part_shapley",
    "gradient_shap",
    "ig_16",
    "ig_32",
    "input_x_gradient",
    "kernel_shap_512",
    "part_lime_1000",
    "part_occlusion",
    "rise_512",
    "smoothgrad_16",
)
BOUNDARY_COMPUTE_METHODS = tuple(method for method in BOUNDARY_METHODS if method != "endpoint_m")


def get_method(method_id: str) -> MethodSpec:
    """Resolve a registered method ID."""

    try:
        return METHODS[method_id]
    except KeyError as error:
        raise KeyError(f"unknown attribution method: {method_id}") from error


def validate_compute_methods(method_ids: tuple[str, ...]) -> tuple[MethodSpec, ...]:
    """Reject analysis-only methods from a compute plan."""

    methods = tuple(get_method(method_id) for method_id in method_ids)
    invalid = [method.method_id for method in methods if method.analysis_only]
    if invalid:
        raise ValueError(f"analysis-only methods cannot be compute members: {invalid}")
    return methods


def decaf_trajectory(
    method_id: str,
    grid: Any,
    response: Any,
    endpoint: Any | None = None,
    *,
    axis: int = -1,
    epsilon: float = 0.02,
) -> dict[str, Any]:
    """Evaluate a registered DECAF schedule through the authoritative core."""

    method = get_method(method_id)
    if method.family != "decaf" or method.stages is None:
        raise ValueError(f"method is not a DECAF trajectory schedule: {method_id}")
    try:
        stage_count = len(grid)
    except TypeError as error:
        raise TypeError("grid must be a sized trajectory") from error
    if stage_count != method.stages:
        raise ValueError(f"{method_id} requires {method.stages} stages, received {stage_count}")
    return trajectory_scores(grid, response, endpoint, epsilon, axis=axis)


__all__ = [
    "BOUNDARY_COMPUTE_METHODS",
    "BOUNDARY_METHODS",
    "FULL50K_METHODS",
    "FUNNYBIRDS_SUPPLEMENT_METHODS",
    "FUNNYBIRDS_PRIMARY_METHODS",
    "LARGE_MODEL_METHODS",
    "MAIN_METHODS",
    "METHODS",
    "MethodSpec",
    "VERIFY_BOUNDARY_METHODS",
    "VERIFY_LARGE_MODEL_METHODS",
    "VERIFY_MAIN_METHODS",
    "VERIFY_RESUME_METHODS",
    "decaf_trajectory",
    "get_method",
    "validate_compute_methods",
]
