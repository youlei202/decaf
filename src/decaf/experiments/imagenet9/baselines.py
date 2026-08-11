"""Registered ImageNet-9 saliency baselines and compatibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BaselineSpec:
    """Access and query contract for one baseline."""

    method_id: str
    access: str
    nominal_queries: int
    requires_gradients: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


BASELINES = {
    item.method_id: item
    for item in (
        BaselineSpec("input_x_gradient", "white_box", 2, True),
        BaselineSpec("integrated_gradients", "white_box", 16, True),
        BaselineSpec("smoothgrad", "white_box", 16, True),
        BaselineSpec("blur_ig", "white_box", 12, True),
        BaselineSpec("occlusion", "forward_only", 49, False),
        BaselineSpec("rise", "forward_only", 256, False),
    )
}


def baseline_plan(method_ids: list[str]) -> list[dict[str, object]]:
    """Resolve a method list and reject unknown or duplicate entries."""

    if len(method_ids) != len(set(method_ids)):
        raise ValueError("baseline method IDs must be unique")
    unknown = sorted(set(method_ids) - set(BASELINES))
    if unknown:
        raise ValueError(f"unknown ImageNet-9 baselines: {unknown}")
    return [BASELINES[method_id].as_dict() for method_id in method_ids]


def method_model_compatible(method_id: str, *, gradient_access: bool) -> bool:
    """Return whether the registered access requirement is satisfied."""

    specification = BASELINES[method_id]
    return gradient_access or not specification.requires_gradients


__all__ = [
    "BASELINES",
    "BaselineSpec",
    "baseline_plan",
    "method_model_compatible",
]
