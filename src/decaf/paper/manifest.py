"""Typed loading and validation for paper replay manifests."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ManifestError(ValueError):
    """Raised when a paper manifest is incomplete or internally inconsistent."""


SOURCE_MISSING_CONTRACT_FIELDS = (
    "missing_item",
    "why_it_matters",
    "reproducible_scope",
    "required_recovery_action",
)


@dataclass(frozen=True)
class RawInput:
    """One machine-readable member of a sealed reference archive."""

    run_id: str
    member: str


@dataclass(frozen=True)
class VisualAsset:
    """A paper figure or table and its replay contract."""

    asset_id: str
    kind: str
    number: int
    title: str
    run_ids: tuple[str, ...]
    raw_inputs: tuple[RawInput, ...]
    generator: str
    tex_target: str
    status: str
    generation_contract: Mapping[str, Any]
    headline_assertions: tuple[Mapping[str, Any], ...]
    source_note: str | None = None


@dataclass(frozen=True)
class VisualManifest:
    """Validated collection of all current paper assets."""

    schema_version: int
    assets: Mapping[str, VisualAsset]


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot load YAML manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError(f"manifest {path} must contain a mapping")
    return payload


def _expected_asset_ids() -> set[str]:
    figures = {f"figure_{number:02d}" for number in range(1, 13)}
    tables = {f"table_{number:02d}" for number in range(1, 17)}
    return figures | tables


def load_visual_manifest(path: str | Path) -> VisualManifest:
    """Load and fully validate the figure/table provenance map."""

    source = Path(path)
    payload = _load_yaml(source)
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, Mapping):
        raise ManifestError("visual manifest requires an assets mapping")
    ids = set(raw_assets)
    expected = _expected_asset_ids()
    if ids != expected:
        missing = sorted(expected - ids)
        extra = sorted(ids - expected)
        raise ManifestError(
            f"visual asset IDs differ from the paper contract: missing={missing}, extra={extra}"
        )

    assets: dict[str, VisualAsset] = {}
    targets: set[str] = set()
    assertion_ids: set[str] = set()
    for asset_id, raw in raw_assets.items():
        if not isinstance(raw, Mapping):
            raise ManifestError(f"{asset_id} must be a mapping")
        kind = str(raw.get("kind", ""))
        number = int(raw.get("number", 0))
        expected_id = f"{kind}_{number:02d}"
        if kind not in {"figure", "table"} or expected_id != asset_id:
            raise ManifestError(f"{asset_id} has inconsistent kind/number")
        run_ids = tuple(str(item) for item in raw.get("run_ids", ()))
        inputs: list[RawInput] = []
        for item in raw.get("raw_inputs", ()):
            if not isinstance(item, Mapping) or not item.get("run_id") or not item.get("member"):
                raise ManifestError(f"{asset_id} contains an invalid raw input")
            value = RawInput(str(item["run_id"]), str(item["member"]))
            if value.run_id not in run_ids:
                raise ManifestError(f"{asset_id} input references undeclared run {value.run_id}")
            inputs.append(value)
        contract = raw.get("generation_contract")
        if not isinstance(contract, Mapping) or not contract.get("operation"):
            raise ManifestError(f"{asset_id} requires a generation contract")
        assertions = raw.get("headline_assertions", [])
        if not isinstance(assertions, list):
            raise ManifestError(f"{asset_id} headline assertions must be a list")
        for assertion in assertions:
            if not isinstance(assertion, Mapping) or not assertion.get("id"):
                raise ManifestError(f"{asset_id} contains an invalid headline assertion")
            assertion_id = str(assertion["id"])
            if assertion_id in assertion_ids:
                raise ManifestError(f"duplicate headline assertion ID: {assertion_id}")
            assertion_ids.add(assertion_id)
            if "expected" not in assertion:
                raise ManifestError(
                    f"{asset_id}/{assertion_id} must declare a numerical expectation"
                )
        tex_target = str(raw.get("tex_target", ""))
        if not tex_target.endswith(".tex") or tex_target in targets:
            raise ManifestError(f"{asset_id} has an invalid or duplicate TeX target")
        targets.add(tex_target)
        status = str(raw.get("status", "ready"))
        contract_status = str(contract.get("status", ""))
        expected_contract_status = (
            "source_missing" if status == "source_missing" else "replay_derived"
        )
        if contract_status != expected_contract_status:
            raise ManifestError(
                f"{asset_id} generation contract status must be "
                f"{expected_contract_status}, received {contract_status}"
            )
        if status == "source_missing" and inputs:
            raise ManifestError(f"{asset_id} is source_missing but declares raw inputs")
        if status == "source_missing":
            missing_gap_fields = [
                field
                for field in SOURCE_MISSING_CONTRACT_FIELDS
                if not str(contract.get(field, "")).strip()
            ]
            if missing_gap_fields:
                raise ManifestError(
                    f"{asset_id} source-missing contract omits {', '.join(missing_gap_fields)}"
                )
        if status != "source_missing" and not inputs:
            raise ManifestError(f"{asset_id} must declare at least one machine-readable input")
        assets[asset_id] = VisualAsset(
            asset_id=asset_id,
            kind=kind,
            number=number,
            title=str(raw.get("title", asset_id)),
            run_ids=run_ids,
            raw_inputs=tuple(inputs),
            generator=str(raw.get("generator", "")),
            tex_target=tex_target,
            status=status,
            generation_contract=dict(contract),
            headline_assertions=tuple(assertions),
            source_note=str(raw["source_note"]) if raw.get("source_note") else None,
        )
    return VisualManifest(schema_version=int(payload.get("schema_version", 0)), assets=assets)


def load_representative_cases(path: str | Path) -> Mapping[str, Any]:
    """Load deterministic representative-case rules and frozen resolutions."""

    payload = _load_yaml(Path(path))
    cases = payload.get("cases")
    if not isinstance(cases, Mapping):
        raise ManifestError("representative case manifest requires a cases mapping")
    expected = {"figure_02", "figure_03", "figure_04"}
    if set(cases) != expected:
        raise ManifestError(f"representative cases must be exactly {sorted(expected)}")
    for case_id, case in cases.items():
        if not isinstance(case, Mapping) or not isinstance(case.get("rule"), Mapping):
            raise ManifestError(f"{case_id} requires a machine-readable rule")
        if not isinstance(case.get("resolved"), Mapping):
            raise ManifestError(f"{case_id} requires resolved metadata")
    return payload


def import_generator(dotted_name: str) -> Callable[..., str]:
    """Resolve a manifest generator only after its dotted path has been validated."""

    module_name, separator, attribute = dotted_name.rpartition(".")
    if not separator or not module_name.startswith("decaf.paper."):
        raise ManifestError(f"paper generator must live below decaf.paper: {dotted_name}")
    generator = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(generator):
        raise ManifestError(f"paper generator is not callable: {dotted_name}")
    return generator


def repository_root() -> Path:
    """Return the repository root from this installed source tree."""

    return Path(__file__).resolve().parents[3]


__all__ = [
    "ManifestError",
    "RawInput",
    "VisualAsset",
    "VisualManifest",
    "import_generator",
    "load_representative_cases",
    "load_visual_manifest",
    "repository_root",
]
