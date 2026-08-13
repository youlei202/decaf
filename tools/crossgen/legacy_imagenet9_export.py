"""Exact-unit ImageNet-9 trajectory export through the frozen historical runtime.

The historical formal archive sealed sample-level DECAF summaries but not raw
stage probabilities.  This verification-only adapter therefore calls the
frozen historical preprocessing, reveal, probability, and checkpoint-loading
code to regenerate only factual/counterfactual scores and path identity.  All
new decomposition is performed by the current repository.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.experiments.common import atomic_json, atomic_text
from decaf.experiments.imagenet9.gpu_models import (
    canonical_tensor_identity,
    preprocess_image,
)
from decaf.experiments.imagenet9.pairs import normalize_wide_manifest
from tools.crossgen.schema import (
    NEUTRAL_COLUMNS,
    sha256_file,
    trapezoid_weights,
    validate_trajectory_record,
)

# Imports from the evidence repository must never create bytecode beside the
# frozen sources.
sys.dont_write_bytecode = True

HISTORICAL_REPOSITORY = Path("/work/Users/leiyo/GitHub/covariance-matched-markov-revelation")
HISTORICAL_SOURCE = HISTORICAL_REPOSITORY / "src"
HISTORICAL_CONFIG = HISTORICAL_REPOSITORY / "configs/decaf_imagenet9_v1/formal_8b200.yaml"
DATASET_ROOT = Path("/work/Users/leiyo/decaf_imagenet9_v1_data")
HISTORICAL_MANIFEST = DATASET_ROOT / "manifests/paired_variants.parquet"
HISTORICAL_RESULTS = Path("/work/Users/leiyo/decaf_imagenet9_v1_results")
HISTORICAL_DESCRIPTORS = HISTORICAL_RESULTS / "inventory/model_descriptors.json"
HISTORICAL_STAGE_LEDGER = HISTORICAL_RESULTS / "stage_ledger.jsonl"
HISTORICAL_PACKAGE = (
    HISTORICAL_RESULTS
    / "packages/decaf_imagenet9_v1_20260808T164818Z_lightweight.zip"
)
HISTORICAL_PACKAGE_SHA256 = (
    "3bae5ac670f6731d8a7832c3f9d7051e308a3f322c6192068bc11868be3821cc"
)
HISTORICAL_PACKAGE_PREFIX = "decaf_imagenet9_v1"
HISTORICAL_PACKAGE_MANIFEST_MEMBER = (
    f"{HISTORICAL_PACKAGE_PREFIX}/PACKAGE_MANIFEST.json"
)
HISTORICAL_PACKAGE_CODE_PREFIX = f"{HISTORICAL_PACKAGE_PREFIX}/code"
HISTORICAL_MODULE_NAMESPACE = "cmr.decaf_imagenet9_v1"
WEIGHT_CACHE_ROOT = Path("/work/Users/leiyo/decaf_imagenet9_v1_ready/cache")
DEFAULT_OUTPUT_ROOT = Path("/work/Users/leiyo/decaf_cross_generation_equivalence/v2")

REFERENCE_RUN = "I9"
ALPHA = tuple(float(value) for value in np.linspace(0.0, 1.0, 9))
EPSILON = 0.02
PATCH_SEED = 7101
PATCH_GRID = (8, 8)
REVEAL_PATHS = ("blend", "patch_A", "patch_B")
PAIR_TYPES = ("same_rand", "same_next")
SOURCE_PAIR_IDS = (
    "00/n02107574_44618",
    "00/n02113186_16989",
    "01/n02017213_25091",
    "01/n02033041_45006",
    "02/n03417042_09345",
    "02/n04335435_04855",
    "03/n01734418_27631",
    "03/n01735189_46211",
)


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """One historical/current model identity pair."""

    historical_model_id: str
    current_model_id: str
    checkpoint: Path
    checkpoint_sha256: str

    @property
    def sample_path(self) -> Path:
        return HISTORICAL_RESULTS / "decaf_deep/jobs" / self.historical_model_id / "sample.parquet"


MODEL_BINDINGS = (
    ModelBinding(
        historical_model_id="tv_resnet18",
        current_model_id="tv_resnet18_imagenet1k_v1",
        checkpoint=WEIGHT_CACHE_ROOT / "torch/hub/checkpoints/resnet18-f37072fd.pth",
        checkpoint_sha256=("f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
    ),
    ModelBinding(
        historical_model_id="ft_resnet50_original_s7101",
        current_model_id="ft_resnet50_original_s7101",
        checkpoint=(HISTORICAL_RESULTS / "training/checkpoints/ft_resnet50_original_s7101/best.pt"),
        checkpoint_sha256=("fd88804bae846b971fbcac05236c82b2fc385a3ce1357d1aabd5b87dd5134130"),
    ),
    ModelBinding(
        historical_model_id="ft_vit_b_16_original_s7101",
        current_model_id="ft_vit_b_16_original_s7101",
        checkpoint=(HISTORICAL_RESULTS / "training/checkpoints/ft_vit_b_16_original_s7101/best.pt"),
        checkpoint_sha256=("55d9c142dab8b4936971421c97939d95eab6e718c18d65b977545eab37fa95ef"),
    ),
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.part{destination.suffix}")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@lru_cache(maxsize=1)
def _historical_package_provenance() -> dict[str, Any]:
    """Validate and describe the sealed historical source snapshot.

    The historical repository stores this experiment namespace as untracked
    material, so its Git HEAD cannot bind the executed code.  The delivered
    lightweight ZIP is the source authority for this verification bridge.
    """

    package = HISTORICAL_PACKAGE.resolve()
    if package.is_symlink() or not package.is_file():
        raise FileNotFoundError(f"sealed ImageNet-9 package is missing or unsafe: {package}")
    package_sha256 = sha256_file(package)
    if package_sha256 != HISTORICAL_PACKAGE_SHA256:
        raise ValueError(
            "sealed ImageNet-9 package SHA-256 changed: "
            f"{package_sha256} != {HISTORICAL_PACKAGE_SHA256}"
        )

    with zipfile.ZipFile(package) as archive:
        try:
            manifest_bytes = archive.read(HISTORICAL_PACKAGE_MANIFEST_MEMBER)
            manifest = json.loads(manifest_bytes)
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("sealed ImageNet-9 package manifest is invalid") from error
        if not isinstance(manifest, dict):
            raise TypeError("sealed ImageNet-9 package manifest must be an object")
        members = manifest.get("members")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("namespace") != HISTORICAL_PACKAGE_PREFIX
            or manifest.get("lightweight") is not True
            or manifest.get("source_layout") != "code/cmr"
            or not isinstance(members, list)
            or manifest.get("recorded_member_count") != len(members)
            or manifest.get("manifest_self_entry")
            != "excluded_by_design_to_avoid_self_hash_recursion"
        ):
            raise ValueError("sealed ImageNet-9 package manifest contract changed")

        actual_files = {
            info.filename for info in archive.infolist() if not info.is_dir()
        }
        expected_files = {HISTORICAL_PACKAGE_MANIFEST_MEMBER}
        member_records: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(members):
            if not isinstance(record, dict):
                raise TypeError(f"sealed package member[{index}] is not an object")
            relative = record.get("path")
            expected_sha256 = record.get("sha256")
            expected_bytes = record.get("bytes")
            if (
                not isinstance(relative, str)
                or not relative
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0
                or relative in member_records
            ):
                raise ValueError(f"sealed package member[{index}] has invalid identity")
            archive_member = f"{HISTORICAL_PACKAGE_PREFIX}/{relative}"
            try:
                payload = archive.read(archive_member)
            except KeyError as error:
                raise ValueError(f"sealed package member is missing: {relative}") from error
            observed_sha256 = hashlib.sha256(payload).hexdigest()
            if len(payload) != expected_bytes or observed_sha256 != expected_sha256:
                raise ValueError(f"sealed package member identity changed: {relative}")
            expected_files.add(archive_member)
            member_records[relative] = {
                "archive_member": archive_member,
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            }
        if actual_files != expected_files:
            raise ValueError("sealed ImageNet-9 package file inventory differs from its manifest")

    module_prefix = "code/cmr/decaf_imagenet9_v1/"
    source_members = {
        relative: record
        for relative, record in sorted(member_records.items())
        if relative.startswith(module_prefix) and relative.endswith(".py")
    }
    required = {
        f"{module_prefix}{name}.py"
        for name in ("__init__", "data", "decaf", "models", "reveal", "run")
    }
    if not required.issubset(source_members):
        raise ValueError("sealed ImageNet-9 package lacks required runtime modules")
    return {
        "path": str(package),
        "sha256": package_sha256,
        "manifest_member": HISTORICAL_PACKAGE_MANIFEST_MEMBER,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "recorded_member_count": len(member_records),
        "zip_import_root": f"{package}/{HISTORICAL_PACKAGE_CODE_PREFIX}",
        "source_authority": "sha256-verified lightweight ZIP; not historical Git HEAD",
        "source_members": source_members,
    }


def _historical_modules() -> tuple[Any, Any, Any]:
    provenance = _historical_package_provenance()
    source = str(provenance["zip_import_root"])
    if source not in sys.path:
        sys.path.insert(0, source)
    from cmr.decaf_imagenet9_v1 import data as historical_data
    from cmr.decaf_imagenet9_v1 import reveal as historical_reveal
    from cmr.decaf_imagenet9_v1 import run as historical_run

    expected_origin = f"{source}/cmr/decaf_imagenet9_v1/"
    loaded = {
        name: getattr(module, "__file__", None)
        for name, module in sys.modules.items()
        if name == HISTORICAL_MODULE_NAMESPACE
        or name.startswith(f"{HISTORICAL_MODULE_NAMESPACE}.")
    }
    if not loaded or any(
        not isinstance(origin, str) or not origin.startswith(expected_origin)
        for origin in loaded.values()
    ):
        raise RuntimeError(
            "ImageNet-9 historical modules were not loaded exclusively from the "
            "SHA-verified package snapshot"
        )

    return historical_data, historical_reveal, historical_run


def load_selection() -> pd.DataFrame:
    """Load and validate the exact eight historical wide-manifest rows."""

    frame = pd.read_parquet(HISTORICAL_MANIFEST)
    selected = frame[frame["pair_id"].astype(str).isin(SOURCE_PAIR_IDS)].copy()
    if len(selected) != len(SOURCE_PAIR_IDS):
        found = set(selected["pair_id"].astype(str))
        raise ValueError(
            f"historical manifest is missing selected IDs: {set(SOURCE_PAIR_IDS) - found}"
        )
    if selected["pair_id"].astype(str).duplicated().any():
        raise ValueError("historical selection contains duplicate pair IDs")
    selected["_selection_index"] = (
        selected["pair_id"]
        .astype(str)
        .map({pair_id: index for index, pair_id in enumerate(SOURCE_PAIR_IDS)})
    )
    selected = selected.sort_values("_selection_index", kind="stable").drop(
        columns="_selection_index"
    )
    if tuple(selected["pair_id"].astype(str)) != SOURCE_PAIR_IDS:
        raise AssertionError("historical selection order changed")
    if set(selected["split"].astype(str)) != {"deep_split"}:
        raise ValueError("all ImageNet-9 bridge pairs must be historical deep_split rows")
    for column in ("mixed_same_path", "mixed_rand_path", "mixed_next_path"):
        missing = [path for path in selected[column].astype(str) if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"selected historical images are missing: {missing}")
    return selected.reset_index(drop=True)


def typed_selection(wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """Expand the exact eight source rows to sixteen current-executor pairs."""

    source = load_selection() if wide is None else wide
    typed = normalize_wide_manifest(source, dataset_root=DATASET_ROOT, expected_rows=8)
    if (
        len(typed) != 16
        or typed["source_pair_id"].nunique() != 8
        or set(typed["pair_type"].astype(str)) != set(PAIR_TYPES)
    ):
        raise AssertionError("ImageNet-9 exact selection did not expand to 16 typed pairs")
    return typed


def sealed_summaries(binding: ModelBinding) -> pd.DataFrame:
    """Read the exact selected epsilon=.02 summaries from one sealed archive."""

    frame = pd.read_parquet(binding.sample_path)
    selected = frame[
        frame["pair_id"].astype(str).isin(SOURCE_PAIR_IDS)
        & np.isclose(
            pd.to_numeric(frame["epsilon"], errors="coerce").to_numpy(dtype=np.float64),
            EPSILON,
            atol=0.0,
            rtol=0.0,
        )
    ].copy()
    keys = ["pair_id", "pair_type", "path"]
    expected = len(SOURCE_PAIR_IDS) * len(PAIR_TYPES) * len(REVEAL_PATHS)
    if len(selected) != expected or selected.duplicated(keys).any():
        raise ValueError(
            f"sealed summaries for {binding.historical_model_id} are not exactly {expected} rows"
        )
    if set(selected["pair_type"].astype(str)) != set(PAIR_TYPES):
        raise ValueError("sealed summaries do not cover both pair types")
    if set(selected["path"].astype(str)) != set(REVEAL_PATHS):
        raise ValueError("sealed summaries do not cover blend and both patch orders")
    return selected.sort_values(keys, kind="stable").reset_index(drop=True)


def _historical_tensor(path: str | Path) -> Any:
    historical_data, _, _ = _historical_modules()
    import torch

    image = historical_data.load_image_224(path, size=224)
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


def build_patch_order_manifest(wide: pd.DataFrame | None = None) -> dict[str, Any]:
    """Reconstruct and seal the exact registered historical patch identities."""

    source = load_selection() if wide is None else wide
    _, historical_reveal, _ = _historical_modules()
    orders: dict[str, dict[str, list[int]]] = {"patch_A": {}, "patch_B": {}}
    tensor_identities: dict[str, dict[str, Any]] = {}
    for row in source.to_dict("records"):
        source_pair_id = str(row["pair_id"])
        plus = _historical_tensor(row["mixed_same_path"])
        current_plus = preprocess_image(row["mixed_same_path"], dataset_root=DATASET_ROOT)
        historical_plus = plus.numpy()
        if not np.array_equal(current_plus, historical_plus):
            raise AssertionError(
                f"preprocessing differs for historical plus endpoint {source_pair_id}"
            )
        for pair_type, variant in (
            ("same_rand", "mixed_rand_path"),
            ("same_next", "mixed_next_path"),
        ):
            typed_id = f"{source_pair_id}__{pair_type}"
            minus = _historical_tensor(row[variant])
            current_minus = preprocess_image(row[variant], dataset_root=DATASET_ROOT)
            historical_minus = minus.numpy()
            if not np.array_equal(current_minus, historical_minus):
                raise AssertionError(
                    f"preprocessing differs for historical minus endpoint {typed_id}"
                )
            tensor_identities[typed_id] = {
                "plus": canonical_tensor_identity(historical_plus),
                "minus": canonical_tensor_identity(historical_minus),
            }
            for label in ("A", "B"):
                order = historical_reveal.difference_energy_patch_order(
                    plus,
                    minus,
                    pair_id=source_pair_id,
                    order=label,
                    seed=PATCH_SEED,
                    grid_size=PATCH_GRID,
                )
                if len(order) != 64 or set(order) != set(range(64)):
                    raise AssertionError(f"historical patch order is invalid: {typed_id}/{label}")
                orders[f"patch_{label}"][typed_id] = list(map(int, order))
    return {
        "schema_version": 1,
        "experiment_family": "imagenet9",
        "reference_run": REFERENCE_RUN,
        "source_pair_ids": list(SOURCE_PAIR_IDS),
        "typed_pair_count": 16,
        "patch_seed": PATCH_SEED,
        "patch_grid": list(PATCH_GRID),
        "orders": orders,
        "preprocessed_endpoint_identities": tensor_identities,
        "identity_source": (
            "deterministic reconstruction by the frozen historical "
            "difference_energy_patch_order implementation"
        ),
        "historical_pair_id_namespace": "base source pair ID (without typed suffix)",
        "current_pair_id_namespace": "base source pair ID plus __same_rand/__same_next",
        "stage_ledger_contains_sample_orders": False,
        "metadata_limit": (
            "stage_ledger.jsonl stores job-level stages and sample.parquet stores "
            "summaries; neither seals per-sample patch permutations"
        ),
        "historical_source_binding": _historical_package_provenance(),
    }


def prepare_bridge(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Path]:
    """Write exact selection, path-identity, and provenance manifests."""

    root = Path(output_root)
    manifests = root / "manifests"
    provenance = root / "provenance"
    manifests.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)
    wide = load_selection()
    typed = typed_selection(wide)
    orders = build_patch_order_manifest(wide)
    wide_path = manifests / "imagenet9_historical_selection.csv"
    typed_path = manifests / "imagenet9_selection.csv"
    order_path = manifests / "imagenet9_historical_patch_orders.json"
    atomic_text(wide_path, wide.to_csv(index=False))
    atomic_text(typed_path, typed.to_csv(index=False))
    atomic_json(order_path, orders)

    bindings = []
    for binding in MODEL_BINDINGS:
        if sha256_file(binding.checkpoint) != binding.checkpoint_sha256:
            raise ValueError(f"checkpoint hash differs: {binding.checkpoint}")
        sealed_summaries(binding)
        bindings.append(
            {
                "historical_model_id": binding.historical_model_id,
                "current_model_id": binding.current_model_id,
                "checkpoint": str(binding.checkpoint),
                "checkpoint_sha256": binding.checkpoint_sha256,
                "sealed_sample": str(binding.sample_path),
                "sealed_sample_sha256": sha256_file(binding.sample_path),
            }
        )
    historical_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HISTORICAL_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    provenance_payload = {
        "schema_version": 1,
        "experiment_family": "imagenet9",
        "reference_run": REFERENCE_RUN,
        "historical_repository": str(HISTORICAL_REPOSITORY),
        "historical_commit": historical_commit,
        "historical_repository_read_only": True,
        "historical_commit_role": (
            "context only; the experiment namespace is untracked and execution is "
            "bound to historical_source_binding"
        ),
        "historical_source_binding": _historical_package_provenance(),
        "historical_config": str(HISTORICAL_CONFIG),
        "historical_config_sha256": sha256_file(HISTORICAL_CONFIG),
        "historical_manifest": str(HISTORICAL_MANIFEST),
        "historical_manifest_sha256": sha256_file(HISTORICAL_MANIFEST),
        "historical_stage_ledger": str(HISTORICAL_STAGE_LEDGER),
        "historical_stage_ledger_sha256": sha256_file(HISTORICAL_STAGE_LEDGER),
        "selection_manifest": str(typed_path),
        "selection_manifest_sha256": sha256_file(typed_path),
        "patch_order_manifest": str(order_path),
        "patch_order_manifest_sha256": sha256_file(order_path),
        "selection_sha256": _canonical_sha256(list(SOURCE_PAIR_IDS)),
        "models": bindings,
        "coverage": {
            "models": 3,
            "source_pairs_per_model": 8,
            "typed_pairs_per_model": 16,
            "reveal_paths": 3,
            "units": 144,
            "stages_per_unit": 9,
        },
    }
    provenance_path = provenance / "imagenet9_bridge.json"
    atomic_json(provenance_path, provenance_payload)
    return {
        "wide_selection": wide_path,
        "selection": typed_path,
        "patch_orders": order_path,
        "provenance": provenance_path,
    }


def _descriptor(binding: ModelBinding) -> dict[str, Any]:
    payload = json.loads(HISTORICAL_DESCRIPTORS.read_text(encoding="utf-8"))
    records = payload.get("models")
    if not isinstance(records, list):
        raise TypeError("historical model descriptor inventory is malformed")
    matches = [
        dict(record)
        for record in records
        if isinstance(record, dict) and str(record.get("model_id")) == binding.historical_model_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"historical descriptor inventory must contain one {binding.historical_model_id}"
        )
    descriptor = matches[0]
    if descriptor["kind"] == "fine_tuned":
        if Path(str(descriptor.get("checkpoint", ""))).resolve() != binding.checkpoint.resolve():
            raise ValueError(
                f"historical descriptor checkpoint differs: {binding.historical_model_id}"
            )
    return descriptor


def _historical_path_spec(path: str) -> tuple[str, str]:
    if path == "blend":
        return "blend", "A"
    if path in {"patch_A", "patch_B"}:
        return "patch", path[-1]
    raise ValueError(f"unknown historical reveal path: {path}")


def export_legacy_stage_scores(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Regenerate raw score pairs through the frozen historical runtime on B200."""

    root = Path(output_root)
    prepare_bridge(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TORCH_HOME", str(WEIGHT_CACHE_ROOT / "torch"))
    _, historical_reveal, historical_run = _historical_modules()
    from decaf.experiments.imagenet9.gpu_runtime import _require_single_b200

    torch, _ = _require_single_b200()
    device = torch.device("cuda:0")
    config = historical_run.load_config(HISTORICAL_CONFIG)
    wide = load_selection()
    labels = wide["true_in9_class"].to_numpy(dtype=np.int64)
    pair_ids = wide["pair_id"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for binding in MODEL_BINDINGS:
        descriptor = _descriptor(binding)
        logits_model = historical_run._load_descriptor(config, descriptor, device)
        classes = int(getattr(logits_model, "num_classes", 1000))
        mapping = None
        if classes != 9:
            mapping = historical_run.load_superclass_mapping(
                Path(config["paths"]["official_repository"])
            )
        model = historical_run._RawPixelProbabilityModel(logits_model, mapping).to(device).eval()
        plus = historical_run._manifest_variant_images(wide, "mixed_same", size=224)
        other_images = {
            "same_rand": historical_run._manifest_variant_images(wide, "mixed_rand", size=224),
            "same_next": historical_run._manifest_variant_images(wide, "mixed_next", size=224),
        }
        for pair_type in PAIR_TYPES:
            minus = other_images[pair_type]
            for reveal_path in REVEAL_PATHS:
                path_kind, order = _historical_path_spec(reveal_path)
                sequence = historical_reveal.generate_reveal_sequence(
                    plus,
                    minus,
                    ALPHA,
                    path=path_kind,
                    pair_id=pair_ids,
                    order=order,
                    seed=PATCH_SEED,
                    sigma=8.0,
                )
                for stage_index, stage in enumerate(sequence):
                    plus_probabilities = historical_run._predict_probabilities(
                        model,
                        torch.as_tensor(stage.plus),
                        device=device,
                        batch_size=16,
                        context=(
                            f"crossgen/{binding.historical_model_id}/{pair_type}/"
                            f"{reveal_path}/{stage_index}/plus"
                        ),
                    )
                    minus_probabilities = historical_run._predict_probabilities(
                        model,
                        torch.as_tensor(stage.minus),
                        device=device,
                        batch_size=16,
                        context=(
                            f"crossgen/{binding.historical_model_id}/{pair_type}/"
                            f"{reveal_path}/{stage_index}/minus"
                        ),
                    )
                    plus_scores = historical_run._true_probability(plus_probabilities, labels)
                    minus_scores = historical_run._true_probability(minus_probabilities, labels)
                    for index, source_pair_id in enumerate(pair_ids):
                        positive = float(plus_scores[index])
                        negative = float(minus_scores[index])
                        rows.append(
                            {
                                "historical_model_id": binding.historical_model_id,
                                "model_id": binding.current_model_id,
                                "checkpoint_sha256": binding.checkpoint_sha256,
                                "source_pair_id": source_pair_id,
                                "pair_id": f"{source_pair_id}__{pair_type}",
                                "pair_type": pair_type,
                                "class_id": int(labels[index]),
                                "reveal_path": reveal_path,
                                "stage_index": stage_index,
                                "alpha": float(stage.alpha),
                                "score_plus": positive,
                                "score_minus": negative,
                                # Use the stored scalar pair's float64 difference
                                # so neutral-schema direct and paired forms agree.
                                "response": positive - negative,
                            }
                        )
                del sequence
        del model, logits_model, plus, other_images
        gc.collect()
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    expected = len(MODEL_BINDINGS) * 16 * len(REVEAL_PATHS) * len(ALPHA)
    keys = ["model_id", "pair_id", "reveal_path", "stage_index"]
    if len(frame) != expected or frame.duplicated(keys).any():
        raise AssertionError(
            f"historical stage exporter produced {len(frame)} rows, expected {expected}"
        )
    for _, unit in frame.groupby(keys[:-1], sort=False):
        ordered = unit.sort_values("stage_index", kind="stable")
        if not np.array_equal(ordered["alpha"].to_numpy(dtype=np.float64), np.asarray(ALPHA)):
            raise AssertionError("historical stage grid differs from the registered alpha grid")
        if abs(float(ordered["response"].iloc[0])) > 1.0e-7:
            raise FloatingPointError("historical shared midpoint response is not numerically zero")
    output = root / "trajectories/imagenet9_legacy_stage_scores.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    return _atomic_parquet(frame.sort_values(keys, kind="stable"), output)


def _dominant(row: pd.Series) -> str:
    values = {name: float(row[name]) for name in ("E", "C", "F")}
    maximum = max(values.values())
    return "|".join(name for name in ("E", "C", "F") if values[name] == maximum)


def build_neutral_record(
    stage_scores: pd.DataFrame,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Join regenerated historical scores to sealed summaries and write neutral rows."""

    required = {
        "historical_model_id",
        "model_id",
        "checkpoint_sha256",
        "source_pair_id",
        "pair_id",
        "pair_type",
        "class_id",
        "reveal_path",
        "stage_index",
        "alpha",
        "score_plus",
        "score_minus",
        "response",
    }
    missing = sorted(required - set(stage_scores.columns))
    if missing:
        raise ValueError(f"historical stage scores are missing columns: {missing}")
    weights = trapezoid_weights(ALPHA)
    rows: list[dict[str, Any]] = []
    for binding in MODEL_BINDINGS:
        sealed = sealed_summaries(binding).set_index(["pair_id", "pair_type", "path"])
        model_scores = stage_scores[
            stage_scores["model_id"].astype(str) == binding.current_model_id
        ]
        for identifiers, unit in model_scores.groupby(
            ["source_pair_id", "pair_id", "pair_type", "class_id", "reveal_path"],
            sort=True,
        ):
            source_pair_id, pair_id, pair_type, class_id, reveal_path = identifiers
            ordered = unit.sort_values("stage_index", kind="stable")
            if len(ordered) != len(ALPHA):
                raise ValueError(f"historical trajectory has the wrong size: {identifiers}")
            summary = sealed.loc[(str(source_pair_id), str(pair_type), str(reveal_path))]
            endpoint_positive = float(ordered["score_plus"].iloc[-1])
            endpoint_negative = float(ordered["score_minus"].iloc[-1])
            endpoint_d = endpoint_positive - endpoint_negative
            historical_gate = bool(summary["endpoint_active"])
            historical_orientation = (
                int(np.sign(float(summary["endpoint_delta"]))) if historical_gate else 0
            )
            metadata = json.dumps(
                {
                    "current_model_id": binding.current_model_id,
                    "current_checkpoint_sha256": binding.checkpoint_sha256,
                    "current_sample_or_pair_id": str(pair_id),
                    "current_factor_or_part_id": f"class_{int(class_id)}",
                    "current_counterfactual_map": str(pair_type),
                    "current_protocol": str(reveal_path),
                    "historical_model_id": binding.historical_model_id,
                    "historical_gate": historical_gate,
                    "historical_orientation": historical_orientation,
                    "historical_dominant": _dominant(summary),
                    "historical_endpoint_delta": float(summary["endpoint_delta"]),
                    "identity_match": True,
                    "raw_score_source": (
                        "fresh SHA-verified packaged historical "
                        "preprocessing/reveal/probability runtime"
                    ),
                    "historical_package_sha256": HISTORICAL_PACKAGE_SHA256,
                    "historical_summary_source": str(binding.sample_path),
                    "patch_order_identity": (
                        "explicit frozen historical order manifest"
                        if str(reveal_path).startswith("patch_")
                        else "registered shared-midpoint blend"
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            unit_id = f"imagenet9::{binding.current_model_id}::{pair_id}::{reveal_path}"
            for row, weight in zip(ordered.to_dict("records"), weights, strict=True):
                rows.append(
                    {
                        "experiment_family": "imagenet9",
                        "reference_run": REFERENCE_RUN,
                        "unit_id": unit_id,
                        "model_id": binding.current_model_id,
                        "checkpoint_sha256": binding.checkpoint_sha256,
                        "sample_or_pair_id": str(pair_id),
                        "factor_or_part_id": f"class_{int(class_id)}",
                        "counterfactual_map": str(pair_type),
                        "protocol": str(reveal_path),
                        "protocol_seed": PATCH_SEED,
                        "stage_index": int(row["stage_index"]),
                        "stage_t": float(row["alpha"]),
                        "quadrature_weight": float(weight),
                        "endpoint_epsilon": EPSILON,
                        "endpoint_score_plus": endpoint_positive,
                        "endpoint_score_minus": endpoint_negative,
                        "endpoint_d": endpoint_d,
                        "stage_score_plus": float(row["score_plus"]),
                        "stage_score_minus": float(row["score_minus"]),
                        "stage_r": float(row["response"]),
                        "historical_M": float(summary["M"]),
                        "historical_E": float(summary["E"]),
                        "historical_C": float(summary["C"]),
                        "historical_F": float(summary["F"]),
                        "historical_Abs": float(summary["Abs"]),
                        "metadata_json": metadata,
                    }
                )
    neutral = pd.DataFrame(rows, columns=NEUTRAL_COLUMNS)
    expected = len(MODEL_BINDINGS) * 16 * len(REVEAL_PATHS) * len(ALPHA)
    expected_units = len(MODEL_BINDINGS) * 16 * len(REVEAL_PATHS)
    if len(neutral) != expected or neutral["unit_id"].nunique() != expected_units:
        raise AssertionError(
            f"neutral ImageNet-9 export has {len(neutral)} rows/"
            f"{neutral['unit_id'].nunique()} units"
        )
    output = Path(output_root) / "trajectories/imagenet9.parquet"
    return _atomic_parquet(validate_trajectory_record(neutral), output)


def export_legacy_trajectory(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    """Run the full historical score-only bridge and neutral export."""

    scores_path = export_legacy_stage_scores(output_root)
    scores = pd.read_parquet(scores_path)
    neutral_path = build_neutral_record(scores, output_root)
    return {"stage_scores": scores_path, "trajectory": neutral_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "export"),
        help="prepare is CPU-only; export requires the single B200",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        prepare_bridge(args.output_root)
        if args.action == "prepare"
        else export_legacy_trajectory(args.output_root)
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
