#!/usr/bin/env python3
"""Prepare public ImageNet-9 assets for the supplementary grid audit.

This creates a new, explicitly identified support; it does not reconstruct the
unavailable historical fine-tuned checkpoints or claim the old split identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import urllib.request
from pathlib import Path

ARCHIVE_URL = (
    "https://github.com/MadryLab/backgrounds_challenge/releases/download/"
    "data/backgrounds_challenge_data.tar.gz"
)
ARCHIVE_SHA256 = "2d66f571b0492986b347e37d2ee498ed34b58012af425d772dacda4422896fe7"
MAPPING_URL = (
    "https://raw.githubusercontent.com/MadryLab/backgrounds_challenge/master/in_to_in9.json"
)
MODEL_WEIGHTS = {
    "resnet18": "IMAGENET1K_V1",
    "resnet50": "IMAGENET1K_V2",
    "resnet152": "IMAGENET1K_V2",
    "resnext50_32x4d": "IMAGENET1K_V2",
    "wide_resnet50_2": "IMAGENET1K_V2",
    "densenet121": "IMAGENET1K_V1",
    "efficientnet_b0": "IMAGENET1K_V1",
    "efficientnet_v2_s": "IMAGENET1K_V1",
    "regnet_y_8gf": "IMAGENET1K_V2",
    "convnext_tiny": "IMAGENET1K_V1",
    "swin_t": "IMAGENET1K_V1",
    "vit_b_16": "IMAGENET1K_V1",
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url: str, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)


def prepare_pairs(root: Path):
    import pandas as pd

    variants = {}
    for variant in ("mixed_same", "mixed_rand", "mixed_next", "original"):
        entries = {}
        for image in sorted((root / "bg_challenge" / variant / "val").glob("*/*.JPEG")):
            class_id = int(image.parent.name.split("_", 1)[0])
            match = re.fullmatch(r"fg_(n\d+_\d+)_bg_n\d+_\d+", image.stem)
            foreground_id = match[1] if match else image.stem
            key = f"{class_id:02d}/{foreground_id}"
            if key in entries:
                raise ValueError(f"duplicate foreground in {variant}: {key}")
            entries[key] = (class_id, image.relative_to(root).as_posix())
        if len(entries) != 4050:
            raise ValueError(f"{variant}: expected 4050 official images, found {len(entries)}")
        variants[variant] = entries
    support = set(variants["mixed_same"])
    if any(set(entries) != support for entries in variants.values()):
        raise ValueError("official variant foreground identities do not match")
    records = []
    for key in sorted(support):
        row = {"pair_id": key, "true_in9_class": variants["mixed_same"][key][0]}
        row.update({f"{variant}_path": entries[key][1] for variant, entries in variants.items()})
        records.append(row)
    frame = pd.DataFrame(records)
    if frame.groupby("true_in9_class").size().to_dict() != dict.fromkeys(range(9), 450):
        raise ValueError("official class support differs from nine classes of 450")
    destination = root / "manifests" / "public_paired_variants.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return destination


def prepare_weights(root: Path, architectures: list[str]):
    import torch
    import torchvision.models as models

    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for architecture in architectures:
        weight_id = MODEL_WEIGHTS[architecture]
        weight = getattr(models.get_model_weights(architecture), weight_id)
        filename = weight.url.rsplit("/", 1)[-1]
        destination = checkpoint_dir / filename
        if not destination.is_file():
            torch.hub.download_url_to_file(
                weight.url, str(destination), hash_prefix=filename.rsplit("-", 1)[-1].split(".")[0]
            )
        actual = sha256(destination)
        expected_prefix = filename.rsplit("-", 1)[-1].split(".")[0]
        if not actual.startswith(expected_prefix):
            raise ValueError(f"official weight digest mismatch: {architecture}")
        records.append(
            {
                "model_id": f"tv_{architecture}_{weight_id.lower()}",
                "architecture": architecture,
                "kind": "pretrained",
                "output_classes": 1000,
                "checkpoint": str(destination),
                "checkpoint_sha256": actual,
                "weight_enum": weight_id,
                "source_url": weight.url,
            }
        )
        print(json.dumps({"prepared_model": architecture, "sha256": actual}), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODEL_WEIGHTS, default=list(MODEL_WEIGHTS))
    parser.add_argument("--source-pairs", type=int, default=450)
    parser.add_argument("--config-output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.source_pairs <= 4050:
        parser.error("source-pairs must be between 1 and 4050")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "downloads" / "backgrounds_challenge_data.tar.gz"
    if args.download_data:
        download(ARCHIVE_URL, archive)
        if archive.stat().st_size != 279581133 or sha256(archive) != ARCHIVE_SHA256:
            raise ValueError("official release archive size or SHA-256 mismatch")
        with tarfile.open(archive) as source:
            source.extractall(root, filter="data")
    mapping = root / "metadata" / "in_to_in9.json"
    download(MAPPING_URL, mapping)
    if sha256(mapping) != "5b72ab44c804129744f5dff9372d8048c3b85da38d48385d73686ee0b0e7b638":
        raise ValueError("ImageNet-9 mapping differs from the paper's official mapping")
    pairs = prepare_pairs(root)
    weights = prepare_weights(root, args.models)
    config = {
        "dataset_root": str(root),
        "pair_manifest": str(pairs),
        "mapping": str(mapping),
        "models": weights,
        "stage_counts": [3, 5, 9, 17, 33, 65],
        "epsilon": 0.02,
        "paths": ["blend", "patch_A"],
        "pair_types": ["same_rand", "same_next"],
        "selection_seed": 20261109,
        "max_source_pairs": args.source_pairs,
        "image_size": 224,
        "blur_sigma": 8.0,
        "patch_grid": [8, 8],
        "patch_seed": 7101,
        "pair_shard_size": 32,
        "pair_batch_size": 16,
        "query_batch_size": 128,
        "support_description": (
            "New deterministic class-balanced public-test subset; public pretrained models only."
        ),
    }
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "source_url": ARCHIVE_URL,
        "archive_sha256": sha256(archive) if archive.is_file() else None,
        "mapping_sha256": sha256(mapping),
        "manifest_sha256": sha256(pairs),
        "source_rows": 4050,
        "selection": "Predefined class-balanced hash order, without looking at scores.",
        "historical_split_reproduced": False,
        "historical_finetunes_available": False,
    }
    (root / "public_asset_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"config": str(args.config_output.resolve()), **provenance}), flush=True)


if __name__ == "__main__":
    main()
