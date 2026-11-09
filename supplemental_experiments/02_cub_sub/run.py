"""Frozen OpenCLIP inference on official SUB images and matched-pair trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
from prepare import checksum, save_json
from protocol import attribute_group, normalized_attribute

VIEW_NAMES = ["clean", "horizontal_flip", "gaussian_blur_1.5", "jpeg_quality_40", "downsample_128"]


def heldout_views(image):
    from PIL import Image, ImageFilter, ImageOps

    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=40)
    encoded.seek(0)
    return [
        image,
        ImageOps.mirror(image),
        image.filter(ImageFilter.GaussianBlur(1.5)),
        Image.open(encoded).convert("RGB"),
        image.resize((128, 128), Image.Resampling.BILINEAR).resize(
            image.size, Image.Resampling.BILINEAR
        ),
    ]


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def cache_receipt(path: Path, run_identity: str, observation_ids: list):
    save_json(
        path.with_suffix(".receipt.json"),
        {
            "sha256": checksum(path),
            "run_identity": run_identity,
            "observation_ids": observation_ids,
        },
    )


def verify_cache(path: Path, run_identity: str, observation_ids: list):
    receipt_path = path.with_suffix(".receipt.json")
    if not receipt_path.exists():
        raise ValueError(f"Cache lacks an identity/hash receipt: {path}")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["run_identity"] != run_identity
        or receipt["observation_ids"] != observation_ids
        or receipt["sha256"] != checksum(path)
    ):
        raise ValueError(f"Cache identity, ordering, or checksum mismatch: {path}")


def freeze_configuration(path: Path, configuration: dict) -> str:
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError(
            "Output exists with a different config/implementation. "
            "Choose another --output directory"
        )
    if not path.exists():
        save_json(path, configuration)
    return checksum(path)


def probability_from_cosine(cosine, indices: list[int], logit_scale: float):
    logits = np.asarray(cosine, dtype=np.float64)[..., indices] * logit_scale
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--stages", type=int, default=9)
    parser.add_argument("--blur-sigma", type=float, default=12.0)
    parser.add_argument(
        "--limit-images", type=int, default=0, help="Smoke only; marks output as incomplete"
    )
    parser.add_argument(
        "--limit-pairs", type=int, default=0, help="Smoke only; marks output as incomplete"
    )
    args = parser.parse_args()
    if args.stages < 2 or args.batch_size < 1 or args.blur_sigma <= 0 or args.cpu_threads < 1:
        parser.error("stages >= 2, batch-size/cpu-threads >= 1, blur-sigma > 0 required")
    import open_clip
    import PIL
    import torch
    import torchvision
    from PIL import Image
    from torchvision.transforms.functional import gaussian_blur

    torch.set_num_threads(args.cpu_threads)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    source_root = args.manifest.resolve().parent
    query_file = source_root / "attribute_clip_queries.csv"
    if checksum(query_file) != manifest["source"]["queries_sha256"]:
        raise ValueError("Official prompt file checksum differs from prepared manifest")
    queries = list(csv.DictReader(query_file.open()))
    attributes = [normalized_attribute(q["attribute"]) for q in queries]
    if len(set(attributes)) != len(attributes):
        raise ValueError("Duplicate attribute prompts")
    dataset_attributes = [
        normalized_attribute(a) for a in manifest["source"]["feature_names"]["attr_label"]
    ]
    if any(name not in attributes for name in dataset_attributes):
        raise ValueError("Released attribute missing from official CLIP prompt vocabulary")
    groups = {
        group: [i for i, attr in enumerate(attributes) if attribute_group(attr) == group]
        for group in sorted({attribute_group(a) for a in attributes})
    }
    attr_query_indices = [attributes.index(a) for a in dataset_attributes]
    birds = manifest["source"]["feature_names"]["bird_label"]
    prompts = [query["value"] for query in queries] + [
        "A photo of a " + bird.replace("_", " ") for bird in birds
    ]
    configuration = {
        "manifest_sha256": checksum(args.manifest),
        "cache_schema": 2,
        "implementation_sha256": {
            name: checksum(Path(__file__).parent / name)
            for name in ["run.py", "protocol.py", "prepare.py"]
        },
        "libraries": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "open_clip": open_clip.__version__,
            "Pillow": PIL.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "model": args.model,
        "pretrained": args.pretrained,
        "stages": args.stages,
        "blur_sigma": args.blur_sigma,
        "views": VIEW_NAMES,
        "attribute_queries": attributes,
        "dataset_attr_query_indices": attr_query_indices,
        "groups": groups,
        "limit_images": args.limit_images,
        "limit_pairs": args.limit_pairs,
        "scalar_score": "softmax(model_logit_scale * cosine) within full CUB attribute group",
        "pair_type": "same_class_group_cross_instance",
        "incomplete_smoke": bool(args.limit_images or args.limit_pairs),
        "source": manifest["source"],
    }
    config_path = out / "config.json"
    torch.manual_seed(20261109)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.eval().to(args.device)
    tokenizer = open_clip.get_tokenizer(args.model)
    with torch.inference_mode():
        encoded = []
        for start in range(0, len(prompts), args.batch_size):
            values = model.encode_text(
                tokenizer(prompts[start : start + args.batch_size]).to(args.device)
            )
            encoded.append(torch.nn.functional.normalize(values.float(), dim=-1))
        text_features = torch.cat(encoded)
        scale = model.logit_scale.exp().item()
    # Hash actual tensor bytes; records the checkpoint even if a remote tag changes.
    weight_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        weight_hash.update(name.encode())
        weight_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    state_hash = weight_hash.hexdigest()
    configuration.update(
        {
            "model_state_sha256": state_hash,
            "preprocess": repr(preprocess),
            "logit_scale": scale,
            "precision": "float32",
            "batch_size": args.batch_size,
        }
    )
    run_identity = freeze_configuration(config_path, configuration)
    environment_path = out / "environment.json"
    if environment_path.exists():
        previous = json.loads(environment_path.read_text())
        if previous["model_state_sha256"] != state_hash:
            raise ValueError("Downloaded model weights differ from the cached run")
    save_json(
        environment_path,
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "open_clip": open_clip.__version__,
            "Pillow": PIL.__version__,
            "python": platform.python_version(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda")
            else None,
            "preprocess": repr(preprocess),
            "logit_scale": scale,
            "model_state_sha256": state_hash,
            "training": "none; frozen public pretrained model; "
            "no SUB-based fitting or prompt selection",
        },
    )

    def encode(tensors):
        results = []
        with torch.inference_mode():
            for start in range(0, len(tensors), args.batch_size):
                batch = torch.stack(tensors[start : start + args.batch_size]).to(args.device)
                feat = torch.nn.functional.normalize(model.encode_image(batch).float(), dim=-1)
                results.append((feat @ text_features.T).cpu().numpy())
        return np.concatenate(results)

    started = time.monotonic()
    rows = manifest["rows"][: args.limit_images or None]
    # Validate selected inputs even when all inference results are cached.
    for row in rows:
        if checksum(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['image_path']}")
    single_file = out / "single_image_scores.npz"
    if single_file.exists():
        verify_cache(single_file, run_identity, [row["row_id"] for row in rows])
        with np.load(single_file) as cache:
            cosines = cache["cosines"]
            row_ids = cache["row_ids"]
        if row_ids.tolist() != [row["row_id"] for row in rows]:
            raise ValueError("Single-image cache row mismatch")
    else:
        chunks = []
        image_batch = max(1, args.batch_size // len(VIEW_NAMES))
        for start in range(0, len(rows), image_batch):
            tensors = []
            for row in rows[start : start + image_batch]:
                path = Path(row["image_path"])
                if checksum(path) != row["image_sha256"]:
                    raise ValueError(f"Image checksum mismatch: {path}")
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                    tensors.extend(preprocess(view) for view in heldout_views(image))
            chunks.append(encode(tensors).reshape(-1, len(VIEW_NAMES), len(prompts)))
            if (start // image_batch) % 20 == 0:
                print(
                    f"single-image inference {min(start + image_batch, len(rows))}/{len(rows)}",
                    flush=True,
                )
        cosines = np.concatenate(chunks)
        row_ids = np.array([row["row_id"] for row in rows])
        atomic_npz(single_file, cosines=cosines, row_ids=row_ids)
        cache_receipt(single_file, run_identity, row_ids.tolist())
    index = {int(row): i for i, row in enumerate(row_ids)}
    pair_rows = [
        pair
        for pair in manifest["pairs"]
        if pair["plus_row"] in index and pair["minus_row"] in index
    ]
    pair_rows = pair_rows[: args.limit_pairs or None]
    if not pair_rows:
        raise ValueError("No matched pairs; increase --limit-images or selected attribute coverage")
    row_lookup = {row["row_id"]: row for row in rows}
    stages = np.linspace(0, 1, args.stages)
    results = []
    for number, pair in enumerate(pair_rows):
        cache_path = out / "pair_cache" / f"{pair['pair_id']}.npz"
        if cache_path.exists():
            verify_cache(
                cache_path, run_identity, [pair["pair_id"], pair["plus_row"], pair["minus_row"]]
            )
            with np.load(cache_path) as cache:
                results.append({name: cache[name] for name in cache.files})
            continue
        branch_tensors = []
        for key in ["plus_row", "minus_row"]:
            with Image.open(row_lookup[pair[key]]["image_path"]) as opened:
                branch_tensors.append(preprocess(opened.convert("RGB")))
        plus, minus = [tensor.to(args.device) for tensor in branch_tensors]
        kernel = 2 * int(3 * args.blur_sigma) + 1
        midpoint = gaussian_blur(
            (plus + minus) / 2, [kernel, kernel], [args.blur_sigma, args.blur_sigma]
        )
        reveal = [midpoint * (1 - t) + endpoint * t for t in stages for endpoint in [plus, minus]]
        trajectory = encode(reveal).reshape(args.stages, 2, len(prompts))
        group_indices = groups[pair["attribute_group"]]
        plus_query = attr_query_indices[pair["plus_attribute"]]
        minus_query = attr_query_indices[pair["minus_attribute"]]
        chosen = group_indices.index(plus_query)
        p = probability_from_cosine(trajectory, group_indices, scale)[..., chosen]
        endpoint_views = np.stack(
            [cosines[index[pair[key]]] for key in ["plus_row", "minus_row"]], axis=1
        )
        endpoint_probs = probability_from_cosine(endpoint_views, group_indices, scale)
        predictions = np.array(group_indices)[endpoint_probs.argmax(axis=-1)]
        cache = {
            "response": p[:, 0] - p[:, 1],
            "clean_scores": endpoint_probs[0, :, chosen],
            "view_scores": endpoint_probs[1:, :, chosen],
            "clean_predictions": predictions[0],
            "view_predictions": predictions[1:],
            "plus_query": np.array(plus_query),
            "minus_query": np.array(minus_query),
        }
        if not np.allclose(p[-1], cache["clean_scores"], atol=2e-5):
            raise ValueError("Batched trajectory endpoint and clean endpoint disagree")
        atomic_npz(cache_path, **cache)
        cache_receipt(
            cache_path, run_identity, [pair["pair_id"], pair["plus_row"], pair["minus_row"]]
        )
        results.append(cache)
        if number % 25 == 0:
            print(f"paired trajectories {number + 1}/{len(pair_rows)}", flush=True)
    atomic_npz(
        out / "paired_scores.npz",
        stages=stages,
        pair_ids=np.array([pair["pair_id"] for pair in pair_rows]),
        **{key: np.stack([row[key] for row in results]) for key in results[0]},
    )
    save_json(out / "evaluated_pairs.json", pair_rows)
    save_json(out / "evaluated_images.json", rows)
    save_json(
        out / "inference_receipt.json",
        {
            "complete": True,
            "images": len(rows),
            "run_identity": run_identity,
            "artifact_sha256": {
                name: checksum(out / name)
                for name in [
                    "paired_scores.npz",
                    "single_image_scores.npz",
                    "evaluated_pairs.json",
                    "evaluated_images.json",
                    "config.json",
                    "environment.json",
                ]
            },
            "pairs": len(pair_rows),
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "query_frames_for_complete_run": len(rows) * len(VIEW_NAMES)
            + len(pair_rows) * args.stages * 2,
            "resumable_pair_cache": True,
            "incomplete_smoke": configuration["incomplete_smoke"],
        },
    )
    print(f"Inference complete: {out}", flush=True)


if __name__ == "__main__":
    main()
