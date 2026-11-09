"""Download a reproducible subset of the public, official SUB release.

Uses ranged Parquet reads for metadata and image paths; downloads selected
original-resolution images from the official HF dataset viewer. Full Parquet
download is also supported, avoiding reliance on the viewer's availability.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from protocol import matched_pairs, select_rows

REPOSITORY = "Jessica-bader/SUB"
REVISION = "6687681ee20f37b375b9ea8354e5841409fd5d7e"
CODE_REVISION = "953a7559878a29e2c7c4a9c79db5e72eee046a85"
HUB = "https://huggingface.co"
VIEWER = "https://datasets-server.huggingface.co"
DEFAULT_ROOT = "/work/Users/leiyo/decaf_supplemental_assets/cub_sub"


def read_url(url: str, headers: dict | None = None, attempts: int = 5) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 16))
    raise RuntimeError("unreachable")


def read_json(url: str) -> dict | list:
    return json.loads(read_url(url))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class RangeReader(io.RawIOBase):
    """Seekable read-only HTTP resource with verified byte ranges."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("negative seek")
        return self.position

    def read(self, size=-1):
        if self.position >= self.size:
            return b""
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if not size:
            return b""
        end = self.position + size - 1
        content = read_url(self.url, {"Range": f"bytes={self.position}-{end}"})
        if len(content) != size:
            raise RuntimeError(f"Range request returned {len(content)} bytes, expected {size}")
        self.position += size
        return content


def resource_metadata(root: Path) -> tuple[list, dict]:
    root.mkdir(parents=True, exist_ok=True)
    listing_path = root / "official_files.json"
    if listing_path.exists():
        listing = json.loads(listing_path.read_text())
    else:
        listing = read_json(
            f"{HUB}/api/datasets/{REPOSITORY}/tree/{REVISION}?recursive=true&limit=1000"
        )
        save_json(listing_path, listing)
    raw = root / "official_dataset_card.md"
    if not raw.exists():
        raw.write_bytes(read_url(f"{HUB}/datasets/{REPOSITORY}/raw/{REVISION}/README.md"))
    import yaml

    card = yaml.safe_load(raw.read_text().split("---")[1])
    feature_names = {}
    for feature in card["dataset_info"]["features"]:
        if feature["name"] != "image":
            names = feature["dtype"]["class_label"]["names"]
            feature_names[feature["name"]] = [names[str(i)] for i in range(len(names))]
    queries = root / "attribute_clip_queries.csv"
    if not queries.exists():
        queries.write_bytes(
            read_url(
                f"https://raw.githubusercontent.com/ExplainableML/sub/{CODE_REVISION}/"
                "CBM_testing/attribute_clip_queries.csv"
            )
        )
    info = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "code_revision": CODE_REVISION,
        "feature_names": feature_names,
        "license": card["license"],
        "total_rows": 38400,
        "queries_sha256": checksum(queries),
        "source_images": "FLUX synthetic; not edited original CUB photographs",
        "pairing_provenance": "No source-pair ID or shared cross-attribute seed is published",
    }
    save_json(root / "source.json", info)
    return [item for item in listing if item["path"].endswith(".parquet")], info


def metadata_index(root: Path, files: list, workers: int) -> list[dict]:
    import pyarrow.parquet as pq

    cache = root / "metadata"
    cache.mkdir(exist_ok=True)

    def one(item):
        path = cache / (Path(item["path"]).stem + ".json")
        if path.exists():
            return json.loads(path.read_text())
        url = f"{HUB}/datasets/{REPOSITORY}/resolve/{REVISION}/{item['path']}"
        local = root / item["path"]
        source = local if local.exists() else RangeReader(url, item["size"])
        parquet = pq.ParquetFile(source)
        columns = parquet.read(columns=["attr_label", "bird_label", "image.path"]).to_pylist()
        result = []
        for index, row in enumerate(columns):
            result.append(
                {
                    "local_row": index,
                    "shard": item["path"],
                    "attr_label": row["attr_label"],
                    "bird_label": row["bird_label"],
                    "published_image_path": (row.get("image") or {}).get("path"),
                }
            )
        save_json(path, result)
        return result

    collected = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, item): item["path"] for item in files}
        for completed, future in enumerate(as_completed(futures), 1):
            collected[futures[future]] = future.result()
            print(f"metadata shard {completed}/{len(files)}", flush=True)
    rows = []
    for name in sorted(collected):
        for row in collected[name]:
            rows.append(dict(row, row_id=len(rows)))
    if len(rows) != 38400:
        raise ValueError(f"Expected official 38,400 rows, obtained {len(rows)}")
    save_json(root / "all_metadata.json", rows)
    return rows


def download_parquets(root: Path, files: list, workers: int) -> None:
    def one(item):
        path = root / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = item["lfs"]["oid"]
        if path.exists() and path.stat().st_size == item["size"] and checksum(path) == expected:
            return
        temporary = path.with_suffix(".part")
        url = f"{HUB}/datasets/{REPOSITORY}/resolve/{REVISION}/{item['path']}"
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if checksum(temporary) != expected:
            raise ValueError(f"SHA256 mismatch: {item['path']}")
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, _ in enumerate(pool.map(one, files), 1):
            print(f"Parquet {completed}/{len(files)}", flush=True)


def materialize_parquet(root: Path, selected: list[dict]) -> list[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    groups = {}
    for row in selected:
        groups.setdefault(row["shard"], []).append(row)
    output = []
    for shard, rows in sorted(groups.items()):
        table = pq.read_table(root / shard, columns=["image"])
        for row in rows:
            image = table["image"][row["local_row"]].as_py()
            path = root / "images" / f"{row['row_id']:05d}.png"
            path.parent.mkdir(exist_ok=True)
            content = image["bytes"]
            decoded = Image.open(io.BytesIO(content))
            if decoded.format != "PNG" or decoded.size != (512, 512):
                raise ValueError("Unexpected source image encoding or dimensions")
            decoded.verify()
            expected = hashlib.sha256(content).hexdigest()
            if not path.exists() or checksum(path) != expected:
                path.write_bytes(content)
            output.append(
                dict(
                    row,
                    image_path=str(path.resolve()),
                    image_sha256=checksum(path),
                    delivery="official_parquet",
                )
            )
        print(f"Extracted selected original PNG bytes from {shard}", flush=True)
    return output


def materialize_viewer(root: Path, selected: list[dict], workers: int) -> list[dict]:
    from PIL import Image

    current = read_json(f"{HUB}/api/datasets/{REPOSITORY}")["sha"]
    if current != REVISION:
        raise RuntimeError(
            "Viewer only serves current release; revision changed. Use --delivery parquet"
        )
    pages = {}
    for row in selected:
        pages.setdefault((row["row_id"] // 100) * 100, []).append(row)

    def one_page(item):
        offset, wanted = item
        done = []
        missing = []
        for row in wanted:
            path = root / "images" / f"{row['row_id']:05d}.jpg"
            if path.exists():
                Image.open(path).verify()
                done.append(
                    dict(
                        row,
                        image_path=str(path.resolve()),
                        image_sha256=checksum(path),
                        delivery="official_hf_viewer",
                    )
                )
            else:
                missing.append(row)
        if not missing:
            return done
        query = urllib.parse.urlencode(
            {
                "dataset": REPOSITORY,
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": 100,
            }
        )
        response = read_json(f"{VIEWER}/rows?{query}")
        found = {x["row_idx"]: x["row"] for x in response["rows"]}
        for row in missing:
            received = found[row["row_id"]]
            if any(received[key] != row[key] for key in ("attr_label", "bird_label")):
                raise ValueError("Viewer metadata differs from pinned Parquet release")
            url = received["image"]["src"]
            if REVISION not in url:
                raise ValueError("Viewer image URL does not identify the pinned release")
            content = read_url(url)
            image = Image.open(io.BytesIO(content))
            if image.size != (512, 512):
                raise ValueError(f"Viewer image was resized: {image.size}")
            image.verify()
            path = root / "images" / f"{row['row_id']:05d}.jpg"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content)
            done.append(
                dict(
                    row,
                    image_path=str(path.resolve()),
                    image_sha256=checksum(path),
                    delivery="official_hf_viewer",
                )
            )
        return done

    output = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_page, item) for item in sorted(pages.items())]
        for completed, future in enumerate(as_completed(futures), 1):
            output.extend(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(f"image pages {completed}/{len(futures)}", flush=True)
    if read_json(f"{HUB}/api/datasets/{REPOSITORY}")["sha"] != REVISION:
        raise RuntimeError("Dataset release changed during viewer download")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--per-cell", type=int, default=4, help="0 keeps all 50 images/cell")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delivery", choices=["viewer", "parquet"], default="parquet")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.per_cell < 0 or args.workers < 1:
        parser.error("per-cell >= 0 and workers >= 1 are required")
    root = args.root.resolve()
    files, info = resource_metadata(root)
    if args.delivery == "parquet" and not args.metadata_only:
        download_parquets(root, files, args.workers)
    rows = metadata_index(root, files, args.workers)
    chosen = select_rows(rows, args.per_cell, args.seed)
    pairs = matched_pairs(chosen, info["feature_names"]["attr_label"], args.seed)
    save_json(
        root / "selection.json",
        {
            "per_cell": args.per_cell,
            "seed": args.seed,
            "selected_rows": len(chosen),
            "pairs": len(pairs),
        },
    )
    if args.metadata_only:
        print(json.dumps({"rows": len(rows), "selected": len(chosen), "pairs": len(pairs)}))
        return
    selected = (
        materialize_parquet(root, chosen)
        if args.delivery == "parquet"
        else materialize_viewer(root, chosen, args.workers)
    )
    selected.sort(key=lambda row: row["row_id"])
    manifest = {
        "source": info,
        "selection": {"per_cell": args.per_cell, "seed": args.seed},
        "rows": selected,
        "pairs": pairs,
    }
    destination = root / f"manifest_per_cell_{args.per_cell}_seed_{args.seed}.json"
    save_json(destination, manifest)
    print(
        f"Manifest: {destination}; {len(selected)} images, {len(pairs)} matched pairs", flush=True
    )


if __name__ == "__main__":
    main()
