"""Render paper TeX exclusively from validated canonical per-asset data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.experiments.common import atomic_text

from .analysis_replay import replay_paper_data
from .manifest import VisualAsset, import_generator, load_visual_manifest, repository_root
from .reference import sha256_file
from .semantic import (
    CANONICAL_SCHEMA_SHA256,
    SemanticDataError,
    canonical_cardinality_sha256,
    canonical_cardinality_text,
    load_canonical_asset,
    semantic_contract,
    semantic_contract_sha256,
)


class PaperRenderError(RuntimeError):
    """Raised when canonical data cannot produce a contract-valid paper asset."""


@dataclass(frozen=True)
class PlotSeries:
    """One numerical series embedded in a TeX picture."""

    label: str
    x: tuple[float, ...]
    y: tuple[float, ...]


@dataclass(frozen=True)
class PlotPanel:
    """One canonical source-backed panel of an emitted figure."""

    title: str
    x_label: str
    y_label: str
    series: tuple[PlotSeries, ...]
    categories: tuple[str, ...] = ()


_MARKER = re.compile(
    r"^% DECAF_SEMANTIC_(?P<kind>GEOMETRY|TABLE) asset=(?P<asset>\S+) "
    r"contract_sha256=(?P<contract>[0-9a-f]{64}) "
    r"schema_sha256=(?P<schema>[0-9a-f]{64}) "
    r"canonical_sha256=(?P<canonical>[0-9a-f]{64}) "
    r"panels=(?P<panels>\d+) data_(?:points|rows)=(?P<rows>\d+) "
    r"cardinality_sha256=(?P<cardinality_hash>[0-9a-f]{64}) "
    r"cardinality=(?P<cardinality>[^\s]+)$",
    re.MULTILINE,
)


def _escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _short(value: Any, *, limit: int = 48) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return f"{number:.6g}" if math.isfinite(number) else "--"
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _context(receipt: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    value = receipt.get("_paper_data_root")
    canonical = receipt.get("canonical")
    if not value or not isinstance(canonical, Mapping):
        raise PaperRenderError("rendering requires canonical replay data and its receipt")
    root = Path(str(value)).resolve()
    if not root.is_dir():
        raise PaperRenderError(f"canonical paper-data root is missing: {root}")
    return root, canonical


def _numeric_x(values: pd.Series) -> tuple[np.ndarray, tuple[str, ...]]:
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.notna().all():
        return numbers.to_numpy(dtype=np.float64), ()
    categories = tuple(dict.fromkeys(values.fillna("--").astype(str)))
    index = {value: position for position, value in enumerate(categories)}
    return values.fillna("--").astype(str).map(index).to_numpy(dtype=np.float64), categories


def _plot_panels(frame: pd.DataFrame) -> tuple[PlotPanel, ...]:
    panels: list[PlotPanel] = []
    for panel_id, local in frame.groupby("panel_id", sort=False):
        x_all, categories = _numeric_x(local["x"])
        local = local.assign(_plot_x=x_all)
        series_rows: list[PlotSeries] = []
        for series_name, group in local.groupby("series", sort=False, dropna=False):
            estimate = pd.to_numeric(group["estimate"], errors="coerce")
            x_values = pd.to_numeric(group["_plot_x"], errors="coerce")
            finite = estimate.notna() & x_values.notna()
            ordered = group.loc[finite].assign(
                _estimate=estimate.loc[finite], _x=x_values.loc[finite]
            )
            ordered = ordered.sort_values("_x", kind="stable")
            if ordered.empty:
                continue
            series_rows.append(
                PlotSeries(
                    label=str(series_name),
                    x=tuple(ordered["_x"].astype(float)),
                    y=tuple(ordered["_estimate"].astype(float)),
                )
            )
        if not series_rows:
            raise PaperRenderError(f"canonical panel {panel_id} has no finite estimates")
        panels.append(
            PlotPanel(
                title=str(panel_id),
                x_label="x",
                y_label="estimate",
                series=tuple(series_rows),
                categories=categories,
            )
        )
    return tuple(panels)


def _coordinate(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _panel_tex(panel: PlotPanel) -> tuple[str, int]:
    all_x = [value for series in panel.series for value in series.x]
    all_y = [value for series in panel.series for value in series.y]
    if not all_x or not all_y or not all(math.isfinite(value) for value in all_x + all_y):
        raise PaperRenderError(f"panel {panel.title!r} has no finite canonical geometry")
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if math.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    if math.isclose(y_min, y_max):
        padding = max(abs(y_min) * 0.05, 0.5)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.06
        y_min -= padding
        y_max += padding

    def scale_x(value: float) -> float:
        return 12.0 + 84.0 * (value - x_min) / (x_max - x_min)

    def scale_y(value: float) -> float:
        return 11.0 + 42.0 * (value - y_min) / (y_max - y_min)

    geometry = [
        r"\setlength{\unitlength}{0.72mm}",
        r"\begin{picture}(100,64)",
        r"\put(12,11){\line(1,0){84}}",
        r"\put(12,11){\line(0,1){42}}",
        rf"\put(2,59){{\makebox(0,0)[l]{{\scriptsize\textbf{{{_escape(panel.title)}}}}}}}",
        rf"\put(12,5){{\makebox(0,0)[l]{{\tiny x [{_short(x_min)},{_short(x_max)}]}}}}",
        rf"\put(12,56){{\makebox(0,0)[l]{{\tiny estimate [{_short(y_min)},{_short(y_max)}]}}}}",
    ]
    if y_min < 0.0 < y_max:
        geometry.append(rf"\put(12,{_coordinate(scale_y(0.0))}){{\line(1,0){{84}}}}")
    points_rendered = 0
    for series_index, series in enumerate(panel.series, start=1):
        points = [(scale_x(x), scale_y(y)) for x, y in zip(series.x, series.y, strict=True)]
        points_rendered += len(points)
        for left, right in zip(points, points[1:], strict=False):
            middle = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
            geometry.append(
                "\\qbezier"
                f"({_coordinate(left[0])},{_coordinate(left[1])})"
                f"({_coordinate(middle[0])},{_coordinate(middle[1])})"
                f"({_coordinate(right[0])},{_coordinate(right[1])})"
            )
        marker = r"\circle*{1.3}" if series_index % 2 else r"\circle{1.6}"
        for x_value, y_value in points:
            geometry.append(rf"\put({_coordinate(x_value)},{_coordinate(y_value)}){{{marker}}}")
        legend = f"{series_index}={_escape(_short(series.label, limit=30))}"
        legend_y = 52.0 - 2.6 * ((series_index - 1) % 15)
        geometry.append(rf"\put(69,{_coordinate(legend_y)}){{\makebox(0,0)[l]{{\tiny {legend}}}}}")
    geometry.append(r"\end{picture}")
    if panel.categories:
        mapping = ", ".join(
            f"{index}={_short(value, limit=22)}" for index, value in enumerate(panel.categories)
        )
        geometry.append(rf"\par\raggedright\tiny x categories: {_escape(mapping)}")
    return (
        "\n".join(
            [
                r"\begin{minipage}[t]{0.48\linewidth}",
                r"\centering",
                *geometry,
                r"\end{minipage}",
            ]
        ),
        points_rendered,
    )


def _canonical_comments(item: Mapping[str, Any]) -> list[str]:
    return [
        f"% canonical_path={item['path']} sha256={item['sha256']} bytes={item['size_bytes']}",
        "% upstream_source_sha256s=" + ",".join(map(str, item["source_sha256s"])),
    ]


def _marker(asset: VisualAsset, item: Mapping[str, Any], *, row_count: int) -> str:
    cardinality = item["panel_cardinality"]
    kind = "GEOMETRY" if asset.kind == "figure" else "TABLE"
    label = "points" if asset.kind == "figure" else "rows"
    return (
        f"% DECAF_SEMANTIC_{kind} asset={asset.asset_id} "
        f"contract_sha256={item['semantic_contract_sha256']} "
        f"schema_sha256={item['schema_sha256']} canonical_sha256={item['sha256']} "
        f"panels={item['panel_count']} data_{label}={row_count} "
        f"cardinality_sha256={canonical_cardinality_sha256(cardinality)} "
        f"cardinality={canonical_cardinality_text(cardinality)}"
    )


def _render_numeric_figure(asset: VisualAsset, frame: pd.DataFrame, item: Mapping[str, Any]) -> str:
    panels = _plot_panels(frame)
    rendered: list[str] = []
    point_count = 0
    for index, panel in enumerate(panels):
        tex, points = _panel_tex(panel)
        rendered.append(tex)
        point_count += points
        rendered.append(r"\par\medskip" if index % 2 else r"\hfill")
    if point_count < 2:
        raise PaperRenderError(f"{asset.asset_id} produced insufficient canonical geometry")
    return "\n".join(
        [
            _marker(asset, item, row_count=point_count),
            *_canonical_comments(item),
            r"\begin{figure}[htbp]",
            r"\centering",
            *rendered,
            rf"\caption{{{_escape(asset.title)}}}",
            rf"\label{{{_escape('decaf:' + asset.asset_id)}}}",
            r"\end{figure}",
            "",
        ]
    )


def _table_records(
    frame: pd.DataFrame, columns: Sequence[str]
) -> tuple[list[str], list[list[Any]]]:
    records = [json.loads(str(value)) for value in frame["record_json"]]
    selected = [column for column in columns if any(column in record for record in records)]
    if not selected:
        raise PaperRenderError("canonical table has no registered display columns")
    return selected, [[record.get(column) for column in selected] for record in records]


def _render_data_table(asset: VisualAsset, frame: pd.DataFrame, item: Mapping[str, Any]) -> str:
    contract = semantic_contract(asset)
    columns, records = _table_records(frame, contract["display_columns"])
    alignment = "l" * len(columns)
    lines = [
        _marker(asset, item, row_count=len(frame)),
        *_canonical_comments(item),
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(rf"\textbf{{{_escape(column)}}}" for column in columns) + " \\\\",
        r"\hline",
    ]
    for record in records:
        lines.append(" & ".join(_escape(_short(value)) for value in record) + " \\\\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{_escape(asset.title)}}}",
            rf"\label{{{_escape('decaf:' + asset.asset_id)}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_data_asset(
    asset: VisualAsset, receipt: Mapping[str, Any], analysis: Mapping[str, Any]
) -> str:
    """Render a figure/table only after validating its canonical semantic frame."""

    del analysis
    root, canonical = _context(receipt)
    try:
        frame, item = load_canonical_asset(root, asset, canonical)
    except SemanticDataError as error:
        raise PaperRenderError(str(error)) from error
    if asset.kind == "figure":
        return _render_numeric_figure(asset, frame, item)
    return _render_data_table(asset, frame, item)


def render_registry_table(
    asset: VisualAsset, receipt: Mapping[str, Any], analysis: Mapping[str, Any]
) -> str:
    """Render a config-derived registry table through the same canonical gate."""

    return render_data_asset(asset, receipt, analysis)


def render_source_missing_asset(
    asset: VisualAsset, receipt: Mapping[str, Any], analysis: Mapping[str, Any]
) -> str:
    """Render the sole explicit historical-source gap without fabricated geometry."""

    del receipt, analysis
    note = asset.source_note or "Historical source was not recovered."
    fields = ("missing_item", "why_it_matters", "reproducible_scope", "required_recovery_action")
    lines = ["status=source_missing", note]
    lines.extend(f"{field}={asset.generation_contract[field]}" for field in fields)
    body = " \\\n".join(_escape(line) for line in lines)
    return (
        f"% DECAF_SOURCE_MISSING asset={asset.asset_id}\n"
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        "  \\fbox{\\begin{minipage}{0.94\\linewidth}\n"
        "    \\textbf{Source unavailable}\\\\\n"
        f"    {body}\n"
        "  \\end{minipage}}\n"
        f"  \\caption{{{_escape(asset.title)}}}\n"
        f"  \\label{{{_escape('decaf:' + asset.asset_id)}}}\n"
        "\\end{figure}\n"
    )


def _parse_cardinality(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        name, separator, count = item.rpartition(":")
        if not separator or not name or name in result:
            raise PaperRenderError(f"invalid semantic cardinality marker: {value}")
        result[name] = int(count)
    return result


def validate_rendered_asset(asset: VisualAsset, content: str) -> str:
    """Enforce semantic contract/schema/cardinality markers in generated TeX."""

    if asset.status == "source_missing":
        expected = f"% DECAF_SOURCE_MISSING asset={asset.asset_id}"
        if expected not in content or r"status=source\_missing" not in content:
            raise PaperRenderError(f"{asset.asset_id} does not record its source gap")
        return "source_missing_recorded"
    match = _MARKER.search(content)
    if match is None or match.group("asset") != asset.asset_id:
        raise PaperRenderError(f"{asset.asset_id} has no semantic render marker")
    if match.group("contract") != semantic_contract_sha256(asset):
        raise PaperRenderError(f"{asset.asset_id} semantic contract hash drifted")
    if match.group("schema") != CANONICAL_SCHEMA_SHA256:
        raise PaperRenderError(f"{asset.asset_id} canonical schema hash drifted")
    cardinality = _parse_cardinality(match.group("cardinality"))
    if canonical_cardinality_sha256(cardinality) != match.group("cardinality_hash"):
        raise PaperRenderError(f"{asset.asset_id} cardinality hash drifted")
    contract = semantic_contract(asset)
    expected_panels = set(contract.get("panels", {"table_body": 1}))
    if set(cardinality) != expected_panels or int(match.group("panels")) != len(cardinality):
        raise PaperRenderError(f"{asset.asset_id} panel structure drifted")
    minima = contract.get("panels", {"table_body": int(contract.get("minimum_rows", 1))})
    if any(cardinality[name] < int(minimum) for name, minimum in minima.items()):
        raise PaperRenderError(f"{asset.asset_id} panel cardinality is below contract")
    if "exact_rows" in contract and sum(cardinality.values()) != int(contract["exact_rows"]):
        raise PaperRenderError(f"{asset.asset_id} exact row cardinality drifted")
    if asset.kind == "figure":
        if match.group("kind") != "GEOMETRY" or r"\begin{picture}" not in content:
            raise PaperRenderError(f"{asset.asset_id} is not canonical data geometry")
        if int(match.group("rows")) < 2 or r"\circle" not in content:
            raise PaperRenderError(f"{asset.asset_id} has insufficient canonical marks")
        return "regenerated_semantic_geometry"
    if match.group("kind") != "TABLE" or r"\begin{tabular}" not in content:
        raise PaperRenderError(f"{asset.asset_id} is not canonical tabular data")
    if int(match.group("rows")) != sum(cardinality.values()):
        raise PaperRenderError(f"{asset.asset_id} tabular row cardinality drifted")
    return "regenerated_semantic_table"


def render_all(
    replay_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    generated_root: str | Path | None = None,
) -> list[Path]:
    """Generate and validate all 28 TeX assets from canonical replay data."""

    repo = Path(repo_root).resolve() if repo_root else repository_root()
    replay = Path(replay_root).resolve()
    receipt = json.loads((replay / "replay_receipt.json").read_text(encoding="utf-8"))
    context = {**receipt, "_paper_data_root": str(replay / receipt["paper_data_directory"])}
    manifest = load_visual_manifest(repo / "paper" / "visual_manifest.yaml")
    output = Path(generated_root).resolve() if generated_root else repo
    paths: list[Path] = []
    artifact_rows: list[dict[str, Any]] = []
    canonical_by_id = {
        str(item["asset_id"]): item for item in receipt.get("canonical", {}).get("artifacts", ())
    }
    for asset in manifest.assets.values():
        generator = import_generator(asset.generator)
        content = generator(asset, context, context)
        classification = validate_rendered_asset(asset, content)
        if generated_root:
            subdirectory = "figures" if asset.kind == "figure" else "tables"
            destination = output / subdirectory / Path(asset.tex_target).name
        else:
            destination = output / asset.tex_target
        atomic_text(destination, content)
        paths.append(destination)
        canonical = canonical_by_id.get(asset.asset_id, {})
        artifact_rows.append(
            {
                "asset_id": asset.asset_id,
                "kind": asset.kind,
                "number": asset.number,
                "manifest_status": asset.status,
                "generated_path": destination.relative_to(output).as_posix(),
                "generated_sha256": sha256_file(destination),
                "generated_bytes": destination.stat().st_size,
                "comparison_status": classification,
                "canonical_path": canonical.get("path", ""),
                "canonical_sha256": canonical.get("sha256", ""),
                "semantic_contract_sha256": canonical.get("semantic_contract_sha256", ""),
                "schema_sha256": canonical.get("schema_sha256", ""),
                "row_count": canonical.get("row_count", ""),
                "panel_cardinality": json.dumps(
                    canonical.get("panel_cardinality", {}), sort_keys=True, separators=(",", ":")
                ),
            }
        )
    verification = replay / "verification"
    verification.mkdir(parents=True, exist_ok=True)
    columns = list(artifact_rows[0])
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(artifact_rows)
    atomic_text(verification / "paper_artifact_diff.csv", buffer.getvalue())
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        "--reference-runs",
        action="append",
        dest="reference_roots",
    )
    parser.add_argument("--replay-root", help="Replay output directory")
    parser.add_argument("--repo-root", help="Repository root containing paper manifests")
    parser.add_argument(
        "--generated-root",
        "--output",
        dest="generated_root",
        help="Alternative root for generated TeX snippets",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Reuse an existing canonical replay receipt",
    )
    parser.add_argument("--materialize-only", action="store_true", help="Skip TeX generation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.render_only and args.materialize_only:
        raise SystemExit("--render-only and --materialize-only are mutually exclusive")
    repo = Path(args.repo_root).resolve() if args.repo_root else repository_root()
    replay_root = (
        Path(args.replay_root).resolve()
        if args.replay_root
        else repo / "verification" / "paper-replay"
    )
    roots = [
        item for raw in (args.reference_roots or ()) for item in str(raw).split(os.pathsep) if item
    ]
    if not args.render_only:
        replay_paper_data(replay_root, reference_root=roots or None, repo_root=repo)
    if not args.materialize_only:
        render_all(replay_root, repo_root=repo, generated_root=args.generated_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PaperRenderError",
    "PlotPanel",
    "PlotSeries",
    "build_parser",
    "main",
    "render_all",
    "render_data_asset",
    "render_registry_table",
    "render_source_missing_asset",
    "validate_rendered_asset",
]
