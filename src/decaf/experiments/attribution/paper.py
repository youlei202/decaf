"""Generate registered attribution paper-data tables from computed metrics."""

from __future__ import annotations

import json
from collections.abc import Callable
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from decaf.core.manifests import sha256_file
from decaf.experiments.attribution.analyze import atomic_csv
from decaf.experiments.common import RunContext, atomic_json, atomic_text

Selector = Callable[[pd.DataFrame], pd.DataFrame]


def _scope(name: str) -> Selector:
    def select(frame: pd.DataFrame) -> pd.DataFrame:
        if "scope" not in frame.columns:
            return frame.iloc[0:0].copy()
        return frame.loc[frame["scope"].astype(str) == name].copy()

    return select


def _scopes(*names: str) -> Selector:
    allowed = frozenset(names)

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        if "scope" not in frame.columns:
            return frame.iloc[0:0].copy()
        return frame.loc[frame["scope"].astype(str).isin(allowed)].copy()

    return select


def _dataset(name: str) -> Selector:
    def select(frame: pd.DataFrame) -> pd.DataFrame:
        if "dataset" not in frame.columns:
            return frame.iloc[0:0].copy()
        return frame.loc[frame["dataset"].astype(str) == name].copy()

    return select


TABLES: tuple[tuple[int, str, str, Selector | None], ...] = (
    (
        2,
        "funnybirds_idsds_attribution",
        "method_results.csv",
        _scopes(
            "idsds_primary",
            "funnybirds_primary",
            "smoke_idsds_primary",
            "smoke_funnybirds_primary",
        ),
    ),
    (3, "dinov2_g_stress_test", "large_model_quality_timing_join", None),
    (
        4,
        "endpoint_m_pairwise",
        "pairwise_differences.csv",
        _scopes(
            "idsds_primary",
            "funnybirds_primary",
            "smoke_idsds_primary",
            "smoke_funnybirds_primary",
        ),
    ),
    (
        6,
        "complete_cross_dataset_attribution",
        "method_results.csv",
        _scopes(
            "idsds_primary",
            "funnybirds_primary",
            "smoke_idsds_primary",
            "smoke_funnybirds_primary",
        ),
    ),
    (
        7,
        "paired_endpoint_trajectory",
        "pairwise_differences.csv",
        _scopes(
            "idsds_primary",
            "funnybirds_primary",
            "smoke_idsds_primary",
            "smoke_funnybirds_primary",
        ),
    ),
    (
        8,
        "architecture_endpoint_ablation",
        "per_model_results.csv",
        _scopes(
            "idsds_primary",
            "funnybirds_primary",
            "smoke_idsds_primary",
            "smoke_funnybirds_primary",
        ),
    ),
    (9, "idsds_full50k", "method_results.csv", _scope("idsds_full50k")),
    (10, "imagenet_compute", "timing_summary.csv", _dataset("imagenet1k_idsds")),
    (
        11,
        "partimagenet_boundary",
        "method_results.csv",
        _scopes("partimagenet_boundary", "smoke_partimagenet_boundary"),
    ),
)


def _source(metrics: Path, name: str) -> pd.DataFrame:
    path = metrics / name
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def _large_model_join(metrics: Path) -> pd.DataFrame:
    quality = _scopes("dinov2_g_quality", "smoke_dinov2_g_quality")(
        _source(metrics, "per_model_results.csv")
    )
    timing = _source(metrics, "timing_summary.csv")
    if not timing.empty:
        timing = timing.loc[timing["model"].astype(str) == "dinov2_vit_g_14"].copy()
    if quality.empty or timing.empty:
        return quality
    keys = ["dataset", "model", "method"]
    return quality.merge(timing, on=keys, how="outer", validate="one_to_one")


def _required_verification_tables(context: RunContext) -> frozenset[int]:
    execution = context.config.get("execution", {})
    if not isinstance(execution, dict) or execution.get("scheduler") != (
        "single_gpu_dynamic_queue"
    ):
        return frozenset()
    return {
        "smoke": frozenset((2, 4, 6, 7, 8)),
        "large-model-smoke": frozenset((3, 10)),
        "boundary-smoke": frozenset((11,)),
    }.get(context.profile, frozenset())


def _latex_cell(value: object) -> str:
    if value is None or (isinstance(value, Real) and pd.isna(value)):
        return ""
    if isinstance(value, Real) and not isinstance(value, bool):
        text = f"{value:.10g}"
    else:
        text = str(value)
    escapes = {
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
    return "".join(escapes.get(character, character) for character in text)


def _latex_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free deterministic LaTeX tabular."""

    columns = [str(column) for column in frame.columns]
    if not columns:
        return "\\begin{tabular}{l}\n\\hline\nNo data \\\\\n\\hline\n\\end{tabular}\n"
    lines = [f"\\begin{{tabular}}{{{'l' * len(columns)}}}", "\\hline"]
    lines.append(" & ".join(_latex_cell(column) for column in columns) + r" \\")
    lines.append("\\hline")
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_cell(value) for value in row) + r" \\")
    lines.extend(("\\hline", "\\end{tabular}"))
    return "\n".join(lines) + "\n"


def paper(context: RunContext) -> dict[str, Any]:
    """Materialize Tables 02--04 and 06--11 without inventing missing values."""

    metrics = context.path / "metrics"
    if context.profile == "paper" and (context.path / "manifests/plan.json").is_file():
        raise RuntimeError(
            "fresh computed-paper rendering is fail-closed: local analysis emits "
            "the complete Spearman summaries but not the historical Table 8 wide, "
            "Table 10 timing+memory+query, or Table 11 15x5 schemas; use the sealed "
            "CPU replay for paper tables until those GPU result converters are "
            "implemented"
        )
    destination = context.path / "paper_data"
    required_nonempty = _required_verification_tables(context)
    manifest_rows: list[dict[str, Any]] = []
    cache: dict[str, pd.DataFrame] = {}
    for number, label, source_name, selector in TABLES:
        if context.profile == "paper" and not (context.path / "manifests/plan.json").is_file():
            from decaf.experiments.attribution.reference import (
                FORMAL_TABLE_SOURCES,
            )

            formal_path = metrics / "formal_tables" / f"table_{number:02d}.csv"
            if not formal_path.is_file():
                raise FileNotFoundError(f"formal Attribution table input is absent: {formal_path}")
            source = pd.read_csv(formal_path)
            selected = source.copy()
            source_label = json.dumps(
                [
                    {"run_id": run_id, "member": member}
                    for run_id, member in FORMAL_TABLE_SOURCES[number]
                ],
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            if source_name == "large_model_quality_timing_join":
                source = cache.setdefault(source_name, _large_model_join(metrics))
            else:
                source = cache.setdefault(source_name, _source(metrics, source_name))
            selected = source.copy() if selector is None else selector(source)
            source_label = (
                "metrics/per_model_results.csv + metrics/timing_summary.csv"
                if source_name == "large_model_quality_timing_join"
                else f"metrics/{source_name}"
            )
        if number in required_nonempty and selected.empty:
            raise RuntimeError(
                f"single-B200 attribution paper table {number} is unexpectedly empty"
            )
        stem = f"table_{number:02d}_{label}"
        csv_path = destination / f"{stem}.csv"
        tex_path = destination / f"{stem}.tex"
        atomic_csv(selected, csv_path)
        atomic_text(tex_path, _latex_table(selected))
        manifest_rows.append(
            {
                "table": number,
                "label": label,
                "source": source_label,
                "rows": len(selected),
                "schema_only": selected.empty,
                "csv": csv_path.name,
                "csv_sha256": sha256_file(csv_path),
                "tex": tex_path.name,
                "tex_sha256": sha256_file(tex_path),
            }
        )
    atomic_json(
        destination / "attribution_tables.json",
        {
            "schema_version": 1,
            "registered_tables": [2, 3, 4, 6, 7, 8, 9, 10, 11],
            "tables": manifest_rows,
        },
    )
    return {
        "table_count": len(manifest_rows),
        "nonempty_tables": sum(not row["schema_only"] for row in manifest_rows),
        "registered_tables": [2, 3, 4, 6, 7, 8, 9, 10, 11],
    }


__all__ = ["TABLES", "paper"]
