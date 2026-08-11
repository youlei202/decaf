"""Family-local panel data for controlled Figures 2--5 and 8--11."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decaf.core.bootstrap import bootstrap_mean
from decaf.core.manifests import atomic_write_json, sha256_file
from decaf.experiments.controlled.analyze import ControlledReferenceBundle

PANEL_COLUMNS = (
    "artifact_id",
    "panel_id",
    "series",
    "x",
    "y",
    "estimate",
    "ci_low",
    "ci_high",
    "n",
    "source_sha256",
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _values(frame: pd.DataFrame, value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str) and value in frame:
        return frame[value].to_numpy()
    return value


def panel_frame(
    frame: pd.DataFrame,
    *,
    artifact_id: str,
    panel_id: str,
    source: Path,
    series: Any,
    x: Any,
    estimate: Any,
    ci_low: Any = None,
    ci_high: Any = None,
    n: Any = None,
) -> pd.DataFrame:
    """Attach the portable paper-data columns while preserving source fields."""

    rows = frame.reset_index(drop=True).copy()
    count = len(rows)
    if count < 1:
        raise ValueError(f"paper panel {artifact_id}/{panel_id} has no rows")
    estimate_values = _values(rows, estimate, np.nan)
    prefix = pd.DataFrame(
        {
            "artifact_id": np.repeat(artifact_id, count),
            "panel_id": np.repeat(panel_id, count),
            "series": _values(rows, series, "all"),
            "x": _values(rows, x, np.arange(count)),
            "y": estimate_values,
            "estimate": estimate_values,
            "ci_low": _values(rows, ci_low, np.nan),
            "ci_high": _values(rows, ci_high, np.nan),
            "n": _values(rows, n, 1),
            "source_sha256": np.repeat(sha256_file(source), count),
        }
    )
    return pd.concat([prefix, rows], axis=1)


def _module_f_bootstrap(bundle: ControlledReferenceBundle) -> pd.DataFrame:
    samples = bundle["c1_bootstrap"]
    primary_geometry = pd.to_numeric(samples["primary_geometry"], errors="coerce").eq(1.0)
    primary_geometry |= samples["primary_geometry"].astype(str).str.lower().isin({"true", "yes"})
    primary = samples.loc[
        samples["module"].eq("F") & primary_geometry & samples["factor"].eq("floor_color")
    ]
    per_model = primary.groupby(["variant", "architecture", "model_id"], as_index=False, sort=True)[
        "F"
    ].mean()
    rows: list[dict[str, Any]] = []
    for (variant, architecture), group in per_model.groupby(["variant", "architecture"], sort=True):
        result = bootstrap_mean(
            group["F"].to_numpy(),
            n_resamples=500,
            confidence_level=0.90,
            seed=7301,
        )
        rows.append(
            {
                "variant": variant,
                "architecture": architecture,
                "estimate": float(result.estimate),
                "ci_low": float(result.lower),
                "ci_high": float(result.upper),
                "n_models": result.n_observations,
                "bootstrap_repetitions": result.n_resamples,
            }
        )
    return pd.DataFrame(rows)


def _evidence_trajectory(
    bundle: ControlledReferenceBundle, selection: Mapping[str, Any]
) -> pd.DataFrame:
    stages = bundle["c1_stages"]
    selected = stages.loc[
        stages["module"].eq("E")
        & stages["primary_geometry"].astype(str).str.lower().isin({"1", "true", "yes"})
        & stages["architecture"].eq(selection["architecture"])
        & pd.to_numeric(stages["p_train"], errors="coerce").eq(
            float(selection["training_correlation"])
        )
        & pd.to_numeric(stages["seed"], errors="coerce").eq(int(selection["seed"]))
        & stages["trajectory_id"].eq(selection["trajectory_id"])
    ].copy()
    grouped = selected.groupby(
        [
            "model_id",
            "architecture",
            "p_train",
            "seed",
            "trajectory_id",
            "epoch",
            "factor",
            "alpha",
        ],
        as_index=False,
        sort=True,
    )[["U_abs", "U_align", "U_opp", "U_null"]].mean()
    grouped["series_id"] = (
        grouped["factor"].astype(str) + "__epoch_" + grouped["epoch"].astype(int).astype(str)
    )
    return grouped


def _all_evidence_trajectories(bundle: ControlledReferenceBundle) -> pd.DataFrame:
    stages = bundle["c1_stages"]
    selected = stages.loc[
        stages["module"].eq("E")
        & stages["primary_geometry"].astype(str).str.lower().isin({"1", "true", "yes"})
    ]
    grouped = selected.groupby(
        [
            "model_id",
            "architecture",
            "p_train",
            "seed",
            "trajectory_id",
            "epoch",
            "factor",
            "alpha",
        ],
        as_index=False,
        sort=True,
    )[["U_abs", "U_align", "U_opp", "U_null"]].mean()
    grouped["series_id"] = (
        grouped["trajectory_id"].astype(str) + "__" + grouped["factor"].astype(str)
    )
    return grouped


def _calibration_transfer(bundle: ControlledReferenceBundle) -> pd.DataFrame:
    curves = bundle["c2_epsilon"]
    return curves.groupby(
        ["task", "architecture", "wall_map", "epsilon"], as_index=False, sort=True
    )[["E", "C", "F", "Abs", "Net", "phi_C"]].mean()


def write_reference_paper_data(
    bundle: ControlledReferenceBundle,
    destination: str | Path,
    *,
    headline_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate compact, provenance-bearing data for all controlled figures."""

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    response = bundle["c0_response"]
    matched = response.loc[
        response["task"].eq("object_shape")
        & response["factor"].isin(("object_shape", "floor_color"))
    ].copy()
    matched["n_models"] = matched.groupby("factor")["model_id"].transform("size")
    artifacts.append(
        _atomic_csv(
            panel_frame(
                matched,
                artifact_id="figure_02",
                panel_id="matched_abs",
                source=bundle.sources["c0_response"],
                series="factor",
                x="model_seed",
                estimate="mean_auc_abs",
                n="n_models",
            ),
            output / "figure_02_matched_abs.csv",
        )
    )
    methods = (
        bundle["c0_methods"]
        .loc[bundle["c0_methods"]["method"].isin(("Abs-CMMR", "Align-CMMR"))]
        .copy()
    )
    artifacts.append(
        _atomic_csv(
            panel_frame(
                methods,
                artifact_id="figure_02",
                panel_id="false_null_order",
                source=bundle.sources["c0_methods"],
                series="method",
                x="method",
                estimate="false_null_order",
                n="eligible_supported_null_pairs",
            ),
            output / "figure_02_false_null.csv",
        )
    )

    correlations = bundle["c1_e_correlations"].copy()
    artifacts.append(
        _atomic_csv(
            panel_frame(
                correlations,
                artifact_id="figure_03",
                panel_id="evidence_correspondence",
                source=bundle.sources["c1_e_correlations"],
                series="metric",
                x="metric",
                estimate="pooled_spearman",
                n="n",
            ),
            output / "figure_03_correlations.csv",
        )
    )
    selection = headline_summary["figure_03"]["representative"]
    trajectory = _evidence_trajectory(bundle, selection)
    artifacts.append(
        _atomic_csv(
            panel_frame(
                trajectory,
                artifact_id="figure_03",
                panel_id="representative_trajectory",
                source=bundle.sources["c1_stages"],
                series="series_id",
                x="alpha",
                estimate="U_align",
                n=1,
            ),
            output / "figure_03_trajectory.csv",
        )
    )
    selection_path = output / "figure_03_selection.json"
    atomic_write_json(selection_path, selection)
    artifacts.append(selection_path)

    validation = bundle["c1_f_validation"]
    regimes = validation.loc[validation["section"].eq("variant_summary")].copy()
    artifacts.append(
        _atomic_csv(
            panel_frame(
                regimes,
                artifact_id="figure_04",
                panel_id="fragility_regimes",
                source=bundle.sources["c1_f_validation"],
                series="variant",
                x="architecture",
                estimate="F_mean",
                n="n_models",
            ),
            output / "figure_04_regimes.csv",
        )
    )
    f_bootstrap = _module_f_bootstrap(bundle)
    artifacts.append(
        _atomic_csv(
            panel_frame(
                f_bootstrap,
                artifact_id="figure_04",
                panel_id="model_bootstrap",
                source=bundle.sources["c1_bootstrap"],
                series="variant",
                x="architecture",
                estimate="estimate",
                ci_low="ci_low",
                ci_high="ci_high",
                n="n_models",
            ),
            output / "figure_04_bootstrap.csv",
        )
    )

    for name, panel_id, source_name, series, x, estimate, count in (
        ("figure_05_epsilon.csv", "epsilon_curves", "c2_epsilon", "task", "epsilon", "C", 1),
        (
            "figure_05_behavior.csv",
            "behavior",
            "c2_behavior",
            "task",
            "epsilon",
            "pairwise_swap_rate",
            1,
        ),
        ("figure_05_seed.csv", "seed_results", "c2_seeds", "task", "seed", "C", 1),
        (
            "figure_05_bootstrap.csv",
            "bootstrap",
            "c2_bootstrap",
            "metric",
            "epsilon",
            "estimate",
            "models",
        ),
    ):
        artifacts.append(
            _atomic_csv(
                panel_frame(
                    bundle[source_name],
                    artifact_id="figure_05",
                    panel_id=panel_id,
                    source=bundle.sources[source_name],
                    series=series,
                    x=x,
                    estimate=estimate,
                    ci_low="ci_low" if "ci_low" in bundle[source_name] else None,
                    ci_high="ci_high" if "ci_high" in bundle[source_name] else None,
                    n=count,
                ),
                output / name,
            )
        )

    artifacts.append(
        _atomic_csv(
            panel_frame(
                response,
                artifact_id="figure_08",
                panel_id="controlled_atlas",
                source=bundle.sources["c0_response"],
                series="factor",
                x="model_id",
                estimate="mean_auc_abs",
                n=1,
            ),
            output / "figure_08_atlas.csv",
        )
    )
    all_evidence = _all_evidence_trajectories(bundle)
    artifacts.append(
        _atomic_csv(
            panel_frame(
                all_evidence,
                artifact_id="figure_09",
                panel_id="all_evidence_trajectories",
                source=bundle.sources["c1_stages"],
                series="series_id",
                x="alpha",
                estimate="U_align",
                n=1,
            ),
            output / "figure_09_all_evidence_trajectories.csv",
        )
    )
    transfer = bundle["c1_geometry"]
    artifacts.append(
        _atomic_csv(
            panel_frame(
                transfer,
                artifact_id="figure_10",
                panel_id="geometry_transfer",
                source=bundle.sources["c1_geometry"],
                series="metric",
                x="geometry",
                estimate="spearman_vs_cmmr",
                n="n_pairs",
            ),
            output / "figure_10_geometry_transfer.csv",
        )
    )
    calibration = _calibration_transfer(bundle)
    artifacts.append(
        _atomic_csv(
            panel_frame(
                calibration,
                artifact_id="figure_11",
                panel_id="calibration_transfer",
                source=bundle.sources["c2_epsilon"],
                series="task",
                x="epsilon",
                estimate="C",
                n=5,
            ),
            output / "figure_11_calibration_transfer.csv",
        )
    )

    receipt = {
        "schema_version": 1,
        "family": "controlled",
        "figures": [2, 3, 4, 5, 8, 9, 10, 11],
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(artifacts)
        ],
        "headline_summary": headline_summary,
    }
    receipt_path = output / "controlled_receipt.json"
    atomic_write_json(receipt_path, receipt)
    return {"artifacts": len(artifacts), "receipt": receipt_path.name}


def write_smoke_paper_data(metrics_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Write one honest CPU-oracle panel for smoke plumbing verification."""

    source = Path(metrics_path)
    frame = pd.read_csv(source)
    panel = panel_frame(
        frame,
        artifact_id="controlled_smoke",
        panel_id="score_oracle",
        source=source,
        series="metric",
        x="model_id",
        estimate="value",
        n="n_values",
    )
    output = Path(destination)
    artifact = _atomic_csv(panel, output / "controlled_smoke_panel.csv")
    receipt = {
        "schema_version": 1,
        "family": "controlled",
        "scope": "cpu_score_oracle",
        "gpu_real_shard_verification": "pending",
        "artifacts": [
            {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        ],
    }
    atomic_write_json(output / "controlled_receipt.json", receipt)
    return {"artifacts": 1, "scope": "cpu_score_oracle"}


__all__ = [
    "PANEL_COLUMNS",
    "panel_frame",
    "write_reference_paper_data",
    "write_smoke_paper_data",
]
