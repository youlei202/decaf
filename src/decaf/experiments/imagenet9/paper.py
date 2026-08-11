"""Run-local paper-data contracts for ImageNet-9 Table 1 and Figures 6, 7, 12."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from decaf.experiments.common import RunContext, atomic_json, atomic_text
from decaf.experiments.imagenet9.baselines import baseline_plan
from decaf.paper.reference import sha256_file

SOURCE_INPUTS: dict[str, tuple[str, ...]] = {
    "figure_6": (
        "results/mechanism_benchmark/fixed_semantic.csv",
        "results/tables/T04_mechanism_benchmark.csv",
        "results/matched_abs/matched_abs_benchmark.csv",
        "results/matched_abs/matched_pairs.parquet",
        "results/tables/T05A_per_bin_auroc.csv",
        "results/mechanism_benchmark/probe_leave_one_architecture_family_out_summary.csv",
    ),
    "figure_7": (
        "results/model_decaf.csv",
        "results/tables/T03_decaf_profiles.csv",
        "results/sample_decaf.parquet",
        "results/stage_ledger.jsonl",
    ),
    "figure_12": (
        "results/tables/T03_decaf_profiles.csv",
        "results/sample_decaf.parquet",
        "results/stage_ledger.jsonl",
    ),
    "table_1": ("configs/decaf_imagenet9_v1/formal_8b200.yaml",),
}


def _registered_source_paths() -> set[str]:
    return {relative for paths in SOURCE_INPUTS.values() for relative in paths}


def validate_reference_replay(context: RunContext) -> dict[str, Any]:
    """Revalidate every sealed input against the analyze-stage receipt."""

    summary_path = context.path / "metrics" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("ImageNet-9 analysis summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("source_mode") != "sealed_reference_replay":
        return {"sealed_reference_validated": False}
    receipt_path = context.path / "receipts" / "reference_replay.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("sealed ImageNet-9 reference receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed" or receipt.get("run_id") != "I9":
        raise ValueError("sealed ImageNet-9 reference receipt identity is invalid")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("sealed ImageNet-9 reference receipt has no input inventory")
    expected = _registered_source_paths()
    observed = {str(item.get("requested_suffix")) for item in inputs}
    if observed != expected or len(inputs) != len(expected):
        raise ValueError("sealed ImageNet-9 reference receipt inventory differs")
    for item in inputs:
        suffix = str(item["requested_suffix"])
        if item.get("run_id") != "I9" or item.get("relative_path") != f"I9/{suffix}":
            raise ValueError(f"sealed ImageNet-9 reference identity differs: {suffix}")
        path = context.path / "reference_data" / str(item["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(f"sealed ImageNet-9 reference input is missing: {suffix}")
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"sealed ImageNet-9 reference size differs: {suffix}")
        if sha256_file(path) != item.get("sha256"):
            raise ValueError(f"sealed ImageNet-9 reference hash differs: {suffix}")
    return {"sealed_reference_validated": True, "reference_input_count": len(inputs)}


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _source_receipts(
    context: RunContext, *, require_complete: bool
) -> dict[str, list[dict[str, Any]]]:
    if require_complete:
        validate_reference_replay(context)
    root = context.path / "reference_data" / "I9"
    receipts: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for asset, relative_paths in SOURCE_INPUTS.items():
        items = []
        for relative in relative_paths:
            path = root / relative
            if not path.is_file():
                missing.append(f"{asset}:{relative}")
                continue
            destination = context.path / "paper_data" / "source_assets" / asset / path.name
            _atomic_copy(path, destination)
            source_sha256 = sha256_file(path)
            if sha256_file(destination) != source_sha256:
                raise RuntimeError(f"ImageNet-9 source copy hash mismatch: {relative}")
            items.append(
                {
                    "path": f"reference_data/I9/{relative}",
                    "materialized_path": str(destination.relative_to(context.path)),
                    "sha256": source_sha256,
                    "size_bytes": path.stat().st_size,
                }
            )
        receipts[asset] = items
    if require_complete and missing:
        raise FileNotFoundError(
            "sealed ImageNet-9 paper replay is missing registered sources: " + ", ".join(missing)
        )
    return receipts


def paper(context: RunContext) -> dict[str, Any]:
    """Write machine-readable panel data; the global renderer owns LaTeX output."""

    summary_path = context.path / "metrics" / "summary.json"
    mechanisms_path = context.path / "metrics" / "mechanism_summary.csv"
    ratios_path = context.path / "metrics" / "protocol_ratios.csv"
    rank_transfer_path = context.path / "metrics" / "protocol_rank_transfer.csv"
    accuracy_path = context.path / "metrics" / "matched_magnitude_accuracy.json"
    required = (
        summary_path,
        mechanisms_path,
        ratios_path,
        rank_transfer_path,
        accuracy_path,
    )
    missing = [str(path.relative_to(context.path)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"analyze must materialize ImageNet-9 paper inputs: {missing}")

    mechanisms = pd.read_csv(mechanisms_path)
    ratios = pd.read_csv(ratios_path)
    accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = pd.DataFrame(baseline_plan(list(map(str, context.config["baselines"]["methods"]))))

    reference_root = context.path / "reference_data" / "I9"
    benchmark_path = reference_root / "results" / "tables" / "T04_mechanism_benchmark.csv"
    matched_path = reference_root / "results" / "matched_abs" / "matched_abs_benchmark.csv"
    profiles_path = reference_root / "results" / "tables" / "T03_decaf_profiles.csv"
    figure6 = (
        pd.read_csv(benchmark_path)
        if benchmark_path.is_file()
        else mechanisms[["model_id", "reveal_path", "E", "C", "F", "Abs"]]
    )
    figure7 = ratios
    figure12 = (
        pd.read_csv(profiles_path)
        if profiles_path.is_file()
        else mechanisms[["model_id", "reveal_path", "F", "Abs", "M"]].copy()
    )
    table1 = methods[["method_id", "access", "nominal_queries", "requires_gradients"]]
    outputs = {
        "figure_6": "paper_data/figure_6_mechanism_benchmark.csv",
        "figure_7": "paper_data/figure_7_protocol_audit.csv",
        "figure_12": "paper_data/figure_12_robustness.csv",
        "table_1": "paper_data/table_1_access_query_structure.csv",
    }
    for key, frame in (
        ("figure_6", figure6),
        ("figure_7", figure7),
        ("figure_12", figure12),
        ("table_1", table1),
    ):
        atomic_text(context.path / outputs[key], frame.to_csv(index=False))
    supporting_outputs: dict[str, dict[str, str]] = {
        "figure_6": {},
        "figure_7": {},
        "figure_12": {},
        "table_1": {},
    }
    if matched_path.is_file():
        destination = "paper_data/figure_6_matched_magnitude.csv"
        atomic_text(context.path / destination, pd.read_csv(matched_path).to_csv(index=False))
        supporting_outputs["figure_6"]["matched_magnitude"] = destination
    rank_destination = "paper_data/figure_7_protocol_rank_transfer.csv"
    atomic_text(
        context.path / rank_destination,
        rank_transfer_path.read_text(encoding="utf-8"),
    )
    supporting_outputs["figure_7"]["protocol_rank_transfer"] = rank_destination
    baseline_summary_path = context.path / "metrics" / "baseline_summary.csv"
    if baseline_summary_path.is_file():
        baseline_destination = "paper_data/table_1_baseline_summary.csv"
        atomic_text(
            context.path / baseline_destination,
            baseline_summary_path.read_text(encoding="utf-8"),
        )
        supporting_outputs["table_1"]["baseline_summary"] = baseline_destination
    sealed_reference = summary.get("source_mode") == "sealed_reference_replay"
    source_inputs = _source_receipts(context, require_complete=sealed_reference)
    receipt = {
        "schema_version": 1,
        "experiment": "imagenet9",
        "source_summary": "metrics/summary.json",
        "assets": outputs,
        "supporting_outputs": supporting_outputs,
        "source_inputs": source_inputs,
        "matched_magnitude_accuracy": accuracy,
        "historical_headlines_asserted_here": sealed_reference,
        "reference_headlines": summary.get("reference_headlines", {}),
        "note": (
            "These are run-local machine-readable panel inputs; the global paper "
            "renderer owns the LaTeX inclusion artifacts."
        ),
    }
    atomic_json(context.path / "paper_data" / "manifest.json", receipt)
    return {
        "paper_assets": len(outputs),
        "source_assets": sum(map(len, source_inputs.values())),
        "manifest": "paper_data/manifest.json",
    }


__all__ = ["paper", "validate_reference_replay"]
