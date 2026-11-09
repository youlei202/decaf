"""Analyze fixed, unfitted DECAF and baseline scores on exactly matched populations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from prepare import checksum, save_json
from protocol import attribute_group, behavior_targets, decompose, normalized_attribute
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def safe_auc(y, score):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def safe_rho(y, score):
    if len(y) < 3 or len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return None
    return float(spearmanr(y, score).statistic)


def cluster_bootstrap_indices(clusters, repetitions: int, seed: int):
    names = np.unique(clusters)
    mapping = {name: np.flatnonzero(clusters == name) for name in names}
    random = np.random.default_rng(seed)
    for _ in range(repetitions):
        yield np.concatenate(
            [mapping[name] for name in random.choice(names, len(names), replace=True)]
        )


def interval(values):
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return [float(v) for v in np.quantile(finite, [0.025, 0.975])] if len(finite) >= 20 else None


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_single(run: Path, config: dict, repetitions: int, seed: int):
    images = json.loads((run / "evaluated_images.json").read_text())
    with np.load(run / "single_image_scores.npz") as cache:
        cosines = cache["cosines"]
        row_ids = cache["row_ids"]
    if row_ids.tolist() != [row["row_id"] for row in images]:
        raise ValueError("Single-image metadata and score order mismatch")
    attr_names = config["attribute_queries"]
    dataset_names = config["source"]["feature_names"]["attr_label"]
    correct = np.empty(cosines.shape[:2], dtype=bool)
    output_rows = []
    for i, image in enumerate(images):
        attr = normalized_attribute(dataset_names[image["attr_label"]])
        true_index = attr_names.index(attr)
        group = attribute_group(attr)
        indices = np.array(config["groups"][group])
        predictions = indices[cosines[i][:, indices].argmax(axis=1)]
        correct[i] = predictions == true_index
        for view, prediction, hit in zip(config["views"], predictions, correct[i], strict=True):
            output_rows.append(
                {
                    "row_id": image["row_id"],
                    "bird_label": image["bird_label"],
                    "true_attribute": attr,
                    "attribute_group": group,
                    "view": view,
                    "predicted_attribute": attr_names[int(prediction)],
                    "correct": int(hit),
                }
            )
    clusters = np.array([image["bird_label"] for image in images])
    bootstrap = list(cluster_bootstrap_indices(clusters, repetitions, seed))
    by_view = []
    for i, view in enumerate(config["views"]):
        values = [float(correct[index, i].mean()) for index in bootstrap]
        by_view.append(
            {
                "view": view,
                "n": len(images),
                "accuracy": float(correct[:, i].mean()),
                "bird_cluster_95ci": interval(values),
            }
        )
    groups = {}
    for i, image in enumerate(images):
        key = attribute_group(dataset_names[image["attr_label"]])
        groups.setdefault(key, []).append(i)
    by_group = [
        {"attribute_group": group, "n": len(index), "accuracy": float(correct[index, 0].mean())}
        for group, index in sorted(groups.items())
    ]
    cells = {}
    for i, image in enumerate(images):
        cells.setdefault((image["bird_label"], image["attr_label"]), []).append(i)
    macro_cell = float(np.mean([correct[index, 0].mean() for index in cells.values()]))
    bird_predictions = cosines[:, 0, len(attr_names) :].argmax(axis=-1)
    bird_accuracy = float(np.mean(bird_predictions == clusters))
    write_csv(run / "single_image_predictions.csv", output_rows)
    write_csv(run / "single_image_accuracy_by_group.csv", by_group)
    return {
        "task": "Official-style modified attribute recognition (group-wise CLIP cosine argmax)",
        "views": by_view,
        "clean_macro_bird_attribute_cell_accuracy": macro_cell,
        "clean_bird_class_accuracy_33_way": bird_accuracy,
        "number_of_cells": len(cells),
        "number_of_attributes": len({r["attr_label"] for r in images}),
        "note": "Uses every released target attribute and all official prompt alternatives "
        "in its group;"
        " does not claim exact replication of paper tables/112-concept CBM filtering.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--change-tolerance", type=float, default=0.20)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    if args.epsilon <= 0 or args.change_tolerance <= 0 or args.bootstrap < 0:
        parser.error("Positive epsilon/tolerance and nonnegative bootstrap required")
    run = args.run.resolve()
    receipt = json.loads((run / "inference_receipt.json").read_text())
    if not receipt["complete"]:
        raise ValueError("Incomplete inference run")
    for name, expected in receipt["artifact_sha256"].items():
        if checksum(run / name) != expected:
            raise ValueError(f"Inference artifact checksum mismatch: {name}")
    config = json.loads((run / "config.json").read_text())
    pairs = json.loads((run / "evaluated_pairs.json").read_text())
    with np.load(run / "paired_scores.npz") as cached:
        data = {name: cached[name] for name in cached.files}
    if len(pairs) != len(data["response"]):
        raise ValueError("Pair manifest and score length mismatch")
    if data["pair_ids"].tolist() != [pair["pair_id"] for pair in pairs]:
        raise ValueError("Pair cache order differs from evaluated pair manifest")
    scores = decompose(data["response"], data["stages"], args.epsilon)
    # Disclosed post-hoc check, applied to every target/population uniformly.
    # Keep the originally specified positive-score comparisons unchanged.
    scores["negative_Abs"] = -scores["Abs"]
    scores["negative_M"] = -scores["M"]
    targets = behavior_targets(
        data["clean_scores"],
        data["view_scores"],
        data["clean_predictions"],
        data["view_predictions"],
        data["plus_query"],
        data["minus_query"],
        args.epsilon,
        args.change_tolerance,
    )
    # Gate is computed from the same clean endpoint used by every comparator.
    if not np.array_equal(scores["active"], targets["active"]):
        raise ValueError("Clean and trajectory endpoint gates differ")
    methods = [
        "E",
        "C",
        "F",
        "Abs",
        "M",
        "signed_endpoint",
        "signed_net",
        "negative_Abs",
        "negative_M",
    ]
    clusters = np.array([pair["bird_label"] for pair in pairs])
    active = scores["active"]
    populations = {
        "all_pairs": np.ones(len(pairs), dtype=bool),
        "endpoint_active": active,
        "endpoint_null": ~active,
    }
    target_names = ["recognition_success", "offpath_reversal", "offpath_instability"]
    metrics = []
    deltas = []
    for population, mask in populations.items():
        if not mask.any():
            continue
        indices = np.flatnonzero(mask)
        boots = list(cluster_bootstrap_indices(clusters[mask], args.bootstrap, args.seed))
        for target_name in target_names:
            if target_name == "offpath_reversal" and population != "endpoint_active":
                # Endpoint orientation is meaningful only outside the null tolerance.
                continue
            y = targets[target_name][mask]
            for method in methods:
                values = scores[method][mask]
                auc = safe_auc(y, values)
                ci = (
                    interval([safe_auc(y[index], values[index]) for index in boots])
                    if auc is not None
                    else None
                )
                metrics.append(
                    {
                        "population": population,
                        "target": target_name,
                        "method": method,
                        "n": len(indices),
                        "positive": int(y.sum()),
                        "bird_clusters": len(np.unique(clusters[mask])),
                        "auroc": auc,
                        "ci_low": ci[0] if ci else None,
                        "ci_high": ci[1] if ci else None,
                        "spearman": safe_rho(y.astype(float), values),
                        "analysis_status": (
                            "posthoc_direction_control"
                            if method.startswith("negative_")
                            else "prespecified_positive_or_signed_score"
                        ),
                    }
                )
            primary = {
                "recognition_success": "E",
                "offpath_reversal": "C",
                "offpath_instability": "F",
            }[target_name]
            for baseline in ["Abs", "M", "negative_Abs", "negative_M"]:
                a, b = scores[primary][mask], scores[baseline][mask]
                aa, bb = safe_auc(y, a), safe_auc(y, b)
                differences = []
                for index in boots:
                    av, bv = safe_auc(y[index], a[index]), safe_auc(y[index], b[index])
                    if av is not None and bv is not None:
                        differences.append(av - bv)
                ci = interval(differences)
                deltas.append(
                    {
                        "population": population,
                        "target": target_name,
                        "component": primary,
                        "baseline": baseline,
                        "auroc_difference": aa - bb if aa is not None and bb is not None else None,
                        "ci_low": ci[0] if ci else None,
                        "ci_high": ci[1] if ci else None,
                        "analysis_status": (
                            "posthoc_direction_control"
                            if baseline.startswith("negative_")
                            else "prespecified_comparison"
                        ),
                    }
                )
    write_csv(run / "paired_metrics.csv", metrics)
    write_csv(run / "paired_auc_differences.csv", deltas)
    pair_records = []
    for i, pair in enumerate(pairs):
        pair_records.append(
            dict(
                pair,
                **{method: float(scores[method][i]) for method in methods},
                endpoint_active=int(active[i]),
                **{name: float(values[i]) for name, values in targets.items() if name != "active"},
            )
        )
    write_csv(run / "pair_scores_and_targets.csv", pair_records)
    single = analyze_single(run, config, args.bootstrap, args.seed)
    summary = {
        "complete": True,
        "incomplete_smoke": config["incomplete_smoke"],
        "model": config["model"],
        "pretrained": config["pretrained"],
        "model_state_sha256": config["model_state_sha256"],
        "source": config["source"],
        "pair_type": config["pair_type"],
        "single_image": single,
        "pairs": len(pairs),
        "endpoint_active": int(active.sum()),
        "endpoint_null": int((~active).sum()),
        "epsilon": args.epsilon,
        "change_tolerance": args.change_tolerance,
        "analysis_source_sha256": checksum(Path(__file__)),
        "inference_run_identity": receipt["run_identity"],
        "input_artifact_sha256": receipt["artifact_sha256"],
        "primary_metric_rows": [
            row
            for row in metrics
            if (row["population"], row["target"])
            in {
                ("all_pairs", "recognition_success"),
                ("endpoint_active", "offpath_reversal"),
                ("endpoint_null", "offpath_instability"),
            }
        ],
        "primary_auc_differences": [
            row
            for row in deltas
            if (row["population"], row["target"])
            in {
                ("all_pairs", "recognition_success"),
                ("endpoint_active", "offpath_reversal"),
                ("endpoint_null", "offpath_instability"),
            }
        ],
        "posthoc_direction_control": {
            "description": "After inspecting the initial results, we added -Abs and -M "
            "uniformly to every reported outcome/population. This is a disclosed sensitivity "
            "analysis, not a prespecified direction choice or a held-out fitted predictor.",
            "original_analysis_archive": "analysis_prespecified/",
            "original_positive_score_results_preserved": True,
            "reason": "Low endpoint margins can predict reversal; comparing only positive "
            "magnitude scores would overstate evidence for additional information in C.",
        },
        "conservation_max_error": float(
            np.max(np.abs(scores["E"] + scores["C"] + scores["F"] - scores["Abs"]))
        ),
        "null_F_equals_Abs_max_error": float(
            np.max(np.abs(scores["F"][~active] - scores["Abs"][~active]))
        )
        if (~active).any()
        else None,
        "uncertainty": {
            "method": "percentile bootstrap clustered by bird class",
            "replicates": args.bootstrap,
            "seed": args.seed,
            "bird_clusters": len(np.unique(clusters)),
        },
        "limitations": [
            "SUB is synthetic imagery based on CUB bird/attribute semantics, "
            "not original real-photo edits.",
            "Paired endpoints share bird class and attribute group but are different "
            "generated instances; nuisance factors can change.",
            "Targets use held-out endpoint transformations; labels are intended SUB attributes, "
            "not manually reverified image attributes.",
            "Endpoint gating is shared with the representation; all score comparisons "
            "use identical subsets.",
            "No causal single-factor isolation or advantage over baselines is assumed.",
            "Existing CLIP pretraining may include CUB bird concepts; "
            "no SUB images are used for fitting.",
        ],
    }
    save_json(run / "summary.json", summary)
    print(
        json.dumps(
            {
                "images": single["views"][0]["n"],
                "clean_attribute_accuracy": single["views"][0]["accuracy"],
                "pairs": len(pairs),
                "null_pairs": int((~active).sum()),
                "output": str(run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
