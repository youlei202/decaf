"""Label-grounded SUB pairing and numerical analysis; no model dependencies."""

from __future__ import annotations

import hashlib
import itertools

import numpy as np


def normalized_attribute(name: str) -> str:
    return name.replace("--", "::")


def attribute_group(name: str) -> str:
    return normalized_attribute(name).split("::", 1)[0]


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_rows(rows: list[dict], per_cell: int, seed: int) -> list[dict]:
    """Select without consulting images, model outputs, or outcome labels."""
    cells: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        cells.setdefault((row["bird_label"], row["attr_label"]), []).append(row)
    selected = []
    for key in sorted(cells):
        group = sorted(cells[key], key=lambda r: stable_key(seed, str(r["row_id"])))
        selected.extend(group if per_cell == 0 else group[:per_cell])
    return selected


def matched_pairs(rows: list[dict], attribute_names: list[str], seed: int) -> list[dict]:
    """Cross-instance pairs, explicitly NOT same-source causal interventions.

    Every unordered pair of observed attribute values in a bird/group is used.
    Within each cell, samples are shuffled by a predetermined hash and zipped.
    A source can therefore be reused across contrasts; uncertainty clusters by bird.
    """
    cells: dict[tuple[int, str], dict[int, list[dict]]] = {}
    for row in rows:
        group = attribute_group(attribute_names[row["attr_label"]])
        cells.setdefault((row["bird_label"], group), {}).setdefault(row["attr_label"], []).append(
            row
        )
    pairs = []
    for (bird, group), attrs in sorted(cells.items()):
        for a, b in itertools.combinations(sorted(attrs), 2):
            aa = sorted(attrs[a], key=lambda r: stable_key(seed, str(r["row_id"])))
            bb = sorted(attrs[b], key=lambda r: stable_key(seed, str(r["row_id"])))
            for x, y in zip(aa, bb, strict=False):
                if x["row_id"] == y["row_id"]:
                    raise ValueError("An endpoint cannot be paired with itself")
                pairs.append(
                    {
                        "pair_id": f"{x['row_id']}-{y['row_id']}",
                        "plus_row": x["row_id"],
                        "minus_row": y["row_id"],
                        "bird_label": bird,
                        "attribute_group": group,
                        "plus_attribute": a,
                        "minus_attribute": b,
                        "pair_type": "same_class_group_cross_instance",
                    }
                )
    return pairs


def decompose(responses: np.ndarray, stages: np.ndarray, epsilon: float) -> dict:
    r = np.asarray(responses, dtype=np.float64)
    t = np.asarray(stages, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != len(t) or len(t) < 2:
        raise ValueError("Expected [pair, stage] responses and at least two stages")
    if not np.all(np.isfinite(r)) or not np.all(np.diff(t) > 0):
        raise ValueError("Non-finite responses or non-increasing stages")
    if t[0] != 0 or t[-1] != 1 or epsilon <= 0:
        raise ValueError("Require stages spanning [0,1] and epsilon > 0")
    if not np.allclose(r[:, 0], 0, atol=2e-6):
        raise ValueError("Reveal must start from identical branch inputs")
    d = r[:, -1]
    active = np.abs(d) >= epsilon
    oriented = r * np.sign(d[:, None])
    integral = np.trapezoid
    absolute = integral(np.abs(r), t, axis=1)
    evidence = integral(np.maximum(oriented, 0), t, axis=1) * active
    contradiction = integral(np.maximum(-oriented, 0), t, axis=1) * active
    fragility = absolute * ~active
    if not np.allclose(evidence + contradiction + fragility, absolute, atol=1e-10):
        raise AssertionError("Conservation failed")
    return {
        "E": evidence,
        "C": contradiction,
        "F": fragility,
        "Abs": absolute,
        "M": np.abs(d),
        "signed_endpoint": d,
        "signed_net": integral(r, t, axis=1),
        "active": active,
    }


def behavior_targets(
    clean_scores: np.ndarray,
    view_scores: np.ndarray,
    clean_predictions: np.ndarray,
    view_predictions: np.ndarray,
    plus_attributes: np.ndarray,
    minus_attributes: np.ndarray,
    epsilon: float,
    change_tolerance: float,
) -> dict:
    """Targets use held-out endpoint views, never reveal-trajectory responses.

    Scores: [pair, branch] and [pair, view, branch]. Predictions are global
    attribute-query indices, so intended attribute recognition is externally labeled.
    """
    d = clean_scores[:, 0] - clean_scores[:, 1]
    dv = view_scores[:, :, 0] - view_scores[:, :, 1]
    expected = np.stack([plus_attributes, minus_attributes], axis=1)
    recognition = np.all(view_predictions == expected[:, None, :], axis=2).mean(axis=1)
    flips = np.any(view_predictions != clean_predictions[:, None, :], axis=(1, 2))
    shift = np.max(np.abs(view_scores - clean_scores[:, None, :]), axis=(1, 2))
    reversal = np.any(dv * np.sign(d[:, None]) <= -epsilon, axis=1)
    return {
        "recognition_fraction": recognition,
        "recognition_success": recognition >= 0.75,
        "offpath_reversal": reversal,
        "offpath_instability": flips | (shift >= change_tolerance),
        "max_offpath_probability_shift": shift,
        "active": np.abs(d) >= epsilon,
    }
