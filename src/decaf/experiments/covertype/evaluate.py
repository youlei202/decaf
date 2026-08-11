"""Static planning, training, legal-query evaluation, and resumable compute."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decaf.core import trajectory_scores
from decaf.core.receipts import (
    finalize_global_receipt,
    load_member_receipt,
    utc_now,
    write_member_receipt,
)
from decaf.experiments.common import RunContext, atomic_json, atomic_text
from decaf.experiments.covertype.baselines import (
    paired_trajectory_baselines,
    permutation_factor_importance,
)
from decaf.experiments.covertype.data import (
    CovertypeDataset,
    load_dataset,
    prepare_dataset,
)
from decaf.experiments.covertype.mechanisms import (
    C_MECHANISMS,
    F_REGIMES,
    MechanismRealization,
    augmented_features,
    legal_query_features,
    realize_module_c,
    realize_module_f,
)
from decaf.experiments.covertype.models import (
    MODEL_FAMILIES,
    build_model,
    fit_model,
    implementation_name,
    predict_positive,
)

FORMAL_MODEL_FAMILIES = MODEL_FAMILIES
FORMAL_SEEDS = (7701, 7702, 7703)
FORMAL_C_STRENGTHS = (0.75, 0.95)
FORMAL_C_MECHANISMS = C_MECHANISMS
FORMAL_F_REGIMES = F_REGIMES
FORMAL_STAGES = tuple(value / 10 for value in range(11))


@dataclass(frozen=True)
class ModelSpec:
    """Identity of one independently trained contextual model."""

    module: str
    regime: str
    model_family: str
    seed: int
    strength: float | None = None

    @property
    def model_id(self) -> str:
        if self.module == "C":
            strength = f"{float(self.strength):.2f}".replace(".", "p")
            return f"c_{self.regime}_p{strength}_{self.model_family}_s{self.seed}"
        return f"f_{self.regime}_{self.model_family}_s{self.seed}"

    def record(self) -> dict[str, Any]:
        return {"model_id": self.model_id, **asdict(self)}


def build_specs(
    *,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
    c_strengths: tuple[float, ...],
    c_mechanisms: tuple[str, ...],
    f_regimes: tuple[str, ...],
) -> tuple[ModelSpec, ...]:
    """Build the registered Cartesian product in deterministic order."""

    unknown = set(families) - set(MODEL_FAMILIES)
    if unknown:
        raise ValueError(f"unknown model families: {sorted(unknown)}")
    specs: list[ModelSpec] = []
    for mechanism in c_mechanisms:
        for strength in c_strengths:
            for family in families:
                for seed in seeds:
                    specs.append(ModelSpec("C", mechanism, family, seed, strength))
    for regime in f_regimes:
        for family in families:
            for seed in seeds:
                specs.append(ModelSpec("F", regime, family, seed))
    identifiers = [spec.model_id for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Covertype model identifiers must be unique")
    return tuple(specs)


def formal_specs() -> tuple[ModelSpec, ...]:
    """Return all 135 preregistered paper models."""

    specs = build_specs(
        families=FORMAL_MODEL_FAMILIES,
        seeds=FORMAL_SEEDS,
        c_strengths=FORMAL_C_STRENGTHS,
        c_mechanisms=FORMAL_C_MECHANISMS,
        f_regimes=FORMAL_F_REGIMES,
    )
    c_count = sum(spec.module == "C" for spec in specs)
    f_count = sum(spec.module == "F" for spec in specs)
    if (c_count, f_count, len(specs)) != (90, 45, 135):
        raise AssertionError("formal Covertype plan must contain C=90, F=45, total=135")
    return specs


def build_formal_plan() -> dict[str, Any]:
    """Return a machine-auditable, data-free static paper plan."""

    specs = formal_specs()
    tree_models = [spec for spec in specs if spec.model_family in {"random_forest", "xgboost"}]
    optional_seed_models = [spec for spec in specs if spec.seed == FORMAL_SEEDS[0]]
    if len(tree_models) != 54 or len(optional_seed_models) != 45:
        raise AssertionError("formal Covertype baseline plan has drifted")
    return {
        "schema_version": 1,
        "experiment": "covertype",
        "profile": "paper",
        "counts": {
            "module_c_models": 90,
            "module_f_models": 45,
            "total_models": 135,
            "model_families": 5,
            "seeds": 3,
        },
        "module_c": {
            "strengths": list(FORMAL_C_STRENGTHS),
            "mechanisms": list(FORMAL_C_MECHANISMS),
            "formula": "2 strengths x 3 mechanisms x 5 families x 3 seeds",
        },
        "module_f": {
            "regimes": list(FORMAL_F_REGIMES),
            "formula": "3 regimes x 5 families x 3 seeds",
        },
        "model_families": list(FORMAL_MODEL_FAMILIES),
        "seeds": list(FORMAL_SEEDS),
        "stages": list(FORMAL_STAGES),
        "baseline_plan": {
            "paired_trajectory_models": 135,
            "permutation_importance_models": 135,
            "native_shap_models": 135,
            "tree_shap_interaction_models": 54,
            "tree_shap_interaction_shards_per_model": 4,
            "tree_shap_interaction_shard_jobs": 216,
            "optional_kernel_shap_models": 45,
            "optional_lime_models": 45,
            "retraining_reference_models": 135,
            "tree_model_ids": [spec.model_id for spec in tree_models],
        },
        "dependency_policy": {
            "xgboost": "optional dependency required by paper compute",
            "smoke": "scikit-learn only",
        },
        "jobs": [spec.record() for spec in specs],
        "audit": {
            "unique_model_ids": len({spec.model_id for spec in specs}) == 135,
            "module_c_count": sum(spec.module == "C" for spec in specs),
            "module_f_count": sum(spec.module == "F" for spec in specs),
        },
    }


def configured_specs(config: dict[str, Any]) -> tuple[ModelSpec, ...]:
    """Resolve the active profile's model subset."""

    protocol = config["protocol"]
    return build_specs(
        families=tuple(str(value) for value in protocol["model_families"]),
        seeds=tuple(int(value) for value in protocol["seeds"]),
        c_strengths=tuple(float(value) for value in protocol["module_c"]["strengths"]),
        c_mechanisms=tuple(str(value) for value in protocol["module_c"]["mechanisms"]),
        f_regimes=tuple(str(value) for value in protocol["module_f"]["regimes"]),
    )


def prepare(context: RunContext) -> dict[str, Any]:
    """Prepare data plus current and formal job manifests."""

    manifest = prepare_dataset(context.path, context.config)
    current = configured_specs(context.config)
    atomic_json(context.path / "manifests" / "formal_plan.json", build_formal_plan())
    atomic_text(
        context.path / "manifests" / "jobs.jsonl",
        "".join(json.dumps(spec.record(), sort_keys=True) + "\n" for spec in current),
    )
    header = "model_id,module,regime,strength,model_family,seed\n"
    rows = [
        ",".join(
            (
                spec.model_id,
                spec.module,
                spec.regime,
                "" if spec.strength is None else str(spec.strength),
                spec.model_family,
                str(spec.seed),
            )
        )
        for spec in current
    ]
    atomic_text(context.path / "manifests" / "model_plan.csv", header + "\n".join(rows) + "\n")
    return {
        "source_kind": manifest["source_kind"],
        "configured_models": len(current),
        "formal_models": 135,
    }


def _realize(spec: ModelSpec, y: np.ndarray, split: str) -> MechanismRealization:
    if spec.module == "C":
        assert spec.strength is not None
        return realize_module_c(
            y,
            strength=spec.strength,
            mechanism=spec.regime,
            seed=spec.seed,
            split=split,
        )
    return realize_module_f(y, regime=spec.regime, seed=spec.seed, split=split)


def _query_effects(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    endpoint_plus = predict_positive(model, legal_query_features(X, context=1, factor=1))
    endpoint_minus = predict_positive(model, legal_query_features(X, context=1, factor=-1))
    alternate_plus = predict_positive(model, legal_query_features(X, context=-1, factor=1))
    alternate_minus = predict_positive(model, legal_query_features(X, context=-1, factor=-1))
    return (
        endpoint_plus - endpoint_minus,
        alternate_plus - alternate_minus,
        4 * len(X),
    )


def _direction_behavior(
    endpoint: np.ndarray,
    alternate: np.ndarray,
    *,
    delta: float,
) -> dict[str, float]:
    signed = np.sign(endpoint) * alternate
    preserve = signed > delta
    collapse = np.abs(alternate) <= delta
    invert = signed < -delta
    other = ~(preserve | collapse | invert)
    return {
        "preserve_rate": float(np.mean(preserve)),
        "collapse_rate": float(np.mean(collapse)),
        "invert_rate": float(np.mean(invert)),
        "other_rate": float(np.mean(other)),
        "mean_opposed_margin": float(np.mean(np.maximum(-signed, 0.0))),
        "Y_C": float(np.mean(invert)),
    }


def _fragility_behavior(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    realization: MechanismRealization,
    endpoint: np.ndarray,
    *,
    epsilon: float,
) -> tuple[dict[str, float | None], int]:
    alternate_plus = predict_positive(model, legal_query_features(X, context=-1, factor=1))
    alternate_minus = predict_positive(model, legal_query_features(X, context=-1, factor=-1))
    actual = realization.context < 0
    if not np.any(actual):
        raise AssertionError("Module F evaluation requires actual negative contexts")
    plus = alternate_plus[actual]
    minus = alternate_minus[actual]
    target = y[actual]
    factor = realization.factor[actual]
    changed = (plus >= 0.5) != (minus >= 0.5)
    excursion = np.abs(plus - minus)
    factual = np.where(factor > 0, plus, minus)
    reversed_score = np.where(factor > 0, minus, plus)
    endpoint_null = np.abs(endpoint[actual]) < epsilon
    null_change = float(np.mean(changed[endpoint_null])) if np.any(endpoint_null) else None
    null_excursion = float(np.mean(excursion[endpoint_null])) if np.any(endpoint_null) else None
    return (
        {
            "pairwise_prediction_change_rate": float(np.mean(changed)),
            "mean_probability_excursion": float(np.mean(excursion)),
            "factual_accuracy_hminus": float(np.mean((factual >= 0.5) == target)),
            "reversed_accuracy_hminus": float(np.mean((reversed_score >= 0.5) == target)),
            "accuracy_drop_after_u_reversal": float(
                np.mean((factual >= 0.5) == target) - np.mean((reversed_score >= 0.5) == target)
            ),
            "null_context_prediction_change_rate": null_change,
            "null_context_probability_excursion": null_excursion,
            "endpoint_null_rate": float(np.mean(endpoint_null)),
            "Y_F": float(np.mean(changed)),
        },
        2 * len(X),
    )


def evaluate_member(
    spec: ModelSpec,
    dataset: CovertypeDataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train and evaluate one genuine classifier using legal context queries."""

    started = time.monotonic()
    train_realization = _realize(spec, dataset.train.y, "train")
    train_X = augmented_features(dataset.train.X, train_realization)
    model = fit_model(
        build_model(spec.model_family, spec.seed, config["models"]),
        train_X,
        dataset.train.y,
    )
    limit = min(int(config["evaluation"]["max_rows"]), len(dataset.test.y))
    X = dataset.test.X[:limit]
    y = dataset.test.y[:limit]
    test_realization = _realize(spec, y, "test")
    factual_X = augmented_features(X, test_realization)
    endpoint, alternate, prediction_rows = _query_effects(model, X)
    grid = np.asarray(config["decaf"]["stages"], dtype=np.float64)
    response = (1.0 - grid[None, :]) * alternate[:, None] + grid[None, :] * endpoint[:, None]
    epsilon = float(config["decaf"]["epsilon"])
    scores = trajectory_scores(grid, response, endpoint=endpoint, epsilon=epsilon)
    audit = scores["numeric_audit"]
    if not audit["passed"]:
        raise AssertionError(f"DECAF numeric audit failed for {spec.model_id}")
    component_means = {
        name: float(np.mean(np.asarray(scores[name], dtype=np.float64)))
        for name in ("M", "E", "C", "F", "Abs", "Net")
    }
    baselines = paired_trajectory_baselines(endpoint, alternate, scores)
    baselines.update(permutation_factor_importance(model, factual_X, y, seed=spec.seed))
    if spec.module == "C":
        behavior: dict[str, Any] = _direction_behavior(
            endpoint,
            alternate,
            delta=float(config["behavior"]["direction_delta"]),
        )
        suffix = "Z"
        behavior_prediction_rows = 0
    else:
        behavior, behavior_prediction_rows = _fragility_behavior(
            model,
            X,
            y,
            test_realization,
            endpoint,
            epsilon=epsilon,
        )
        suffix = "U"
    prediction_rows += behavior_prediction_rows + int(baselines["baseline_prediction_rows"])
    return {
        **spec.record(),
        "dataset_fingerprint": dataset.fingerprint,
        "dataset_source_kind": dataset.source_kind,
        "model_implementation": implementation_name(model),
        "evaluation_rows": limit,
        "epsilon": epsilon,
        **component_means,
        **{f"{name}_{suffix}": value for name, value in component_means.items()},
        "endpoint_active_rate": float(np.mean(scores["endpoint_active"])),
        "endpoint_null_rate_decaf": float(np.mean(~scores["endpoint_active"])),
        **behavior,
        **baselines,
        "decaf_identity_passed": True,
        "decaf_max_abs_error": float(audit["integrated"]["absolute_residual"]),
        "prediction_rows": prediction_rows,
        "wall_seconds": time.monotonic() - started,
    }


def _member_paths(run_path: Path, model_id: str) -> tuple[Path, Path]:
    return (
        run_path / "raw" / "members" / f"{model_id}.json",
        run_path / "receipts" / "members" / f"{model_id}.json",
    )


def _run_member(
    context: RunContext,
    dataset: CovertypeDataset,
    spec: ModelSpec,
) -> tuple[dict[str, Any], bool]:
    artifact_path, receipt_path = _member_paths(context.path, spec.model_id)
    if context.resume and artifact_path.is_file() and receipt_path.is_file():
        receipt = load_member_receipt(receipt_path)
        if receipt["status"] == "completed":
            return json.loads(artifact_path.read_text(encoding="utf-8")), True
    started_at = utc_now()
    write_member_receipt(
        receipt_path,
        spec.model_id,
        "running",
        started_at=started_at,
        details={"module": spec.module, "model_family": spec.model_family},
    )
    try:
        record = evaluate_member(spec, dataset, context.config)
        atomic_json(artifact_path, record)
        write_member_receipt(
            receipt_path,
            spec.model_id,
            "completed",
            started_at=started_at,
            details={
                "artifact": f"raw/members/{spec.model_id}.json",
                "decaf_identity_passed": record["decaf_identity_passed"],
            },
        )
        return record, False
    except Exception as error:
        write_member_receipt(
            receipt_path,
            spec.model_id,
            "failed",
            started_at=started_at,
            details={"module": spec.module, "model_family": spec.model_family},
            error=f"{type(error).__name__}: {error}",
        )
        raise


def compute(context: RunContext) -> dict[str, Any]:
    """Train configured members with atomic receipts and safe resume semantics."""

    if not (context.path / "raw" / "covertype_data.npz").is_file():
        prepare(context)
    dataset = load_dataset(context.path)
    specs = configured_specs(context.config)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    resumed = 0
    workers = min(context.workers, len(specs))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="covertype") as executor:
        futures = {executor.submit(_run_member, context, dataset, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                record, was_resumed = future.result()
                records.append(record)
                resumed += int(was_resumed)
            except Exception as error:
                failures.append(f"{spec.model_id}: {type(error).__name__}: {error}")
    receipts = {
        spec.model_id: load_member_receipt(_member_paths(context.path, spec.model_id)[1])
        for spec in specs
    }
    finalize_global_receipt(
        context.path / "receipts" / "compute_members.json",
        context.path.name,
        receipts,
        expected_members=tuple(spec.model_id for spec in specs),
        details={
            "configured_members": len(specs),
            "completed_members": len(records),
            "resumed_members": resumed,
            "failure_count": len(failures),
        },
    )
    if failures:
        raise RuntimeError("Covertype member failures: " + "; ".join(failures))
    atomic_json(
        context.path / "raw" / "compute_index.json",
        {
            "schema_version": 1,
            "members": sorted(record["model_id"] for record in records),
            "module_c_members": sum(record["module"] == "C" for record in records),
            "module_f_members": sum(record["module"] == "F" for record in records),
        },
    )
    return {
        "completed_members": len(records),
        "module_c_members": sum(record["module"] == "C" for record in records),
        "module_f_members": sum(record["module"] == "F" for record in records),
        "resumed_members": resumed,
        "workers": workers,
    }


__all__ = [
    "FORMAL_C_MECHANISMS",
    "FORMAL_C_STRENGTHS",
    "FORMAL_F_REGIMES",
    "FORMAL_MODEL_FAMILIES",
    "FORMAL_SEEDS",
    "FORMAL_STAGES",
    "ModelSpec",
    "build_formal_plan",
    "build_specs",
    "compute",
    "configured_specs",
    "evaluate_member",
    "formal_specs",
    "prepare",
]
