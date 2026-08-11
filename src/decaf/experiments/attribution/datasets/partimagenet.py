"""PartImageNet boundary and DINOv2-g common-support contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from decaf.experiments.attribution.models import BOUNDARY_MODELS, LARGE_MODEL

BOUNDARY_CANDIDATES_PER_MODEL = 1_024
BOUNDARY_INCLUDED_TOTAL = 3_586
LARGE_MODEL_INCLUDED = 238
SUPPORT_COLUMNS = (
    "analysis_scope",
    "support_set",
    "dataset",
    "subset",
    "model",
    "image_id",
    "number_of_parts",
    "included",
    "exclusion_reason",
)
PART_COLUMNS = (
    "dataset",
    "image_id",
    "model",
    "method",
    "part_group",
    "attribution_score",
)


def _read(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"unsupported PartImageNet table type: {source.suffix}")


def validate_support(frame: pd.DataFrame, *, formal: bool = False) -> pd.DataFrame:
    """Validate frozen PartImageNet support without opening restricted images."""

    missing = sorted(set(SUPPORT_COLUMNS) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"PartImageNet support is invalid; missing={missing}")
    result = frame.copy()
    result["model"] = result["model"].astype(str)
    result["image_id"] = result["image_id"].astype(str)
    allowed = {*BOUNDARY_MODELS, LARGE_MODEL}
    if not set(result["model"]).issubset(allowed):
        raise ValueError("PartImageNet support contains an unregistered model")
    if result.duplicated(["analysis_scope", "model", "image_id"]).any():
        raise ValueError("PartImageNet support contains duplicate scope/model/image rows")
    part_counts = pd.to_numeric(result["number_of_parts"], errors="coerce")
    if part_counts.isna().any() or not bool((part_counts > 0).all()):
        raise ValueError("PartImageNet rows require a positive part count")
    if formal:
        boundary = result.loc[result["model"].isin(BOUNDARY_MODELS)]
        candidates = boundary.groupby("model")["image_id"].nunique().to_dict()
        expected = {model: BOUNDARY_CANDIDATES_PER_MODEL for model in BOUNDARY_MODELS}
        if candidates != expected:
            raise ValueError(f"PartImageNet candidate counts drifted: {candidates}")
        included = int(boundary.loc[boundary["included"].astype(bool)].shape[0])
        if included != BOUNDARY_INCLUDED_TOTAL:
            raise ValueError(f"PartImageNet included count drifted: {included}")
    return result


def load_support(path: str | Path, *, formal: bool = False) -> pd.DataFrame:
    """Load and validate a frozen common-support manifest."""

    return validate_support(_read(path), formal=formal)


def validate_part_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate long-form semantic-part attribution rows."""

    missing = sorted(set(PART_COLUMNS) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"PartImageNet attribution is invalid; missing={missing}")
    result = frame.copy()
    keys = ["dataset", "model", "method", "image_id", "part_group"]
    if result.duplicated(keys).any():
        raise ValueError("PartImageNet attribution contains duplicate part rows")
    scores = pd.to_numeric(result["attribution_score"], errors="coerce")
    if scores.isna().any():
        raise ValueError("PartImageNet attribution contains non-numeric scores")
    return result


__all__ = [
    "BOUNDARY_CANDIDATES_PER_MODEL",
    "BOUNDARY_INCLUDED_TOTAL",
    "LARGE_MODEL_INCLUDED",
    "PART_COLUMNS",
    "SUPPORT_COLUMNS",
    "load_support",
    "validate_part_attribution",
    "validate_support",
]
