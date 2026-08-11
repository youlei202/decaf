"""FunnyBirds common-support and held-out quality contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from decaf.experiments.attribution.models import FUNNYBIRDS_MODELS

SUPPORT_COUNTS = {
    "funnybirds_resnet50": 499,
    "funnybirds_vgg16": 497,
    "funnybirds_vit_b_16": 488,
}
HELDOUT_OPERATORS = ("background_texture", "telea_dilate3")
SUPPORT_COLUMNS = (
    "dataset",
    "model",
    "image_id",
    "correctly_classified",
    "included",
    "exclusion_reason",
)
QUALITY_COLUMNS = ("dataset", "model", "method", "image_id", "spearman")


def _read(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"unsupported FunnyBirds table type: {source.suffix}")


def validate_support(frame: pd.DataFrame, *, formal: bool = False) -> pd.DataFrame:
    """Validate model/image support, optionally enforcing sealed formal counts."""

    missing = sorted(set(SUPPORT_COLUMNS) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"FunnyBirds support is invalid; missing={missing}")
    result = frame.copy()
    result["dataset"] = result["dataset"].astype(str)
    result["model"] = result["model"].astype(str)
    result["image_id"] = result["image_id"].astype(str)
    if set(result["dataset"]) != {"funnybirds"}:
        raise ValueError("FunnyBirds support contains another dataset")
    if not set(result["model"]).issubset(FUNNYBIRDS_MODELS):
        raise ValueError("FunnyBirds support contains an unregistered model")
    if result.duplicated(["model", "image_id"]).any():
        raise ValueError("FunnyBirds support contains duplicate model/image rows")
    if formal:
        included = result.loc[result["included"].astype(bool)]
        counts = included.groupby("model")["image_id"].nunique().to_dict()
        if counts != SUPPORT_COUNTS:
            raise ValueError(f"FunnyBirds support counts drifted: {counts}")
    return result


def load_support(path: str | Path, *, formal: bool = False) -> pd.DataFrame:
    """Load a support table without importing the model stack."""

    return validate_support(_read(path), formal=formal)


def validate_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate one finite Spearman value per model/method/image."""

    missing = sorted(set(QUALITY_COLUMNS) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"FunnyBirds quality is invalid; missing={missing}")
    result = frame.copy()
    if result.duplicated(["dataset", "model", "method", "image_id"]).any():
        raise ValueError("FunnyBirds quality contains duplicate rows")
    values = pd.to_numeric(result["spearman"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("FunnyBirds quality contains non-finite values")
    return result


__all__ = [
    "HELDOUT_OPERATORS",
    "QUALITY_COLUMNS",
    "SUPPORT_COLUMNS",
    "SUPPORT_COUNTS",
    "load_support",
    "validate_quality",
    "validate_support",
]
