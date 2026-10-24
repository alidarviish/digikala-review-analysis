"""Build the cleaned comments table from the raw Parquet copy.

The review text is copied unchanged. Normalization happens later, in one place.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from digikala_reviews.config import (
    RATE_MAX,
    RATE_MIN,
    RATE_ZERO_POLICY,
    RATING_BAND_EDGES,
    RECOMMENDATION_CLASSES,
)
from digikala_reviews.quality import is_missing

CLEANING_RULES = [
    "P1 a value is missing when it is empty, only whitespace / invisible characters, or the text 'nan'",
    "P2 reviews without a body are excluded from every task",
    "P3 reviews without a recommendation label are kept; they can still be used for rating_band",
    "P4 a rate outside [0, 5] leaves rating_band empty; the review is kept for recommendation",
    "P5 identical copies of an id: keep the first copy",
    "P6 copies of an id that differ only in metadata (likes, seller, ...): keep the first copy",
    "P7 ids whose copies disagree on body, rate or recommendation: exclude every copy",
    "P8 a rate of exactly 0 means no rating was recorded: rating_band is left empty, the review is kept",
]

COLUMNS = ["src_row", "id", "product_id", "body", "rate", "recommendation_status", "is_buyer", "created_at"]


def rating_band(rate: pd.Series, zero_policy: str = RATE_ZERO_POLICY) -> pd.Series:
    """Map a numeric rate to low / medium / high. Invalid rates give an empty value."""
    low_edge, high_edge = RATING_BAND_EDGES
    valid = rate.between(RATE_MIN, RATE_MAX)
    if zero_policy == "invalid":
        valid &= rate != 0
    elif zero_policy != "low":
        raise ValueError(f"Unknown zero policy: {zero_policy}")

    band = np.select([rate < low_edge, rate < high_edge], ["low", "medium"], default="high")
    return pd.Series(band, index=rate.index).where(valid)


def find_conflicting_ids(df: pd.DataFrame) -> set:
    """Ids whose copies disagree on the body, the rating or the label."""
    copies = df[df["id"].duplicated(keep=False)].copy()
    copies["body_stripped"] = copies["body"].str.strip()
    differs = copies.groupby("id")[["body_stripped", "rate", "recommendation_status"]].nunique() > 1
    return set(differs.index[differs.any(axis=1)])


def build_clean_table(raw_path: Path, clean_path: Path, excluded_path: Path) -> dict:
    start = time.perf_counter()
    df = pd.read_parquet(raw_path, columns=COLUMNS).sort_values("src_row", ignore_index=True)
    n_raw = len(df)

    labels = df["recommendation_status"]
    unknown_labels = ~is_missing(labels) & ~labels.isin(RECOMMENDATION_CLASSES)
    if unknown_labels.any():
        raise ValueError(f"{unknown_labels.sum()} labels are neither missing nor a known class")

    # Why each row is excluded; the first matching rule wins
    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    reason[df["id"].isin(find_conflicting_ids(df))] = "duplicate_id_conflict"
    reason[reason.isna() & df["id"].duplicated(keep="first")] = "duplicate_id_extra_copy"
    reason[reason.isna() & is_missing(df["body"])] = "missing_body"

    excluded = df.loc[reason.notna(), ["src_row", "id"]].assign(exclusion_reason=reason[reason.notna()])
    excluded.to_parquet(excluded_path, index=False)

    kept = df[reason.isna()]
    rate = pd.to_numeric(kept["rate"])
    label = kept["recommendation_status"]
    clean = pd.DataFrame(
        {
            "src_row": kept["src_row"],
            "id": kept["id"].astype("int64"),
            "product_id": kept["product_id"].astype("int64"),
            "body": kept["body"],
            "rate": rate,
            "rate_is_zero": rate == 0,
            "rating_band": rating_band(rate),
            "recommendation": label.where(label.isin(RECOMMENDATION_CLASSES)),
            "is_buyer": kept["is_buyer"] == "True",
            "created_at_jalali": kept["created_at"],
        }
    )
    clean.to_parquet(clean_path, index=False)

    excluded_counts = excluded["exclusion_reason"].value_counts().to_dict()
    if n_raw - len(excluded) != len(clean):
        raise RuntimeError("Row counts do not add up after cleaning")
    if clean["id"].duplicated().any():
        raise RuntimeError("Duplicate ids remain after cleaning")

    return {
        "rules": CLEANING_RULES,
        "rate_zero_policy": RATE_ZERO_POLICY,
        "rows_raw": n_raw,
        "rows_excluded": excluded_counts,
        "rows_clean": len(clean),
        "recommendation": clean["recommendation"].value_counts(dropna=False).to_dict(),
        "rating_band": clean["rating_band"].value_counts(dropna=False).to_dict(),
        "rate_is_zero": int(clean["rate_is_zero"].sum()),
        "products": int(clean["product_id"].nunique()),
        "seconds": round(time.perf_counter() - start, 1),
    }
