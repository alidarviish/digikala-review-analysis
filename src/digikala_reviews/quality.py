"""Schema and quality checks on the raw data.

Nothing here changes the data. Each check counts problems and keeps a few
examples, so every cleaning decision can be traced back to a number.

Columns are loaded one or a few at a time to keep memory low.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from digikala_reviews.config import RATE_MAX, RATE_MIN, RECOMMENDATION_CLASSES
from digikala_reviews.dates import DATE_PATTERN, JALALI_MONTHS

EXPECTED_COMMENT_COLUMNS = [
    "id", "title", "body", "created_at", "rate", "recommendation_status", "is_buyer", "product_id",
    "advantages", "disadvantages", "likes", "dislikes", "seller_title", "seller_code", "true_to_size_rate",
]  # fmt: skip

# Zero-width characters that make a text look non-empty although nothing is visible
INVISIBLE_CHARS = "​‌‍‎‏﻿"
_REMOVE_INVISIBLE = str.maketrans("", "", INVISIBLE_CHARS)


def is_missing(values: pd.Series) -> pd.Series:
    """True where a text value is missing.

    The CSV was exported with pandas, which wrote missing values as the text "nan".
    So a value is missing when it is empty, only whitespace / invisible characters, or "nan".
    """
    cleaned = values.str.translate(_REMOVE_INVISIBLE).str.strip()
    return (cleaned == "") | (cleaned.str.lower() == "nan")


def load(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def check_schema(path: Path) -> dict:
    first_part = sorted(path.glob("part-*.parquet"))[0]
    columns = [c for c in pd.read_parquet(first_part).columns if c != "src_row"]
    return {
        "columns": columns,
        "missing_expected": [c for c in EXPECTED_COMMENT_COLUMNS if c not in columns],
        "unexpected": [c for c in columns if c not in EXPECTED_COMMENT_COLUMNS],
    }


def check_missing(path: Path, columns: list[str]) -> dict:
    """Per column: how many values are blank, how many are the text 'nan', and the total."""
    result = {}
    for col in columns:
        values = load(path, [col])[col]
        missing = is_missing(values)
        nan_text = values.str.strip().str.lower() == "nan"
        result[col] = {
            "blank": int((missing & ~nan_text).sum()),
            "nan_token": int(nan_text.sum()),
            "missing": int(missing.sum()),
            "missing_pct": round(100 * missing.mean(), 3),
        }
    return result


def check_ids(path: Path) -> dict:
    """id and product_id must be whole numbers."""
    df = load(path, ["id", "product_id"])
    return {col: {"not_a_number": int(pd.to_numeric(df[col], errors="coerce").isna().sum())} for col in df}


def check_rate(path: Path) -> dict:
    df = load(path, ["src_row", "id", "product_id", "rate", "recommendation_status", "body"])
    rate = pd.to_numeric(df["rate"], errors="coerce")
    valid = rate.notna() & np.isfinite(rate)
    in_range = valid & rate.between(RATE_MIN, RATE_MAX)
    out_of_range = valid & ~in_range

    examples = df[out_of_range].head(10).copy()
    examples["body"] = examples["body"].str[:120]
    return {
        "not_a_number": int((~valid).sum()),
        "in_range": int(in_range.sum()),
        "out_of_range": int(out_of_range.sum()),
        "zero": int((in_range & (rate == 0)).sum()),
        "fractional": int((in_range & (rate % 1 != 0)).sum()),
        "distinct_values": int(rate[in_range].nunique()),
        "min": float(rate[in_range].min()),
        "max": float(rate[in_range].max()),
        "out_of_range_examples": examples.to_dict("records"),
    }


def value_counts(path: Path, col: str) -> dict:
    return load(path, [col])[col].value_counts().to_dict()


def check_recommendation_labels(path: Path) -> dict:
    labels = load(path, ["recommendation_status"])["recommendation_status"]
    counts = labels.value_counts()
    missing = is_missing(labels)
    unknown = ~missing & ~labels.isin(RECOMMENDATION_CLASSES)
    return {
        "observed": counts.to_dict(),
        "missing": int(missing.sum()),
        "not_canonical": labels[unknown].value_counts().to_dict(),
    }


def check_dates(path: Path) -> dict:
    """Dates are Persian (Jalali) text like '23 شهریور 1402'. Only the format is checked here."""
    dates = load(path, ["created_at"])["created_at"]
    parts = dates.str.extract(DATE_PATTERN)
    day, month, year = pd.to_numeric(parts[0]), parts[1], pd.to_numeric(parts[2])
    matches = parts[0].notna()
    return {
        "not_matching_format": int((~matches).sum()),
        "unknown_month": int((matches & ~month.isin(JALALI_MONTHS)).sum()),
        "invalid_day": int((matches & ~day.between(1, 31)).sum()),
        "examples_not_matching": dates[~matches].head(5).tolist(),
        "rows_per_year": year.value_counts().sort_index().astype(int).to_dict(),
    }


def check_duplicate_ids(path: Path) -> dict:
    """Find ids that appear more than once and describe how their copies differ."""
    ids = load(path, ["id"])["id"]
    duplicated_ids = ids[ids.duplicated()].unique().tolist()
    if not duplicated_ids:
        return {"duplicated_ids": 0}

    # Only the few thousand duplicated rows are loaded with all their columns
    rows = pd.read_parquet(path, filters=[("id", "in", duplicated_ids)])
    rows["body_stripped"] = rows["body"].str.strip()
    content = ["body_stripped", "rate", "recommendation_status"]
    other = [c for c in rows.columns if c not in ("src_row", "id", "body_stripped")]

    differs = rows.groupby("id").nunique() > 1  # True where the copies of an id disagree on a column
    identical = ~differs[other].any(axis=1)
    content_differs = differs[content].any(axis=1)

    conflict_ids = differs.index[content_differs][:4]
    examples = rows[rows["id"].isin(conflict_ids)].sort_values(["id", "src_row"])
    examples = examples[["src_row", "id", "product_id", "rate", "recommendation_status", "likes", "body"]]
    examples["body"] = examples["body"].str[:80]

    return {
        "duplicated_ids": len(duplicated_ids),
        "rows_involved": int(ids.isin(duplicated_ids).sum()),
        "extra_rows": int(ids.duplicated().sum()),
        "max_rows_per_id": int(ids.value_counts().max()),
        "groups_fully_identical": int(identical.sum()),
        "groups_metadata_only": int((~identical & ~content_differs).sum()),
        "groups_content_differs": int(content_differs.sum()),
        "groups_differing_by_column": differs[other].sum().loc[lambda s: s > 0].astype(int).to_dict(),
        "content_conflict_examples": examples.to_dict("records"),
    }


def check_products(comments_path: Path, products_path: Path) -> dict:
    products = load(products_path, ["id", "Category1", "Category2", "sub_category"])
    per_id = products.groupby("id").agg(
        rows=("id", "size"),
        categories=("Category1", "nunique"),
        sub_categories=("sub_category", "nunique"),
    )
    repeated = per_id[per_id["rows"] > 1]
    conflicting = (repeated["categories"] > 1) | (repeated["sub_categories"] > 1)

    reviewed = set(load(comments_path, ["product_id"])["product_id"].unique())
    known = set(per_id.index)
    return {
        "rows": len(products),
        "distinct_ids": len(per_id),
        "ids_with_several_rows": len(repeated),
        "max_rows_per_id": int(per_id["rows"].max()),
        "ids_with_conflicting_category": int(conflicting.sum()),
        "reviewed_products": len(reviewed),
        "reviewed_products_missing_from_file": len(reviewed - known),
        "missing_values": check_missing(products_path, list(products.columns)),
    }


def run_quality_gate(comments_path: Path, products_path: Path) -> dict:
    schema = check_schema(comments_path)
    if schema["missing_expected"]:
        raise ValueError(f"Expected columns are missing: {schema['missing_expected']}")

    return {
        "missing_value_rule": "empty, only whitespace / invisible characters, or the text 'nan'",
        "comments": {
            "rows": len(load(comments_path, ["src_row"])),
            "schema": schema,
            "missing_values": check_missing(comments_path, schema["columns"]),
            "ids": check_ids(comments_path),
            "rate": check_rate(comments_path),
            "recommendation_status": check_recommendation_labels(comments_path),
            "is_buyer": value_counts(comments_path, "is_buyer"),
            "true_to_size_rate": value_counts(comments_path, "true_to_size_rate"),
            "created_at": check_dates(comments_path),
            "duplicate_ids": check_duplicate_ids(comments_path),
        },
        "products": check_products(comments_path, products_path),
    }
