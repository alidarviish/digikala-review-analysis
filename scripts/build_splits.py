"""Split the cleaned reviews by product into train / validation / test and check the balance.

Writes data/splits/split_assignment.parquet (id, product_id, split) and
reports/split_report.json. The standard CSV files are built from this in the next step.

The same split is also made with two other seeds, only to check that the balance does not
depend on a lucky seed. Those splits are not saved.

Usage:
    uv run python scripts/build_splits.py
"""

import json
import time
from datetime import UTC, datetime

import pandas as pd
import pyarrow.parquet as pq
import sklearn

from digikala_reviews.config import (
    COMMENTS_CLEAN_PARQUET,
    PRODUCTS_RAW_DIR,
    REPORTS_DIR,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
    SPLITS_DIR,
)
from digikala_reviews.dates import parse_jalali
from digikala_reviews.normalize import normalize
from digikala_reviews.split import assign_splits, joint_label, products_in_several_splits, share_table

ASSIGNMENT_PATH = SPLITS_DIR / "split_assignment.parquet"
REPORT_PATH = REPORTS_DIR / "split_report.json"
STABILITY_SEEDS = [2027, 2028]  # only for the balance check, never saved
CHUNK_ROWS = 250_000


def text_facts():
    """Words per review after normalization, read in chunks so the texts never sit in memory twice."""
    words = []
    for batch in pq.ParquetFile(COMMENTS_CLEAN_PARQUET).iter_batches(batch_size=CHUNK_ROWS, columns=["body"]):
        words.append(batch.to_pandas()["body"].map(lambda text: len(normalize(text).split())))
    return pd.concat(words, ignore_index=True)


def product_categories():
    """Most frequent category of each product; ties broken alphabetically (as in notebook 01)."""
    products = pd.read_parquet(PRODUCTS_RAW_DIR, columns=["id", "sub_category"])
    products["id"] = products["id"].astype("int64")
    pairs = products.value_counts(["id", "sub_category"]).reset_index()
    pairs = pairs.sort_values(["id", "count", "sub_category"], ascending=[True, False, True])
    return pairs.drop_duplicates("id").set_index("id")["sub_category"]


def balance(data, split):
    """Share tables for both targets, the year and the product category."""
    return {
        "recommendation": share_table(data["recommendation"], split),
        "rating_band": share_table(data["rating_band"], split),
        "year": share_table(data["year"].astype("Int64").astype(str), split),
        "category": share_table(data["category"], split),
    }


def sizes(data, split):
    return {
        name: {
            "reviews": int((split == name).sum()),
            "reviews_pct": round(100 * (split == name).mean(), 2),
            "products": int(data.loc[split == name, "product_id"].nunique()),
        }
        for name in SPLIT_FRACTIONS
    }


def main():
    start = time.perf_counter()
    columns = ["id", "product_id", "recommendation", "rating_band", "created_at_jalali"]
    data = pd.read_parquet(COMMENTS_CLEAN_PARQUET, columns=columns)
    data["words"] = text_facts().to_numpy()
    data["year"] = parse_jalali(data.pop("created_at_jalali"))["year"]
    data["category"] = data["product_id"].map(product_categories())
    print(f"Read {len(data):,} reviews ({time.perf_counter() - start:.0f}s)")

    # Reviews with no text left after normalization have nothing to learn from
    empty = data["words"] == 0
    empty_ids = data.loc[empty, "id"].tolist()
    data = data.loc[~empty].reset_index(drop=True)
    print(f"Left out {len(empty_ids)} reviews that are empty after normalization")
    if len(empty_ids) != 7:  # 7 in reports/normalization_report.json; a change means the input changed
        print("  WARNING: expected 7, check the cleaned data and the normalizer version")

    strata = joint_label(data["recommendation"], data["rating_band"])
    split = assign_splits(data["product_id"], strata, SPLIT_SEED)
    overlap = products_in_several_splits(data["product_id"], split)
    assert overlap == 0, f"{overlap} products are in more than one part"
    print(f"Split done ({time.perf_counter() - start:.0f}s)")

    tables = balance(data, split)
    tables["label_combination"] = share_table(strata, split)
    words = data.groupby(split)["words"].agg(["mean", "median"]).round(2).reindex(list(SPLIT_FRACTIONS))

    stability = {}
    for seed in STABILITY_SEEDS:
        other = assign_splits(data["product_id"], strata, seed)
        other_tables = balance(data, other)
        stability[str(seed)] = {
            "sizes_pct": {name: round(float(100 * (other == name).mean()), 2) for name in SPLIT_FRACTIONS},
            "max_gap_pp": {key: round(float(t["max_gap_pp"].max()), 3) for key, t in other_tables.items()},
            "same_part_as_main_split_pct": round(100 * (other == split).mean(), 2),
        }
        print(f"Stability check with seed {seed} done ({time.perf_counter() - start:.0f}s)")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    data[["id", "product_id"]].assign(split=split).to_parquet(ASSIGNMENT_PATH, index=False)

    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "StratifiedGroupKFold, 20 folds, groups = product_id, "
        "strata = recommendation x rating_band (missing as its own class); "
        "14 folds train, 3 validation, 3 test, which folds go where is drawn with the seed",
        "seed": SPLIT_SEED,
        "sklearn_version": sklearn.__version__,
        "reviews": len(data),
        "left_out_empty_after_normalization": len(empty_ids),
        "left_out_ids": empty_ids,
        "products_in_several_parts": overlap,
        "sizes": sizes(data, split),
        "max_gap_pp": {key: round(float(t["max_gap_pp"].max()), 3) for key, t in tables.items()},
        "words_per_review": words.to_dict(orient="index"),
        "shares_pct": {key: t.round(2).to_dict(orient="index") for key, t in tables.items()},
        "stability_seeds": stability,
        "seconds": round(time.perf_counter() - start),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, size in report["sizes"].items():
        reviews, pct, products = size["reviews"], size["reviews_pct"], size["products"]
        print(f"{name:>10}: {reviews:>9,} reviews ({pct}%), {products:,} products")
    print(f"Products in more than one part: {overlap}")
    print(f"Largest gap to all data (percentage points): {report['max_gap_pp']}")
    for seed, result in stability.items():
        print(f"Seed {seed}: sizes {result['sizes_pct']}, largest gaps {result['max_gap_pp']}")
    print(f"Saved {ASSIGNMENT_PATH.relative_to(SPLITS_DIR.parent.parent)} and reports/{REPORT_PATH.name}")
    print(f"Done in {report['seconds']}s")


if __name__ == "__main__":
    main()
