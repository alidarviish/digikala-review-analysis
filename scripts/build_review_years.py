"""Write the year of every train and validation review, for the time split of E10.

The cache has no dates, so this adds data/cache/review_years.parquet (columns id and year,
the Jalali year) and review_years_info.json with its hash. Only reviews in the train and
validation cache are kept; the test set is not part of this experiment. Run locally after
scripts/cache_normalized.py, then copy both files next to the cache on the other machine.

Usage:
    uv run python scripts/build_review_years.py
"""

import json

import pandas as pd

from digikala_reviews.config import CACHE_DIR, COMMENTS_CLEAN_PARQUET
from digikala_reviews.dates import parse_jalali
from digikala_reviews.ingest import file_sha256
from digikala_reviews.provenance import check_cache

CACHE_FILES = ["train_supervised.parquet", "validation.parquet"]
OUT = CACHE_DIR / "review_years.parquet"


def main():
    data = check_cache(CACHE_DIR, CACHE_FILES)
    ids = pd.concat([pd.read_parquet(CACHE_DIR / name, columns=["id"])["id"] for name in CACHE_FILES])
    dates = pd.read_parquet(COMMENTS_CLEAN_PARQUET, columns=["id", "created_at_jalali"])
    dates = dates[dates["id"].isin(ids)]
    years = pd.DataFrame({"id": dates["id"], "year": parse_jalali(dates["created_at_jalali"])["year"]})
    assert len(years) == len(ids) == ids.nunique(), "a cached review is missing or repeated"
    assert years["year"].notna().all(), "a review without a readable date"
    years["year"] = years["year"].astype("int16")
    years.to_parquet(OUT, index=False)

    info = {
        "rows": len(years),
        "sha256": file_sha256(OUT),
        "cache_sha256": data["sha256"],
        "reviews_per_year": {str(k): int(v) for k, v in years["year"].value_counts().sort_index().items()},
    }
    (CACHE_DIR / "review_years_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"{len(years):,} reviews written to {OUT.name}")
    for year, count in info["reviews_per_year"].items():
        print(f"  {year}: {count:,}")


if __name__ == "__main__":
    main()
