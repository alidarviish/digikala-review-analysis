"""What the reviews removed by cleaning rule P7 look like (report only).

P7 removed every copy of an id whose copies disagree on the body, the rate or the
recommendation label: 245 ids, 490 rows (reports/cleaning_report.json). This sorts those ids
by what actually differs, to judge whether a future rebuild could keep one copy of some of
them. Each id falls in the first of these that fits:

  label        the copies disagree on the recommendation label (one may have none)
  text edit    the bodies differ after normalization (src/digikala_reviews/normalize.py)
  rate 0       only the rate differs, and one copy has 0 (no rating recorded, rule P8)
  two rates    only the rate differs, between two real rates
  spacing      the bodies differ only before normalization (spaces, invisible characters);
               rate and label agree

Also counted: every combination of differing fields, whether the copies share a product and a
creation time, which copy is newer when they do not, and which split each id's product is in.
Only ids and counts are written, no text. Nothing in the data changes: putting rows back would
mean training every model again for at most 0.008% of the reviews; the report is for a future
rebuild. These rows were removed before the split, so they are in no part of the data; the
split of their product is only counted. Creation times are compared as stored (text).

Writes reports/p7_conflicts.json and prints a summary.

Usage:
    uv run python scripts/analyze_p7_conflicts.py
"""

from datetime import UTC, datetime

import pandas as pd

from digikala_reviews.config import (
    COMMENTS_RAW_DIR,
    EXCLUDED_ROWS_PARQUET,
    REPORTS_DIR,
    SPLITS_DIR,
)
from digikala_reviews.normalize import NORMALIZER_VERSION, normalize
from digikala_reviews.provenance import git_commit
from digikala_reviews.quality import is_missing
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "p7_conflicts.json"
COLUMNS = ["src_row", "id", "product_id", "body", "rate", "recommendation_status", "created_at"]
ORDER = ["label", "text edit", "rate 0", "two rates", "spacing"]
EXAMPLES = 5


def product_key(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("Int64").astype(str)


def category(diff: dict, rates: list[float]) -> str:
    if diff["label"]:
        return "label"
    if diff["normalized body"]:
        return "text edit"
    if diff["rate"]:
        return "rate 0" if min(rates) == 0 else "two rates"
    assert diff["body"], "an id in P7 must differ on something"
    return "spacing"


def main():
    excluded = pd.read_parquet(EXCLUDED_ROWS_PARQUET)
    conflict = excluded[excluded["exclusion_reason"] == "duplicate_id_conflict"]
    ids = set(conflict["id"])
    parts = []
    for path in sorted(COMMENTS_RAW_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=COLUMNS)
        parts.append(part[part["id"].isin(ids)])
    rows = pd.concat(parts, ignore_index=True).sort_values("src_row", ignore_index=True)
    assert set(rows["src_row"]) == set(conflict["src_row"]), "the raw rows do not match excluded_rows"
    assignment = pd.read_parquet(SPLITS_DIR / "split_assignment.parquet", columns=["product_id", "split"])
    # the raw copy keeps every column as text, the split assignment has numbers: compare as numbers
    assignment["product_id"] = product_key(assignment["product_id"])
    product_split = assignment.drop_duplicates("product_id").set_index("product_id")["split"]

    rows["label"] = rows["recommendation_status"].where(~is_missing(rows["recommendation_status"]), "none")
    rows["rate_value"] = pd.to_numeric(rows["rate"], errors="coerce")
    rows["body_stripped"] = rows["body"].fillna("").str.strip()
    rows["body_normalized"] = rows["body"].fillna("").map(normalize)

    found, combos, examples = [], {}, {c: [] for c in ORDER}
    for review_id, group in rows.groupby("id", sort=True):
        diff = {
            "body": group["body_stripped"].nunique() > 1,
            "normalized body": group["body_normalized"].nunique() > 1,
            "rate": group["rate_value"].nunique(dropna=False) > 1,
            "label": group["label"].nunique() > 1,
        }
        name = category(diff, group["rate_value"].fillna(-1).tolist())
        combo = " + ".join(k for k, v in diff.items() if v)
        combos[combo] = combos.get(combo, 0) + 1
        newer_last = None
        if group["created_at"].nunique() > 1:  # is the copy with the larger src_row also the newer one?
            newer_last = bool(group["created_at"].iloc[-1] >= group["created_at"].max())
        found.append(
            {
                "id": review_id,
                "category": name,
                "copies": len(group),
                "same_product": group["product_id"].nunique() == 1,
                "same_created_at": group["created_at"].nunique() == 1,
                "last_copy_is_newest": newer_last,
                "one_label_missing": diff["label"] and (group["label"] == "none").any(),
                "split": str(product_split.get(product_key(group["product_id"]).iloc[0], "none")),
            }
        )
        if len(examples[name]) < EXAMPLES:
            examples[name].append(str(review_id))
    table = pd.DataFrame(found)

    result = {
        "ids": int(len(table)),
        "rows": int(len(rows)),
        "normalizer_version": NORMALIZER_VERSION,
        "categories": {c: int((table["category"] == c).sum()) for c in ORDER},
        "differing_fields": dict(sorted(combos.items(), key=lambda kv: -kv[1])),
        "label_cases_with_one_label_missing": int(table["one_label_missing"].sum()),
        "same_product": int(table["same_product"].sum()),
        "same_created_at": int(table["same_created_at"].sum()),
        "different_created_at_last_copy_is_newest": int((table["last_copy_is_newest"] == True).sum()),  # noqa: E712
        "different_created_at_last_copy_is_older": int((table["last_copy_is_newest"] == False).sum()),  # noqa: E712
        "split_of_product": table["split"].value_counts().to_dict(),
        "by_category": {
            c: {
                "same_product": int(part["same_product"].sum()),
                "same_created_at": int(part["same_created_at"].sum()),
                "split_of_product": part["split"].value_counts().to_dict(),
            }
            for c, part in table.groupby("category")
        },
        "example_ids": examples,
    }
    save_json(
        OUTPUT,
        {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **result, "code": git_commit()},
    )
    print(f"{result['ids']} ids, {result['rows']} rows")
    for c in ORDER:
        print(f"  {c:<10} {result['categories'][c]}")
    print("differing fields: " + "; ".join(f"{k} {v}" for k, v in result["differing_fields"].items()))
    print(f"label cases with one label missing: {result['label_cases_with_one_label_missing']}")
    print(
        f"same product {result['same_product']}, same creation time {result['same_created_at']}; "
        f"when the times differ, the later row is newest {result['different_created_at_last_copy_is_newest']}"
        f" and older {result['different_created_at_last_copy_is_older']} times"
    )
    print(f"split of the product: {result['split_of_product']}")
    assert result["split_of_product"].get("none", 0) < result["ids"], (
        "no product matched the split assignment"
    )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
