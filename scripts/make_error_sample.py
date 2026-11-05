"""Make the file for reading a sample of rating band errors by hand, and its key.

The plan, the codes and the expected results are in reports/error_sample_plan.md. Five
groups of validation reviews, 20 each, chosen with seed 2026 from the final FaBERT run's
saved labels with offsets:

  A  recommended, band medium, predicted high
  B  no_idea, band medium, predicted low or high
  C  recommended, band high, predicted medium
  D  recommended, band low, predicted medium or high
  E  control: any review with both labels whose band FaBERT got right

Each text at most once, at most one review per product, and no text that is also in train
(those repeat texts are hypothesis H4, already measured). 10 of the 100 reviews are shown a
second time. The reader sees only the row number and the original text, in shuffled order.

Writes two files in data/error_sample/ (not in git):
  to_label.csv  row, text and three empty columns (reading, stars, tags); the only file to open
  key.csv       row, id, product_id, group, rate, labels, predictions, repeat_of; do not open it
                before reading is finished

The test set is not read; nothing is trained.

Usage:
    uv run python scripts/make_error_sample.py
"""

import json

import numpy as np
import pandas as pd
from analyze_rate_recommendation import BAND, REC, load_fabert, load_recommendation
from analyze_rating_boundaries import CACHE_FILE, SPLIT_FILE, check_rates, load_validation

from digikala_reviews.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.standard_files import read_split_file

OUT_DIR = DATA_DIR / "error_sample"
PER_GROUP = 20
REPEATS = 10
GROUPS = {
    "A": "recommended, band medium, predicted high",
    "B": "no_idea, band medium, predicted low or high",
    "C": "recommended, band high, predicted medium",
    "D": "recommended, band low, predicted medium or high",
    "E": "control: band predicted right",
}


def group_masks(rec, band, pred) -> dict:
    """Which reviews belong to each group (reviews with both labels only)."""
    both = pd.notna(rec) & pd.notna(band)
    return {
        "A": both & (rec == "recommended") & (band == "medium") & (pred == "high"),
        "B": both & (rec == "no_idea") & (band == "medium") & (pred != "medium"),
        "C": both & (rec == "recommended") & (band == "high") & (pred == "medium"),
        "D": both & (rec == "recommended") & (band == "low") & (pred != "low"),
        "E": both & (band == pred),
    }


def choose(table: pd.DataFrame, masks: dict, seed: int) -> pd.DataFrame:
    """PER_GROUP reviews per group, each text and each product at most once overall."""
    rng = np.random.default_rng(seed)
    chosen, seen_texts, seen_products = [], set(), set()
    for name, mask in masks.items():
        rows = table[mask & ~table["text_seen_in_train"].to_numpy(dtype=bool)]
        taken = 0
        for position in rng.permutation(len(rows)):
            row = rows.iloc[position]
            text = " ".join(str(row["body"]).split())
            if text in seen_texts or row["product_id"] in seen_products:
                continue
            seen_texts.add(text)
            seen_products.add(row["product_id"])
            chosen.append(row.to_dict() | {"group": name})
            taken += 1
            if taken == PER_GROUP:
                break
        assert taken == PER_GROUP, f"group {name}: only {taken} reviews could be chosen"
    sample = pd.DataFrame(chosen)
    sample["repeat_of_id"] = 0
    repeats = sample.iloc[rng.choice(len(sample), REPEATS, replace=False)].copy()
    repeats["repeat_of_id"] = repeats["id"]
    out = pd.concat([sample, repeats], ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    out.insert(0, "row", np.arange(1, len(out) + 1))
    first = out[out["repeat_of_id"] == 0]
    first_row = dict(zip(first["id"], first["row"], strict=True))
    out["repeat_of"] = [first_row[i] if r else 0 for i, r in zip(out["id"], out["repeat_of_id"], strict=True)]
    return out.drop(columns="repeat_of_id")


def main():
    to_label, key = OUT_DIR / "to_label.csv", OUT_DIR / "key.csv"
    assert not to_label.exists(), f"{to_label} already exists; delete it only if nothing is written in it"
    manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["class_counts"][SPLIT_FILE]
    validation, _ = load_validation(verify_csv=False)
    check_rates(validation, counts[BAND])
    rec, _ = load_recommendation(validation, False, counts[REC])
    predicted, _ = load_fabert(validation)
    band = validation[BAND].to_numpy()
    assert (pd.isna(predicted[BAND]) == pd.isna(band)).all(), "band predictions empty for labelled reviews"

    seen = pd.read_parquet(CACHE_DIR / CACHE_FILE, columns=["id", "text_seen_in_train"])
    assert (seen["id"].to_numpy() == validation["id"].to_numpy()).all(), "row order differs"
    body = read_split_file(SPLIT_FILE)[["id", "body"]]  # the original text, as the buyer wrote it
    body = validation[["id"]].merge(body, on="id", how="left", validate="one_to_one")
    assert body["body"].notna().all(), "a validation id has no body in validation.csv"

    table = pd.DataFrame(
        {
            "id": validation["id"].to_numpy(),
            "product_id": validation["product_id"].to_numpy(),
            "rate": validation["rate"].to_numpy(),
            "band": band,
            "recommendation": rec,
            "band_predicted": predicted[BAND],
            "recommendation_predicted": predicted[REC],
            "text_seen_in_train": seen["text_seen_in_train"].to_numpy(),
            "body": body["body"].to_numpy(),
        }
    )
    masks = group_masks(rec, band, predicted[BAND])
    sizes = {name: int(m.sum()) for name, m in masks.items()}
    sample = choose(table, masks, SPLIT_SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so that Excel shows the Persian text correctly
    sample[["row", "body"]].rename(columns={"body": "text"}).assign(reading="", stars="", tags="").to_csv(
        to_label, index=False, encoding="utf-8-sig"
    )
    key_columns = [
        "row",
        "id",
        "product_id",
        "group",
        "rate",
        "band",
        "recommendation",
        "band_predicted",
        "recommendation_predicted",
        "repeat_of",
    ]
    sample[key_columns].to_csv(key, index=False, encoding="utf-8")
    print(f"group sizes in validation (before leaving out texts seen in train): {sizes}")
    print(f"{len(sample)} rows to read in {to_label}")
    print(f"Key (do not open before reading is done): {key}")


if __name__ == "__main__":
    main()
