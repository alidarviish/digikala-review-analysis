"""Compare my hand labels of the validation sample with the original labels and with FaBERT.

Reads data/label_sample/to_label.csv (filled in) and data/label_sample/key.csv, and the kept
FaBERT validation predictions with offsets (artifacts/transformer_final/validation_predictions.parquet).
For the first showing of every review: agreement of my label with the original label, overall
and per original label, and the full table of counts; agreement of FaBERT with my label and
with the original label. For the repeated reviews: how often I gave the same label twice.
Where my label and the original differ, the buyer's star rating (never shown while labeling)
is read as a third opinion: high = recommended, medium = no idea, low = not recommended.

The sample has fixed numbers per original label, so the overall agreement is also given
weighted by the share of each label in the whole validation set.

Writes reports/label_sample.json and prints a summary.

Usage:
    uv run python scripts/score_label_sample.py
"""

from datetime import UTC, datetime

import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, DATA_DIR, RECOMMENDATION_CLASSES, REPORTS_DIR
from digikala_reviews.label_sample import agreement, confusion, read_labels
from digikala_reviews.provenance import git_commit
from digikala_reviews.standard_files import read_split_file
from digikala_reviews.tuning import save_json

FOLDER = DATA_DIR / "label_sample"
RATING_SAYS = {"high": "recommended", "medium": "no_idea", "low": "not_recommended"}


def main():
    answers = pd.read_csv(FOLDER / "to_label.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(FOLDER / "key.csv")
    table = key.merge(answers[["row", "my_label"]].astype({"row": int}), on="row", validate="one_to_one")
    assert len(table) == len(key), "every row of the key must be in the labeled file"
    table["my_label"] = read_labels(table["my_label"])

    first = table[table["repeat_of"] == 0].copy()
    repeats = table[table["repeat_of"] > 0].merge(
        first[["row", "my_label"]], left_on="repeat_of", right_on="row", suffixes=("", "_first")
    )
    fabert = pd.read_parquet(ARTIFACTS_DIR / "transformer_final" / "validation_predictions.parquet")
    fabert = fabert.set_index("id")["recommendation__predicted_with_offsets"]
    first["fabert"] = fabert.loc[first["id"]].to_numpy()

    validation = read_split_file("validation.csv")
    shares = validation["recommendation_status"].value_counts(normalize=True)
    # the buyer's own star rating, never shown while labeling, as a third opinion
    rating = validation.set_index("id")["rating_band"].loc[first["id"]]
    first["rating_says"] = rating.map(RATING_SAYS).to_numpy()
    disagree = first[first["my_label"] != first["original_label"]].dropna(subset=["rating_says"])
    mine = agreement(first, "original_label")
    weighted = float(sum(shares[label] * mine[label]["agree"] for label in RECOMMENDATION_CLASSES))
    fabert_vs_original = float((first["fabert"] == first["original_label"]).mean())
    # the reviews FaBERT gets "wrong": does the hand label side with the model or with the original?
    errors = first[first["fabert"] != first["original_label"]]
    result = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reviews": int(len(first)),
        "my_label_vs_original": mine,
        "my_label_vs_original_weighted_by_validation_shares": weighted,
        "original_label_by_my_label": confusion(first),
        "fabert_vs_my_label": agreement(first, "fabert"),
        "fabert_vs_original": fabert_vs_original,
        "fabert_errors": {
            "rows": int(len(errors)),
            "my_label_sides_with_fabert": float((errors["my_label"] == errors["fabert"]).mean()),
            "my_label_sides_with_original": float((errors["my_label"] == errors["original_label"]).mean()),
            "by_original_label": errors["original_label"].value_counts().to_dict(),
        },
        "disagreements_with_a_rating": {
            "rows": int(len(disagree)),
            "rating_sides_with_my_label": float((disagree["rating_says"] == disagree["my_label"]).mean()),
            "rating_sides_with_original": float(
                (disagree["rating_says"] == disagree["original_label"]).mean()
            ),
            "rating_mapping": RATING_SAYS,
        },
        "repeats": {
            "rows": int(len(repeats)),
            "same_label": float((repeats["my_label"] == repeats["my_label_first"]).mean()),
        },
        "code": git_commit(),
    }
    save_json(REPORTS_DIR / "label_sample.json", result)

    print(f"{len(first)} reviews, {len(repeats)} repeats")
    print(f"My label = original label: {mine['agree']:.1%} of the sample, {weighted:.1%} weighted by shares")
    for label in RECOMMENDATION_CLASSES:
        counts = result["original_label_by_my_label"][label]
        print(f"  original {label}: {mine[label]['agree']:.1%} of {mine[label]['rows']}, my labels {counts}")
    agree = result["fabert_vs_my_label"]["agree"]
    print(f"FaBERT = my label: {agree:.1%}; FaBERT = original label: {fabert_vs_original:.1%}")
    wrong = result["fabert_errors"]
    print(
        f"Where FaBERT differs from the original label ({wrong['rows']} reviews), my label sides with "
        f"FaBERT {wrong['my_label_sides_with_fabert']:.1%}, with the original "
        f"{wrong['my_label_sides_with_original']:.1%}; by original label {wrong['by_original_label']}"
    )
    rated = result["disagreements_with_a_rating"]
    print(
        f"Where I disagree with the original label ({rated['rows']} with a rating), the rating sides with "
        f"me {rated['rating_sides_with_my_label']:.1%}, "
        f"with the original {rated['rating_sides_with_original']:.1%}"
    )
    print(f"Same label on the repeated reviews: {result['repeats']['same_label']:.1%} of {len(repeats)}")


if __name__ == "__main__":
    main()
