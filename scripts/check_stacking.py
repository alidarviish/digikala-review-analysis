"""Does a small model on top of both models and both heads beat the weighted mean of FaBERT and TF-IDF?

For each head, a logistic regression reads 12 numbers per review: the log probabilities of the
final FaBERT and TF-IDF models for both heads (recommendation and rating band). It is fitted
on one half of validation by product, with its offsets, and predicts the other half, then the
other way round. It is compared on the same reviews with the weighted mean of the two models
for that head alone (scripts/check_ensemble.py, cross-fitted the same way) and with FaBERT
alone (its saved labels with cross-fitted offsets).

Writes reports/stacking.json. Nothing is trained but the small models, and the test set is
not read. The plan and the expected results are in reports/stacking_plan.md.

Usage:
    uv run python scripts/check_stacking.py
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from check_ensemble import log_proba_columns

from digikala_reviews.config import ARTIFACTS_DIR, FINAL_TRANSFORMER_DIR, REPORTS_DIR
from digikala_reviews.ensemble import cross_fitted_blend, cross_fitted_stack, stack_features
from digikala_reviews.experiments import compare, product_half
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json


def summary(found):
    low, high = found["difference_interval_95"]
    return {
        "variant_macro_f1": found["variant_scores"]["macro_f1"],
        "reference_macro_f1": found["reference_scores"]["macro_f1"],
        "difference": found["macro_f1_difference"],
        "difference_interval_95": [low, high],
        "direction": found["direction"],
    }


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    fabert = pd.read_parquet(ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "validation_predictions.parquet")
    tfidf = pd.read_parquet(ARTIFACTS_DIR / "tfidf_final" / "validation_predictions.parquet")
    ids = validation["id"].to_numpy()
    assert (fabert["id"].to_numpy() == ids).all() and (tfidf["id"].to_numpy() == ids).all(), "row order"
    features = stack_features([log_proba_columns(table, head) for table in (fabert, tfidf) for head in HEADS])

    result = {}
    for head, h in HEADS.items():
        rows = validation[h["column"]].notna().to_numpy()
        labels = np.array(h["labels"])
        y_true = validation[h["column"]].to_numpy()[rows]
        codes = np.array([h["labels"].index(v) for v in y_true])
        halves = product_half(validation["product_id"].to_numpy()[rows])
        stacked, _ = cross_fitted_stack(features[rows], codes, halves)
        blended, _ = cross_fitted_blend(
            log_proba_columns(fabert, head)[rows], log_proba_columns(tfidf, head)[rows], codes, halves
        )
        alone = fabert[f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str)
        against_blend = compare(head, y_true, labels[stacked], labels[blended])
        result[head] = {
            "reviews": int(rows.sum()),
            "stacking_against_blend": summary(against_blend),
            "stacking_against_fabert": summary(compare(head, y_true, labels[stacked], alone)),
            "f1_per_class": {
                label: {
                    "blend": against_blend["reference_scores"]["per_class"][label]["f1"],
                    "stacking": against_blend["variant_scores"]["per_class"][label]["f1"],
                }
                for label in h["labels"]
            },
        }

    save_json(
        REPORTS_DIR / "stacking.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "features": "log probabilities of FaBERT and TF-IDF for both heads, floored at -20",
            "heads": result,
            "code": git_commit(),
        },
    )
    for head, r in result.items():
        for name, key in (("blend", "stacking_against_blend"), ("FaBERT alone", "stacking_against_fabert")):
            s = r[key]
            low, high = s["difference_interval_95"]
            print(
                f"{head}: stacking {s['variant_macro_f1']:.4f} against {name} {s['reference_macro_f1']:.4f}, "
                f"{s['difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({s['direction']})"
            )


if __name__ == "__main__":
    main()
