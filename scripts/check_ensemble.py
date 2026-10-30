"""Do combined models score better than one? Two checks on the saved validation predictions.

1. The final FaBERT and TF-IDF models: the mean of their probabilities, weight w for FaBERT
   and 1 - w for TF-IDF, w from 0 to 1 in steps of 0.1. The weight and the offsets are tuned
   on one half of validation by product and scored on the other, and the other way round,
   then compared with FaBERT alone (its saved labels with cross-fitted offsets) on the same
   reviews.
2. Three seeds of the FaBERT pilot (step 41, 500,000 reviews): the mean of their
   probabilities against each seed alone, both with cross-fitted offsets. This shows how much
   training several final models would add.

Writes reports/ensemble.json. Nothing is trained and the test set is not read. The plan and
the expected results are in reports/ensemble_plan.md.

Usage:
    uv run python scripts/check_ensemble.py
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, FINAL_TRANSFORMER_DIR, REPORTS_DIR
from digikala_reviews.ensemble import cross_fitted_blend, mean_probabilities
from digikala_reviews.experiments import compare, cross_fitted_offsets, product_half
from digikala_reviews.metrics import macro_f1_from_codes
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

PILOT = "transformer_pilot_whole"
PILOT_TAGS = [f"sbunlp__fabert__seed{s}__n500000__checks10" for s in (2026, 2027, 2028)]


def log_proba_columns(table, head):
    return table[[f"{head}__log_proba__{label}" for label in HEADS[head]["labels"]]].to_numpy(dtype=float)


def final_models(validation):
    fabert = pd.read_parquet(ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "validation_predictions.parquet")
    tfidf = pd.read_parquet(ARTIFACTS_DIR / "tfidf_final" / "validation_predictions.parquet")
    ids = validation["id"].to_numpy()
    assert (fabert["id"].to_numpy() == ids).all() and (tfidf["id"].to_numpy() == ids).all(), "row order"
    result = {}
    for head, h in HEADS.items():
        rows = validation[h["column"]].notna().to_numpy()
        labels = np.array(h["labels"])
        y_true = validation[h["column"]].to_numpy()[rows]
        codes = np.array([h["labels"].index(v) for v in y_true])
        halves = product_half(validation["product_id"].to_numpy()[rows])
        log_f, log_t = log_proba_columns(fabert, head)[rows], log_proba_columns(tfidf, head)[rows]
        alone = fabert[f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str)  # as saved
        blended, weights = cross_fitted_blend(log_f, log_t, codes, halves)
        found = compare(head, y_true, labels[blended], alone)
        result[head] = {
            "reviews": int(rows.sum()),
            "fabert_weight_per_half": weights,
            "fabert_macro_f1": found["reference_scores"]["macro_f1"],
            "blend_macro_f1": found["variant_scores"]["macro_f1"],
            "difference": found["macro_f1_difference"],
            "difference_interval_95": found["difference_interval_95"],
            "direction": found["direction"],
            "f1_per_class": {
                label: {
                    "fabert": found["reference_scores"]["per_class"][label]["f1"],
                    "blend": found["variant_scores"]["per_class"][label]["f1"],
                }
                for label in h["labels"]
            },
        }
    return result


def pilot_seeds(validation):
    by_id = validation.set_index("id")
    runs = [np.load(ARTIFACTS_DIR / PILOT / f"{tag}.npz") for tag in PILOT_TAGS]
    ids = runs[0]["ids"]
    assert all((r["ids"] == ids).all() for r in runs), "the seeds must score the same reviews"
    rows = by_id.loc[ids]
    result = {}
    for head, h in HEADS.items():
        labeled = rows[h["column"]].notna().to_numpy()
        labels = np.array(h["labels"])
        y_true = rows[h["column"]].to_numpy()[labeled]
        codes = np.array([h["labels"].index(v) for v in y_true])
        halves = product_half(rows["product_id"].to_numpy()[labeled])
        singles = [cross_fitted_offsets(r[head][labeled], codes, halves)[0] for r in runs]
        combined, _ = cross_fitted_offsets(
            mean_probabilities([r[head][labeled] for r in runs]), codes, halves
        )
        single_f1 = [macro_f1_from_codes(codes, s, len(labels)) for s in singles]
        against_each = [compare(head, y_true, labels[combined], labels[s]) for s in singles]
        result[head] = {
            "single_macro_f1": single_f1,
            "combined_macro_f1": macro_f1_from_codes(codes, combined, len(labels)),
            "difference_to_each_seed": [c["macro_f1_difference"] for c in against_each],
            "interval_95_to_each_seed": [c["difference_interval_95"] for c in against_each],
            "direction_to_each_seed": [c["direction"] for c in against_each],
        }
    return result


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    result = {
        "final_fabert_and_tfidf": final_models(validation),
        "three_pilot_seeds": pilot_seeds(validation),
    }
    save_json(
        REPORTS_DIR / "ensemble.json",
        {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **result, "code": git_commit()},
    )
    print("FaBERT and TF-IDF combined, against FaBERT alone (both with cross-fitted offsets):")
    for head, r in result["final_fabert_and_tfidf"].items():
        low, high = r["difference_interval_95"]
        w = r["fabert_weight_per_half"]
        print(
            f"  {head}: {r['blend_macro_f1']:.4f} against {r['fabert_macro_f1']:.4f}, "
            f"{r['difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({r['direction']}); "
            f"FaBERT weight {w['tuned_on_a']:.1f} and {w['tuned_on_b']:.1f}"
        )
    print("Three pilot seeds combined, against each seed alone:")
    for head, r in result["three_pilot_seeds"].items():
        cells = [
            f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            for d, (lo, hi) in zip(r["difference_to_each_seed"], r["interval_95_to_each_seed"], strict=True)
        ]
        print(
            f"  {head}: {r['combined_macro_f1']:.4f}; seeds alone "
            + ", ".join(f"{s:.4f}" for s in r["single_macro_f1"])
            + "; differences "
            + ", ".join(cells)
        )


if __name__ == "__main__":
    main()
