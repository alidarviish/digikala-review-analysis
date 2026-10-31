"""Do the three final models make different mistakes? And does adding TookaBERT-Large to the
combination of FaBERT and TF-IDF help?

Reads the saved validation predictions of the final FaBERT, TookaBERT-Large and TF-IDF models
(their labels with cross-fitted offsets, as saved). For each head:

1. The share of reviews all three get right, all three get wrong, and the rest, where a
   combination could help; per true class, and for each pair of models the share of either
   model's mistakes that both make.
2. The perfect-chooser bound: the Macro F1 if, for every review, the right model were picked
   whenever one of them is right. No real combination can score above it.
3. FaBERT and Large averaged, then combined with TF-IDF as in step 60 (weight and offsets
   tuned on one half of validation by product, scored on the other), against the kept
   combination of FaBERT and TF-IDF on the same reviews.

Writes reports/error_overlap.json. Nothing is trained and the test set is not read. The plan
and the expected results are in reports/error_overlap_plan.md.

Usage:
    uv run python scripts/check_error_overlap.py
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, FINAL_TRANSFORMER_DIR, REPORTS_DIR
from digikala_reviews.ensemble import cross_fitted_blend, error_overlap, mean_probabilities, oracle
from digikala_reviews.experiments import compare, product_half
from digikala_reviews.metrics import macro_f1_from_codes
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

MODELS = {
    "fabert": FINAL_TRANSFORMER_DIR,
    "large": "transformer_large",
    "tfidf": "tfidf_final",
}


def log_proba_columns(table, head):
    return table[[f"{head}__log_proba__{label}" for label in HEADS[head]["labels"]]].to_numpy(dtype=float)


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    ids = validation["id"].to_numpy()
    tables = {}
    for name, folder in MODELS.items():
        tables[name] = pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet")
        assert (tables[name]["id"].to_numpy() == ids).all(), f"row order of {name}"
    result = {}
    for head, h in HEADS.items():
        rows = validation[h["column"]].notna().to_numpy()
        labels = np.array(h["labels"])
        y_true = validation[h["column"]].to_numpy()[rows].astype(str)
        codes = np.array([h["labels"].index(v) for v in y_true])
        halves = product_half(validation["product_id"].to_numpy()[rows])
        saved = {
            name: t[f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str)
            for name, t in tables.items()
        }
        best = oracle(y_true, saved, "fabert")
        log = {name: log_proba_columns(t, head)[rows] for name, t in tables.items()}
        two, two_weights = cross_fitted_blend(log["fabert"], log["tfidf"], codes, halves)
        transformers = mean_probabilities([log["fabert"], log["large"]])
        three, three_weights = cross_fitted_blend(transformers, log["tfidf"], codes, halves)
        found = compare(head, y_true, labels[three], labels[two])
        code_of = {label: i for i, label in enumerate(h["labels"])}
        result[head] = {
            "reviews": int(rows.sum()),
            "overlap": error_overlap(y_true, saved),
            "overlap_per_true_class": {
                label: error_overlap(
                    y_true[y_true == label], {n: p[y_true == label] for n, p in saved.items()}
                )
                for label in h["labels"]
            },
            "macro_f1": {
                name: macro_f1_from_codes(codes, np.array([code_of[v] for v in p]), len(labels))
                for name, p in saved.items()
            },
            "perfect_chooser_macro_f1": macro_f1_from_codes(
                codes, np.array([code_of[v] for v in best]), len(labels)
            ),
            "fabert_and_tfidf": {"macro_f1": found["reference_scores"]["macro_f1"], "weights": two_weights},
            "three_models": {
                "macro_f1": found["variant_scores"]["macro_f1"],
                "transformer_weights": three_weights,
                "difference": found["macro_f1_difference"],
                "difference_interval_95": found["difference_interval_95"],
                "direction": found["direction"],
                "f1_per_class": {
                    label: {
                        "fabert_and_tfidf": found["reference_scores"]["per_class"][label]["f1"],
                        "three_models": found["variant_scores"]["per_class"][label]["f1"],
                    }
                    for label in h["labels"]
                },
            },
        }
    save_json(
        REPORTS_DIR / "error_overlap.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "heads": result,
            "code": git_commit(),
        },
    )
    for head, r in result.items():
        o = r["overlap"]
        print(f"{head} ({r['reviews']:,} reviews):")
        print(
            f"  all right {o['all_right']:.1%}, all wrong {o['all_wrong']:.1%}, "
            f"some right {o['some_right']:.1%}"
        )
        for label, c in r["overlap_per_true_class"].items():
            print(f"    true {label}: all wrong {c['all_wrong']:.1%}, some right {c['some_right']:.1%}")
        print(
            "  pairs, both wrong as a share of either wrong: "
            + ", ".join(f"{k} {v:.1%}" for k, v in o["both_wrong_share_of_either_wrong"].items())
        )
        print(
            "  Macro F1: "
            + ", ".join(f"{k} {v:.4f}" for k, v in r["macro_f1"].items())
            + f"; perfect chooser {r['perfect_chooser_macro_f1']:.4f}"
        )
        t = r["three_models"]
        low, high = t["difference_interval_95"]
        print(
            f"  three models {t['macro_f1']:.4f} against FaBERT and TF-IDF "
            f"{r['fabert_and_tfidf']['macro_f1']:.4f}: "
            f"{t['difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({t['direction']})"
        )


if __name__ == "__main__":
    main()
