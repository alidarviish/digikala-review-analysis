"""A chooser with more information: can gradient-boosted trees combine the final models
better than the weighted mean?

For each head, a non-linear model (scikit-learn's HistGradientBoostingClassifier) learns the
class from:
- the log probabilities of the models for both heads, floored at -20,
- the length of the text in words and in characters,
- for each pair of models, whether their most likely classes for this head agree.

Two versions: FaBERT and TF-IDF only, and all three final models (with TookaBERT-Large). Each
is fitted on one half of validation by product and predicts the other half, with offsets
searched on predictions it did not fit (scripts in src/digikala_reviews/ensemble.py), and is
compared with the kept combination of FaBERT and TF-IDF (step 60) on the same reviews.

Writes reports/chooser.json. Nothing outside validation is read; the test set is not read.
The plan and the expected results are in reports/chooser_plan.md.

Usage:
    uv run python scripts/check_chooser.py
"""

import time
from datetime import UTC, datetime
from itertools import combinations

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, FINAL_TRANSFORMER_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.ensemble import cross_fitted_blend, cross_fitted_chooser, stack_features
from digikala_reviews.experiments import compare, product_half
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

MODELS = {"fabert": FINAL_TRANSFORMER_DIR, "large": "transformer_large", "tfidf": "tfidf_final"}
VERSIONS = {"fabert_and_tfidf": ["fabert", "tfidf"], "three_models": ["fabert", "large", "tfidf"]}


def log_proba_columns(table, head):
    return table[[f"{head}__log_proba__{label}" for label in HEADS[head]["labels"]]].to_numpy(dtype=float)


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    ids = validation["id"].to_numpy()
    tables = {}
    for name, folder in MODELS.items():
        tables[name] = pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet")
        assert (tables[name]["id"].to_numpy() == ids).all(), f"row order of {name}"
    text = validation["text"].astype(str)
    lengths = np.column_stack([text.str.split().str.len().to_numpy(), text.str.len().to_numpy()])
    result = {}
    for head, h in HEADS.items():
        rows = validation[h["column"]].notna().to_numpy()
        labels = np.array(h["labels"])
        y_true = validation[h["column"]].to_numpy()[rows].astype(str)
        codes = np.array([h["labels"].index(v) for v in y_true])
        products = validation["product_id"].to_numpy()[rows]
        log = {name: log_proba_columns(t, head)[rows] for name, t in tables.items()}
        kept, _ = cross_fitted_blend(log["fabert"], log["tfidf"], codes, product_half(products))
        result[head] = {"reviews": int(rows.sum()), "versions": {}}
        for version, names in VERSIONS.items():
            start = time.perf_counter()
            both_heads = [log_proba_columns(tables[n], other)[rows] for n in names for other in HEADS]
            agree = [
                (log[a].argmax(axis=1) == log[b].argmax(axis=1)).astype(float)
                for a, b in combinations(names, 2)
            ]
            features = np.column_stack([stack_features(both_heads), lengths[rows], *agree])
            chosen, _ = cross_fitted_chooser(features, codes, products, SPLIT_SEED)
            found = compare(head, y_true, labels[chosen], labels[kept])
            result[head]["versions"][version] = {
                "features": int(features.shape[1]),
                "kept_macro_f1": found["reference_scores"]["macro_f1"],
                "chooser_macro_f1": found["variant_scores"]["macro_f1"],
                "difference": found["macro_f1_difference"],
                "difference_interval_95": found["difference_interval_95"],
                "direction": found["direction"],
                "f1_per_class": {
                    label: {
                        "kept": found["reference_scores"]["per_class"][label]["f1"],
                        "chooser": found["variant_scores"]["per_class"][label]["f1"],
                    }
                    for label in h["labels"]
                },
                "minutes": round((time.perf_counter() - start) / 60, 1),
            }
            r = result[head]["versions"][version]
            low, high = r["difference_interval_95"]
            print(
                f"{head}, {version}: chooser {r['chooser_macro_f1']:.4f} "
                f"against kept {r['kept_macro_f1']:.4f}, "
                f"{r['difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({r['direction']}), {r['minutes']} min",
                flush=True,
            )
            print(
                "  F1 per class, kept -> chooser: "
                + ", ".join(
                    f"{k} {v['kept']:.3f} -> {v['chooser']:.3f}" for k, v in r["f1_per_class"].items()
                ),
                flush=True,
            )
    save_json(
        REPORTS_DIR / "chooser.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "heads": result,
            "code": git_commit(),
        },
    )


if __name__ == "__main__":
    main()
