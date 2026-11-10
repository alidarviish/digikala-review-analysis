"""Answer "uncertain" for the least confident 10% of reviews (plan: reports/uncertain_plan.md).

The final system mixes FaBERT's and TF-IDF's probabilities with a weight, then adds per-class
offsets (D58). Its confidence in a review is the mixed probability of the class it answers. On
each product half of validation, the weight, the offsets and the confidence threshold (the
10th percentile) are fitted on the other half, as in step 60; the reviews under the threshold
are answered "uncertain". For the test set and the offline script, the three are fitted once
on all of validation and saved under "for_test".

Reports, per head, on the whole validation set: the share answered "uncertain", the scores on
the answered reviews against all reviews, what the "uncertain" reviews are, and the coverage
curve. The system's labels must equal step 60's (its recorded Macro F1). The test set is not
read.

Writes reports/uncertain.json and prints a summary.

Usage:
    uv run python scripts/uncertain_output.py
"""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, FINAL_TRANSFORMER_DIR, REPORTS_DIR
from digikala_reviews.ensemble import WEIGHTS, blend
from digikala_reviews.experiments import predict_with_offsets, product_half, search_offsets
from digikala_reviews.metrics import classification_report, macro_f1_from_codes
from digikala_reviews.provenance import check_cache, git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "uncertain.json"
UNCERTAIN_SHARE = 0.10
COVERAGES = [round(c, 2) for c in np.arange(0.50, 1.001, 0.05)]


def fit(log_a, log_b, codes, n_classes):
    """The weight and offsets with the best Macro F1 on these reviews (as cross_fitted_blend),
    and the confidence threshold that leaves UNCERTAIN_SHARE of them under it."""
    best = None
    for weight in WEIGHTS:
        mixed = blend(log_a, log_b, weight)
        offsets = search_offsets(mixed, codes)
        score = macro_f1_from_codes(codes, predict_with_offsets(mixed, offsets), n_classes)
        if best is None or score > best[0]:  # the first best weight on ties, as in step 60
            best = (score, weight, offsets)
    _, weight, offsets = best
    _, confidence = answer(log_a, log_b, weight, offsets)
    return float(weight), offsets, float(np.quantile(confidence, UNCERTAIN_SHARE))


def answer(log_a, log_b, weight, offsets):
    """The class the system answers, and the mixed probability of that class."""
    mixed = blend(log_a, log_b, weight)
    predicted = predict_with_offsets(mixed, offsets)
    return predicted, np.exp(mixed[np.arange(len(predicted)), predicted])


def scores(codes, predicted, labels) -> dict:
    report = classification_report(np.array(labels)[codes], np.array(labels)[predicted], labels)
    return {
        "reviews": report["rows"],
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "recall": {c: v["recall"] for c, v in report["per_class"].items()},
    }


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    ensemble = json.loads((REPORTS_DIR / "ensemble.json").read_text(encoding="utf-8"))[
        "final_fabert_and_tfidf"
    ]
    tables = {}
    for name, folder in [("fabert", FINAL_TRANSFORMER_DIR), ("tfidf", "tfidf_final")]:
        table = pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet")
        assert table["id"].is_unique and set(table["id"]) == set(validation["id"]), f"{name}: ids differ"
        tables[name] = validation[["id"]].merge(table, on="id", how="left", validate="one_to_one")

    heads = {}
    for head, spec in HEADS.items():
        labels = spec["labels"]
        n = len(labels)
        rows = validation[spec["column"]].notna().to_numpy()
        codes = np.array([labels.index(v) for v in validation[spec["column"]].to_numpy()[rows]])
        log = {
            name: t[[f"{head}__log_proba__{c}" for c in labels]].to_numpy(dtype=float)[rows]
            for name, t in tables.items()
        }
        half_a = product_half(validation["product_id"].to_numpy()[rows])
        predicted = np.empty(len(codes), dtype=int)
        confidence = np.empty(len(codes))
        uncertain = np.empty(len(codes), dtype=bool)
        per_half = {}
        for name, tune_on in [("tuned_on_a", half_a), ("tuned_on_b", ~half_a)]:
            weight, offsets, threshold = fit(log["fabert"][tune_on], log["tfidf"][tune_on], codes[tune_on], n)
            p, c = answer(log["fabert"][~tune_on], log["tfidf"][~tune_on], weight, offsets)
            predicted[~tune_on], confidence[~tune_on] = p, c
            uncertain[~tune_on] = c < threshold
            per_half[name] = {"fabert_weight": weight, "threshold": threshold}
        recorded = ensemble[head]["blend_macro_f1"]
        found = macro_f1_from_codes(codes, predicted, n)
        assert abs(found - recorded) < 1e-9, (
            f"{head}: the system's Macro F1 {found} is not step 60's {recorded}"
        )

        weight, offsets, threshold = fit(log["fabert"], log["tfidf"], codes, n)
        answered = ~uncertain
        right = predicted == codes
        order = np.argsort(-confidence, kind="stable")
        curve = []
        for coverage in COVERAGES:
            kept = order[: int(round(coverage * len(codes)))]
            curve.append(
                {
                    "coverage": coverage,
                    "accuracy": float(right[kept].mean()),
                    "macro_f1": macro_f1_from_codes(codes[kept], predicted[kept], n),
                }
            )
        base = np.bincount(codes, minlength=n) / len(codes)
        unsure = np.bincount(codes[uncertain], minlength=n) / max(uncertain.sum(), 1)
        heads[head] = {
            "share_uncertain": float(uncertain.mean()),
            "all_reviews": scores(codes, predicted, labels),
            "answered_reviews": scores(codes[answered], predicted[answered], labels),
            "uncertain_reviews": {
                "reviews": int(uncertain.sum()),
                "system_right": float(right[uncertain].mean()),
                "true_class_share": dict(zip(labels, unsure.round(6).tolist(), strict=True)),
                "true_class_share_all_reviews": dict(zip(labels, base.round(6).tolist(), strict=True)),
                "times_its_share": dict(zip(labels, (unsure / base).round(3).tolist(), strict=True)),
            },
            "per_half": per_half,
            "coverage_curve": curve,
            "for_test": {
                "fabert_weight": weight,
                "offsets": dict(zip(labels, np.asarray(offsets).tolist(), strict=True)),
                "threshold": threshold,
            },
        }

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/uncertain_plan.md",
            "uncertain_share": UNCERTAIN_SHARE,
            "heads": heads,
            "data": data,
            "code": git_commit(),
        },
    )
    for head, e in heads.items():
        a, k, u = e["all_reviews"], e["answered_reviews"], e["uncertain_reviews"]
        print(f"\n=== {head}: {e['share_uncertain']:.1%} answered 'uncertain'")
        print(f"  all reviews:      accuracy {a['accuracy']:.4f}, Macro F1 {a['macro_f1']:.4f}")
        print(f"  answered reviews: accuracy {k['accuracy']:.4f}, Macro F1 {k['macro_f1']:.4f}")
        print(
            "  recall, all -> answered: "
            + ", ".join(f"{c} {a['recall'][c]:.3f} -> {k['recall'][c]:.3f}" for c in a["recall"])
        )
        print(
            f"  uncertain reviews: {u['reviews']:,}, the system right on {u['system_right']:.1%}; "
            "times their share: " + ", ".join(f"{c} {v:.2f}" for c, v in u["times_its_share"].items())
        )
        print(
            "  coverage curve: "
            + ", ".join(f"{p['coverage']:.0%} {p['accuracy']:.3f}" for p in e["coverage_curve"])
        )
        t = e["for_test"]
        print(f"  for the test set: FaBERT weight {t['fabert_weight']}, threshold {t['threshold']:.4f}")
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    main()
