"""How much of the rating band error sits in reviews whose two buyer labels pull apart, and
are the hard reviews of the two heads the same reviews?

Each review can carry two labels chosen by the buyer: the recommendation, and the rate the
band comes from. The rate recommendation analysis found that among reviews rated 3, those
FaBERT reads as high were recommended by the buyer 81% of the time. This script measures:

1. The band, by the buyer's recommendation label (recommended, not_recommended, no_idea,
   none): Macro F1 when every band is present, recall of each band, and the counts of the
   table buyer label x true band x predicted band. This is a description, not a new metric:
   no review is dropped from any official score.
2. The two hard classes (no_idea and medium): on reviews with both labels, the table true
   recommendation x true band with each head's error rate and the rate of both being wrong;
   the share of each head's errors that fall on the other head's hard class; and whether a
   band mistake is more likely when the recommendation is also mistaken (the difference
   P(band wrong | recommendation wrong) - P(band wrong | recommendation right)).

Only saved validation predictions of the final FaBERT run (labels with offsets) are read;
nothing is trained. The labels and the rate group the reviews, they are never inputs. The
test set is not read.

Checks: the same data checks as the rate recommendation analysis (cache hash, manifest
counts, band follows from the rate, recommendation classes and counts, optional comparison
with validation.csv, one-to-one join, predictions empty exactly where a head has no label,
predicted labels are known classes), and both heads must score the Macro F1 recorded in
reports/ensemble.json on all their labelled reviews.

Intervals: product-level cluster bootstrap, 1000 rounds, seed 2026. Groups with fewer than
1000 reviews or 100 products are marked "small".

Writes reports/label_conflict.json and prints a summary.

Usage:
    uv run python scripts/analyze_label_conflict.py
    uv run python scripts/analyze_label_conflict.py --verify-csv
"""

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from analyze_rate_recommendation import BAND, REC, load_fabert, load_recommendation
from analyze_rating_boundaries import (
    LABELS,
    SPLIT_FILE,
    check_rates,
    cluster_bootstrap,
    load_validation,
    macro_f1_of,
    size,
    weighted_matrix,
)

from digikala_reviews.config import RECOMMENDATION_CLASSES, REPORTS_DIR
from digikala_reviews.metrics import classification_report
from digikala_reviews.provenance import git_commit
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "label_conflict.json"
NONE = "none"  # no recommendation label
BUYER_GROUPS = [*RECOMMENDATION_CLASSES, NONE]


def band_codes(labels) -> np.ndarray:
    position = {label: i for i, label in enumerate(LABELS)}
    return np.array([position[v] for v in labels], dtype=int)


def band_group(y_true, y_pred, products) -> dict:
    """Band scores of one group of reviews: Macro F1 (only when every band is present),
    recall of each band, with product-level bootstrap intervals, and the confusion counts."""
    result = size(len(y_true), products)
    result["true_band_share"] = {b: float(np.mean(y_true == b)) for b in LABELS}
    result["all_bands_present"] = bool(all(v > 0 for v in result["true_band_share"].values()))
    t, p = band_codes(y_true), band_codes(y_pred)
    counts = weighted_matrix(t, p, np.ones(len(t)))
    result["confusion"] = {
        f"true {LABELS[i]}": {f"predicted {LABELS[j]}": int(counts[i, j]) for j in range(len(LABELS))}
        for i in range(len(LABELS))
    }

    def statistic(w):
        m = weighted_matrix(t, p, w)
        out = {}
        for i, b in enumerate(LABELS):
            total = m[i].sum()
            out[f"recall {b}"] = float(m[i, i] / total) if total > 0 else np.nan
        if result["all_bands_present"]:
            out["macro_f1"] = macro_f1_of(m)
        return out

    interval = cluster_bootstrap(products, statistic)
    found = statistic(np.ones(len(t)))
    for key, value in found.items():
        result[key] = {"value": value, "interval_95": interval[key]}
    if not result["all_bands_present"]:
        result["note"] = "not every band is in this group: Macro F1 is not reported"
    return result


def rate_with_interval(flags: dict, products) -> dict:
    """Mean of each 0/1 array in flags, with product-level bootstrap intervals."""
    arrays = {k: v.astype(float) for k, v in flags.items()}

    def statistic(w):
        return {k: float((w * a).sum() / w.sum()) for k, a in arrays.items()}

    interval = cluster_bootstrap(products, statistic)
    return {k: {"value": float(a.mean()), "interval_95": interval[k]} for k, a in arrays.items()}


def conditional_difference(event, given, products) -> dict:
    """P(event | given) - P(event | not given), on the same draws of products."""
    e, g = event.astype(float), given.astype(float)

    def statistic(w):
        wg, wn = w * g, w * (1 - g)
        return {
            "given": float((wg * e).sum() / wg.sum()),
            "not_given": float((wn * e).sum() / wn.sum()),
            "difference": float((wg * e).sum() / wg.sum() - (wn * e).sum() / wn.sum()),
        }

    interval = cluster_bootstrap(products, statistic)
    found = statistic(np.ones(len(e)))
    return {k: {"value": v, "interval_95": interval[k]} for k, v in found.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-csv", action="store_true", help="compare the cache columns with validation.csv"
    )
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["class_counts"][SPLIT_FILE]
    validation, source = load_validation(args.verify_csv)
    rate_checks = check_rates(validation, counts[BAND])
    buyer_all, rec_record = load_recommendation(validation, args.verify_csv, counts[REC])
    predicted, join = load_fabert(validation)
    band_all = validation[BAND].to_numpy()
    for name, truth, classes in [(BAND, band_all, LABELS), (REC, buyer_all, RECOMMENDATION_CLASSES)]:
        assert (pd.isna(predicted[name]) == pd.isna(truth)).all(), (
            f"FaBERT {name} predictions are empty for other reviews than those without a label"
        )
        has = pd.notna(truth)
        unknown = sorted(set(np.unique(predicted[name][has].astype(str))) - set(classes))
        assert not unknown, f"FaBERT {name} predictions have unknown labels: {unknown[:10]}"

    # both heads must score what the final run recorded, on all their labelled reviews
    saved = json.loads((REPORTS_DIR / "ensemble.json").read_text(encoding="utf-8"))
    saved = saved["final_fabert_and_tfidf"]
    scores = {}
    for head, name, truth, classes in [
        ("rating_band", BAND, band_all, LABELS),
        ("recommendation", REC, buyer_all, RECOMMENDATION_CLASSES),
    ]:
        has = pd.notna(truth)
        found = classification_report(truth[has].astype(str), predicted[name][has].astype(str), classes)
        scores[head] = found["macro_f1"]
        assert int(has.sum()) == saved[head]["reviews"], f"{head}: labelled reviews differ from ensemble.json"
        assert abs(found["macro_f1"] - saved[head]["fabert_macro_f1"]) < 1e-9, (
            f"{head}: FaBERT scores {found['macro_f1']}, ensemble.json {saved[head]['fabert_macro_f1']}"
        )

    products_all = validation["product_id"].to_numpy()
    rows = pd.notna(band_all)  # reviews with a band
    band = band_all[rows].astype(str)
    band_pred = predicted[BAND][rows].astype(str)
    products = products_all[rows]
    buyer = np.where(pd.isna(buyer_all[rows]), NONE, buyer_all[rows]).astype(str)

    result = {
        "reviews_with_band": int(rows.sum()),
        "data_checks": {
            "source": source,
            "rates": rate_checks,
            "recommendation": rec_record,
            "prediction_join": join,
            "fabert_macro_f1": scores,
        },
        "1_band_by_buyer_label": {},
        "2_hard_classes": {},
    }

    # 1. the band, by the buyer's recommendation label
    result["1_band_by_buyer_label"]["all reviews with a band"] = band_group(band, band_pred, products)
    for g in BUYER_GROUPS:
        inside = buyer == g
        result["1_band_by_buyer_label"][g] = band_group(band[inside], band_pred[inside], products[inside])

    # 2. reviews with both labels: are the hard reviews of the two heads the same?
    both = rows & pd.notna(buyer_all)
    t_band = band_all[both].astype(str)
    t_rec = buyer_all[both].astype(str)
    band_wrong = predicted[BAND][both].astype(str) != t_band
    rec_wrong = predicted[REC][both].astype(str) != t_rec
    p_both = products_all[both]
    cells = {}
    for r in RECOMMENDATION_CLASSES:
        for b in LABELS:
            inside = (t_rec == r) & (t_band == b)
            cells[f"{r} x {b}"] = {
                **size(int(inside.sum()), p_both[inside]),
                **rate_with_interval(
                    {
                        "recommendation wrong": rec_wrong[inside],
                        "band wrong": band_wrong[inside],
                        "both wrong": rec_wrong[inside] & band_wrong[inside],
                    },
                    p_both[inside],
                ),
            }
    no_idea, medium = t_rec == "no_idea", t_band == "medium"
    result["2_hard_classes"] = {
        "reviews_with_both_labels": int(both.sum()),
        "by_true_labels": cells,
        "band_wrong_given_recommendation_wrong": conditional_difference(band_wrong, rec_wrong, p_both),
    }
    # shares of each head's errors that fall on the other head's hard class
    rec_errors = rec_wrong.nonzero()[0]
    band_errors = band_wrong.nonzero()[0]
    medium_errors = (band_wrong & medium).nonzero()[0]
    no_idea_errors = (rec_wrong & no_idea).nonzero()[0]
    result["2_hard_classes"]["where_errors_fall"] = {
        "recommendation errors on true medium": rate_with_interval(
            {"share": medium[rec_errors]}, p_both[rec_errors]
        )["share"],
        "band errors on true no_idea": rate_with_interval(
            {"share": no_idea[band_errors]}, p_both[band_errors]
        )["share"],
        "no_idea errors on true medium": rate_with_interval(
            {"share": medium[no_idea_errors]}, p_both[no_idea_errors]
        )["share"],
        "medium errors on true no_idea": rate_with_interval(
            {"share": no_idea[medium_errors]}, p_both[medium_errors]
        )["share"],
        "medium share among all reviews with both labels": float(medium.mean()),
        "no_idea share among all reviews with both labels": float(no_idea.mean()),
    }

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {
                "model": "FaBERT final run, saved labels with offsets",
                "intervals": "product-level cluster bootstrap, percentiles 2.5 and 97.5, "
                "1000 rounds, seed 2026",
                "verify_csv": args.verify_csv,
            },
            **result,
            "minutes": round((time.perf_counter() - start) / 60, 1),
            "code": git_commit(),
        },
    )
    summary(result)
    print(f"saved {OUTPUT} ({(time.perf_counter() - start) / 60:.1f} min)")


def v(x, form=".3f") -> str:
    return f"{x['value']:{form}} [{x['interval_95'][0]:{form}}, {x['interval_95'][1]:{form}}]"


def summary(result):
    c = result["data_checks"]
    print(
        f"data: {c['rates']['rows']:,} validation rows, {result['reviews_with_band']:,} with a band; "
        f"FaBERT Macro F1 {c['fabert_macro_f1']} (match ensemble.json)"
    )
    print("1. band by the buyer's recommendation label: Macro F1; recall low / medium / high [95%]:")
    for g, x in result["1_band_by_buyer_label"].items():
        mf = v(x["macro_f1"], ".4f") if "macro_f1" in x else "n/a"
        shares = " / ".join(f"{x['true_band_share'][b]:.2f}" for b in LABELS)
        recall = " / ".join(v(x[f"recall {b}"]) for b in LABELS)
        flag = "  (small)" if x["small"] else ""
        print(f"   {g:<24} rows {x['rows']:>8,}  band shares {shares}  Macro F1 {mf}")
        print(f"   {'':<24} recall {recall}{flag}")
    h = result["2_hard_classes"]
    print(f"2. reviews with both labels: {h['reviews_with_both_labels']:,}")
    print("   true recommendation x true band: rows; recommendation wrong / band wrong / both wrong [95%]:")
    for cell, x in h["by_true_labels"].items():
        flag = "  (small)" if x["small"] else ""
        print(
            f"   {cell:<26} rows {x['rows']:>8,}  {v(x['recommendation wrong'])} / {v(x['band wrong'])} / "
            f"{v(x['both wrong'])}{flag}"
        )
    w = h["where_errors_fall"]
    print(
        f"   base shares: medium {w['medium share among all reviews with both labels']:.3f}, "
        f"no_idea {w['no_idea share among all reviews with both labels']:.3f}"
    )
    for key in [
        "recommendation errors on true medium",
        "band errors on true no_idea",
        "no_idea errors on true medium",
        "medium errors on true no_idea",
    ]:
        print(f"   {key:<40} {v(w[key])}")
    d = h["band_wrong_given_recommendation_wrong"]
    print(
        f"   band wrong when recommendation wrong {v(d['given'])}, when right {v(d['not_given'])}, "
        f"difference {v(d['difference'], '+.3f')}"
    )


if __name__ == "__main__":
    main()
