"""Do buyers who rate 3 to 3.99 still recommend the product? (hypothesis H2')

The rating boundary analysis found that most medium mistakes are reviews rated exactly 3,
and that 41.8% of them are read as high. H2' says the text of these reviews reads positive.
The buyer's own recommendation is a label the buyer chose, apart from the rate, so it shows
whether the buyer meant the review as positive. This script reports, by rate:

1. the share of recommended, not_recommended and no_idea among reviews with a
   recommendation label (the buyer's choice)
2. for medium reviews (rate 2.5 to below 4): the same shares split by what FaBERT predicted
   for the band (low, medium or high), and the difference between predicted high and
   predicted medium
3. the share of reviews that FaBERT's recommendation head reads as recommended (the text
   alone), by rate, as a second view. The saved labels with offsets exist only for reviews
   with a buyer's label (the offsets are fitted on labelled reviews), so this part uses the
   same reviews as part 1; the script checks that the empty predictions are exactly the
   reviews without a label.

Only saved validation predictions of the final FaBERT run are read; nothing is trained. The
rate and the recommendation label are used only to group and describe the reviews, never as
inputs. The test set is not read.

Data and checks: the rate, product_id and band come from analyze_rating_boundaries.py's
load_validation and check_rates (cache hash, manifest counts, band follows from the rate,
optional comparison with validation.csv). The recommendation label is read from the same
cache file in the same row order; its values must be the known classes or empty, and its
counts must equal the split manifest. The predictions are joined by id one to one, and the
band labels must score the Macro F1 recorded in reports/ensemble.json.

Intervals: product-level cluster bootstrap, 1000 rounds, seed 2026, as in the rating boundary
analysis. A difference between two subgroups is computed on the same draw of products.
Groups with fewer than 1000 reviews or 100 products are marked "small".

Writes reports/rate_recommendation.json and prints a summary.

Usage:
    uv run python scripts/analyze_rate_recommendation.py
    uv run python scripts/analyze_rate_recommendation.py --verify-csv
"""

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from analyze_rating_boundaries import (
    CACHE_FILE,
    LABELS,
    SPLIT_FILE,
    check_rates,
    cluster_bootstrap,
    load_validation,
    rate_group,
    size,
)

from digikala_reviews.config import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    FINAL_TRANSFORMER_DIR,
    RECOMMENDATION_CLASSES,
    REPORTS_DIR,
)
from digikala_reviews.metrics import classification_report
from digikala_reviews.provenance import git_commit
from digikala_reviews.standard_files import read_split_file
from digikala_reviews.tuning import save_json

BAND = "rating_band"
REC = "recommendation_status"  # the column in the data
REC_HEAD = "recommendation"  # the head's name in the prediction files
OUTPUT = REPORTS_DIR / "rate_recommendation.json"


# ---------------------------------------------------------------- data


def load_recommendation(
    validation: pd.DataFrame, verify_csv: bool, manifest_counts: dict
) -> tuple[np.ndarray, dict]:
    """The buyer's recommendation label of every validation row, in the cache's row order."""
    assert REC in pq.read_schema(CACHE_DIR / CACHE_FILE).names, f"{CACHE_FILE} has no {REC} column"
    table = pd.read_parquet(CACHE_DIR / CACHE_FILE, columns=["id", REC])
    assert (table["id"].to_numpy() == validation["id"].to_numpy()).all(), (
        "row order differs from the band data"
    )
    labels = table[REC]
    known = labels.isna() | labels.isin(RECOMMENDATION_CLASSES)
    assert known.all(), f"unknown {REC} values: {sorted(labels[~known].unique())[:10]}"
    counts = {label: int((labels == label).sum()) for label in RECOMMENDATION_CLASSES}
    counts["missing"] = int(labels.isna().sum())
    for key, value in counts.items():
        assert value == manifest_counts[key], f"{REC} {key}: {value:,}, manifest {manifest_counts[key]:,}"
    record = {"counts": counts}
    if verify_csv:
        csv = read_split_file(SPLIT_FILE)[["id", REC]]
        csv = table[["id"]].merge(csv, on="id", how="left", validate="one_to_one")
        a, b = labels, csv[REC]
        record["cache_vs_csv_mismatches"] = int((~((a == b) | (a.isna() & b.isna()))).sum())
        assert record["cache_vs_csv_mismatches"] == 0, "cache and csv differ in the recommendation label"
    return labels.to_numpy(dtype=object), record


def load_fabert(validation: pd.DataFrame) -> tuple[dict, dict]:
    """FaBERT's saved validation labels for both heads, joined to validation by id."""
    path = ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "validation_predictions.parquet"
    columns = ["id", f"{BAND}__predicted_with_offsets", f"{REC_HEAD}__predicted_with_offsets"]
    absent = [c for c in columns if c not in pq.read_schema(path).names]
    assert not absent, f"{path} lacks {absent}"
    table = pd.read_parquet(path, columns=columns)
    assert table["id"].is_unique, "duplicate ids in the predictions"
    both = validation[["id"]].merge(
        table[["id"]], on="id", how="outer", indicator=True, validate="one_to_one"
    )
    join = {
        "rows": int(len(table)),
        "only_in_validation": int((both["_merge"] == "left_only").sum()),
        "only_in_predictions": int((both["_merge"] == "right_only").sum()),
    }
    assert join["only_in_validation"] == 0 and join["only_in_predictions"] == 0, f"join: {join}"
    table = validation[["id"]].merge(table, on="id", how="left", validate="one_to_one")
    assert (table["id"].to_numpy() == validation["id"].to_numpy()).all(), "the join changed the row order"
    predicted = {
        BAND: table[f"{BAND}__predicted_with_offsets"].to_numpy(dtype=object),
        REC: table[f"{REC_HEAD}__predicted_with_offsets"].to_numpy(dtype=object),
    }
    return predicted, join


# ---------------------------------------------------------------- group reports


def shares(labels, classes, products) -> dict:
    """Share of each class in a group, with product-level bootstrap intervals."""
    labels = np.asarray(labels, dtype=object)
    result = size(len(labels), products)
    hit = {c: (labels == c).astype(float) for c in classes}

    def statistic(w):
        return {c: float((w * h).sum() / w.sum()) for c, h in hit.items()}

    interval = cluster_bootstrap(products, statistic)
    for c in classes:
        result[c] = {"share": float(hit[c].mean()), "interval_95": interval[c]}
    return result


def recommended_difference(labels, in_a, in_b, products) -> dict:
    """Share recommended in subgroup a minus subgroup b, on the same draws of products."""
    keep = in_a | in_b  # only the products of the two subgroups are drawn
    rec = (np.asarray(labels, dtype=object)[keep] == "recommended").astype(float)
    a, b = in_a[keep].astype(float), in_b[keep].astype(float)

    def statistic(w):
        wa, wb = w * a, w * b
        share_a = (wa * rec).sum() / wa.sum() if wa.sum() > 0 else np.nan
        share_b = (wb * rec).sum() / wb.sum() if wb.sum() > 0 else np.nan
        return {"difference": float(share_a - share_b)}

    interval = cluster_bootstrap(np.asarray(products)[keep], statistic)
    return {"difference": statistic(np.ones(len(rec)))["difference"], "interval_95": interval["difference"]}


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
    buyer, rec_record = load_recommendation(validation, args.verify_csv, counts[REC])
    predicted, join = load_fabert(validation)
    # train_transformer.py saves a label with offsets only where the head has a true label
    for name, truth in [(BAND, validation[BAND].to_numpy()), (REC, buyer)]:
        assert (pd.isna(predicted[name]) == pd.isna(truth)).all(), (
            f"FaBERT {name} predictions are empty for other reviews than those without a label"
        )

    rows = validation[BAND].notna().to_numpy()  # reviews with a valid rate
    rate = validation["rate"].to_numpy(dtype=float)[rows]
    band = validation[BAND].to_numpy()[rows].astype(str)
    products = validation["product_id"].to_numpy()[rows]
    buyer = buyer[rows]
    band_pred = predicted[BAND][rows].astype(str)
    rec_pred = predicted[REC][rows]  # empty where the buyer gave no label
    labelled = pd.notna(buyer)
    for name, values, classes in [
        ("band", band_pred, LABELS),
        ("recommendation", rec_pred[labelled].astype(str), RECOMMENDATION_CLASSES),
    ]:
        unknown = sorted(set(np.unique(values)) - set(classes))
        assert not unknown, f"FaBERT {name} predictions have unknown labels: {unknown[:10]}"

    # the band labels must be the ones the final run scored
    saved = json.loads((REPORTS_DIR / "ensemble.json").read_text(encoding="utf-8"))
    saved = saved["final_fabert_and_tfidf"][BAND]["fabert_macro_f1"]
    found = classification_report(band, band_pred, LABELS)["macro_f1"]
    assert abs(found - saved) < 1e-9, f"FaBERT band labels score {found}, ensemble.json {saved}"

    groups = rate_group(rate)
    medium = band == "medium"
    whole = np.isclose(rate, np.round(rate))
    # medium holds one whole rate (3) and fractional rates on both sides of it
    span = np.select(
        [medium & whole, medium & ~whole & (rate < 3.0), medium & ~whole & (rate > 3.0)],
        ["whole 3", "fractional [2.50, 3.00)", "fractional [3.00, 4.00)"],
        default="",
    )
    assert (span[medium] != "").all() and (span[~medium] == "").all()

    result = {
        "reviews_with_band": int(rows.sum()),
        "with_recommendation_label": int(labelled.sum()),
        "data_checks": {
            "source": source,
            "rates": rate_checks,
            "recommendation": rec_record,
            "prediction_join": join,
            "fabert_band_macro_f1": found,
        },
        "1_buyer_label_by_rate": {},
        "2_medium_by_band_prediction": {},
        "3_text_read_as_recommended_by_rate": {},
    }
    for group in sorted(pd.unique(groups)):
        inside = groups == group
        part = inside & labelled
        result["1_buyer_label_by_rate"][str(group)] = {
            "without_label": int((inside & ~labelled).sum()),
            **shares(buyer[part], RECOMMENDATION_CLASSES, products[part]),
        }
        result["3_text_read_as_recommended_by_rate"][str(group)] = shares(
            rec_pred[part].astype(str), RECOMMENDATION_CLASSES, products[part]
        )
    for name in ["whole 3", "fractional [2.50, 3.00)", "fractional [3.00, 4.00)"]:
        inside = (span == name) & labelled
        part = {}
        for label in LABELS:
            mine = inside & (band_pred == label)
            part[f"predicted {label}"] = shares(buyer[mine], RECOMMENDATION_CLASSES, products[mine])
        part["recommended: predicted high minus predicted medium"] = recommended_difference(
            buyer[inside],
            band_pred[inside] == "high",
            band_pred[inside] == "medium",
            products[inside],
        )
        result["2_medium_by_band_prediction"][name] = part

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {
                "model": f"FaBERT final run ({FINAL_TRANSFORMER_DIR}), saved labels with offsets",
                "intervals": "product-level cluster bootstrap, percentiles 2.5 and 97.5, "
                "1000 rounds, seed 2026",
                "buyer_label": "reviews without a recommendation label are left out of part 1 and 2",
                "verify_csv": args.verify_csv,
            },
            **result,
            "minutes": round((time.perf_counter() - start) / 60, 1),
            "code": git_commit(),
        },
    )
    summary(result)
    print(f"saved {OUTPUT} ({(time.perf_counter() - start) / 60:.1f} min)")


def share_text(g) -> str:
    return " / ".join(
        f"{g[c]['share']:.3f} [{g[c]['interval_95'][0]:.3f}, {g[c]['interval_95'][1]:.3f}]"
        for c in RECOMMENDATION_CLASSES
    )


def summary(result):
    c = result["data_checks"]
    print(
        f"data: {c['rates']['rows']:,} validation rows, {result['reviews_with_band']:,} with a band, "
        f"{result['with_recommendation_label']:,} of them with a recommendation label"
    )
    print(f"      recommendation counts {c['recommendation']['counts']}")
    if "cache_vs_csv_mismatches" in c["recommendation"]:
        print(
            f"      recommendation, cache against validation.csv, values that differ: "
            f"{c['recommendation']['cache_vs_csv_mismatches']}"
        )
    print(f"      FaBERT band Macro F1 {c['fabert_band_macro_f1']:.4f} (matches ensemble.json)")
    for key, title in [
        ("1_buyer_label_by_rate", "1. buyer's label by rate: recommended / not_recommended / no_idea [95%]"),
        (
            "3_text_read_as_recommended_by_rate",
            "3. FaBERT recommendation head by rate: recommended / not / no_idea [95%]",
        ),
    ]:
        print(title + ":")
        for group, g in result[key].items():
            flag = "  (small)" if g["small"] else ""
            print(f"   {group:<26} rows {g['rows']:>8,} products {g['products']:>6,}  {share_text(g)}{flag}")
    print("2. medium reviews, buyer's label by FaBERT's band prediction: recommended / not / no_idea [95%]:")
    for name, part in result["2_medium_by_band_prediction"].items():
        for label in LABELS:
            g = part[f"predicted {label}"]
            flag = "  (small)" if g["small"] else ""
            print(
                f"   {name:<24} predicted {label:<6} rows {g['rows']:>7,} products {g['products']:>6,}  "
                f"{share_text(g)}{flag}"
            )
        d = part["recommended: predicted high minus predicted medium"]
        print(
            f"   {name:<24} recommended, high minus medium {d['difference']:+.3f} "
            f"[{d['interval_95'][0]:+.3f}, {d['interval_95'][1]:+.3f}]"
        )


if __name__ == "__main__":
    main()
