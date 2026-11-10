"""The optional extensions scored on the whole validation set (plan: reports/extensions_plan.md).

1. Exact rate: FaBERT's rate head, the TF-IDF ridge regression and a constant (the training mean),
   on the validation reviews with a real rate: MAE, RMSE, MSE, R2, Spearman; the residuals by true
   rate, by recommendation class and by review length; a 2D histogram of predicted against true
   rate (for the plot). The rate model's two classification heads against the kept FaBERT run.
2. Frozen encoder: against FaBERT and TF-IDF.
3 to 5. The students at 0.80 and 0.95, with band guesses and with the natural pool, at the end of
   their epoch, against the supervised model, the control and the official student (both at the end
   of their epoch); the rule of the official plan (more than 0.12 point over the control, interval
   above zero). The quality of the band guesses against the reference labels (read only here, after
   training), and how many natural reviews were accepted.

Every classification model must first score the Macro F1 recorded for it. Labels with offsets
(cross-fitted, as saved); differences with intervals that draw products (1,000 rounds, seed 2026).
The test set is not read: scripts/final_evaluation_extensions.py does that, once.

Writes reports/extensions_validation.json and prints a summary.

Usage:
    uv run python scripts/check_extensions.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_rating_boundaries import load_validation  # noqa: E402
from check_semi_supervised import scores  # noqa: E402
from recheck_decisions import confusion, macro_f1  # noqa: E402

from digikala_reviews.analysis import length_bucket, word_count  # noqa: E402
from digikala_reviews.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CACHE_DIR,
    FINAL_TRANSFORMER_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
)
from digikala_reviews.provenance import check_cache, git_commit  # noqa: E402
from digikala_reviews.pseudo import quality  # noqa: E402
from digikala_reviews.standard_files import read_split_file  # noqa: E402
from digikala_reviews.tfidf import HEADS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

OUTPUT = REPORTS_DIR / "extensions_validation.json"
ROUNDS, SEED, MIN_LEAD = 1000, 2026, 0.0012
# name: (folder, predictions file, report, key of the recorded scores in it)
MODELS = {
    "tfidf": ("tfidf_final", "validation_predictions.parquet", None, None),
    "fabert": (
        FINAL_TRANSFORMER_DIR,
        "validation_predictions.parquet",
        "transformer_final/fabert_no_early_stop.json",
        "heads",
    ),
    "rate model": (
        "transformer_rate",
        "validation_predictions.parquet",
        "transformer_final/rate.json",
        "heads",
    ),
    "frozen": (
        "transformer_frozen",
        "validation_predictions.parquet",
        "transformer_final/frozen.json",
        "heads",
    ),
    "control": (
        "semi_supervised_control",
        "validation_predictions_last.parquet",
        "semi_supervised/control.json",
        "last_check",
    ),
    "student 0.90": (
        "semi_supervised_student",
        "validation_predictions_last.parquet",
        "semi_supervised/student.json",
        "last_check",
    ),
    "student 0.80": (
        "semi_supervised_student_t080",
        "validation_predictions_last.parquet",
        "semi_supervised/student_t080.json",
        "last_check",
    ),
    "student 0.95": (
        "semi_supervised_student_t095",
        "validation_predictions_last.parquet",
        "semi_supervised/student_t095.json",
        "last_check",
    ),
    "student both heads": (
        "semi_supervised_student_both",
        "validation_predictions_last.parquet",
        "semi_supervised/student_both.json",
        "last_check",
    ),
    "student natural": (
        "semi_supervised_student_natural",
        "validation_predictions_last.parquet",
        "semi_supervised/student_natural.json",
        "last_check",
    ),
}
PAIRS = [
    ("rate model", "fabert"),
    ("frozen", "fabert"),
    ("frozen", "tfidf"),
    ("student 0.80", "control"),
    ("student 0.95", "control"),
    ("student both heads", "control"),
    ("student natural", "control"),
    ("student 0.80", "student 0.90"),
    ("student 0.95", "student 0.90"),
    ("student both heads", "student 0.90"),
    ("student natural", "student 0.90"),
    ("student 0.80", "fabert"),
    ("student 0.95", "fabert"),
    ("student both heads", "fabert"),
    ("student natural", "fabert"),
]
STUDENTS = ["student 0.80", "student 0.95", "student both heads", "student natural"]
RATE_BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 5.01)]  # true rate, low end included


def read(relative) -> dict:
    return json.loads((REPORTS_DIR / relative).read_text(encoding="utf-8"))


def rate_scores(true, predicted) -> dict:
    error = predicted - true
    mse = float(np.mean(error**2))
    return {
        "reviews": int(len(true)),
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1 - mse / np.var(true)),
        "spearman": None if np.ptp(predicted) == 0 else float(spearmanr(true, predicted).statistic),
    }


def residuals_by(true, predicted, groups) -> dict:
    found = {}
    for group in pd.unique(groups):
        inside = groups == group
        error = predicted[inside] - true[inside]
        found[str(group)] = {
            "reviews": int(inside.sum()),
            "mean_residual": float(error.mean()),
            "mae": float(np.abs(error).mean()),
        }
    return found


def rate_bin(true) -> np.ndarray:
    names = np.empty(len(true), dtype=object)
    for low, high in RATE_BINS:
        names[(true >= low) & (true < high)] = f"{low:g}" if high - low < 1 else f"{low:g} to {high:g}"
    return names


def histogram(true, predicted) -> dict:
    counts, x_edges, y_edges = np.histogram2d(
        true, predicted, bins=[np.arange(0.75, 5.5, 0.5), np.arange(0, 5.26, 0.25)]
    )
    return {
        "true_edges": x_edges.tolist(),
        "predicted_edges": y_edges.tolist(),
        "counts": counts.astype(int).tolist(),
    }


def rate_section(table, predictions, train_mean) -> dict:
    rated = table["rate"].to_numpy(dtype=float) > 0
    true = table["rate"].to_numpy(dtype=float)[rated]
    lengths = length_bucket(word_count(table["text"].to_numpy()[rated]))
    classes = table["recommendation_status"].fillna("none").to_numpy()[rated]
    found = {}
    predictions = {**predictions, "constant (training mean)": np.full(len(table), train_mean)}
    for name, predicted in predictions.items():
        p = predicted[rated]
        found[name] = {
            "scores": rate_scores(true, p),
            "residuals_by_true_rate": residuals_by(true, p, rate_bin(true)),
            "residuals_by_recommendation": residuals_by(true, p, classes),
            "residuals_by_length": residuals_by(true, p, lengths),
        }
        if not name.startswith("constant"):
            found[name]["histogram"] = histogram(true, p)
    return found


def classification(table, tables, records, product_index, pairs=PAIRS) -> dict:
    heads = {}
    for head, spec in HEADS.items():
        labels = spec["labels"]
        rows = table[spec["column"]].notna().to_numpy()
        y_true = table[spec["column"]].to_numpy()[rows].astype(str)
        codes = np.array([labels.index(v) for v in y_true])
        entry, predicted = {"models": {}}, {}
        for name, t in tables.items():
            found = t[f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str)
            entry["models"][name] = scores(y_true, found, labels, head)
            if records.get(name) is not None:
                recorded = records[name][head]["with_offsets"]["macro_f1"]
                assert abs(entry["models"][name]["macro_f1"] - recorded) < 1e-9, (
                    f"{head} {name}: Macro F1 differs"
                )
            predicted[name] = np.array([labels.index(v) for v in found])
        _, products = np.unique(product_index[rows], return_inverse=True)
        n_products = int(products.max()) + 1
        rng = np.random.default_rng(SEED)
        draws = {pair: [] for pair in pairs}
        for _ in range(ROUNDS):
            w = np.bincount(rng.integers(0, n_products, n_products), minlength=n_products)[products].astype(
                float
            )
            f1 = {name: macro_f1(confusion(codes, p, w, len(labels))) for name, p in predicted.items()}
            for a, b in pairs:
                draws[(a, b)].append(f1[a] - f1[b])
        entry["differences"] = {}
        for (a, b), found in draws.items():
            entry["differences"][f"{a} minus {b}"] = {
                "difference": entry["models"][a]["macro_f1"] - entry["models"][b]["macro_f1"],
                "interval_95_products": np.percentile(found, [2.5, 97.5]).tolist(),
            }
        heads[head] = entry
    return heads


def print_rate(rate) -> None:
    for name, r in rate.items():
        s = r["scores"]
        spearman = "-" if s["spearman"] is None else f"{s['spearman']:.3f}"
        print(f"  {name:<26} MAE {s['mae']:.3f}  RMSE {s['rmse']:.3f}  R2 {s['r2']:.3f}  Spearman {spearman}")


def main():
    rates, _ = load_validation(verify_csv=False)
    table = load_cache("validation.parquet", extra_columns=["product_id"]).merge(
        rates[["id", "rate"]], on="id", how="left", validate="one_to_one"
    )
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    tables, records = {}, {}
    for name, (folder, file, report, key) in MODELS.items():
        found = pd.read_parquet(ARTIFACTS_DIR / folder / file)
        assert found["id"].is_unique and set(found["id"]) == set(table["id"]), f"{name}: ids differ"
        tables[name] = table[["id"]].merge(found, on="id", how="left", validate="one_to_one")
        records[name] = read(report)[key] if report else None
        if key == "last_check":
            records[name] = records[name]["heads"]
    ridge = read("extensions/rate_ridge.json")
    ridge_table = table[["id"]].merge(
        pd.read_parquet(ARTIFACTS_DIR / "tfidf_final" / "rate_validation_predictions.parquet"),
        on="id",
        how="left",
    )
    rate = rate_section(
        table,
        {
            "FaBERT rate head": tables["rate model"]["rate__predicted"].to_numpy(dtype=float),
            "TF-IDF ridge": ridge_table["rate__predicted"].to_numpy(dtype=float),
        },
        ridge["training_mean_rate"],
    )
    _, product_index = np.unique(table["product_id"].to_numpy(), return_inverse=True)
    heads = classification(table, tables, records, product_index)

    band = pd.read_parquet(REPORTS_DIR / "semi_supervised" / "pseudo_labels_band.parquet")
    reference = read_split_file(SPLITS_DIR / "audit" / "masked_labels_reference.csv")
    band = band.merge(reference[["id", "rating_band"]], on="id", how="left", validate="one_to_one")
    band = band[band["rating_band"].notna()]
    band_quality = quality(
        band["label"], band["p"], band["rating_band"], [0.8, 0.9, 0.95], HEADS["rating_band"]["labels"]
    )
    band_quality["reference_class_shares"] = band["rating_band"].value_counts(normalize=True).to_dict()
    natural = read("semi_supervised/teacher_natural.json")
    rule = {}
    for name in STUDENTS:
        lead = heads["recommendation"]["differences"][f"{name} minus control"]
        rule[name] = bool(lead["interval_95_products"][0] > 0 and lead["difference"] > MIN_LEAD)

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/extensions_plan.md",
            "rate": rate,
            "heads": heads,
            "band_guesses": {"hidden_reviews_with_a_reference_band": int(len(band)), "quality": band_quality},
            "natural_guesses": {k: natural[k] for k in ("reviews", "share_at_least_0.9", "guessed_classes")},
            "pseudo_labeled_reviews": {
                name: read(MODELS[name][2])["pseudo_labeled_reviews"] for name in STUDENTS
            },
            "rule_pseudo_labels_help": rule,
            "data": data,
            "code": git_commit(),
        },
    )
    print("=== exact rate (validation, reviews with a real rate)")
    print_rate(rate)
    for head, e in heads.items():
        print(f"\n=== {head}, Macro F1 with offsets")
        for name, m in e["models"].items():
            print(f"  {name:<20} {m['macro_f1']:.4f}")
        for pair, d in e["differences"].items():
            low, high = d["interval_95_products"]
            print(f"  {pair}: {d['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
    q = band_quality["0.9"]
    print(
        f"\nband guesses at 0.9: kept {q['share_kept']:.1%}, right {q['accuracy']:.1%}, "
        f"classes {q['kept_classes']}"
    )
    print(
        f"natural pool: {natural['share_at_least_0.9']:.1%} reach 0.9, classes {natural['guessed_classes']}"
    )
    print(
        "pseudo-labels help by the rule: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in rule.items())
    )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
