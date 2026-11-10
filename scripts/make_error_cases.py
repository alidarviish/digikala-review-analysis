"""Pick validation mistakes of the final system in the categories the project names, for the
error analysis of the final report (reports/final_report.md).

Every pick is a review the final system (FaBERT and TF-IDF mixed, labels with offsets tuned on
the other product half, as in step 60) gets wrong. Three per category, chosen with seed 2026
from the reviews that fit its rule, in this order; a review, a text or a product is used once:

  negation       recommendation wrong; the text has a negation word (CATEGORY_WORDS)
  sarcasm        true not_recommended, predicted recommended; the text has a strongly positive word
  short          recommendation wrong; at most 3 words
  mixed          recommendation wrong; the text has "ولی" or "اما"
  spelling       recommendation wrong; the text has a misspelling of resources/spelling_fixes.csv
  duplicate      recommendation wrong; the same text is also in train (text_seen_in_train)
  contradictory  recommendation wrong; the same text is in validation with another recommendation
  boundary       rating band wrong by one band; the rate is within 0.25 of a band edge
  disagreement   TF-IDF and FaBERT disagree on the recommendation and one of them is right:
                 two where FaBERT is right, two where TF-IDF is right (the final system may be either)

Texts are shown normalized (as the models read them) and only if they have at most 40 words.
For each review: both labels, the rate, the label and probability of TF-IDF, FaBERT and the
final system for both heads, and the words that pushed TF-IDF most towards its answer.

Writes reports/error_cases.json and reports/error_cases.md. The test set is not read.

Usage:
    uv run python scripts/make_error_cases.py
"""

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_rating_boundaries import load_validation  # noqa: E402

from digikala_reviews.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CACHE_DIR,
    FINAL_TRANSFORMER_DIR,
    PROJECT_ROOT,
    RATING_BAND_EDGES,
    REPORTS_DIR,
)
from digikala_reviews.ensemble import blend, cross_fitted_blend  # noqa: E402
from digikala_reviews.experiments import product_half  # noqa: E402
from digikala_reviews.metrics import macro_f1_from_codes  # noqa: E402
from digikala_reviews.predictor import top_words  # noqa: E402
from digikala_reviews.provenance import git_commit  # noqa: E402
from digikala_reviews.tfidf import HEADS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

SEED, PER_CATEGORY, MAX_WORDS = 2026, 3, 40
NEGATION = {
    "نه",
    "نیست",
    "نبود",
    "نیستم",
    "نداره",
    "ندارد",
    "نداشت",
    "نکرد",
    "نکنه",
    "نمیکنه",
    "نخرید",
    "نبودم",
}
POSITIVE = {"عالی", "عالیه", "بهترین", "فوق‌العاده", "فوق", "محشر", "محشره", "عالی‌ه"}
CONTRAST = {"ولی", "اما"}
CATEGORIES = [
    "negation",
    "sarcasm",
    "short",
    "mixed",
    "spelling",
    "duplicate",
    "contradictory",
    "boundary",
    "disagreement",
]


def predictions(validation, recorded):
    """For each head: codes of TF-IDF and FaBERT (saved labels with offsets), of the final
    system (cross-fitted as in step 60), and the probabilities of all three."""
    found = {}
    tables = {
        name: validation[["id"]].merge(
            pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet"), on="id", how="left"
        )
        for name, folder in [("tfidf", "tfidf_final"), ("fabert", FINAL_TRANSFORMER_DIR)]
    }
    for head, spec in HEADS.items():
        labels = spec["labels"]
        rows = validation[spec["column"]].notna().to_numpy()
        codes = np.array([labels.index(v) for v in validation[spec["column"]].to_numpy()[rows]])
        log = {
            n: t[[f"{head}__log_proba__{c}" for c in labels]].to_numpy(dtype=float)[rows]
            for n, t in tables.items()
        }
        half_a = product_half(validation["product_id"].to_numpy()[rows])
        final_codes, weights = cross_fitted_blend(log["fabert"], log["tfidf"], codes, half_a)
        found_f1 = macro_f1_from_codes(codes, final_codes, len(labels))
        assert abs(found_f1 - recorded[head]["blend_macro_f1"]) < 1e-9, (
            f"{head}: the final system's labels do not give step 60's Macro F1"
        )
        mixed = np.empty_like(log["fabert"])
        for name, tuned_on in [("tuned_on_a", half_a), ("tuned_on_b", ~half_a)]:
            mixed[~tuned_on] = blend(log["fabert"][~tuned_on], log["tfidf"][~tuned_on], weights[name])
        head_found = {"rows": rows, "true": codes, "final": final_codes, "proba": {"final": np.exp(mixed)}}
        for n, t in tables.items():
            saved = t[f"{head}__predicted_with_offsets"].to_numpy()[rows]
            head_found[n] = np.array([labels.index(v) for v in saved])
            head_found["proba"][n] = np.exp(log[n])
        found[head] = head_found
    return found


def per_review(found, n_reviews):
    """Head results spread back to one row per validation review (-1 where a head has no label)."""
    out = {}
    for head, f in found.items():
        full = {k: np.full(n_reviews, -1) for k in ("true", "final", "tfidf", "fabert")}
        for k in full:
            full[k][f["rows"]] = f[k]
        proba = {}
        for k, p in f["proba"].items():
            proba[k] = np.full((n_reviews, p.shape[1]), np.nan)
            proba[k][f["rows"]] = p
        out[head] = {**full, "proba": proba}
    return out


def main():
    ensemble = json.loads((REPORTS_DIR / "ensemble.json").read_text(encoding="utf-8"))
    recorded = ensemble["final_fabert_and_tfidf"]
    rates, _ = load_validation(verify_csv=False)
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    seen = pd.read_parquet(CACHE_DIR / "validation.parquet", columns=["id", "text_seen_in_train"])
    validation = validation.merge(rates[["id", "rate"]], on="id", how="left", validate="one_to_one")
    validation = validation.merge(seen, on="id", how="left", validate="one_to_one")
    found = per_review(predictions(validation, recorded), len(validation))
    rec, band = found["recommendation"], found["rating_band"]

    words = validation["text"].str.split()
    tokens = words.map(set)
    n_words = words.map(len).to_numpy()
    with open(PROJECT_ROOT / "resources" / "spelling_fixes.csv", encoding="utf-8") as f:
        misspelled = {r["wrong"] for r in csv.DictReader(f) if r["decision"] == "fix"}
    labels_of_text = (
        validation[validation["recommendation_status"].notna()]
        .groupby("text")["recommendation_status"]
        .nunique()
    )
    conflicting = validation["text"].map(labels_of_text).fillna(0).to_numpy() > 1
    negated = tokens.map(lambda t: bool(t & NEGATION) or any(w.startswith("نمی") for w in t)).to_numpy()
    rec_wrong = (rec["true"] >= 0) & (rec["final"] != rec["true"])
    rate = validation["rate"].to_numpy(dtype=float)
    near_edge = np.zeros(len(validation), dtype=bool)
    for edge in RATING_BAND_EDGES:
        near_edge |= np.abs(rate - edge) <= 0.25
    rules = {
        "negation": rec_wrong & negated,
        "sarcasm": (rec["true"] == 1)
        & (rec["final"] == 0)
        & tokens.map(lambda t: bool(t & POSITIVE)).to_numpy(),
        "short": rec_wrong & (n_words <= 3),
        "mixed": rec_wrong & tokens.map(lambda t: bool(t & CONTRAST)).to_numpy(),
        "spelling": rec_wrong & tokens.map(lambda t: bool(t & misspelled)).to_numpy(),
        "duplicate": rec_wrong & validation["text_seen_in_train"].to_numpy(dtype=bool),
        "contradictory": rec_wrong & conflicting,
        "boundary": (band["true"] >= 0) & (np.abs(band["final"] - band["true"]) == 1) & near_edge,
    }
    tfidf_right = (rec["true"] >= 0) & (rec["tfidf"] == rec["true"]) & (rec["fabert"] != rec["true"])
    fabert_right = (rec["true"] >= 0) & (rec["fabert"] == rec["true"]) & (rec["tfidf"] != rec["true"])

    rng = np.random.default_rng(SEED)
    used_ids, used_texts, used_products, picks = set(), set(), set(), []
    readable = n_words <= MAX_WORDS

    def take(mask, category, k, note=""):
        candidates = np.flatnonzero(mask & readable)
        rng.shuffle(candidates)
        taken = 0
        for i in candidates:
            row = validation.iloc[i]
            if row["id"] in used_ids or row["text"] in used_texts or row["product_id"] in used_products:
                continue
            used_ids.add(row["id"]), used_texts.add(row["text"]), used_products.add(row["product_id"])
            picks.append((category, int(i), note, int(mask.sum())))
            taken += 1
            if taken == k:
                return
        raise SystemExit(f"{category}: only {taken} reviews fit")

    for category in CATEGORIES[:-1]:
        take(rules[category], category, PER_CATEGORY)
    take(fabert_right, "disagreement", 2, "FaBERT right, TF-IDF wrong")
    take(tfidf_right, "disagreement", 2, "TF-IDF right, FaBERT wrong")

    tfidf = joblib.load(ARTIFACTS_DIR / "tfidf_final" / "model.joblib")
    names = tfidf["vectorizer"].get_feature_names_out()
    cases = []
    for category, i, note, pool in picks:
        row = validation.iloc[i]
        x = tfidf["vectorizer"].transform([row["text"]])
        case = {
            "category": category,
            "note": note,
            "reviews_fitting_the_rule": pool,
            "id": int(row["id"]),
            "text": row["text"],
            "rate": None if np.isnan(rate[i]) else float(rate[i]),
            "seen_in_train": bool(row["text_seen_in_train"]),
            "heads": {},
        }
        for head, spec in HEADS.items():
            labels, h = spec["labels"], found[head]
            if h["true"][i] < 0:
                case["heads"][head] = {"true": None}
                continue
            model = tfidf["models"][head]
            order = [list(model.classes_.astype(str)).index(c) for c in labels]
            entry = {"true": labels[h["true"][i]]}
            for name in ("tfidf", "fabert", "final"):
                code = h[name][i]
                entry[name] = {"label": labels[code], "p": round(float(h["proba"][name][i][code]), 3)}
            entry["tfidf_words"] = [
                w["term"] for w in top_words(x[0], model.coef_[order], names, h["tfidf"][i], n=4)
            ]
            case["heads"][head] = entry
        cases.append(case)

    save_json(
        REPORTS_DIR / "error_cases.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": SEED,
            "rules": {
                "negation_words": sorted(NEGATION),
                "positive_words": sorted(POSITIVE),
                "max_words": MAX_WORDS,
            },
            "cases": cases,
            "code": git_commit(),
        },
    )
    lines = ["# Error cases (validation, final system)", "", "Made by `scripts/make_error_cases.py`.", ""]
    for n, c in enumerate(cases, 1):
        lines.append(f"## {n}. {c['category']}{' (' + c['note'] + ')' if c['note'] else ''}")
        lines.append("")
        lines.append(f"> {c['text']}")
        lines.append("")
        lines.append(f"rate {c['rate']}; text seen in train: {'yes' if c['seen_in_train'] else 'no'}")
        lines.append("")
        lines.append("| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |")
        lines.append("|---|---|---|---|---|---|")
        for head, e in c["heads"].items():
            if e["true"] is None:
                lines.append(f"| {head} | none | | | | |")
                continue
            cells = [f"{e[m]['label']} ({e[m]['p']:.2f})" for m in ("tfidf", "fabert", "final")]
            lines.append(
                f"| {head} | {e['true']} | " + " | ".join(cells) + f" | {', '.join(e['tfidf_words'])} |"
            )
        lines.append("")
    (REPORTS_DIR / "error_cases.md").write_text("\n".join(lines), encoding="utf-8")
    for category in CATEGORIES:
        pool = next(p for c, _, _, p in picks if c == category)
        print(f"  {category:<14} {sum(c == category for c, *_ in picks)} picked from {pool:,}")
    print(f"saved {REPORTS_DIR / 'error_cases.json'} and error_cases.md")


if __name__ == "__main__":
    main()
