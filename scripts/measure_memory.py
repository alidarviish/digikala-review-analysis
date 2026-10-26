"""Measure how much memory and time each TF-IDF setup needs, before any run on the full data.

Each setup is trained on 100k, 200k and 400k training reviews (recommendation head),
every run in its own fresh process so the memory numbers do not mix. The growth from
100k to 400k is then used to predict the need for the full training set.

Also compares float32 with float64 on one setup, and each lighter setup with the plain
word 1-2-gram setup on the same validation reviews, with a paired bootstrap interval for
the difference in Macro F1. Writes reports/memory_measurements.json.

Usage:
    uv run python scripts/measure_memory.py
"""

import json
import math
import multiprocessing
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR
from digikala_reviews.resources import available_memory_gb, require_free_memory

SAMPLE_SEED = 2026
SIZES = [100_000, 200_000, 400_000]
VALIDATION_ROWS = 100_000
SAFE_LIMIT_GB = 3.0  # a full run is allowed only below this predicted peak
# About 1 million columns, close to the vocabulary size of the plain setup on the full data.
# More columns mean fewer collisions, but logistic regression keeps several weight-sized
# vectors in memory, so 2**21 alone would add about 1.5 GB.
HASH_FEATURES = 2**20
SETUPS = {
    "word 1-gram": {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2},
    "word 1-2-gram": {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2},
    "word 1-2-gram, min_df 5": {"analyzer": "word", "ngram_range": (1, 2), "min_df": 5},
    "hashing word 1-2-gram": {"hashing": True, "analyzer": "word", "ngram_range": (1, 2)},
    "char_wb 2-5-gram": {"analyzer": "char_wb", "ngram_range": (2, 5), "min_df": 2},
    "char_wb 3-4-gram": {"analyzer": "char_wb", "ngram_range": (3, 4), "min_df": 2},
}
REFERENCE = "word 1-2-gram"  # the lighter setups are compared with this one
BOOTSTRAP_ROUNDS = 300
TRAIN_SAMPLE = CACHE_DIR / "memory_sample_train.parquet"
VALIDATION_SAMPLE = CACHE_DIR / "memory_sample_validation.parquet"


def make_samples():
    """Shuffled samples of labeled reviews; the first n rows of a sample are a random sample of size n."""
    for source, out, rows in [
        ("train_supervised.parquet", TRAIN_SAMPLE, max(SIZES)),
        ("validation.parquet", VALIDATION_SAMPLE, VALIDATION_ROWS),
    ]:
        table = pd.read_parquet(CACHE_DIR / source, columns=["text", "recommendation_status"])
        table = table[table["recommendation_status"].notna()]
        table.sample(n=rows, random_state=SAMPLE_SEED).to_parquet(out, index=False)
        del table


def make_vectorizer(setup_name, dtype):
    """Plain TF-IDF, or hashing followed by the same TF-IDF weighting (no vocabulary kept)."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    setup = dict(SETUPS[setup_name])
    if setup.pop("hashing", False):
        hashing = HashingVectorizer(
            n_features=HASH_FEATURES, alternate_sign=False, norm=None, dtype=dtype, **setup
        )
        return make_pipeline(hashing, TfidfTransformer(sublinear_tf=True))
    return TfidfVectorizer(sublinear_tf=True, dtype=dtype, **setup)


def one_run(setup_name, rows, dtype_name):
    """Runs in a fresh process. Returns time, size, peak memory and predictions of one setup."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    from digikala_reviews.resources import peak_memory_gb

    program = peak_memory_gb()  # Python and the libraries, the same for any data size
    train = pd.read_parquet(TRAIN_SAMPLE).head(rows)
    validation = pd.read_parquet(VALIDATION_SAMPLE)
    before = peak_memory_gb()

    start = time.perf_counter()
    vectorizer = make_vectorizer(setup_name, getattr(np, dtype_name))
    x_train = vectorizer.fit_transform(train["text"])
    if hasattr(vectorizer, "stop_words_"):
        vectorizer.stop_words_ = None  # sklearn keeps every dropped rare term here; not needed
    vectorize_seconds = time.perf_counter() - start

    start = time.perf_counter()
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, train["recommendation_status"])
    fit_seconds = time.perf_counter() - start

    predicted = model.predict(vectorizer.transform(validation["text"]))
    return {
        "setup": setup_name,
        "rows": rows,
        "dtype": dtype_name,
        "features": int((x_train.getnnz(axis=0) > 0).sum()),  # columns that are used
        "nonzeros": int(x_train.nnz),
        "vectorize_seconds": round(vectorize_seconds, 1),
        "fit_seconds": round(fit_seconds, 1),
        "program_gb": round(program, 3),
        "memory_before_gb": round(before, 3),
        "peak_memory_gb": round(peak_memory_gb(), 3),
        "macro_f1_validation_sample": round(
            f1_score(validation["recommendation_status"], predicted, average="macro"), 4
        ),
        "predictions": list(predicted),
    }


def run_in_fresh_process(*args):
    require_free_memory(2.0, f"measuring {args}")
    context = multiprocessing.get_context("spawn")
    with context.Pool(1, maxtasksperchild=1) as pool:
        return pool.apply(one_run, args)


def predict_peak(runs, full_rows):
    """Fit growth = a * rows^b on the extra memory of the runs, then read it at full_rows."""
    rows = np.array([r["rows"] for r in runs], dtype=float)
    extra = np.array([max(r["peak_memory_gb"] - r["memory_before_gb"], 1e-3) for r in runs])
    slope, intercept = np.polyfit(np.log(rows), np.log(extra), 1)
    last = runs[-1]
    # The loaded texts grow with the number of rows; the program itself does not
    loaded_data = (last["memory_before_gb"] - last["program_gb"]) * full_rows / last["rows"]
    peak = last["program_gb"] + loaded_data + math.exp(intercept) * full_rows**slope
    return round(float(peak), 2), round(float(slope), 2)


def paired_bootstrap(truth, predicted_a, predicted_b, seed=SAMPLE_SEED):
    """95% interval of Macro F1(a) - Macro F1(b), resampling the same reviews for both."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    truth, predicted_a, predicted_b = map(np.asarray, (truth, predicted_a, predicted_b))
    differences = []
    for _ in range(BOOTSTRAP_ROUNDS):
        rows = rng.integers(0, len(truth), len(truth))
        differences.append(
            f1_score(truth[rows], predicted_a[rows], average="macro")
            - f1_score(truth[rows], predicted_b[rows], average="macro")
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def main():
    start = time.perf_counter()
    print(f"Free memory: {available_memory_gb():.1f} GB")
    make_samples()
    full_rows = len(pd.read_parquet(CACHE_DIR / "train_supervised.parquet", columns=["id"]))

    results, predictions = [], {}
    for setup_name in SETUPS:
        runs = []
        for rows in SIZES:
            run = run_in_fresh_process(setup_name, rows, "float32")
            runs.append(run)
            print(
                f"  {setup_name:>16}, {rows:>7,} rows: {run['features']:>9,} features, "
                f"peak {run['peak_memory_gb']:.2f} GB, {run['vectorize_seconds'] + run['fit_seconds']:.0f}s, "
                f"macro F1 {run['macro_f1_validation_sample']}"
            )
        results += runs
        peak, growth = predict_peak(runs, full_rows)
        predictions[setup_name] = {
            "predicted_peak_gb_full_data": peak,
            "growth_exponent": growth,
            "safe_for_full_run": peak < SAFE_LIMIT_GB,
        }
        print(f"  -> predicted peak on {full_rows:,} rows: {peak} GB (growth exponent {growth})")

    print("float32 against float64 (word 1-2-gram, 400k rows)")
    float64_run = run_in_fresh_process("word 1-2-gram", max(SIZES), "float64")
    float32_run = next(r for r in results if r["setup"] == "word 1-2-gram" and r["rows"] == max(SIZES))
    results.append(float64_run)
    dtype_check = {
        "macro_f1_float32": float32_run["macro_f1_validation_sample"],
        "macro_f1_float64": float64_run["macro_f1_validation_sample"],
        "peak_gb_float32": float32_run["peak_memory_gb"],
        "peak_gb_float64": float64_run["peak_memory_gb"],
    }
    print(f"  {dtype_check}")

    truth = pd.read_parquet(VALIDATION_SAMPLE)["recommendation_status"]
    largest = {r["setup"]: r for r in results if r["rows"] == max(SIZES) and r["dtype"] == "float32"}
    comparisons = {}
    print(f"Each setup against {REFERENCE} ({max(SIZES):,} rows, Macro F1 difference, 95% interval)")
    for setup_name, run in largest.items():
        if setup_name == REFERENCE:
            continue
        low, high = paired_bootstrap(truth, run["predictions"], largest[REFERENCE]["predictions"])
        difference = run["macro_f1_validation_sample"] - largest[REFERENCE]["macro_f1_validation_sample"]
        comparisons[setup_name] = {"difference": round(difference, 4), "interval_95": [low, high]}
        print(f"  {setup_name:>24}: {difference:+.4f}  [{low:+.4f}, {high:+.4f}]")
    low, high = paired_bootstrap(truth, float64_run["predictions"], float32_run["predictions"])
    dtype_check["difference_interval_95"] = [low, high]
    for run in results:
        run.pop("predictions")

    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "recommendation head, LogisticRegression(max_iter=1000), sublinear TF-IDF; "
        "validation sample of 100k labeled reviews; each run in a fresh process",
        "full_training_rows": full_rows,
        "safe_limit_gb": SAFE_LIMIT_GB,
        "runs": results,
        "predictions": predictions,
        "float32_vs_float64": dtype_check,
        "against_reference": {"reference": REFERENCE, "setups": comparisons},
        "seconds": round(time.perf_counter() - start),
    }
    out = REPORTS_DIR / "memory_measurements.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved reports/{out.name} ({report['seconds']}s)")


if __name__ == "__main__":
    main()
