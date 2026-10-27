"""Fit the final TF-IDF model of each head and check it once more as a whole.

The final model is the stage 3 reference with every change that was adopted (see "How a
difference is judged" in reports/tfidf_experiment_plan.md):

- recommendation: spelling fixes (E8), conflicting labels removed from training (E11),
  per-class decision offsets (E1)
- rating band: character n-grams next to the words (E2), conflicting labels removed (E11),
  per-class decision offsets (E1)

Changes are not always additive, so the combination is scored again on validation against
the reference, with and without offsets. The offsets of each validation review are tuned on
the other half of validation (by product), as in E1; the offsets for the test set are tuned
on the whole validation set and saved. Training scores are reported next to validation
scores to show how much the model fits its own training data. The test set is not read.

Usage:
    python scripts/final_tfidf.py
"""

import json
import time
from datetime import UTC, datetime

import numpy as np

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.experiments import (
    compare,
    cross_fitted_offsets,
    majority_rows,
    product_half,
    search_offsets,
)
from digikala_reviews.ingest import file_sha256
from digikala_reviews.metrics import bootstrap_macro_f1, classification_report
from digikala_reviews.provenance import check_cache, environment, git_commit
from digikala_reviews.resources import peak_memory_gb_if_known
from digikala_reviews.spelling import SPELLING_FIXES, fix_spelling, load_fixes
from digikala_reviews.tfidf import HEADS, drop_pruned_terms, head_rows, load_cache, make_vectorizer
from digikala_reviews.tuning import fingerprint, finished, fit_logistic, save_json

RESULTS_DIR = REPORTS_DIR / "tfidf_final"
CACHE_FILES = ["train_supervised.parquet", "validation.parquet"]
SETUP = "word123_min5_persian"
CHAR_PART = {"analyzer": "char_wb", "ngram_range": (3, 4), "min_df": 5}  # as in E2
TRAIN_SCORE_ROWS = 300_000  # training reviews scored for the train-vs-validation check
FINAL = {
    "recommendation": {"C": 0.5, "class_weight": "balanced", "spelling": True, "char": False},
    "rating_band": {"C": 1.0, "class_weight": None, "spelling": False, "char": True},
}


def features(train_texts, validation_texts, char):
    start = time.perf_counter()
    vectorizer = make_vectorizer(SETUP, extra_parts=[CHAR_PART] if char else ())
    x_train = vectorizer.fit_transform(train_texts)
    drop_pruned_terms(vectorizer)
    x_validation = vectorizer.transform(validation_texts)
    print(f"  features: {x_train.shape[1]:,} columns ({time.perf_counter() - start:.0f}s)", flush=True)
    return x_train, x_validation


def fit(head, x_train, y_train, x_validation, rows):
    settings = FINAL[head]
    start = time.perf_counter()
    model, converged = fit_logistic(x_train, y_train, settings["C"], settings["class_weight"])
    log_proba = model.predict_log_proba(x_validation[rows])
    minutes = round((time.perf_counter() - start) / 60, 1)
    print(f"  fitted in {minutes} min, converged: {converged}", flush=True)
    return model, log_proba, list(model.classes_.astype(str)), bool(converged), minutes


def with_offsets(log_proba, classes, y_validation, half_a):
    """Labels with cross-fitted offsets, and the offsets for the test set (tuned on all of validation)."""
    code = {label: i for i, label in enumerate(classes)}
    true_codes = np.array([code[v] for v in y_validation])
    predicted, per_half = cross_fitted_offsets(log_proba, true_codes, half_a)
    for_test = search_offsets(log_proba, true_codes)
    return np.array(classes)[predicted], {
        "offsets_per_half": {k: dict(zip(classes, v.tolist(), strict=True)) for k, v in per_half.items()},
        "offsets_for_test": dict(zip(classes, for_test.tolist(), strict=True)),
    }


def main():
    context = {
        "code": git_commit(),
        "environment": environment(),
        "data": check_cache(CACHE_DIR, CACHE_FILES),
    }
    print(f"Code {context['code']['commit']}, data checked against cache_info.json", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = load_cache(CACHE_FILES[0])
    validation = load_cache(CACHE_FILES[1], extra_columns=["product_id"])
    train_texts, validation_texts = train.pop("text"), validation.pop("text")
    fixes = load_fixes()

    for head, settings in FINAL.items():
        spec = {
            "head": head,
            "final": settings,
            "base": fingerprint(SETUP, context["data"]),
            "char_part": CHAR_PART,
            "fixes_sha256": file_sha256(SPELLING_FIXES),
            "train_score_rows": TRAIN_SCORE_ROWS,
            "seed": SPLIT_SEED,
        }
        spec = json.loads(json.dumps(spec))
        path = RESULTS_DIR / f"final__{head}.json"
        if finished(path, spec):
            print(f"{head}: already done", flush=True)
            continue
        print(f"{head}:", flush=True)
        train_rows = head_rows(train, head)
        y_train = train[HEADS[head]["column"]].to_numpy()[train_rows]
        rows = head_rows(validation, head)
        y_validation = validation[HEADS[head]["column"]].to_numpy()[rows]
        half_a = product_half(validation["product_id"].to_numpy()[rows])
        labels = HEADS[head]["labels"]

        # the reference, refitted here so that every comparison uses the same machine
        print(" reference", flush=True)
        x_train, x_validation = features(train_texts, validation_texts, char=False)
        _, ref_log_proba, ref_classes, ref_converged, ref_minutes = fit(
            head, x_train[train_rows], y_train, x_validation, rows
        )
        reference = np.array(ref_classes)[np.argmax(ref_log_proba, axis=1)]
        reference_offsets, _ = with_offsets(ref_log_proba, ref_classes, y_validation, half_a)
        x_train = x_validation = None

        # the final model
        print(" final", flush=True)
        if settings["spelling"]:
            final_train_texts = fix_spelling(train_texts, fixes)
            final_validation_texts = fix_spelling(validation_texts, fixes)
        else:
            final_train_texts, final_validation_texts = train_texts, validation_texts
        x_train, x_validation = features(final_train_texts, final_validation_texts, settings["char"])
        keep = majority_rows(train_texts.to_numpy()[train_rows], y_train)  # as tested in E11
        model, log_proba, classes, converged, minutes = fit(
            head, x_train[train_rows[keep]], y_train[keep], x_validation, rows
        )
        plain = np.array(classes)[np.argmax(log_proba, axis=1)]
        final, offsets = with_offsets(log_proba, classes, y_validation, half_a)

        # training scores on a fixed sample of the reviews the model was trained on
        sample = np.random.default_rng(SPLIT_SEED).permutation(len(keep))[:TRAIN_SCORE_ROWS]
        train_sample = train_rows[keep][np.sort(sample)]
        train_codes = np.argmax(model.predict_log_proba(x_train[train_sample]), axis=1)
        y_sample = y_train[keep][np.sort(sample)]
        train_report = classification_report(y_sample, np.array(classes)[train_codes], labels)

        result = {
            "head": head,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fingerprint": spec,
            "features": int(x_train.shape[1]),
            "training_rows": {"labeled": int(len(train_rows)), "after_removing_conflicts": int(len(keep))},
            "converged": {"reference": ref_converged, "final": converged},
            "minutes": {"reference": ref_minutes, "final": minutes},
            "validation": {
                "final_with_offsets": classification_report(y_validation, final, labels),
                "final_with_offsets_interval_95": bootstrap_macro_f1(y_validation, final, labels),
                "final_without_offsets": classification_report(y_validation, plain, labels),
                "reference": classification_report(y_validation, reference, labels),
                "reference_with_offsets": classification_report(y_validation, reference_offsets, labels),
            },
            "final_vs_reference": compare(head, y_validation, final, reference),
            "final_vs_reference_with_offsets": compare(head, y_validation, final, reference_offsets),
            "final_without_offsets_vs_reference": compare(head, y_validation, plain, reference),
            "train_sample": train_report,
            "train_minus_validation_macro_f1": train_report["macro_f1"]
            - classification_report(y_validation, plain, labels)["macro_f1"],
            **offsets,
            "process_peak_memory_gb": peak_memory_gb_if_known(),
            **context,
        }
        save_json(path, result)
        x_train = x_validation = None
        for name in [
            "final_vs_reference",
            "final_vs_reference_with_offsets",
            "final_without_offsets_vs_reference",
        ]:
            low, high = result[name]["difference_interval_95"]
            print(
                f"  {name}: {result[name]['macro_f1_difference']:+.4f} [{low:+.4f}, {high:+.4f}]", flush=True
            )
        print(
            f"  final macro F1 {result['validation']['final_with_offsets']['macro_f1']:.4f}, "
            f"train {train_report['macro_f1']:.4f}",
            flush=True,
        )
    print("Done", flush=True)


if __name__ == "__main__":
    main()
