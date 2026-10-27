"""Fit the chosen TF-IDF model of both heads and keep everything later steps need.

Decision after scripts/final_tfidf.py (see "The final model" in
reports/tfidf_experiment_plan.md): both heads use the stage 3 reference features (word 1-3
grams, min_df 5, Persian tokens) with per-class decision offsets (E1) and nothing else. So
one vectorizer, fitted on the training texts, serves both heads.

Saved, so that the analyses and the test evaluation never refit:
- artifacts/tfidf_final/model.joblib: the vectorizer, one logistic regression per head and
  the offsets for the test set (tuned on all labeled validation reviews)
- artifacts/tfidf_final/validation_predictions.parquet: for every validation review, the
  log-probabilities of each head, the label without offsets and the label with cross-fitted
  offsets (each review scored with offsets tuned on the other half of validation)
- artifacts/tfidf_final/train_unlabeled_predictions.parquet: log-probabilities for the
  training reviews that have no label for a head (how much the semi-supervised step has);
  one row per review and head, with the columns of the other head's classes left empty
- reports/tfidf_final/chosen_model.json: scores, offsets, hashes of the saved files

The test set is not read.

Usage:
    python scripts/fit_final_tfidf.py
"""

import json
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.experiments import cross_fitted_offsets, product_half, search_offsets
from digikala_reviews.ingest import file_sha256
from digikala_reviews.metrics import bootstrap_macro_f1, classification_report
from digikala_reviews.provenance import check_cache, environment, git_commit
from digikala_reviews.resources import peak_memory_gb_if_known
from digikala_reviews.tfidf import HEADS, drop_pruned_terms, head_rows, load_cache, make_vectorizer
from digikala_reviews.tuning import fingerprint, finished, fit_logistic, save_json

OUT_DIR = ARTIFACTS_DIR / "tfidf_final"
SUMMARY = REPORTS_DIR / "tfidf_final" / "chosen_model.json"
CACHE_FILES = ["train_supervised.parquet", "validation.parquet"]
SETUP = "word123_min5_persian"
CHOSEN = {
    "recommendation": {"C": 0.5, "class_weight": "balanced"},
    "rating_band": {"C": 1.0, "class_weight": None},
}


def write_parquet(table: pd.DataFrame, path):
    temporary = path.with_suffix(".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def main():
    context = {
        "code": git_commit(),
        "environment": environment(),
        "data": check_cache(CACHE_DIR, CACHE_FILES),
    }
    spec = {"chosen": CHOSEN, "base": fingerprint(SETUP, context["data"]), "seed": SPLIT_SEED}
    spec = json.loads(json.dumps(spec))
    if finished(SUMMARY, spec):
        print("Already done", flush=True)
        return
    print(f"Code {context['code']['commit']}, data checked against cache_info.json", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    train = load_cache(CACHE_FILES[0])
    validation = load_cache(CACHE_FILES[1], extra_columns=["product_id"])

    start = time.perf_counter()
    vectorizer = make_vectorizer(SETUP)
    x_train = vectorizer.fit_transform(train["text"])
    drop_pruned_terms(vectorizer)
    x_validation = vectorizer.transform(validation["text"])
    print(f"features: {x_train.shape[1]:,} columns ({time.perf_counter() - start:.0f}s)", flush=True)

    models, offsets_for_test, heads = {}, {}, {}
    val_out = pd.DataFrame({"id": validation["id"].to_numpy()})
    unlabeled_out = []
    for head, settings in CHOSEN.items():
        print(f"{head}:", flush=True)
        column, labels = HEADS[head]["column"], HEADS[head]["labels"]
        train_rows = head_rows(train, head)
        start = time.perf_counter()
        model, converged = fit_logistic(
            x_train[train_rows], train[column].to_numpy()[train_rows], settings["C"], settings["class_weight"]
        )
        minutes = round((time.perf_counter() - start) / 60, 1)
        print(f"  fitted in {minutes} min, converged: {converged}", flush=True)
        classes = list(model.classes_.astype(str))

        log_proba = model.predict_log_proba(x_validation)  # every validation review, labeled or not
        rows = head_rows(validation, head)
        y_validation = validation[column].to_numpy()[rows]
        code = {label: i for i, label in enumerate(classes)}
        true_codes = np.array([code[v] for v in y_validation])
        half_a = product_half(validation["product_id"].to_numpy()[rows])
        crossed, per_half = cross_fitted_offsets(log_proba[rows], true_codes, half_a)
        for_test = search_offsets(log_proba[rows], true_codes)

        plain = np.array(classes)[np.argmax(log_proba, axis=1)]
        with_offsets = np.full(len(validation), None, dtype=object)  # only labeled reviews have one
        with_offsets[rows] = np.array(classes)[crossed]
        labeled_offsets = with_offsets[rows].astype(str)
        for i, label in enumerate(classes):
            val_out[f"{head}__log_proba__{label}"] = log_proba[:, i].astype(np.float32)
        val_out[f"{head}__predicted"] = plain
        val_out[f"{head}__predicted_with_offsets"] = with_offsets

        unlabeled = np.setdiff1d(np.arange(len(train)), train_rows)
        if len(unlabeled):
            part = pd.DataFrame({"id": train["id"].to_numpy()[unlabeled], "head": head})
            unlabeled_proba = model.predict_log_proba(x_train[unlabeled])
            for i, label in enumerate(classes):
                part[f"log_proba__{label}"] = unlabeled_proba[:, i].astype(np.float32)
            unlabeled_out.append(part)

        models[head] = model
        offsets_for_test[head] = dict(zip(classes, for_test.tolist(), strict=True))
        heads[head] = {
            "settings": settings,
            "converged": bool(converged),
            "minutes": minutes,
            "training_rows": int(len(train_rows)),
            "unlabeled_training_rows": int(len(unlabeled)),
            "validation_without_offsets": classification_report(y_validation, plain[rows], labels),
            "validation_with_offsets": classification_report(y_validation, labeled_offsets, labels),
            "validation_with_offsets_interval_95": bootstrap_macro_f1(y_validation, labeled_offsets, labels),
            "offsets_per_half": {k: dict(zip(classes, v.tolist(), strict=True)) for k, v in per_half.items()},
            "offsets_for_test": offsets_for_test[head],
        }
        print(
            f"  macro F1 without offsets {heads[head]['validation_without_offsets']['macro_f1']:.4f}, "
            f"with offsets {heads[head]['validation_with_offsets']['macro_f1']:.4f}",
            flush=True,
        )

    model_path = OUT_DIR / "model.joblib"
    joblib.dump(
        {"vectorizer": vectorizer, "models": models, "offsets_for_test": offsets_for_test, "setup": SETUP},
        model_path.with_suffix(".tmp"),
        compress=3,
    )
    model_path.with_suffix(".tmp").replace(model_path)
    val_path = OUT_DIR / "validation_predictions.parquet"
    write_parquet(val_out, val_path)
    unlabeled_path = OUT_DIR / "train_unlabeled_predictions.parquet"
    unlabeled_table = pd.concat(unlabeled_out, ignore_index=True) if unlabeled_out else pd.DataFrame()
    write_parquet(unlabeled_table, unlabeled_path)

    save_json(
        SUMMARY,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fingerprint": spec,
            "features": int(x_train.shape[1]),
            "heads": heads,
            "files": {
                str(p.relative_to(ARTIFACTS_DIR.parent)): {"sha256": file_sha256(p), "size": p.stat().st_size}
                for p in [model_path, val_path, unlabeled_path]
            },
            "process_peak_memory_gb": peak_memory_gb_if_known(),
            **context,
        },
    )
    print("Done", flush=True)


if __name__ == "__main__":
    main()
