"""Run E8 to E12 of reports/tfidf_experiment_plan.md, the experiments added after stage 1.

Made to run on a second machine next to stage 3 (see notebooks/molab_tfidf_extra.py), so the
reference models are fitted again here instead of being read from the stage 1 and 2 runs.
Each result is saved in reports/tfidf_extra/ as soon as it is ready, and finished ones are
skipped when the script is run again. The test set is never read.

Usage:
    python scripts/experiment_tfidf_extra.py --experiments reference E11 E8
    python scripts/experiment_tfidf_extra.py --experiments E12 E9_E10
"""

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.experiments import compare, majority_rows, nested_samples, pool_splits
from digikala_reviews.ingest import file_sha256
from digikala_reviews.metrics import bootstrap_macro_f1, classification_report
from digikala_reviews.provenance import check_cache, environment, git_commit
from digikala_reviews.resources import peak_memory_gb
from digikala_reviews.spelling import SPELLING_FIXES, fix_spelling, load_fixes
from digikala_reviews.tfidf import HEADS, drop_pruned_terms, head_rows, load_cache, make_vectorizer
from digikala_reviews.tuning import fingerprint, finished, fit_logistic, save_json

RESULTS_DIR = REPORTS_DIR / "tfidf_extra"
PREDICTIONS_DIR = ARTIFACTS_DIR / "tfidf_extra"  # reference predictions, so a restart need not refit
CACHE_FILES = ["train_supervised.parquet", "validation.parquet"]
YEARS_FILE = CACHE_DIR / "review_years.parquet"
SETUP = "word123_min5_persian"
# The stage 3 reference of each head (best of stages 1 and 2), with its stage 2 score on Colab
REFERENCES = {
    "recommendation": {"C": 0.5, "class_weight": "balanced", "stage2_macro_f1": 0.6802},
    "rating_band": {"C": 1.0, "class_weight": None, "stage2_macro_f1": 0.6459},
}
SAMPLE_SIZES = [10_000, 100_000, 1_000_000]
ALL_STEPS = ["reference", "E11", "E8", "E12", "E9_E10"]


class Runner:
    def __init__(self, context):
        self.context = context
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.train = load_cache(CACHE_FILES[0], extra_columns=["product_id"])
        self.validation = load_cache(CACHE_FILES[1], extra_columns=["product_id"])
        self.train_texts, self.validation_texts = self.train.pop("text"), self.validation.pop("text")
        ids = np.ascontiguousarray(self.validation["id"].to_numpy())
        self.validation_ids_sha256 = hashlib.sha256(ids.tobytes()).hexdigest()
        self.base = fingerprint(SETUP, context["data"])
        self.x_train = self.x_validation = None  # reference features, built when first needed

    # ---- bookkeeping -------------------------------------------------------------------------
    def spec(self, experiment, head, variant, extra=None):
        return {
            "experiment": experiment,
            "head": head,
            "variant": variant,
            "base": self.base,
            "reference": REFERENCES[head],
            "seed": SPLIT_SEED,
            **(extra or {}),
        }

    def path(self, experiment, head, variant):
        return RESULTS_DIR / f"{experiment}__{head}__{variant}.json"

    def done(self, experiment, head, variant, extra=None):
        return finished(self.path(experiment, head, variant), self.spec(experiment, head, variant, extra))

    def save(self, experiment, head, variant, result, spec_extra=None):
        record = {
            "experiment": experiment,
            "head": head,
            "variant": variant,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fingerprint": self.spec(experiment, head, variant, spec_extra),
            **result,
            "process_peak_memory_gb": peak_memory(),
            **self.context,
        }
        save_json(self.path(experiment, head, variant), record)
        print(f"  {experiment} {head} {variant}: {summary(result)}", flush=True)

    # ---- shared steps ------------------------------------------------------------------------
    def labels(self, table, head):
        rows = head_rows(table, head)
        return rows, table[HEADS[head]["column"]].to_numpy()[rows]

    def features(self, train_texts, validation_texts, what):
        start = time.perf_counter()
        vectorizer = make_vectorizer(SETUP)
        x_train = vectorizer.fit_transform(train_texts)
        drop_pruned_terms(vectorizer)
        x_validation = vectorizer.transform(validation_texts)
        seconds = time.perf_counter() - start
        print(f"  features for {what}: {x_train.shape[1]:,} columns ({seconds:.0f}s)", flush=True)
        return x_train, x_validation

    def reference_features(self):
        if self.x_train is None:
            self.x_train, self.x_validation = self.features(
                self.train_texts, self.validation_texts, "reference"
            )
        return self.x_train, self.x_validation

    def fit_predict(self, head, x_train, y_train, x_validation):
        reference = REFERENCES[head]
        start = time.perf_counter()
        model, converged = fit_logistic(x_train, y_train, reference["C"], reference["class_weight"])
        return model.predict(x_validation), converged, round((time.perf_counter() - start) / 60, 1)

    # ---- the reference -----------------------------------------------------------------------
    def cache_path(self, head):
        return PREDICTIONS_DIR / f"reference_predictions__{head}.npz"

    def reference_labels(self, head):
        """Reference predictions for the labeled validation reviews of a head (fitted once)."""
        path, expected = (
            self.cache_path(head),
            json.dumps(self.spec("reference", head, "fit"), sort_keys=True),
        )
        if path.exists():
            saved = np.load(path, allow_pickle=False)
            if str(saved["fingerprint"]) == expected and str(saved["ids"]) == self.validation_ids_sha256:
                return saved["labels"].astype(str)
        x_train, x_validation = self.reference_features()
        train_rows, y_train = self.labels(self.train, head)
        rows, y_validation = self.labels(self.validation, head)
        predicted, converged, minutes = self.fit_predict(
            head, x_train[train_rows], y_train, x_validation[rows]
        )
        temporary = path.with_suffix(".tmp")
        with open(temporary, "wb") as file:
            np.savez(file, fingerprint=expected, ids=self.validation_ids_sha256, labels=predicted.astype(str))
        os.replace(temporary, path)

        labels = HEADS[head]["labels"]
        result = scores(head, y_validation, predicted, converged, minutes)
        stage2 = REFERENCES[head]["stage2_macro_f1"]
        result["stage2_macro_f1_on_colab"] = stage2
        result["reproduces_stage2"] = bool(round(result["scores"]["macro_f1"], 3) == round(stage2, 3))
        result["labels"] = labels
        self.save("reference", head, "fit", result)
        return predicted.astype(str)

    def reference(self):
        for head in HEADS:
            self.reference_labels(head)

    # ---- E11: conflicting labels removed from training -----------------------------------------
    def conflicts_removed(self):
        for head in HEADS:
            if self.done("E11", head, "majority_only"):
                continue
            reference = self.reference_labels(head)
            x_train, x_validation = self.reference_features()
            train_rows, y_train = self.labels(self.train, head)
            rows, y_validation = self.labels(self.validation, head)
            keep = majority_rows(self.train_texts.to_numpy()[train_rows], y_train)
            predicted, converged, minutes = self.fit_predict(
                head, x_train[train_rows[keep]], y_train[keep], x_validation[rows]
            )
            removed = pd.Series(np.delete(y_train, keep)).value_counts()
            result = compare(head, y_validation, predicted, reference)
            result.update(
                verdict(result, converged),
                minutes=minutes,
                removed_rows=int(len(y_train) - len(keep)),
                removed_share=round(1 - len(keep) / len(y_train), 5),
                removed_per_class={str(k): int(v) for k, v in removed.items()},
            )
            self.save("E11", head, "majority_only", result)

    # ---- E8: spelling fixes -------------------------------------------------------------------
    def spelling(self):
        extra = {"fixes_sha256": file_sha256(SPELLING_FIXES)}
        heads = [h for h in HEADS if not self.done("E8", h, "spelling_fixed", extra)]
        if not heads:
            return
        references = {h: self.reference_labels(h) for h in heads}
        fixes = load_fixes()
        train_fixed = fix_spelling(self.train_texts, fixes)
        validation_fixed = fix_spelling(self.validation_texts, fixes)
        changed = {
            "train": float((train_fixed != self.train_texts).mean()),
            "validation": float((validation_fixed != self.validation_texts).mean()),
        }
        print(f"  spelling fixes change {changed['train']:.2%} of train reviews", flush=True)
        self.x_train = self.x_validation = None  # make room for the new features
        x_train, x_validation = self.features(train_fixed, validation_fixed, "E8")
        for head in heads:
            train_rows, y_train = self.labels(self.train, head)
            rows, y_validation = self.labels(self.validation, head)
            predicted, converged, minutes = self.fit_predict(
                head, x_train[train_rows], y_train, x_validation[rows]
            )
            result = compare(head, y_validation, predicted, references[head])
            result.update(
                verdict(result, converged), minutes=minutes, fixes=len(fixes), reviews_changed_share=changed
            )
            self.save("E8", head, "spelling_fixed", result, extra)

    # ---- E12: learning curve ------------------------------------------------------------------
    def learning_curve(self):
        extra = {"sizes": SAMPLE_SIZES}
        for head in HEADS:
            train_rows, y_train = self.labels(self.train, head)
            rows, y_validation = self.labels(self.validation, head)
            for size, positions in nested_samples(len(train_rows), SAMPLE_SIZES, SPLIT_SEED).items():
                variant = f"n{size}"
                if self.done("E12", head, variant, extra):
                    continue
                sample = train_rows[positions]
                x_train, x_validation = self.features(
                    self.train_texts.iloc[sample], self.validation_texts.iloc[rows], f"E12 {head} {variant}"
                )
                predicted, converged, minutes = self.fit_predict(
                    head, x_train, y_train[positions], x_validation
                )
                result = scores(head, y_validation, predicted, converged, minutes)
                result.update(training_reviews=int(size), features=int(x_train.shape[1]))
                self.save("E12", head, variant, result, extra)

    # ---- E9 and E10: other ways to split ------------------------------------------------------
    def other_splits(self):
        extra = {"years_sha256": file_sha256(YEARS_FILE)}
        jobs = [("E9", "by_product"), ("E9", "random"), ("E10", "by_time")]
        if all(self.done(e, h, v, extra) for e, v in jobs for h in HEADS):
            return
        info = json.loads((CACHE_DIR / "review_years_info.json").read_text(encoding="utf-8"))
        assert info["sha256"] == extra["years_sha256"], "review_years.parquet differs from the one built"
        pool = pd.concat([self.train, self.validation], ignore_index=True)
        texts = pd.concat([self.train_texts, self.validation_texts], ignore_index=True)
        years = pd.read_parquet(YEARS_FILE)
        year = pool[["id"]].merge(years, on="id", how="left", validate="one_to_one")["year"]
        assert year.notna().all(), "a review without a year"
        newest = (year == year.max()).to_numpy()
        splits = pool_splits(pool["product_id"].to_numpy(), newest, SPLIT_SEED)
        self.x_train = self.x_validation = None  # make room
        for experiment, name in jobs:
            if all(self.done(experiment, h, name, extra) for h in HEADS):
                continue
            in_validation = splits[name]
            train_part, validation_part = pool[~in_validation], pool[in_validation]
            train_texts, validation_texts = texts[~in_validation], texts[in_validation]
            seen = float(validation_texts.isin(set(train_texts)).mean())
            x_train, x_validation = self.features(train_texts, validation_texts, f"{experiment} {name}")
            for head in HEADS:
                train_rows, y_train = self.labels(train_part, head)
                rows, y_validation = self.labels(validation_part, head)
                predicted, converged, minutes = self.fit_predict(
                    head, x_train[train_rows], y_train, x_validation[rows]
                )
                result = scores(head, y_validation, predicted, converged, minutes)
                result.update(
                    split=name,
                    newest_year=int(year.max()),
                    train_reviews=int(len(train_rows)),
                    validation_reviews=int(len(rows)),
                    validation_share_of_pool=round(float(in_validation.mean()), 4),
                    validation_texts_seen_in_train=round(seen, 4),
                    products_in_both_parts=int(
                        len(set(train_part["product_id"]) & set(validation_part["product_id"]))
                    ),
                )
                self.save(experiment, head, name, result, extra)
            x_train = x_validation = None


def peak_memory():
    """Peak memory of this process in GB, or None where the system does not report it (molab)."""
    try:
        return round(peak_memory_gb(), 2)
    except RuntimeError:
        return None


def scores(head, y_true, predicted, converged, minutes):
    labels = HEADS[head]["labels"]
    report = classification_report(y_true, predicted, labels)
    return {
        "scores": report,
        "macro_f1_interval_95": bootstrap_macro_f1(y_true, predicted, labels),
        "converged": bool(converged),
        "minutes": minutes,
        "verdict": "report only",
    }


def verdict(result, converged):
    """E8 and E11 are adopted only if the whole interval of the difference is above zero."""
    return {"converged": bool(converged), "verdict": result["direction"] if converged else "not converged"}


def summary(result):
    if "macro_f1_difference" in result:
        low, high = result["difference_interval_95"]
        return f"{result['macro_f1_difference']:+.4f} [{low:+.4f}, {high:+.4f}] {result['verdict']}"
    low, high = result["macro_f1_interval_95"]
    return f"macro F1 {result['scores']['macro_f1']:.4f} [{low:.4f}, {high:.4f}] ({result['minutes']} min)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    steps = parser.parse_args().experiments
    context = {
        "code": git_commit(),
        "environment": environment(),
        "data": check_cache(CACHE_DIR, CACHE_FILES),
    }
    print(f"Code {context['code']['commit']}, data checked against cache_info.json", flush=True)
    runner = Runner(context)
    actions = {
        "reference": runner.reference,
        "E11": runner.conflicts_removed,
        "E8": runner.spelling,
        "E12": runner.learning_curve,
        "E9_E10": runner.other_splits,
    }
    for step in ALL_STEPS:  # always in this order, whatever order they were asked in
        if step in steps:
            actions[step]()
    print("Done", flush=True)


if __name__ == "__main__":
    main()
