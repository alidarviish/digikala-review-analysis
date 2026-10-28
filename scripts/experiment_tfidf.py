"""Run the TF-IDF experiments written down in reports/tfidf_experiment_plan.md.

Each experiment changes one thing in the reference model of a head (the best run of
scripts/tune_tfidf.py) and is scored on the same validation reviews as the reference, with
a paired bootstrap interval for the difference in Macro F1. Results are saved one by one in
reports/tfidf_experiments/; finished ones are skipped when the script is run again. The
test set is never read here.

Usage:
    python scripts/experiment_tfidf.py [--experiments E1 E4 ...]

E13 (added after E6) is run the same way, on the reference features.
"""

import argparse
import hashlib
import json
import os
import time
import warnings
from datetime import UTC, datetime

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from digikala_reviews.clean import rating_band
from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR
from digikala_reviews.experiments import (
    compare,
    cross_fitted_offsets,
    most_common_words,
    ordinal_probabilities,
    product_half,
    search_offsets,
)
from digikala_reviews.metrics import bootstrap_mean_difference
from digikala_reviews.provenance import check_cache, environment, git_commit
from digikala_reviews.resources import peak_memory_gb_if_known, require_free_memory
from digikala_reviews.tfidf import (
    FEATURE_SETUPS,
    HEADS,
    TOKEN_PATTERNS,
    drop_pruned_terms,
    head_rows,
    load_cache,
    make_vectorizer,
)
from digikala_reviews.tuning import MAX_ITER, best_run, fingerprint, finished, fit_logistic, save_json

RUNS_DIR = REPORTS_DIR / "tfidf_runs"
EXPERIMENTS_DIR = REPORTS_DIR / "tfidf_experiments"
PREDICTIONS_DIR = ARTIFACTS_DIR / "tfidf"  # reference predictions, so a restart need not refit them
CACHE_FILES = ["train_supervised.parquet", "validation.parquet"]
ALL_EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E13"]
REPORT_ONLY = {"E6", "E7", "E13"}  # the plan never adopts these, whatever they show
CHAR_PART = {"analyzer": "char_wb", "ngram_range": (3, 4), "min_df": 5}
STOPWORD_COUNT = 100
NEGATION_WORDS = ["نه", "نیست", "نبود", "نداره", "نکنید", "نخرید", "نیستم", "نشد", "نکرد"]
# Experiments on the reference features, and experiments that need features of their own
REFERENCE_FEATURE_JOBS = [
    ("E1", "offsets"),
    ("E4", "ordinal"),
    ("E6", "svc_C0.1"),
    ("E6", "svc_C1"),
    ("E6", "complement_nb_alpha0.1"),
    ("E6", "complement_nb_alpha1"),
    ("E7", "zero_as_low"),
    ("E13", "svc_C0.1_with_offsets"),
]
BAND_ONLY = {"E4", "E7"}
# What each experiment changes, as stored in its fingerprint
SPECS = {
    "E1": {
        "offset_grid": "-2..2 step 0.1, 3 passes, first class fixed at 0",
        "halves": "product_id even/odd",
    },
    "E2": {"extra_part": CHAR_PART},
    "E3": {"removed_words": f"{STOPWORD_COUNT} words (tokens with a letter) in the most training reviews"},
    "E4": {"method": "Frank and Hall, two binary logistic regressions"},
    "E5": {"sublinear_tf": False},
    "E6": {"svc_C": [0.1, 1.0], "complement_nb_alpha": [0.1, 1.0], "svc_max_iter": MAX_ITER},
    "E7": {"rate_zero_policy": "low"},
    "E13": {
        "svc_C": 0.1,
        "svc_max_iter": MAX_ITER,
        "offsets": "as E1, on the SVM decision values and on the reference log probabilities",
        "compared_with": "the reference with offsets",
    },
}


def reference_runs(data):
    """For each head, the best run of tune_tfidf.py, after checking that both stages are complete."""
    references = {}
    for head in HEADS:
        runs = []
        for path in sorted(RUNS_DIR.glob(f"{head}__*.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            if run["fingerprint"] != fingerprint(run["setup"], data):
                raise SystemExit(f"{path.name} was made with other settings or data")
            runs.append(run)
        stage2 = [r for r in runs if r["stage"] == "regularization"]
        if len(stage2) < 4:
            raise SystemExit(
                f"Finish both stages of tune_tfidf.py first ({head}: {len(stage2)} of 4 stage 2 runs)"
            )
        references[head] = best_run(runs)
    return references


def fit_quietly(model, x, y):
    """Fit any sklearn model and report whether it converged (no ConvergenceWarning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    return model, not any(issubclass(w.category, ConvergenceWarning) for w in caught)


class Experiments:
    def __init__(self, todo, context):
        self.todo, self.context = set(todo), context
        self.references = reference_runs(context["data"])
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.train = load_cache(CACHE_FILES[0], extra_columns=["rate"])
        self.validation = load_cache(CACHE_FILES[1], extra_columns=["rate", "product_id"])
        self.train_texts, self.validation_texts = self.train.pop("text"), self.validation.pop("text")
        ids = self.validation["id"].to_numpy()
        self.validation_ids_sha256 = hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()
        self.reference_labels = {}  # head -> reference predictions for every validation review

    # ---- bookkeeping -------------------------------------------------------------------------
    def path(self, experiment, head, variant):
        return EXPERIMENTS_DIR / f"{experiment}__{head}__{variant}.json"

    def spec(self, experiment, head, variant):
        reference = self.references[head]
        return {
            "experiment": experiment,
            "variant": variant,
            "reference_run": reference["run"],
            "reference_fingerprint": reference["fingerprint"],
            "spec": json.loads(json.dumps(SPECS[experiment])),
        }

    def pending(self, experiment, head, variant):
        if experiment not in self.todo or (experiment in BAND_ONLY and head != "rating_band"):
            return False
        return not finished(self.path(experiment, head, variant), self.spec(experiment, head, variant))

    def save(self, experiment, head, variant, result, converged=True, extra=None):
        if not converged:
            result["verdict"] = "not converged"
        elif experiment in REPORT_ONLY:
            result["verdict"] = "report only"
        else:
            result["verdict"] = result["direction"]  # adopted only if "better"
        reference = self.references[head]
        record = {
            "experiment": experiment,
            "head": head,
            "variant": variant,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "reference_model": {k: reference[k] for k in ["run", "setup", "C", "class_weight"]},
            "converged": bool(converged),
            "fingerprint": self.spec(experiment, head, variant),
            **result,
            **(extra or {}),
            "process_peak_memory_gb": peak_memory_gb_if_known(),
            **self.context,
        }
        save_json(self.path(experiment, head, variant), record)
        low, high = result["difference_interval_95"]
        difference, verdict = result["macro_f1_difference"], result["verdict"]
        print(
            f"  {experiment} {head} {variant}: {difference:+.4f} [{low:+.4f}, {high:+.4f}] {verdict}",
            flush=True,
        )

    # ---- features and the reference model ----------------------------------------------------
    def features(self, setup, **options):
        require_free_memory(8.0 if options.get("extra_parts") else 5.0, f"building features for {setup}")
        start = time.perf_counter()
        vectorizer = make_vectorizer(setup, **options)
        x_train = vectorizer.fit_transform(self.train_texts)
        drop_pruned_terms(vectorizer)
        x_validation = vectorizer.transform(self.validation_texts)
        shown = {k: (f"{len(v)} words" if k == "stop_words" else v) for k, v in options.items()}
        seconds = time.perf_counter() - start
        print(f"  features {setup} {shown or ''}: {x_train.shape[1]:,} columns ({seconds:.0f}s)", flush=True)
        return x_train, x_validation

    def labels(self, data, head):
        rows = head_rows(data, head)
        return rows, data[HEADS[head]["column"]].to_numpy()[rows]

    def reference_cache(self, head):
        return PREDICTIONS_DIR / f"reference_predictions__{head}.npz"

    def load_reference(self, head):
        """Cached reference predictions, if they belong to this run, these settings and these reviews."""
        path = self.reference_cache(head)
        if not path.exists():
            return None
        saved = np.load(path, allow_pickle=False)
        expected = json.dumps(self.references[head]["fingerprint"], sort_keys=True)
        if (
            str(saved["run"]) != self.references[head]["run"]
            or str(saved["fingerprint"]) != expected
            or str(saved["validation_ids_sha256"]) != self.validation_ids_sha256
        ):
            return None
        return saved["labels"], saved["log_proba"], list(saved["classes"])

    def fit_reference(self, head, x_train, x_validation):
        """Fit the reference model and predict every validation review (E7 needs the zero-rated ones too)."""
        reference = self.references[head]
        train_rows, y_train = self.labels(self.train, head)
        model, _ = fit_logistic(x_train[train_rows], y_train, reference["C"], reference["class_weight"])
        log_proba = model.predict_log_proba(x_validation)
        labels = model.classes_[np.argmax(log_proba, axis=1)].astype(str)  # argmax before rounding
        temporary = self.reference_cache(head).with_suffix(".tmp")
        with open(temporary, "wb") as file:
            np.savez(
                file,
                run=reference["run"],
                fingerprint=json.dumps(reference["fingerprint"], sort_keys=True),
                validation_ids_sha256=self.validation_ids_sha256,
                labels=labels,
                log_proba=log_proba.astype(np.float32),
                classes=model.classes_.astype(str),
            )
        os.replace(temporary, self.reference_cache(head))
        return labels, log_proba.astype(np.float32), list(model.classes_.astype(str))

    # ---- experiments on the reference features (E1, E4, E6, E7) ------------------------------
    def run_on_reference_features(self):
        """Get each head's reference predictions (cached or fitted), then run E1, E4, E6 and E7."""
        setups = {}
        for head in HEADS:
            setups.setdefault(self.references[head]["setup"], []).append(head)
        for setup, heads in setups.items():
            jobs = {h: [(e, v) for e, v in REFERENCE_FEATURE_JOBS if self.pending(e, h, v)] for h in heads}
            cached = {h: self.load_reference(h) for h in heads}
            needs_matrix = {h: [j for j in jobs[h] if j[0] != "E1"] for h in heads}
            x_train = x_validation = None
            if any(needs_matrix.values()) or any(c is None for c in cached.values()):
                x_train, x_validation = self.features(setup)
            for head in heads:
                labels_all, log_proba_all, classes = cached[head] or self.fit_reference(
                    head, x_train, x_validation
                )
                self.reference_labels[head] = labels_all
                rows, y_validation = self.labels(self.validation, head)
                if ("E1", "offsets") in jobs[head]:
                    self.offsets(head, rows, y_validation, log_proba_all[rows], classes)
                if not needs_matrix[head]:
                    continue
                train_rows, y_train = self.labels(self.train, head)
                x_train_head, x_validation_head = x_train[train_rows], x_validation[rows]
                for experiment, variant in needs_matrix[head]:
                    if experiment == "E4":
                        self.ordinal(
                            head, x_train_head, y_train, x_validation_head, y_validation, labels_all[rows]
                        )
                    elif experiment == "E6":
                        data = (x_train_head, y_train, x_validation_head, y_validation, labels_all[rows])
                        self.other_model(variant, head, *data)
                    elif experiment == "E13":
                        data = (x_train_head, y_train, x_validation_head, y_validation, rows)
                        self.svm_with_offsets(head, *data, log_proba_all[rows], classes)
                x_train_head = x_validation_head = None  # free the copies before E7 and the next head
                if ("E7", "zero_as_low") in needs_matrix[head]:
                    self.zero_as_low(head, x_train, x_validation, labels_all)
            x_train = x_validation = None

    def offsets(self, head, rows, y_validation, log_proba, classes):
        code = {label: i for i, label in enumerate(classes)}
        true_codes = np.array([code[v] for v in y_validation])
        half_a = product_half(self.validation["product_id"].to_numpy()[rows])
        predicted_codes, per_half = cross_fitted_offsets(log_proba, true_codes, half_a)
        result = compare(
            head, y_validation, np.array(classes)[predicted_codes], self.reference_labels[head][rows]
        )
        extra = {
            "offsets_per_half": {k: dict(zip(classes, v.tolist(), strict=True)) for k, v in per_half.items()},
            "offsets_for_test": dict(
                zip(classes, search_offsets(log_proba, true_codes).tolist(), strict=True)
            ),
            "rows_per_half": {"a": int(half_a.sum()), "b": int((~half_a).sum())},
            "note": "each review is predicted with offsets tuned on the other half of validation",
        }
        self.save("E1", head, "offsets", result, extra=extra)

    def ordinal(self, head, x_train_head, y_train, x_validation_head, y_validation, reference_labels):
        reference = self.references[head]
        order = HEADS[head]["labels"]  # low, medium, high
        codes = np.array([order.index(v) for v in y_train])
        probabilities, converged = [], True
        for threshold in (0, 1):  # above low, above medium
            model, ok = fit_logistic(
                x_train_head, codes > threshold, reference["C"], reference["class_weight"]
            )
            probabilities.append(model.predict_proba(x_validation_head)[:, 1])
            converged &= ok
        predicted = np.array(order)[np.argmax(ordinal_probabilities(*probabilities), axis=1)]
        result = compare(head, y_validation, predicted, reference_labels)
        true_codes = np.array([order.index(v) for v in y_validation])

        def two_steps(labels):
            return np.abs(np.array([order.index(v) for v in labels]) - true_codes) == 2

        variant_errors, reference_errors = two_steps(predicted), two_steps(reference_labels)
        extra = {
            "two_step_errors": {
                "variant_share": float(variant_errors.mean()),
                "reference_share": float(reference_errors.mean()),
                "difference": float(variant_errors.mean() - reference_errors.mean()),
                "difference_interval_95": bootstrap_mean_difference(variant_errors, reference_errors),
            }
        }
        self.save("E4", head, "ordinal", result, converged, extra)

    def other_model(
        self, variant, head, x_train_head, y_train, x_validation_head, y_validation, reference_labels
    ):
        if variant.startswith("svc"):
            c = float(variant.split("_C")[1])
            model = LinearSVC(C=c, class_weight=self.references[head]["class_weight"], max_iter=MAX_ITER)
        else:
            model = ComplementNB(alpha=float(variant.split("alpha")[1]))  # takes no class weights
        model, converged = fit_quietly(model, x_train_head, y_train)
        result = compare(head, y_validation, model.predict(x_validation_head), reference_labels)
        self.save("E6", head, variant, result, converged)

    def svm_with_offsets(
        self, head, x_train_head, y_train, x_validation_head, y_validation, rows, log_proba, classes
    ):
        """E13: the E6 SVM (C = 0.1) against the reference, both with offsets tuned the E1 way.

        E6 compared the SVM with the reference without offsets, while offsets alone add about
        two points to the reference. Here both get offsets, each tuned on the other product half.
        """
        model = LinearSVC(C=0.1, class_weight=self.references[head]["class_weight"], max_iter=MAX_ITER)
        model, converged = fit_quietly(model, x_train_head, y_train)
        assert list(model.classes_.astype(str)) == classes, "SVM and reference classes differ"
        code = {label: i for i, label in enumerate(classes)}
        true_codes = np.array([code[v] for v in y_validation])
        half_a = product_half(self.validation["product_id"].to_numpy()[rows])
        svm_codes, svm_offsets = cross_fitted_offsets(
            model.decision_function(x_validation_head), true_codes, half_a
        )
        reference_codes, reference_offsets = cross_fitted_offsets(log_proba, true_codes, half_a)
        labels = np.array(classes)
        result = compare(head, y_validation, labels[svm_codes], labels[reference_codes])

        def named(per_half):
            return {k: dict(zip(classes, v.tolist(), strict=True)) for k, v in per_half.items()}

        extra = {
            "svm_offsets_per_half": named(svm_offsets),
            "reference_offsets_per_half": named(reference_offsets),
            "note": "both models predict each review with offsets tuned on the other half of validation",
        }
        self.save("E13", head, "svc_C0.1_with_offsets", result, converged, extra)

    def zero_as_low(self, head, x_train, x_validation, reference_labels_all):
        """Train with a rating of 0 counted as low; score on both readings of the validation labels."""
        reference = self.references[head]
        validation_band = rating_band(self.validation["rate"], "low")
        real = self.validation["rating_band"].notna().to_numpy()
        assert (validation_band.to_numpy()[real] == self.validation["rating_band"].to_numpy()[real]).all()

        train_band = rating_band(self.train["rate"], "low")
        train_rows = np.flatnonzero(train_band.notna().to_numpy())
        model, converged = fit_logistic(
            x_train[train_rows], train_band.to_numpy()[train_rows], reference["C"], reference["class_weight"]
        )
        all_rated = np.flatnonzero(validation_band.notna().to_numpy())
        real_ratings = head_rows(self.validation, head)  # a rating above 0, the reference's own rows
        results = {}
        for name, rows, truth in [
            ("zeros_as_low", all_rated, validation_band.to_numpy()[all_rated]),
            ("real_ratings_only", real_ratings, self.validation["rating_band"].to_numpy()[real_ratings]),
        ]:
            results[name] = compare(
                head, truth, model.predict(x_validation[rows]), reference_labels_all[rows]
            )
        extra = {
            "scored_with_zeros_as_low": results["zeros_as_low"],
            "train_rows": int(len(train_rows)),
            "note": "the headline compares both models on reviews with a real rating; the zeros-as-low "
            "scoring favours the variant by construction, since the reference never learned to call a "
            "zero low. Training lacks the zero-rated reviews without a recommendation label (not in "
            "train_supervised.csv), while validation includes them.",
        }
        self.save("E7", head, "zero_as_low", results["real_ratings_only"], converged, extra)

    # ---- experiments with features of their own (E5, E3, E2) ---------------------------------
    def run_with_new_features(self, experiment, variant, make_options):
        setups = {}
        for head in HEADS:
            if self.pending(experiment, head, variant):
                setups.setdefault(self.references[head]["setup"], []).append(head)
        for setup, heads in setups.items():
            options, extra = make_options(setup)
            x_train, x_validation = self.features(setup, **options)  # once for every head of this setup
            for head in heads:
                reference = self.references[head]
                train_rows, y_train = self.labels(self.train, head)
                rows, y_validation = self.labels(self.validation, head)
                model, converged = fit_logistic(
                    x_train[train_rows], y_train, reference["C"], reference["class_weight"]
                )
                predicted = model.predict(x_validation[rows])
                result = compare(head, y_validation, predicted, self.reference_labels[head][rows])
                self.save(experiment, head, variant, result, converged, extra)
            x_train = x_validation = None

    def stopword_options(self, setup):
        token = next(p["token"] for p in FEATURE_SETUPS[setup] if p["analyzer"] == "word")
        words = most_common_words(self.train_texts, TOKEN_PATTERNS[token], STOPWORD_COUNT)
        negations = [w for w in words if w in NEGATION_WORDS or w.startswith("نمی")]
        return {"stop_words": words}, {
            "removed_words": words,
            "negation_words_removed": negations,
            "note": "words are removed before n-grams are built, so a bigram can join the words on "
            "both sides of a removed word",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=ALL_EXPERIMENTS, default=ALL_EXPERIMENTS)
    args = parser.parse_args()

    context = {
        "code": git_commit(),
        "environment": environment(),
        "data": check_cache(CACHE_DIR, CACHE_FILES),
    }
    print(f"Code {context['code']['commit']}, data checked against cache_info.json", flush=True)
    runner = Experiments(args.experiments, context)
    for head, run in runner.references.items():
        print(
            f"Reference for {head}: {run['run']} (macro F1 {run['validation']['macro_f1']:.4f})", flush=True
        )

    runner.run_on_reference_features()
    runner.run_with_new_features("E5", "plain_tf", lambda setup: ({"sublinear_tf": False}, {}))
    runner.run_with_new_features("E3", "frequent_words_removed", runner.stopword_options)
    runner.run_with_new_features("E2", "word_plus_char", lambda setup: ({"extra_parts": [CHAR_PART]}, {}))
    print("Done", flush=True)


if __name__ == "__main__":
    main()
