"""Search TF-IDF + logistic regression settings on the full training data, scored on validation.

Two stages:
    features        every feature setup with C = 1, with and without class weights
    regularization  for each head, its best setup and class weighting with other values of C

Every fitted model is scored on the full validation set and saved right away as
reports/tfidf_runs/<run>.json. A run whose file exists is skipped only if it was made with
the same settings, data and library version (its "fingerprint"); otherwise the script stops
instead of mixing old and new results. So after an interruption the same command continues
where it stopped. The test set is never read here.

Usage:
    python scripts/tune_tfidf.py --stage features [--setups word12_min5 word12_min5_persian]
    python scripts/tune_tfidf.py --stage regularization
"""

import argparse
import json
import time
from datetime import UTC, datetime

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR
from digikala_reviews.metrics import bootstrap_macro_f1, classification_report, ordinal_distance_error
from digikala_reviews.provenance import check_cache, environment, git_commit
from digikala_reviews.resources import peak_memory_gb, require_free_memory
from digikala_reviews.tfidf import (
    FEATURE_SETUPS,
    HEADS,
    drop_pruned_terms,
    head_rows,
    load_cache,
    make_vectorizer,
)
from digikala_reviews.tuning import (
    MAX_ITER,
    best_run,
    fingerprint,
    finished,
    fit_logistic,
    run_name,
    save_json,
)

RUNS_DIR = REPORTS_DIR / "tfidf_runs"
CACHE_FILES = ["train_supervised.parquet", "validation.parquet"]
CLASS_WEIGHTS = [None, "balanced"]
OTHER_C_VALUES = [0.25, 0.5, 2.0, 4.0]
# Free memory needed before a setup starts, from the measurements in reports/memory_measurements.json
MEMORY_NEEDED_GB = {"word123_min5_persian": 7.0}
DEFAULT_MEMORY_GB = 5.0


def run_setup(stage, setup, todo, context):
    """Build the features of one setup once, then fit and score every pending model on them."""
    expected = fingerprint(setup, context["data"])
    pending = [t for t in todo if not finished(RUNS_DIR / f"{run_name(*t)}.json", expected)]
    print(f"{setup}: {len(todo) - len(pending)} done, {len(pending)} to run", flush=True)
    if not pending:
        return
    require_free_memory(MEMORY_NEEDED_GB.get(setup, DEFAULT_MEMORY_GB), f"building features for {setup}")

    start = time.perf_counter()
    train, validation = load_cache(CACHE_FILES[0]), load_cache(CACHE_FILES[1])
    vectorizer = make_vectorizer(setup)
    x_train = vectorizer.fit_transform(train.pop("text"))
    drop_pruned_terms(vectorizer)
    x_validation = vectorizer.transform(validation.pop("text"))
    load_and_vectorize_seconds = time.perf_counter() - start
    print(f"  features: {x_train.shape[1]:,} columns ({load_and_vectorize_seconds:.0f}s)", flush=True)

    current_head = x_train_head = x_validation_head = None
    for head, setup_name, class_weight, c in sorted(pending, key=lambda t: t[0]):
        if head != current_head:  # keep only one head's rows in memory at a time
            current_head = head
            x_train_head = x_validation_head = None  # free the previous head's copy first
            column, labels = HEADS[head]["column"], HEADS[head]["labels"]
            train_rows, validation_rows = head_rows(train, head), head_rows(validation, head)
            x_train_head, x_validation_head = x_train[train_rows], x_validation[validation_rows]
            y_train = train[column].to_numpy()[train_rows]
            y_validation = validation[column].to_numpy()[validation_rows]

        start = time.perf_counter()
        model, converged = fit_logistic(x_train_head, y_train, c, class_weight)
        fit_seconds = time.perf_counter() - start
        predicted = model.predict(x_validation_head)

        scores = classification_report(y_validation, predicted, labels)
        scores["macro_f1_interval_95"] = bootstrap_macro_f1(y_validation, predicted, labels)
        if head == "rating_band":
            scores["ordinal"] = ordinal_distance_error(y_validation, predicted)

        name = run_name(head, setup_name, class_weight, c)
        save_json(
            RUNS_DIR / f"{name}.json",
            {
                "run": name,
                "stage": stage,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "head": head,
                "setup": setup_name,
                "setup_parts": FEATURE_SETUPS[setup_name],
                "C": c,
                "class_weight": class_weight,
                "max_iter": MAX_ITER,
                "iterations": int(model.n_iter_.max()),
                "converged": bool(converged),
                "features": int(x_train.shape[1]),
                "train_rows": int(len(train_rows)),
                "validation_rows": int(len(validation_rows)),
                "load_and_vectorize_seconds": round(load_and_vectorize_seconds, 1),
                "fit_seconds": round(fit_seconds, 1),
                "process_peak_memory_gb": round(peak_memory_gb(), 2),  # highest so far in this process
                "fingerprint": expected,
                "validation": scores,
                **context,
            },
        )
        low, high = scores["macro_f1_interval_95"]
        print(
            f"  {name}: macro F1 {scores['macro_f1']:.4f} [{low:.4f}, {high:.4f}] "
            f"({fit_seconds / 60:.1f} min, peak so far {peak_memory_gb():.1f} GB)"
            + ("" if converged else "  NOT CONVERGED, left out of the selection"),
            flush=True,
        )


def best_of_features_stage(data):
    """For each head, the converged stage 1 run with the highest validation Macro F1.

    Every stage 1 run must exist with the current fingerprint. Ties go to fewer features,
    then to no class weights, so the choice never depends on file order.
    """
    best = {}
    for head in HEADS:
        candidates = []
        for setup in FEATURE_SETUPS:
            for class_weight in CLASS_WEIGHTS:
                name = run_name(head, setup, class_weight, 1.0)
                if not finished(RUNS_DIR / f"{name}.json", fingerprint(setup, data)):
                    raise SystemExit(f"Stage 1 is not complete: {name} is missing")
                run = json.loads((RUNS_DIR / f"{name}.json").read_text(encoding="utf-8"))
                candidates.append(run)
                if not run["converged"]:
                    print(f"  {name} did not converge and is left out")
        best[head] = best_run(candidates)
        if best[head] is None:
            raise SystemExit(f"No converged stage 1 run for {head}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["features", "regularization"], required=True)
    parser.add_argument("--setups", nargs="+", choices=list(FEATURE_SETUPS), default=list(FEATURE_SETUPS))
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    context = {
        "code": git_commit(),
        "environment": environment(),
        "data": check_cache(CACHE_DIR, CACHE_FILES),
    }
    print(f"Code {context['code']['commit']}, data checked against cache_info.json", flush=True)

    if args.stage == "features":
        for setup in args.setups:
            todo = [(head, setup, weight, 1.0) for head in HEADS for weight in CLASS_WEIGHTS]
            run_setup("features", setup, todo, context)
    else:
        best = best_of_features_stage(context["data"])
        for head, run in best.items():
            print(f"{head}: best is {run['run']} (macro F1 {run['validation']['macro_f1']:.4f})")
        for setup in sorted({run["setup"] for run in best.values()}):
            todo = [
                (head, setup, run["class_weight"], c)
                for head, run in best.items()
                if run["setup"] == setup
                for c in OTHER_C_VALUES
            ]
            run_setup("regularization", setup, todo, context)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
