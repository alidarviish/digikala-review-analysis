"""Helpers shared by the TF-IDF tuning and experiment scripts.

A saved run carries a "fingerprint": everything besides its name that changes the result
(feature settings, iteration limit, input data, library version). Results are reused or
compared only when their fingerprints match, so old and new results never mix silently.
"""

import json
import os
import warnings

import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from digikala_reviews.tfidf import FEATURE_SETUPS, TOKEN_PATTERNS

MAX_ITER = 2000


def run_name(head, setup, class_weight, c):
    return f"{head}__{setup}__{class_weight or 'unweighted'}__C{c:g}"


def fingerprint(setup, data):
    """Everything besides the run name that changes a result. Two runs compare only if these match."""
    parts = []
    for part in FEATURE_SETUPS[setup]:
        part = dict(part)
        if "token" in part:
            part["token_pattern"] = TOKEN_PATTERNS[part["token"]]
        parts.append(part)
    return json.loads(
        json.dumps(
            {
                "setup_parts": parts,
                "max_iter": MAX_ITER,
                "data_sha256": data["sha256"],
                "normalizer_version": data["normalizer_version"],
                "sklearn": sklearn.__version__,
            }
        )
    )  # through JSON so tuples become lists, as in the saved files


def finished(path, expected_fingerprint):
    """True if this result is already saved with the same fingerprint; stops on a stale one."""
    if not path.exists():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["fingerprint"] != expected_fingerprint:
        raise SystemExit(f"{path.name} was made with other settings or data; move its folder aside")
    return True


def save_json(path, record):
    """Write to a temporary file first, so an interruption never leaves half a file behind."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fit_logistic(x, y, c, class_weight):
    """Fit logistic regression; also report whether it really converged."""
    model = LogisticRegression(C=c, class_weight=class_weight, max_iter=MAX_ITER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    converged = model.n_iter_.max() < MAX_ITER and not any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )
    return model, bool(converged)


def best_run(runs):
    """Highest validation Macro F1 among converged runs; ties go to fewer features, then to no weights."""
    converged = [r for r in runs if r["converged"]]
    if not converged:
        return None
    return max(
        converged, key=lambda r: (r["validation"]["macro_f1"], -r["features"], r["class_weight"] is None)
    )
