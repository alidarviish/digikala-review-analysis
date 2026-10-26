"""Evaluation measures shared by every model, so TF-IDF and the Transformer are scored the same way.

Required by the project: Macro F1, per-class precision / recall / F1, the confusion
matrix, balanced accuracy and, for the ordered rating bands, the ordinal distance error.
A bootstrap interval shows how much a score could move by chance.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from digikala_reviews.config import RATING_BAND_CLASSES


def classification_report(y_true, y_pred, labels: list[str]) -> dict:
    """Every required score for one head, in a form that can be saved as JSON."""
    unknown = (set(y_true) | set(y_pred)) - set(labels)
    if unknown:
        raise ValueError(f"labels outside {labels}: {sorted(unknown)}")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "rows": int(len(y_true)),
        "macro_f1": float(np.mean(f1)),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "balanced_accuracy": float(np.mean(np.diag(matrix) / matrix.sum(axis=1))),  # mean recall
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": {"labels": labels, "rows_are_true_class": matrix.tolist()},
    }


def ordinal_distance_error(y_true, y_pred, order: list[str] = RATING_BAND_CLASSES) -> dict:
    """How many bands a prediction is off on average (low vs high counts as 2 steps)."""
    code = {label: i for i, label in enumerate(order)}
    distance = np.abs(np.array([code[v] for v in y_pred]) - np.array([code[v] for v in y_true]))
    return {
        "mean_distance": float(distance.mean()),
        "share_off_by_two": float((distance == 2).mean()),
    }


def macro_f1_from_codes(true_codes: np.ndarray, predicted_codes: np.ndarray, n_classes: int) -> float:
    """Macro F1 from integer class codes, fast enough to repeat a thousand times."""
    matrix = np.bincount(true_codes * n_classes + predicted_codes, minlength=n_classes**2)
    matrix = matrix.reshape(n_classes, n_classes)
    correct = np.diag(matrix).astype(float)
    predicted, actual = matrix.sum(axis=0), matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(predicted + actual > 0, 2 * correct / (predicted + actual), 0.0)
    return float(f1.mean())


def bootstrap_macro_f1(
    y_true, y_pred, labels: list[str], rounds: int = 1000, seed: int = 2026
) -> list[float]:
    """95% interval of Macro F1, resampling the evaluated reviews with replacement."""
    code = {label: i for i, label in enumerate(labels)}
    true_codes = np.array([code[v] for v in y_true])
    predicted_codes = np.array([code[v] for v in y_pred])
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(rounds):
        rows = rng.integers(0, len(true_codes), len(true_codes))
        scores.append(macro_f1_from_codes(true_codes[rows], predicted_codes[rows], len(labels)))
    low, high = np.percentile(scores, [2.5, 97.5])
    return [float(low), float(high)]
