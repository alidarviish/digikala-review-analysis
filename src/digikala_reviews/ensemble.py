"""Combining the probabilities of several models (reports/ensemble_plan.md)."""

import numpy as np

from digikala_reviews.experiments import predict_with_offsets, search_offsets
from digikala_reviews.metrics import macro_f1_from_codes

WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def mean_probabilities(log_probas) -> np.ndarray:
    """Log of the mean of the models' probabilities."""
    stacked = np.stack([np.asarray(lp, dtype=float) for lp in log_probas])
    return np.logaddexp.reduce(stacked, axis=0) - np.log(len(stacked))


def blend(log_a, log_b, weight: float) -> np.ndarray:
    """Log of weight * p_a + (1 - weight) * p_b."""
    log_a, log_b = np.asarray(log_a, dtype=float), np.asarray(log_b, dtype=float)
    if weight == 1.0:
        return log_a
    if weight == 0.0:
        return log_b
    return np.logaddexp(np.log(weight) + log_a, np.log(1 - weight) + log_b)


def cross_fitted_blend(log_a, log_b, codes, half_a, weights=WEIGHTS):
    """Predict each half with the weight and offsets that score best (Macro F1) on the other
    half, so no review is scored with settings tuned on it. Returns the predicted codes and,
    for each half used for tuning, the weight chosen there."""
    n_classes = np.asarray(log_a).shape[1]
    predicted = np.empty(len(codes), dtype=int)
    chosen = {}
    for name, tune_on in [("tuned_on_a", half_a), ("tuned_on_b", ~half_a)]:
        best = None
        for weight in weights:
            blended = blend(log_a[tune_on], log_b[tune_on], weight)
            offsets = search_offsets(blended, codes[tune_on])
            score = macro_f1_from_codes(codes[tune_on], predict_with_offsets(blended, offsets), n_classes)
            if best is None or score > best[0]:  # the first best weight on ties
                best = (score, weight, offsets)
        _, weight, offsets = best
        predicted[~tune_on] = predict_with_offsets(blend(log_a[~tune_on], log_b[~tune_on], weight), offsets)
        chosen[name] = float(weight)
    return predicted, chosen
