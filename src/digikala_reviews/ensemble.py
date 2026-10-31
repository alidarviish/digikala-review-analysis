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


def stack_features(log_probas) -> np.ndarray:
    """The log probabilities of several models and heads side by side, one row per review,
    floored at -20 so a near-zero probability does not dominate the fit."""
    return np.hstack([np.maximum(np.asarray(lp, dtype=float), -20.0) for lp in log_probas])


def cross_fitted_stack(features, codes, half_a, c: float = 1.0):
    """Stacking: a logistic regression learns the class from the features (the probabilities of
    both models for both heads), fitted on one half of validation together with its offsets,
    and predicts the other half, then the other way round. Returns the predicted codes and the
    log probabilities of every review from the model that did not see it."""
    from sklearn.linear_model import LogisticRegression

    features = np.asarray(features, dtype=float)
    n_classes = int(codes.max()) + 1
    predicted = np.empty(len(codes), dtype=int)
    log_proba = np.empty((len(codes), n_classes))
    for fit_on in (half_a, ~half_a):
        model = LogisticRegression(C=c, max_iter=2000)
        model.fit(features[fit_on], codes[fit_on])
        assert list(model.classes_) == list(range(n_classes)), "every class must be in each half"
        offsets = search_offsets(model.predict_log_proba(features[fit_on]), codes[fit_on])
        log_proba[~fit_on] = model.predict_log_proba(features[~fit_on])
        predicted[~fit_on] = predict_with_offsets(log_proba[~fit_on], offsets)
    return predicted, log_proba


def error_overlap(y_true, predictions: dict) -> dict:
    """How the mistakes of several models overlap: the share of reviews all of them get right,
    all of them get wrong, and the rest (some right, some wrong), where a combination could
    help; and for each pair, the share of either model's mistakes that both make."""
    y_true = np.asarray(y_true)
    right = {name: np.asarray(p) == y_true for name, p in predictions.items()}
    stacked = np.vstack(list(right.values()))
    names = list(right)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            either = ~right[a] | ~right[b]
            pairs[f"{a} & {b}"] = float((~right[a] & ~right[b])[either].mean()) if either.any() else None
    return {
        "all_right": float(stacked.all(axis=0).mean()),
        "all_wrong": float((~stacked).all(axis=0).mean()),
        "some_right": float((stacked.any(axis=0) & ~stacked.all(axis=0)).mean()),
        "both_wrong_share_of_either_wrong": pairs,
    }


def oracle(y_true, predictions: dict, fallback: str) -> np.ndarray:
    """The labels of a perfect chooser: the true label wherever at least one model has it,
    otherwise the fallback model's label. No real combination can score above this."""
    y_true = np.asarray(y_true)
    any_right = np.vstack([np.asarray(p) == y_true for p in predictions.values()]).any(axis=0)
    return np.where(any_right, y_true, np.asarray(predictions[fallback]))


def chooser_model(seed: int):
    """Gradient-boosted trees: they can learn rules such as "on short texts where the models
    disagree, trust this one", which a linear model on the probabilities cannot."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.1, random_state=seed
    )


def cross_fitted_chooser(features, codes, product_id, seed: int):
    """A non-linear model learns the class from the features on one half of validation (by
    product) and predicts the other half, then the other way round. Its offsets are searched
    on predictions it did not fit: within the fitting half, a second split by product gives
    each review a prediction from a model that did not see it. Returns the predicted codes and
    the log probabilities of every review from the model that did not see it."""
    features, codes = np.asarray(features, dtype=float), np.asarray(codes)
    product_id = np.asarray(product_id)
    assert np.issubdtype(product_id.dtype, np.integer), "product ids must be whole numbers"
    half_a = product_id % 2 == 0
    inner = (product_id // 2) % 2 == 0  # a second split by product, independent of the first
    n_classes = int(codes.max()) + 1
    predicted = np.empty(len(codes), dtype=int)
    log_proba = np.empty((len(codes), n_classes))
    for fit_on in (half_a, ~half_a):
        unseen = np.empty((int(fit_on.sum()), n_classes))
        part = inner[fit_on]
        x, y = features[fit_on], codes[fit_on]
        for sub in (part, ~part):
            model = chooser_model(seed).fit(x[sub], y[sub])
            assert list(model.classes_) == list(range(n_classes)), "every class must be in each part"
            unseen[~sub] = np.log(np.maximum(model.predict_proba(x[~sub]), 1e-12))
        offsets = search_offsets(unseen, y)
        model = chooser_model(seed).fit(x, y)
        log_proba[~fit_on] = np.log(np.maximum(model.predict_proba(features[~fit_on]), 1e-12))
        predicted[~fit_on] = predict_with_offsets(log_proba[~fit_on], offsets)
    return predicted, log_proba

