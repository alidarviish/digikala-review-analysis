"""Building blocks of the TF-IDF experiments (see reports/tfidf_experiment_plan.md)."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from digikala_reviews.metrics import macro_f1_from_codes

OFFSET_GRID = np.round(np.arange(-2.0, 2.01, 0.1), 2)


def predict_with_offsets(log_proba: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Class codes after adding one offset per class to the log probabilities (E1)."""
    return np.argmax(log_proba + offsets, axis=1)


def search_offsets(log_proba: np.ndarray, true_codes: np.ndarray, passes: int = 3) -> np.ndarray:
    """Offsets that maximize Macro F1, found one class at a time (the first class stays at 0)."""
    n_classes = log_proba.shape[1]
    offsets = np.zeros(n_classes)
    for _ in range(passes):
        for k in range(1, n_classes):
            scores = []
            for value in OFFSET_GRID:
                trial = offsets.copy()
                trial[k] = value
                predicted = predict_with_offsets(log_proba, trial)
                scores.append(macro_f1_from_codes(true_codes, predicted, n_classes))
            offsets[k] = OFFSET_GRID[int(np.argmax(scores))]  # first best value on ties
    return offsets


def cross_fitted_offsets(log_proba: np.ndarray, true_codes: np.ndarray, half_a: np.ndarray):
    """Predict each half of validation with offsets tuned on the other half (E1).

    Returns the predicted codes and the offsets found on each half, so no review is scored
    with offsets that were tuned on it.
    """
    predicted = np.empty(len(true_codes), dtype=int)
    offsets = {}
    for name, tune_on in [("tuned_on_a", half_a), ("tuned_on_b", ~half_a)]:
        found = search_offsets(log_proba[tune_on], true_codes[tune_on])
        predicted[~tune_on] = predict_with_offsets(log_proba[~tune_on], found)
        offsets[name] = found
    return predicted, offsets


def ordinal_probabilities(p_above_low: np.ndarray, p_above_medium: np.ndarray) -> np.ndarray:
    """Frank and Hall: class probabilities from two "above this band?" models (E4)."""
    low = 1 - p_above_low
    medium = np.clip(p_above_low - p_above_medium, 0, None)
    high = p_above_medium
    return np.column_stack([low, medium, high])


def product_half(product_id: np.ndarray) -> np.ndarray:
    """Splits validation into two halves by product (even or odd id), so no product is in both."""
    product_id = np.asarray(product_id)
    assert np.issubdtype(product_id.dtype, np.integer), "product ids must be whole numbers"
    return product_id % 2 == 0


def most_common_words(texts, token_pattern: str, k: int) -> list[str]:
    """The k words that appear in the most texts (each text counts a word once) (E3).

    Tokens are lowercased as the vectorizer does, and only tokens with at least one letter
    count as words, so punctuation marks and emoji are not removed by this experiment.
    """
    counter = CountVectorizer(token_pattern=token_pattern, lowercase=True, binary=True)
    in_texts = np.asarray(counter.fit_transform(texts).sum(axis=0)).ravel()
    words = counter.get_feature_names_out()
    order = sorted(range(len(words)), key=lambda i: (-in_texts[i], words[i]))
    return [str(words[i]) for i in order if any(ch.isalpha() for ch in words[i])][:k]
