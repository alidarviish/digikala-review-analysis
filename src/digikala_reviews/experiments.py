"""Building blocks of the TF-IDF experiments (see reports/tfidf_experiment_plan.md)."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from digikala_reviews.metrics import (
    bootstrap_difference,
    classification_report,
    macro_f1_from_codes,
    ordinal_distance_error,
)
from digikala_reviews.tfidf import HEADS

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


def compare(head: str, y_true, variant, reference) -> dict:
    """Scores of a variant next to the reference on the same reviews, with the paired interval."""
    labels = HEADS[head]["labels"]
    scores = classification_report(y_true, variant, labels)
    reference_scores = classification_report(y_true, reference, labels)
    result = {
        "variant_scores": scores,
        "reference_scores": reference_scores,
        "macro_f1_difference": scores["macro_f1"] - reference_scores["macro_f1"],
        "difference_interval_95": bootstrap_difference(y_true, variant, reference, labels),
    }
    if head == "rating_band":
        scores["ordinal"] = ordinal_distance_error(y_true, variant)
        reference_scores["ordinal"] = ordinal_distance_error(y_true, reference)
    low, high = result["difference_interval_95"]
    result["direction"] = "better" if low > 0 else "worse" if high < 0 else "no measurable difference"
    return result


def majority_rows(texts, labels) -> np.ndarray:
    """Positions of the rows to keep when conflicting labels are removed (E11).

    A text that comes with more than one label keeps only the rows with its most common
    label. If the top labels of a text are tied, all its rows are kept, since there is no
    majority to trust. Texts with a single label are never touched.
    """
    table = pd.DataFrame({"text": np.asarray(texts), "label": np.asarray(labels)})
    counts = table.groupby(["text", "label"]).size().rename("n").reset_index()
    top = counts.groupby("text")["n"].transform("max")
    tied = (counts["n"] == top).groupby(counts["text"]).transform("sum") > 1
    counts["keep"] = (counts["n"] == top) | tied
    keep = table.merge(counts[["text", "label", "keep"]], on=["text", "label"], how="left")["keep"]
    return np.flatnonzero(keep.to_numpy())


def nested_samples(n_rows: int, sizes: list[int], seed: int) -> dict[int, np.ndarray]:
    """Random samples of row positions, each sample inside the next larger one (E12)."""
    order = np.random.default_rng(seed).permutation(n_rows)
    return {size: np.sort(order[:size]) for size in sizes if size < n_rows}


def pool_splits(product_id, newest, seed: int) -> dict[str, np.ndarray]:
    """Three ways to choose about the same number of validation reviews from one pool (E9, E10).

    `newest` marks the reviews of the latest year; their number sets the size of every split.
    Returns, for each way, True for the reviews that go to validation:
    by_product (whole products drawn at random), random (single reviews drawn at random)
    and by_time (the newest reviews).
    """
    product_id, newest = np.asarray(product_id), np.asarray(newest, dtype=bool)
    n_rows, size = len(product_id), int(newest.sum())
    rng = np.random.default_rng(seed)
    random = np.zeros(n_rows, dtype=bool)
    random[rng.permutation(n_rows)[:size]] = True
    products, reviews = np.unique(product_id, return_counts=True)
    order = rng.permutation(len(products))
    chosen = products[order][np.cumsum(reviews[order]) <= size]  # whole products, up to the size
    return {"by_product": np.isin(product_id, chosen), "random": random, "by_time": newest}
