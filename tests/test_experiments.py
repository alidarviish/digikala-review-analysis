import numpy as np

from digikala_reviews.experiments import (
    cross_fitted_offsets,
    most_common_words,
    ordinal_probabilities,
    predict_with_offsets,
    product_half,
    search_offsets,
)
from digikala_reviews.metrics import bootstrap_difference, bootstrap_mean_difference, macro_f1_from_codes


def test_offsets_change_the_decision():
    log_proba = np.log([[0.6, 0.4], [0.55, 0.45]])
    assert predict_with_offsets(log_proba, np.zeros(2)).tolist() == [0, 0]
    assert predict_with_offsets(log_proba, np.array([0.0, 0.3])).tolist() == [0, 1]


def test_offset_search_helps_an_under_predicted_class():
    rng = np.random.default_rng(0)
    true = rng.choice(3, 5_000, p=[0.8, 0.1, 0.1])
    proba = np.full((5_000, 3), 0.1)
    proba[np.arange(5_000), true] += rng.uniform(0.0, 0.5, 5_000)  # true class only sometimes on top
    proba[:, 0] += 0.25  # a model that leans to the big class
    log_proba = np.log(proba / proba.sum(axis=1, keepdims=True))
    before = macro_f1_from_codes(true, predict_with_offsets(log_proba, np.zeros(3)), 3)
    offsets = search_offsets(log_proba, true)
    after = macro_f1_from_codes(true, predict_with_offsets(log_proba, offsets), 3)
    assert offsets[0] == 0 and after > before


def test_ordinal_probabilities_by_hand():
    result = ordinal_probabilities(np.array([0.9, 0.2]), np.array([0.3, 0.25]))
    np.testing.assert_allclose(result[0], [0.1, 0.6, 0.3])
    np.testing.assert_allclose(result[1], [0.8, 0.0, 0.25])  # a negative medium is cut to 0


def test_product_half_keeps_products_together():
    halves = product_half(np.array([10, 10, 11, 12, 11]))
    assert halves.tolist() == [True, True, False, True, False]


def test_most_common_words_counts_each_text_once():
    texts = ["خوب خوب خوب", "خوب بود", "بد بود", "بد"]
    assert most_common_words(texts, r"\w+", 2) == ["بد", "بود"]  # tie at 2 texts, then alphabetical


def test_most_common_words_skips_punctuation_and_lowercases():
    texts = ["Good !", "good !", "خوب !"]
    assert most_common_words(texts, r"\w+|[^\w\s]", 1) == ["good"]


def test_cross_fitting_never_uses_a_rows_own_half():
    rng = np.random.default_rng(2)
    true = rng.choice(3, 2_000, p=[0.7, 0.2, 0.1])
    log_proba = np.log(rng.dirichlet([1, 1, 1], 2_000))
    half_a = rng.random(2_000) < 0.5
    predicted, offsets = cross_fitted_offsets(log_proba, true, half_a)
    expected_b = predict_with_offsets(log_proba[~half_a], offsets["tuned_on_a"])
    expected_a = predict_with_offsets(log_proba[half_a], offsets["tuned_on_b"])
    assert (predicted[~half_a] == expected_b).all() and (predicted[half_a] == expected_a).all()


def test_mean_difference_interval():
    a, b = np.ones(1_000), np.zeros(1_000)
    assert bootstrap_mean_difference(a, b, rounds=50) == [1.0, 1.0]


def test_bootstrap_difference_is_zero_for_identical_models():
    labels = ["a", "b"]
    y = ["a", "b"] * 500
    low, high = bootstrap_difference(y, y, y, labels, rounds=50)
    assert low == high == 0
