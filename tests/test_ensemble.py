import numpy as np
import pytest

from digikala_reviews.ensemble import blend, cross_fitted_blend, mean_probabilities


def test_mean_probabilities_averages_in_probability_space():
    a = np.log(np.array([[0.8, 0.2], [0.5, 0.5]]))
    b = np.log(np.array([[0.4, 0.6], [0.1, 0.9]]))
    assert np.exp(mean_probabilities([a, b])) == pytest.approx(np.array([[0.6, 0.4], [0.3, 0.7]]))


def test_blend_weights_the_two_models():
    a = np.log(np.array([[0.8, 0.2]]))
    b = np.log(np.array([[0.2, 0.8]]))
    assert np.exp(blend(a, b, 0.75)) == pytest.approx(np.array([[0.65, 0.35]]))
    assert blend(a, b, 1.0) == pytest.approx(a)
    assert blend(a, b, 0.0) == pytest.approx(b)


def test_cross_fitted_blend_finds_the_better_model_and_never_tunes_on_the_scored_half():
    rng = np.random.default_rng(0)
    n = 4000
    codes = rng.integers(0, 3, n)
    good = np.full((n, 3), np.log(0.1))
    good[np.arange(n), codes] = np.log(0.8)  # always right
    noise = np.log(rng.dirichlet(np.ones(3), n))  # no information
    half_a = rng.random(n) < 0.5
    predicted, chosen = cross_fitted_blend(good, noise, codes, half_a)
    assert (predicted == codes).all()
    assert set(chosen) == {"tuned_on_a", "tuned_on_b"}
    assert all(w >= 0.5 for w in chosen.values())
