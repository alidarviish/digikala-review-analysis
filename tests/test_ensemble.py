import numpy as np
import pytest

from digikala_reviews.ensemble import (
    blend,
    cross_fitted_blend,
    cross_fitted_chooser,
    cross_fitted_stack,
    error_overlap,
    mean_probabilities,
    oracle,
    stack_features,
)


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


def test_stacking_uses_the_other_head_when_it_carries_the_answer():
    rng = np.random.default_rng(1)
    n = 6000
    codes = rng.integers(0, 3, n)
    useless = np.log(rng.dirichlet(np.ones(3), n))  # this head's own model knows nothing
    other = np.full((n, 3), np.log(0.1))
    other[np.arange(n), codes] = np.log(0.8)  # the other head's model points to the answer
    half_a = rng.random(n) < 0.5
    alone, _ = cross_fitted_stack(stack_features([useless]), codes, half_a)
    together, log_proba = cross_fitted_stack(stack_features([useless, other]), codes, half_a)
    assert np.mean(together == codes) > 0.95
    assert np.mean(alone == codes) < 0.5
    assert np.exp(log_proba).sum(axis=1) == pytest.approx(np.ones(n))


def test_stack_features_floors_the_log_probabilities():
    a = np.log(np.array([[1e-30, 1 - 1e-30]]))
    assert stack_features([a, a]).shape == (1, 4)
    assert stack_features([a]).min() == pytest.approx(-20.0)


def test_error_overlap_and_oracle():
    y = np.array(["a", "a", "b", "b"])
    predictions = {"x": np.array(["a", "b", "a", "b"]), "y": np.array(["a", "a", "a", "a"])}
    found = error_overlap(y, predictions)
    assert found["all_right"] == pytest.approx(0.25)  # review 0
    assert found["all_wrong"] == pytest.approx(0.25)  # review 2
    assert found["some_right"] == pytest.approx(0.5)  # reviews 1 and 3
    assert found["both_wrong_share_of_either_wrong"]["x & y"] == pytest.approx(1 / 3)
    assert oracle(y, predictions, "x").tolist() == ["a", "a", "a", "b"]


def test_chooser_learns_which_model_to_trust_from_another_feature():
    rng = np.random.default_rng(2)
    n = 8000
    codes = rng.integers(0, 3, n)
    short = rng.random(n) < 0.5  # on short texts model a is right, on long texts model b
    wrong = (codes + rng.integers(1, 3, n)) % 3  # either other class, so the pair alone says nothing
    a = np.full((n, 3), np.log(0.1))
    a[np.arange(n), np.where(short, codes, wrong)] = np.log(0.8)
    b = np.full((n, 3), np.log(0.1))
    b[np.arange(n), np.where(short, wrong, codes)] = np.log(0.8)
    product_id = rng.integers(0, 2000, n)
    features = np.column_stack([a, b])
    without, _ = cross_fitted_chooser(features, codes, product_id, seed=1)
    with_length, log_proba = cross_fitted_chooser(
        np.column_stack([features, short]), codes, product_id, seed=1
    )
    assert np.mean(with_length == codes) > 0.95
    assert np.mean(without == codes) < 0.6  # without the length it cannot tell them apart
    assert np.exp(log_proba).sum(axis=1) == pytest.approx(np.ones(n), abs=1e-6)
