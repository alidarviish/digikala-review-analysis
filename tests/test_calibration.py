import numpy as np
import pytest

from digikala_reviews.calibration import above, brier, fit_temperature, mean_loss, reliability, scale


def made_up(n=20_000, sharpness=3.0, seed=0):
    """Three classes; the true class is drawn from honest probabilities, and the model reports
    the same scores multiplied by `sharpness`, so it is too sure by a known factor."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 3)) * 1.5
    honest = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    codes = np.array([rng.choice(3, p=np.exp(row)) for row in honest])
    reported = scale(logits * sharpness, 1.0)
    return reported, honest, codes


def test_scale_keeps_the_order_and_sums_to_one():
    log_proba = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]))
    for t in (0.5, 1.0, 2.0):
        scaled = scale(log_proba, t)
        assert np.exp(scaled).sum(axis=1) == pytest.approx([1, 1])
        assert (scaled.argmax(axis=1) == log_proba.argmax(axis=1)).all()
    assert scale(log_proba, 1.0) == pytest.approx(log_proba)
    assert np.exp(scale(log_proba, 1e6)) == pytest.approx(np.full((2, 3), 1 / 3), abs=1e-4)


def test_fit_temperature_finds_how_much_too_sure_the_model_is():
    reported, honest, codes = made_up(sharpness=3.0)
    t = fit_temperature(reported, codes)
    assert t == pytest.approx(3.0, rel=0.1)
    assert mean_loss(scale(reported, t), codes) < mean_loss(reported, codes)
    assert fit_temperature(honest, codes) == pytest.approx(1.0, rel=0.1)


def test_reliability_and_brier_get_better_after_scaling():
    reported, _, codes = made_up(sharpness=3.0)
    t = fit_temperature(reported, codes)
    before, after = reliability(reported, codes), reliability(scale(reported, t), codes)
    assert after["ece"] < before["ece"] / 3
    assert sum(b["share"] for b in before["bins"]) == pytest.approx(1.0)
    assert brier(scale(reported, t), codes) < brier(reported, codes)


def test_reliability_of_a_perfectly_sure_and_right_model_is_zero():
    codes = np.array([0, 1, 2, 1])
    log_proba = np.log(np.eye(3)[codes] * (1 - 3e-12) + 1e-12)
    assert reliability(log_proba, codes)["ece"] == pytest.approx(0.0, abs=1e-9)
    assert brier(log_proba, codes) == pytest.approx(0.0, abs=1e-9)


def test_above_counts_sure_reviews_and_their_accuracy():
    log_proba = np.log(np.array([[0.95, 0.03, 0.02], [0.5, 0.3, 0.2], [0.05, 0.92, 0.03]]))
    found = above(log_proba, np.array([0, 0, 0]), 0.9)
    assert found["share"] == pytest.approx(2 / 3)
    assert found["accuracy"] == pytest.approx(0.5)
    assert found["mean_probability"] == pytest.approx((0.95 + 0.92) / 2)
    assert above(log_proba, np.array([0, 0, 0]), 0.99)["accuracy"] is None
