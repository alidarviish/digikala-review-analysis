import numpy as np
import pytest

from digikala_reviews.finetune import (
    IGNORE,
    fixed_sample,
    focal_weights,
    label_codes,
    linear_warmup,
    mean_log_loss,
)


def test_fixed_sample_is_sorted_repeatable_and_bounded():
    a, b = fixed_sample(1_000, 50, seed=1), fixed_sample(1_000, 50, seed=1)
    assert (a == b).all() and (np.diff(a) > 0).all() and len(a) == 50
    assert fixed_sample(10, 50, seed=1).tolist() == list(range(10))


def test_label_codes_mark_missing_labels():
    codes = label_codes(["b", None, "a", float("nan")], ["a", "b"])
    assert codes.tolist() == [1, IGNORE, 0, IGNORE]


def test_linear_warmup_rises_then_falls_to_zero():
    factors = [linear_warmup(s, total=10, warmup=2) for s in range(10)]
    assert factors[0] == 0.5 and factors[1] == 1.0
    assert factors[2] == 1.0 and factors[-1] == 0.125
    assert all(a >= b for a, b in zip(factors[1:], factors[2:], strict=False))


def test_mean_log_loss_is_the_cross_entropy_of_the_true_class():
    log_proba = np.log(np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))
    assert np.isclose(mean_log_loss(log_proba, np.array([0, 2])), -(np.log(0.5) + np.log(0.8)) / 2)


def test_focal_weights_move_weight_to_hard_reviews():
    p = np.array([0.95, 0.2])
    assert focal_weights(p, 0).tolist() == [1.0, 1.0]  # the usual loss
    assert focal_weights(p, 1) == pytest.approx([0.05, 0.8])
    assert focal_weights(p, 2) == pytest.approx([0.0025, 0.64])
