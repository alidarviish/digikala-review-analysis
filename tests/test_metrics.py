import numpy as np
import pytest
from sklearn.metrics import f1_score

from digikala_reviews.metrics import (
    bootstrap_macro_f1,
    classification_report,
    macro_f1_from_codes,
    ordinal_distance_error,
)

LABELS = ["low", "medium", "high"]
TRUE = ["low", "low", "medium", "high", "high", "high"]
PRED = ["low", "medium", "medium", "high", "low", "high"]


def test_report_by_hand():
    report = classification_report(TRUE, PRED, LABELS)
    # low: 1 correct of 2 true, 2 predicted -> P 0.5, R 0.5
    assert report["per_class"]["low"]["precision"] == pytest.approx(0.5)
    assert report["per_class"]["low"]["recall"] == pytest.approx(0.5)
    assert report["per_class"]["high"]["recall"] == pytest.approx(2 / 3)
    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["confusion_matrix"]["rows_are_true_class"][2] == [1, 0, 2]
    assert report["macro_f1"] == pytest.approx(f1_score(TRUE, PRED, average="macro"))


def test_ordinal_distance_by_hand():
    result = ordinal_distance_error(TRUE, PRED)
    # distances: 0, 1, 0, 0, 2, 0
    assert result["mean_distance"] == pytest.approx(3 / 6)
    assert result["share_off_by_two"] == pytest.approx(1 / 6)


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    true, pred = rng.integers(0, 3, 5_000), rng.integers(0, 3, 5_000)
    assert macro_f1_from_codes(true, pred, 3) == pytest.approx(f1_score(true, pred, average="macro"))


def test_bootstrap_interval_contains_the_score():
    rng = np.random.default_rng(1)
    true = rng.choice(LABELS, 3_000)
    pred = np.where(rng.random(3_000) < 0.7, true, rng.choice(LABELS, 3_000))
    low, high = bootstrap_macro_f1(true, pred, LABELS, rounds=200)
    assert low < f1_score(true, pred, average="macro") < high
    assert high - low < 0.06
