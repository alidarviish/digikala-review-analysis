import numpy as np

from digikala_reviews.analysis import (
    calibration,
    confident_share,
    label_ceiling,
    length_bucket,
    sample_spread,
    scores_by_group,
    word_count,
)

LABELS = ["a", "b"]


def test_length_buckets_cover_every_count():
    counts = word_count(["x", "x y", "x y z w", "w " * 40])
    assert length_bucket(counts).tolist() == ["1-1", "2-3", "4-7", "32+"]


def test_ceiling_uses_the_most_common_label_of_each_text():
    texts = ["good", "good", "good", "bad", "bad", "once"]
    labels = ["a", "a", "b", "b", "b", "a"]
    found = label_ceiling(texts, labels, LABELS)
    assert found["repeated_rows"] == 5  # "once" is not repeated
    assert found["texts_with_conflicting_labels"] == 1
    assert found["ceiling"]["accuracy"] == 4 / 5  # the single "b" copy of "good" cannot be right
    assert found["ceiling"]["recall"] == {"a": 1.0, "b": 2 / 3}
    assert found["rows_mask"].tolist() == [True, True, True, True, True, False]


def test_ceiling_tie_goes_to_the_first_label_for_every_copy():
    found = label_ceiling(["t", "t"], ["b", "a"], LABELS)
    assert found["ceiling"]["accuracy"] == 0.5
    assert found["ceiling"]["recall"] == {"a": 1.0, "b": 0.0}


def test_scores_by_group_splits_rows():
    result = scores_by_group(["a", "b", "a", "b"], ["a", "b", "b", "b"], ["x", "x", "y", "y"], LABELS)
    assert result["x"]["macro_f1"] == 1.0 and result["x"]["share"] == 0.5
    assert result["y"]["recall"] == {"a": 0.0, "b": 1.0}
    assert scores_by_group(["a"], ["a"], ["x"], LABELS, min_rows=2) == {}


def test_sample_spread_is_zero_when_every_answer_is_right():
    y = ["a", "b"] * 500
    result = sample_spread(y, y, LABELS, [100], draws=20)
    assert result["full"] == 1.0 and result["100"]["std"] == 0.0


def test_calibration_of_a_model_that_is_right_as_often_as_it_says():
    confidence = np.array([0.95] * 100 + [0.55] * 100)
    correct = np.array([1] * 95 + [0] * 5 + [1] * 55 + [0] * 45)
    result = calibration(confidence, correct)
    assert result["expected_calibration_error"] < 1e-9
    assert [b["rows"] for b in result["bins"]] == [100, 100]


def test_confident_share_counts_kept_rows_and_accuracy():
    result = confident_share([0.5, 0.85, 0.99], ["a", "b", "a"], [0.8, 0.9], correct=[True, False, True])
    assert result["0.8"]["kept"] == 2 and result["0.8"]["accuracy"] == 0.5
    assert result["0.9"]["predicted_classes"] == {"a": 1.0}
    assert "accuracy" not in confident_share([0.9], ["a"], [0.5])["0.5"]
