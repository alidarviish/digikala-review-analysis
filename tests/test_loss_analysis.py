import numpy as np
import pytest

from digikala_reviews.loss_analysis import label_entropy_floor, loss_by_group, near_key, repeats, row_loss


def test_near_key_ignores_spacing_punctuation_digits_and_stretched_letters():
    assert near_key("عاااالی!!") == near_key("عالی") == near_key("عا لی 2")
    assert near_key("خوب") != near_key("بد")


def test_row_loss_is_minus_log_of_the_true_class():
    log_proba = np.log(np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))
    assert row_loss(log_proba, np.array([0, 2])) == pytest.approx([np.log(2), -np.log(0.8)])


def test_floor_is_the_entropy_of_each_text_labels():
    keys = ["a", "a", "a", "a", "b", "c", "c"]
    codes = [0, 0, 1, 1, 2, 1, 1]
    floor = label_entropy_floor(keys, codes, n_classes=3, min_copies=2)
    assert floor[:4] == pytest.approx([np.log(2)] * 4)  # half and half
    assert np.isnan(floor[4])  # one copy: no floor
    assert floor[5:] == pytest.approx([0.0, 0.0])  # always the same label
    assert np.isnan(label_entropy_floor(keys, codes, 3, min_copies=3)[5])


def test_loss_by_group_shares_add_up():
    found = loss_by_group(np.array([1.0, 1.0, 2.0]), ["x", "x", "y"])
    assert found["x"]["share_of_loss"] == pytest.approx(0.5)
    assert found["y"]["mean_loss"] == pytest.approx(2.0)
    assert found["x"]["share_of_rows"] == pytest.approx(2 / 3)


def test_repeats_counts_earlier_copies_and_mixed_labels():
    found = repeats(["a", "a", "b", "c", "c"], ["x", "y", "x", "x", None])
    assert found["different_texts"] == 3
    assert found["rows_repeating_an_earlier_text"] == pytest.approx(2 / 5)
    assert found["rows_of_texts_with_more_than_one_label"] == pytest.approx(2 / 5)  # only "a"
