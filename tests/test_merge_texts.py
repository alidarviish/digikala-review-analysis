import numpy as np
import pytest

from digikala_reviews.merge_texts import class_shares, label_counts, merge, mixed

TEXTS = ["a", "a", "a", "b", "c", "c"]
REC = ["x", "x", "y", None, "y", "y"]
BAND = ["low", None, "low", "high", "high", "high"]
HEADS = {"rec": (REC, ["x", "y"]), "band": (BAND, ["low", "high"])}


def test_label_counts_skip_rows_without_a_label():
    found = label_counts(np.array([0, 0, 1]), np.array([1, -1, 0]), n_texts=2, n_classes=2)
    assert found.tolist() == [[0, 1], [1, 0]]


def test_merge_keeps_every_label_and_copy():
    texts, copies, counts = merge(TEXTS, HEADS)
    assert list(texts) == ["a", "b", "c"]
    assert copies.tolist() == [3, 1, 2]
    assert counts["rec"].tolist() == [[2, 1], [0, 0], [0, 2]]
    assert counts["band"].tolist() == [[2, 0], [0, 1], [0, 2]]
    assert counts["rec"].sum() == sum(v is not None for v in REC)


def test_class_shares_as_now_and_once_per_text():
    _, _, counts = merge(TEXTS, HEADS)
    shares = class_shares(counts["rec"], ["x", "y"])
    assert shares["x"]["copies_counted"] == pytest.approx(2 / 5)  # as on the separate rows
    assert shares["x"]["each_text_once"] == pytest.approx((2 / 3 + 0) / 2)  # "b" has no label


def test_mixed_counts_texts_with_more_than_one_label():
    _, _, counts = merge(TEXTS, HEADS)
    found = mixed(counts["rec"])
    assert found["texts_with_a_label"] == 2
    assert found["share_of_texts"] == pytest.approx(1 / 2)  # only "a"
    assert found["share_of_rows"] == pytest.approx(3 / 5)
