import numpy as np
import pytest

from digikala_reviews.merge_texts import (
    class_shares,
    label_counts,
    merge,
    mixed,
    soft_targets,
    weighted_soft_loss,
)

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


def test_soft_targets_shares_and_weights():
    _, _, counts = merge(TEXTS, HEADS)
    shares, weights = soft_targets(counts["rec"], "copies")
    assert shares[0] == pytest.approx([2 / 3, 1 / 3])
    assert shares[1] == pytest.approx([0, 0])  # "b" has no recommendation
    assert weights.tolist() == [3, 0, 2]
    assert soft_targets(counts["rec"], "once")[1].tolist() == [1, 0, 1]


def test_merged_loss_with_copies_equals_loss_on_separate_rows():
    rng = np.random.default_rng(0)
    texts = rng.choice(list("abcdefg"), 200)
    labels = rng.choice(["x", "y", "z", None], 200)
    merged_texts, _, counts = merge(texts, {"h": (labels, ["x", "y", "z"])})
    logits = {t: rng.normal(size=3) for t in merged_texts}
    log_p = {t: v - np.log(np.exp(v).sum()) for t, v in logits.items()}

    labeled = [i for i, v in enumerate(labels) if v is not None]
    code = {"x": 0, "y": 1, "z": 2}
    separate = -np.mean([log_p[texts[i]][code[labels[i]]] for i in labeled])

    shares, weights = soft_targets(counts["h"], "copies")
    merged_log_p = np.array([log_p[t] for t in merged_texts])
    assert weighted_soft_loss(merged_log_p, shares, weights) == pytest.approx(separate)


def test_soft_loss_with_one_hot_shares_is_plain_cross_entropy():
    log_p = np.log(np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))
    one_hot = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    expected = -(np.log(0.5) + np.log(0.8)) / 2
    assert weighted_soft_loss(log_p, one_hot, np.ones(2)) == pytest.approx(expected)
