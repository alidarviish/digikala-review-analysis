import numpy as np
import pytest

from digikala_reviews.cards import confusion, f1_per_class, loss_comparison, pooled_comparison, row_shares


def test_confusion_rows_are_the_true_class():
    found = confusion([0, 0, 1, 2], [0, 1, 1, 0], n_classes=3)
    assert found.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]
    assert row_shares(found)[0].tolist() == [50.0, 50.0, 0.0]


def test_f1_per_class_matches_the_definition():
    matrix = np.array([[8, 2], [1, 9]])
    precision, recall = 8 / 9, 8 / 10
    assert f1_per_class(matrix)[0] == pytest.approx(2 * precision * recall / (precision + recall))
    assert f1_per_class(np.zeros((2, 2))).tolist() == [0.0, 0.0]  # no review of a class: F1 0, not NaN


def make_runs(seed=0, n=400, seeds=2):
    rng = np.random.default_rng(seed)
    true = rng.integers(0, 3, n)
    reference = np.where(rng.random((seeds, n)) < 0.7, true, rng.integers(0, 3, (seeds, n)))
    variant = np.where(rng.random((seeds, n)) < 0.8, true, rng.integers(0, 3, (seeds, n)))
    groups = {"length": rng.choice(["short", "long"], n)}
    return true, reference, variant, groups


def test_pooled_matrices_are_the_sum_over_seeds():
    true, reference, variant, groups = make_runs()
    found = pooled_comparison(true, reference, variant, groups, 3)
    expected = sum(confusion(true, reference[s], 3) for s in range(2))
    assert (found["reference_matrix"] == expected).all()
    change = f1_per_class(found["variant_matrix"]) - f1_per_class(found["reference_matrix"])
    assert found["f1_change"] == pytest.approx(change)


def test_pooled_group_change_matches_the_group_alone():
    true, reference, variant, groups = make_runs()
    found = pooled_comparison(true, reference, variant, groups, 3)
    inside = groups["length"] == "short"
    ref = sum(confusion(true[inside], reference[s][inside], 3) for s in range(2))
    var = sum(confusion(true[inside], variant[s][inside], 3) for s in range(2))
    short = found["groups"]["length"]["short"]
    assert short["change"] == pytest.approx(f1_per_class(var).mean() - f1_per_class(ref).mean())
    assert short["share"] == pytest.approx(inside.mean())
    low, high = short["interval"]
    assert low < short["change"] < high


def test_intervals_find_a_clear_improvement_and_no_change():
    true, reference, variant, groups = make_runs(n=3000)
    found = pooled_comparison(true, reference, variant, groups, 3)
    assert (found["f1_change_interval"][:, 0] > 0).all()  # the variant is right more often
    same = pooled_comparison(true, reference, reference, groups, 3)
    assert np.abs(same["f1_change_interval"]).max() == 0.0


def test_loss_comparison_is_paired():
    rng = np.random.default_rng(1)
    reference = rng.exponential(0.5, (2, 1000))
    found = loss_comparison(reference, reference - 0.01)  # every review 0.01 lower
    assert found["change"] == pytest.approx(-0.01)
    assert found["interval"] == pytest.approx([-0.01, -0.01])  # paired: no spread at all
    assert found["reference"] == pytest.approx(reference.mean())
