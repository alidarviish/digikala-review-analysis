import numpy as np
import pytest

from digikala_reviews.difficulty import curriculum_order, hardest, review_difficulty


def test_hardest_picks_the_lowest_p_among_labeled_reviews():
    p = np.array([0.9, 0.1, np.nan, 0.5, 0.2, 0.8, 0.7, 0.6, 0.3, 0.4, 0.95])
    chosen = hardest(p, 0.2)  # 20% of the 10 labeled reviews
    assert chosen.tolist() == [False, True, False, False, True, False, False, False, False, False, False]
    assert not hardest(p, 0.0).any()
    assert not hardest(p, 1.0)[2]  # never a review without a label


def test_review_difficulty_averages_the_heads_with_a_label():
    found = review_difficulty({"a": [0.8, np.nan, np.nan], "b": [0.4, 0.3, np.nan]})
    assert found[:2] == pytest.approx([(0.2 + 0.6) / 2, 0.7])
    assert np.isnan(found[2])


def test_curriculum_goes_from_easy_to_hard_group_by_group():
    rng = np.random.default_rng(0)
    difficulty = rng.random(1000)
    difficulty[[3, 7]] = np.nan
    order = curriculum_order(difficulty, groups=10, seed=1)
    assert sorted(order.tolist()) == list(range(1000))  # every review once
    known = [d for d in difficulty[order] if not np.isnan(d)]
    groups = [np.mean(known[i : i + 100]) for i in range(0, 900, 100)]
    assert groups == sorted(groups)  # each group harder than the one before
    assert (curriculum_order(difficulty, 10, seed=1) == order).all()  # repeatable
    assert not (curriculum_order(difficulty, 10, seed=2) == order).all()  # the seed changes it
