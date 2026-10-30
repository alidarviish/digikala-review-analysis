import numpy as np
import pytest

from digikala_reviews.difficulty import curriculum_order, gradual_order, hardest, review_difficulty


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


def test_gradual_order_starts_easy_and_ends_mixed():
    rng = np.random.default_rng(0)
    difficulty = rng.random(20_000)
    difficulty[:50] = np.nan
    order = gradual_order(difficulty, seed=1)
    assert sorted(order.tolist()) == list(range(20_000))  # every review once
    rank = np.argsort(np.argsort(difficulty)) / 20_000  # 0 = easiest
    first, last = rank[order[:1000]], rank[order[-4000:]]
    known_first = first[~np.isnan(difficulty[order[:1000]])]
    # the first 5% of the epoch: only reviews from the easier half, most of them very easy
    assert known_first.max() < 0.5 and known_first.mean() < 0.2
    # at the end the hardest tenth makes up more than its tenth, but all difficulties are mixed
    hardest_share = np.mean(last >= 0.9)
    assert 0.12 < hardest_share < 0.2
    assert np.mean(last < 0.5) > 0.35
    assert (gradual_order(difficulty, seed=1) == order).all()  # repeatable
    assert not (gradual_order(difficulty, seed=2) == order).all()  # the seed changes it
