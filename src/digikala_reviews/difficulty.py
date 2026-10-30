"""How hard each training review is, and two ways to use it (reports/difficulty_plan.md).

The difficulty of a review for a head is the probability p that a model which never saw the
review gives to its label (out of fold): the lower p, the harder the review.
"""

import numpy as np


def hardest(p_true, fraction: float) -> np.ndarray:
    """True for the given fraction of reviews with the lowest p; reviews without a label (NaN)
    are never chosen, and the fraction is of the reviews that have a label."""
    p_true = np.asarray(p_true, dtype=float)
    labeled = np.flatnonzero(~np.isnan(p_true))
    count = int(round(fraction * len(labeled)))
    chosen = np.zeros(len(p_true), dtype=bool)
    if count:
        order = labeled[np.argsort(p_true[labeled], kind="stable")]
        chosen[order[:count]] = True
    return chosen


def review_difficulty(p_true_by_head: dict) -> np.ndarray:
    """1 - p, averaged over the heads that have a label; NaN when no head has one."""
    stacked = np.vstack([1 - np.asarray(p, dtype=float) for p in p_true_by_head.values()])
    counts = (~np.isnan(stacked)).sum(axis=0)
    sums = np.nansum(stacked, axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def curriculum_order(difficulty, groups: int, seed: int) -> np.ndarray:
    """Positions of the reviews from easy to hard: the reviews are cut into equal groups by
    difficulty, the groups follow each other from the easiest, and the order inside a group is
    random. Reviews with no difficulty (NaN) are spread at random over the whole order."""
    difficulty = np.asarray(difficulty, dtype=float)
    rng = np.random.default_rng(seed)
    key = np.empty(len(difficulty))
    known = np.flatnonzero(~np.isnan(difficulty))
    ranked = known[np.argsort(difficulty[known], kind="stable")]
    for group, part in enumerate(np.array_split(ranked, groups)):
        key[part] = group + rng.random(len(part))  # random order inside the group
    unknown = np.isnan(difficulty)
    key[unknown] = rng.random(unknown.sum()) * groups
    return np.argsort(key, kind="stable")
