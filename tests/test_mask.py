import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from digikala_reviews.mask import select_masked, stable_hash


def test_stable_hash_matches_the_protocol_by_hand():
    by_hand = int(hashlib.sha256(b"digikala-semi-supervised-v1|53672599").hexdigest()[:16], 16)
    assert stable_hash("53672599") == by_hand == 6706765163302456473


@pytest.fixture(scope="module")
def reviews():
    rng = np.random.default_rng(0)
    n = 10_000
    return pd.DataFrame(
        {
            "id": rng.choice(10**8, size=n, replace=False),
            "recommendation": rng.choice(
                ["recommended", "not_recommended", "no_idea", None], size=n, p=[0.6, 0.1, 0.1, 0.2]
            ),
            "split": rng.choice(["train", "validation", "test"], size=n, p=[0.7, 0.15, 0.15]),
        }
    )


@pytest.fixture(scope="module")
def masked(reviews):
    return select_masked(reviews["id"], reviews["recommendation"], reviews["split"])


def test_exactly_floor_20_percent_per_class(reviews, masked):
    train = reviews["split"] == "train"
    for label in ["recommended", "not_recommended", "no_idea"]:
        in_class = train & (reviews["recommendation"] == label)
        assert masked[in_class].sum() == math.floor(0.20 * in_class.sum())


def test_never_masks_validation_test_or_unlabeled(reviews, masked):
    assert not masked[reviews["split"] != "train"].any()
    assert not masked[reviews["recommendation"].isna()].any()


def test_masked_rows_are_the_smallest_hashes(reviews, masked):
    train = reviews[(reviews["split"] == "train") & (reviews["recommendation"] == "no_idea")]
    hashes = train["id"].astype(str).map(stable_hash)
    assert hashes[masked[train.index]].max() < hashes[~masked[train.index]].min()


def test_same_result_in_any_row_order(reviews, masked):
    shuffled = reviews.sample(frac=1, random_state=1)
    again = select_masked(shuffled["id"], shuffled["recommendation"], shuffled["split"])
    assert again.sort_index().equals(masked.sort_index())
