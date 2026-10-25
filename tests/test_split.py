import numpy as np
import pandas as pd
import pytest

from digikala_reviews.split import (
    assign_splits,
    folds_per_split,
    joint_label,
    products_in_several_splits,
    share_table,
)


@pytest.fixture(scope="module")
def reviews():
    """3,000 made-up products with 1-40 reviews each and random labels."""
    rng = np.random.default_rng(0)
    per_product = rng.integers(1, 41, size=3_000)
    product_id = np.repeat(np.arange(3_000), per_product)
    n = len(product_id)
    recommendation = rng.choice(
        ["recommended", "not_recommended", "no_idea", None], size=n, p=[0.6, 0.1, 0.1, 0.2]
    )
    rating_band = rng.choice(["low", "medium", "high", None], size=n, p=[0.1, 0.2, 0.6, 0.1])
    data = pd.DataFrame(
        {"product_id": product_id, "recommendation": recommendation, "rating_band": rating_band}
    )
    data["strata"] = joint_label(data["recommendation"], data["rating_band"])
    return data


@pytest.fixture(scope="module")
def split(reviews):
    return assign_splits(reviews["product_id"], reviews["strata"], seed=2026)


def test_folds_per_split():
    assert folds_per_split() == ["train"] * 14 + ["validation"] * 3 + ["test"] * 3


def test_folds_per_split_rejects_uneven_fractions():
    with pytest.raises(ValueError):
        folds_per_split({"train": 0.72, "validation": 0.14, "test": 0.14})


def test_joint_label_keeps_missing_as_a_class():
    labels = joint_label(pd.Series(["recommended", None]), pd.Series([None, "low"]))
    assert labels.tolist() == ["recommended|missing", "missing|low"]


def test_no_product_in_two_parts(reviews, split):
    assert products_in_several_splits(reviews["product_id"], split) == 0


def test_every_review_gets_a_part(reviews, split):
    assert len(split) == len(reviews)
    assert set(split) == {"train", "validation", "test"}


def test_sizes_close_to_70_15_15(split):
    shares = split.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.02)
    assert shares["validation"] == pytest.approx(0.15, abs=0.02)
    assert shares["test"] == pytest.approx(0.15, abs=0.02)


def test_label_shares_balanced(reviews, split):
    table = share_table(reviews["strata"], split)
    assert table["max_gap_pp"].max() < 2.0


def test_same_seed_same_split(reviews, split):
    again = assign_splits(reviews["product_id"], reviews["strata"], seed=2026)
    assert again.equals(split)


def test_other_seed_other_split(reviews, split):
    other = assign_splits(reviews["product_id"], reviews["strata"], seed=7)
    assert not other.equals(split)


def test_share_table_by_hand():
    values = pd.Series(["a", "a", "b", None])
    split = pd.Series(["train", "train", "test", "test"])
    table = share_table(values, split)
    assert table.loc["a", "train"] == 100 and table.loc["a", "test"] == 0
    assert table.loc["b", "test"] == 50 and table.loc["missing", "test"] == 50
    assert table.loc["a", "all"] == 50
    assert table.loc["a", "max_gap_pp"] == 50
