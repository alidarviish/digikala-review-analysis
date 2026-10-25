"""Train / validation / test split by product, balanced on both targets.

Every product goes to exactly one part, so the test set measures how the models do on
products they have never seen. Reviews of one product are often alike, and a random split
by review would let a model score well just by remembering products.

How it works: StratifiedGroupKFold cuts the products into 20 folds of about 5% each and
keeps the share of every label combination (recommendation x rating band, 16 in total)
about the same in every fold. 14 folds become train, 3 validation and 3 test (70/15/15).

sklearn places the biggest products first and always in the same folds, whatever the seed.
So which folds become train, validation and test is also drawn with the seed; otherwise the
biggest products would land in train on every seed.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from digikala_reviews.config import SPLIT_FRACTIONS

N_FOLDS = 20


def folds_per_split(fractions: dict[str, float] = SPLIT_FRACTIONS, n_folds: int = N_FOLDS) -> list[str]:
    """Which part each fold goes to: ['train'] * 14 + ['validation'] * 3 + ['test'] * 3."""
    counts = {name: round(share * n_folds) for name, share in fractions.items()}
    exact = all(abs(share * n_folds - counts[name]) < 1e-9 for name, share in fractions.items())
    if not exact or sum(counts.values()) != n_folds:
        raise ValueError(f"{fractions} cannot be made from {n_folds} equal folds")
    return [name for name, count in counts.items() for _ in range(count)]


def joint_label(recommendation: pd.Series, rating_band: pd.Series) -> pd.Series:
    """Both targets in one label, with missing values as their own class ('recommended|high')."""
    return recommendation.fillna("missing").astype(str) + "|" + rating_band.fillna("missing").astype(str)


def assign_splits(product_id: pd.Series, strata: pd.Series, seed: int) -> pd.Series:
    """Return 'train', 'validation' or 'test' for every review; all reviews of a product get the same."""
    strata_codes, _ = pd.factorize(strata)  # numbers are faster than strings for sklearn
    fold = np.empty(len(product_id), dtype=np.int8)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for i, (_, fold_rows) in enumerate(cv.split(np.zeros(len(product_id)), strata_codes, groups=product_id)):
        fold[fold_rows] = i
    fold_to_part = np.array(folds_per_split())[np.random.default_rng(seed).permutation(N_FOLDS)]
    split = fold_to_part[fold]
    return pd.Series(split, index=product_id.index, name="split")


def products_in_several_splits(product_id: pd.Series, split: pd.Series) -> int:
    """Number of products whose reviews ended up in more than one part (must be 0)."""
    return int((split.groupby(product_id).nunique() > 1).sum())


def share_table(values: pd.Series, split: pd.Series) -> pd.DataFrame:
    """Percentage of each value inside every part and in all data, plus the largest gap to all data."""
    table = pd.crosstab(values.fillna("missing"), split, normalize="columns") * 100
    table["all"] = values.fillna("missing").value_counts(normalize=True) * 100
    parts = [name for name in SPLIT_FRACTIONS if name in table.columns]
    table["max_gap_pp"] = table[parts].sub(table["all"], axis=0).abs().max(axis=1)
    return table[[*parts, "all", "max_gap_pp"]]
