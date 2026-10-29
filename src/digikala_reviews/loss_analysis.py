"""Where a model's validation loss comes from, and how much of it no model could remove.

The loss of one review is -log p(true label). A text that comes with different labels sets
a floor: a model that sees only the text gives every copy the same probabilities, and the
best it can do is the share of each label among the copies. The cross-entropy of those shares
(the entropy of the labels of the text) is that floor.
"""

import re

import numpy as np
import pandas as pd

# Everything that is not a letter is dropped and runs of the same letter are shortened to one,
# so "عاااالی!!" and "عالی" get the same key. Used to find texts that are nearly the same.
NOT_A_LETTER = re.compile(r"[\W\d_]+")
REPEATED_LETTER = re.compile(r"(.)\1+")


def near_key(text: str) -> str:
    """A key shared by texts that differ only in spacing, punctuation, digits or stretched letters."""
    return REPEATED_LETTER.sub(r"\1", NOT_A_LETTER.sub("", text))


def row_loss(log_proba: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """-log p(true label) of every row."""
    return -log_proba[np.arange(len(codes)), codes]


def label_entropy_floor(keys, codes, n_classes: int, min_copies: int) -> np.ndarray:
    """For rows whose text has at least min_copies copies: the entropy of that text's labels.

    Rows of rarer texts get NaN. The shares are measured on the same rows they are applied to,
    so the floor is slightly optimistic, less so for texts with many copies.
    """
    table = pd.DataFrame({"key": np.asarray(keys), "code": np.asarray(codes)})
    counts = pd.crosstab(table["key"], table["code"]).reindex(columns=range(n_classes), fill_value=0)
    totals = counts.sum(axis=1)
    shares = counts.div(totals, axis=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(shares > 0, shares * np.log(shares), 0.0).sum(axis=1)
    per_key = pd.Series(np.where(totals.to_numpy() >= min_copies, entropy, np.nan), index=counts.index)
    return per_key.loc[table["key"]].to_numpy()


def loss_by_group(loss: np.ndarray, groups) -> dict:
    """Rows, mean loss and share of the total loss in each group."""
    loss, groups = np.asarray(loss, dtype=float), np.asarray(groups)
    total = loss.sum()
    result = {}
    for group in pd.unique(groups):
        inside = groups == group
        result[str(group)] = {
            "rows": int(inside.sum()),
            "share_of_rows": float(inside.mean()),
            "mean_loss": float(loss[inside].mean()),
            "share_of_loss": float(loss[inside].sum() / total),
        }
    return result


def repeats(texts, codes=None) -> dict:
    """How many rows repeat a text that came earlier, and how many texts carry more than one label."""
    texts = pd.Series(np.asarray(texts))
    result = {
        "rows": int(len(texts)),
        "different_texts": int(texts.nunique()),
        "rows_repeating_an_earlier_text": float(texts.duplicated().mean()),
    }
    if codes is not None:
        labels_per_text = pd.Series(np.asarray(codes)).groupby(texts.to_numpy()).nunique()
        mixed = labels_per_text[labels_per_text > 1].index
        result["rows_of_texts_with_more_than_one_label"] = float(texts.isin(mixed).mean())
    return result
