"""A small validation sample labeled again by hand, to measure how noisy the labels are.

The sample has a fixed number of reviews per original recommendation label, each text at most
once, plus a few reviews shown a second time to measure how consistent the labeler is.
The labeler sees only the text, in shuffled order; the key (original label, which rows repeat
which) is written to a separate file.
"""

import numpy as np
import pandas as pd

from digikala_reviews.config import RECOMMENDATION_CLASSES

PER_CLASS = {"recommended": 100, "not_recommended": 50, "no_idea": 50}
REPEATS = 20
# What the labeler types, and what it means
CODES = {"1": "recommended", "2": "not_recommended", "3": "no_idea"}


def choose_sample(validation: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rows to label: PER_CLASS reviews per label (each text once), then REPEATS of them again.

    Returns a table with row (the number the labeler sees), id, body, original label and
    repeat_of (the row it repeats, or 0), in shuffled order.
    """
    rng = np.random.default_rng(seed)
    labeled = validation[validation["recommendation_status"].notna()]
    chosen, seen_texts = [], set()
    for label, count in PER_CLASS.items():
        rows = labeled[labeled["recommendation_status"] == label]
        taken = 0
        for position in rng.permutation(len(rows)):
            text = " ".join(str(rows["body"].iloc[position]).split())
            if text in seen_texts:
                continue
            seen_texts.add(text)
            chosen.append(rows.iloc[position])
            taken += 1
            if taken == count:
                break
        assert taken == count, f"not enough different texts for {label}"
    sample = pd.DataFrame(chosen)[["id", "body", "recommendation_status"]].reset_index(drop=True)
    sample = sample.rename(columns={"recommendation_status": "original_label"})
    sample["repeat_of_id"] = 0

    repeats = sample.iloc[rng.choice(len(sample), REPEATS, replace=False)].copy()
    repeats["repeat_of_id"] = repeats["id"]
    table = pd.concat([sample, repeats], ignore_index=True)
    table = table.iloc[rng.permutation(len(table))].reset_index(drop=True)
    table.insert(0, "row", np.arange(1, len(table) + 1))

    first = table[table["repeat_of_id"] == 0]
    first_row = dict(zip(first["id"], first["row"], strict=True))
    pairs = zip(table["id"], table["repeat_of_id"], strict=True)
    table["repeat_of"] = [first_row[i] if repeat else 0 for i, repeat in pairs]
    return table.drop(columns="repeat_of_id")


def read_labels(answers: pd.Series) -> pd.Series:
    """The labeler's codes (1, 2, 3) as label names; anything else is an error."""
    codes = answers.astype(str).str.strip()
    bad = sorted(set(codes) - set(CODES))
    assert not bad, f"labels must be 1, 2 or 3; found {bad}"
    return codes.map(CODES)


def agreement(table: pd.DataFrame, other: str) -> dict:
    """Share of rows where my_label equals the other column, overall and per original label."""
    result = {"rows": int(len(table)), "agree": float((table["my_label"] == table[other]).mean())}
    for label in RECOMMENDATION_CLASSES:
        part = table[table["original_label"] == label]
        result[label] = {"rows": int(len(part)), "agree": float((part["my_label"] == part[other]).mean())}
    return result


def confusion(table: pd.DataFrame) -> dict:
    """How many reviews of each original label got each of my labels."""
    counts = {}
    for label in RECOMMENDATION_CLASSES:
        rows = table["original_label"] == label
        counts[label] = {m: int((rows & (table["my_label"] == m)).sum()) for m in RECOMMENDATION_CLASSES}
    return counts
