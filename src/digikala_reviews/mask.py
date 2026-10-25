"""The shared masking rule for the semi-supervised experiment.

This follows the project protocol word for word, so every team masks the same rows:
inside each recommendation class of the training split, rows are sorted by
stable_hash(id) and the first floor(MASK_RATE * class_count) rows are masked.
Validation and test rows are never masked.
"""

import hashlib
import math

import pandas as pd

from digikala_reviews.config import MASK_RATE, MASK_SEED


def stable_hash(row_id: str) -> int:
    """Copied from the project protocol; do not change."""
    key = f"{MASK_SEED}|{row_id}".encode("utf-8")  # noqa: UP012 (kept as written in the protocol)
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def select_masked(ids: pd.Series, recommendation: pd.Series, split: pd.Series) -> pd.Series:
    """True for the training rows whose recommendation label is hidden for the experiment.

    `ids` are the review ids, used as text ("53672599") inside the hash.
    """
    # A float id would become "53672599.0" in the hash and silently pick other rows
    assert pd.api.types.is_integer_dtype(ids) or pd.api.types.is_string_dtype(ids), (
        "ids must be whole numbers"
    )
    labeled_train = pd.DataFrame({"id": ids, "recommendation": recommendation})
    labeled_train = labeled_train[(split == "train") & recommendation.notna()]
    labeled_train["hash"] = labeled_train["id"].astype(str).map(stable_hash)
    # id breaks a tie between two equal hashes, which is extremely unlikely
    labeled_train = labeled_train.sort_values(["recommendation", "hash", "id"])

    position = labeled_train.groupby("recommendation").cumcount()
    class_count = labeled_train.groupby("recommendation")["id"].transform("size")
    n_masked = class_count.map(lambda count: math.floor(MASK_RATE * count))
    masked_index = labeled_train.index[position < n_masked]

    masked = pd.Series(False, index=ids.index, name="masked")
    masked[masked_index] = True
    return masked
