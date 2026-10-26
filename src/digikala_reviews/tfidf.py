"""TF-IDF features for the baseline models.

The vocabulary and the IDF weights are learned from the training texts only and then
applied unchanged to validation and test.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union

from digikala_reviews.config import CACHE_DIR, RATING_BAND_CLASSES, RECOMMENDATION_CLASSES

HEADS = {
    "recommendation": {"column": "recommendation_status", "labels": RECOMMENDATION_CLASSES},
    "rating_band": {"column": "rating_band", "labels": RATING_BAND_CLASSES},
}

ZWNJ = "‌"  # half-space, written as a code so it can be seen in the source
LETTER = r"[\wً-ٰٟ]"  # a letter or digit, or an Arabic vowel mark that belongs to it
EMOJI_TAIL = r"[︎️\U0001f3fb-\U0001f3ff]*"  # emoji style and skin tone marks stay attached
TOKEN_PATTERNS = {
    # sklearn's default: words of 2+ letters; drops emoji, punctuation and one-letter words,
    # and splits words at the half-space (می‌خواهم -> می, خواهم)
    "sklearn": r"(?u)\b\w\w+\b",
    # words of any length, kept whole across half-spaces, plus every emoji or punctuation mark
    "persian": rf"(?u){LETTER}+(?:{ZWNJ}{LETTER}+)*|[^\w\s{ZWNJ}ً-ٰٟ]{EMOJI_TAIL}",
}

# Each setup is a list of parts; several parts would be joined side by side.
# Character n-grams are left out: alone they scored 3 points lower than word 1-2-grams
# (reports/memory_measurements.json), and joined with words they would not fit in 12 GB.
FEATURE_SETUPS = {
    "word12_min2": [{"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "token": "sklearn"}],
    "word12_min5": [{"analyzer": "word", "ngram_range": (1, 2), "min_df": 5, "token": "sklearn"}],
    "word12_min5_persian": [{"analyzer": "word", "ngram_range": (1, 2), "min_df": 5, "token": "persian"}],
    "word123_min5_persian": [{"analyzer": "word", "ngram_range": (1, 3), "min_df": 5, "token": "persian"}],
}


def make_vectorizer(setup_name: str, sublinear_tf: bool = True, stop_words=None, extra_parts=()):
    """The vectorizer of a named setup. The other arguments are used only by the experiments."""
    parts = []
    for part in [*FEATURE_SETUPS[setup_name], *extra_parts]:
        options = {k: v for k, v in part.items() if k != "token"}
        if part["analyzer"] == "word":
            options["token_pattern"] = TOKEN_PATTERNS[part["token"]]
            options["stop_words"] = stop_words
        parts.append(TfidfVectorizer(sublinear_tf=sublinear_tf, dtype=np.float32, **options))
    return parts[0] if len(parts) == 1 else make_union(*parts)


def drop_pruned_terms(vectorizer) -> None:
    """sklearn keeps every term it dropped as rare; that set is large and never used again."""
    for part in getattr(vectorizer, "transformer_list", [(None, vectorizer)]):
        if hasattr(part[1], "stop_words_"):
            part[1].stop_words_ = None


def load_cache(name: str, cache_dir=CACHE_DIR, extra_columns=()) -> pd.DataFrame:
    columns = ["id", "text", "recommendation_status", "rating_band", *extra_columns]
    return pd.read_parquet(cache_dir / name, columns=columns)


def head_rows(table: pd.DataFrame, head: str) -> np.ndarray:
    """Positions of the rows that have a label for this head."""
    return np.flatnonzero(table[HEADS[head]["column"]].notna().to_numpy())


def n_features(matrix: sparse.spmatrix) -> int:
    return int(matrix.shape[1])
