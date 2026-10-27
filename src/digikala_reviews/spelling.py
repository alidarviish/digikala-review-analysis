"""Fixes for frequent misspellings (E8 in reports/tfidf_experiment_plan.md).

resources/spelling_fixes.csv lists misspelled words found in the training reviews, each with
its correction and a decision made by hand. Rows marked `fix` are applied; rows marked `keep`
look misspelled but are not (spoken forms, formal verbs) and are listed only so that the
decision can be seen. How the list was built is described in the plan.
"""

import csv
import re

import pandas as pd

from digikala_reviews.config import PROJECT_ROOT
from digikala_reviews.tfidf import LETTER, ZWNJ

SPELLING_FIXES = PROJECT_ROOT / "resources" / "spelling_fixes.csv"


def load_fixes(path=SPELLING_FIXES) -> dict[str, str]:
    """Misspelled word -> correction, for the rows marked `fix`."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["decision"] == "fix"]
    fixes = {row["wrong"]: row["correct"] for row in rows}
    assert len(fixes) == len(rows), "a word is listed twice"
    assert all(fixes.values()), "a fix without a correction"
    return fixes


def fix_spelling(texts: pd.Series, fixes: dict[str, str]) -> pd.Series:
    """Replace listed words in normalized texts, but only where they stand as whole words.

    A listed word inside a longer word, or joined to another word by a half-space, is left
    alone, since the TF-IDF features see that as a different word too.
    """
    words = "|".join(re.escape(w) for w in sorted(fixes, key=len, reverse=True))
    edge = f"(?:{LETTER}|{ZWNJ})"
    pattern = re.compile(rf"(?<!{edge})(?:{words})(?!{edge})")
    return texts.str.replace(pattern, lambda match: fixes[match.group(0)], regex=True)
