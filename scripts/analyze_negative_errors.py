"""What the negative reviews that all three models get wrong, with no known cause, have in
common (plan: reports/reading_models_plan.md, third addition).

For `not_recommended` (recommendation head) and `low` (rating band head), on the same
872,795 validation reviews as scripts/compare_reading_models.py:

- group "unexplained": all three models wrong and none of the known causes
- group "right": all three models right

For each, the labels the three models gave (when all wrong), the rate of the reviews, the
length in words, and the words that set the unexplained reviews apart from the right ones:
the log-odds ratio of each word with an informative Dirichlet prior from all compared
reviews (Monroe, Colaresi and Quinn, 2008), as a z-score; the 40 words most typical of each
group, each seen in at least 30 reviews of the two groups together, with their counts.

Words come from the project's `persian` token pattern on the normalized text (emoji and
punctuation are tokens), counted once per review. Report only. The test set is not read.

Writes reports/negative_errors.json and prints a summary.

Usage:
    uv run python scripts/analyze_negative_errors.py
"""

import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_reading_models import (  # noqa: E402
    MODELS,
    SELECTION_ROWS,
    comparison_mask,
    known_causes,
    load_predictions,
)

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR  # noqa: E402
from digikala_reviews.provenance import check_cache, git_commit  # noqa: E402
from digikala_reviews.tfidf import HEADS, TOKEN_PATTERNS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

OUTPUT = REPORTS_DIR / "negative_errors.json"
TOKEN = re.compile(TOKEN_PATTERNS["persian"])
TARGETS = {"recommendation": "not_recommended", "rating_band": "low"}
MIN_REVIEWS = 30
TOP = 40


def words_of(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def review_counts(texts) -> Counter:
    counts = Counter()
    for t in texts:
        counts.update(words_of(t))
    return counts


def distinctive(a: Counter, n_a: int, b: Counter, n_b: int, prior: Counter, n_prior: int) -> list[dict]:
    """Log-odds ratio with an informative Dirichlet prior (per-review counts), as z-scores.
    Positive: typical of a; negative: typical of b."""
    if n_a == 0 or n_b == 0:
        return []
    scale = 1000.0 / max(n_prior, 1)  # prior strength: about 1,000 reviews' worth
    alpha0 = n_prior * scale
    rows = []
    for w in set(a) | set(b):
        if a[w] + b[w] < MIN_REVIEWS:
            continue
        aw = prior[w] * scale  # > 0: the prior counts every compared review, both groups included
        rest_a, rest_b = n_a + alpha0 - a[w] - aw, n_b + alpha0 - b[w] - aw
        if rest_a <= 0 or rest_b <= 0:  # a word in every review: no odds to compare
            continue
        la = np.log((a[w] + aw) / rest_a)
        lb = np.log((b[w] + aw) / rest_b)
        variance = 1.0 / (a[w] + aw) + 1.0 / (b[w] + aw)
        rows.append(
            {
                "word": w,
                "z": float((la - lb) / np.sqrt(variance)),
                "reviews_unexplained": int(a[w]),
                "share_unexplained": round(a[w] / n_a, 4),
                "reviews_right": int(b[w]),
                "share_right": round(b[w] / n_b, 4),
            }
        )
    return sorted(rows, key=lambda r: -r["z"])


def describe(mask, texts, rate, labels_by_model) -> dict:
    n = int(mask.sum())
    lengths = np.array([len(t.split()) for t in texts[mask]])
    r = rate[mask]
    whole = np.isclose(r, np.round(r))
    buckets = np.where(
        np.isnan(r),
        "no rate",
        np.where(
            whole,
            [f"{x:.0f}" for x in np.nan_to_num(r)],
            [f"{np.floor(x):.0f} to {np.floor(x) + 1:.0f}, fractional" for x in np.nan_to_num(r)],
        ),
    )
    rates = pd.Series(buckets).value_counts().sort_index()
    out = {
        "reviews": n,
        "words_median": float(np.median(lengths)) if n else None,
        "words_mean": round(float(lengths.mean()), 2) if n else None,
        "rate_counts": {str(k): int(v) for k, v in rates.items()},
    }
    if labels_by_model is not None:
        stacked = np.column_stack([v[mask] for v in labels_by_model.values()])
        same = (stacked == stacked[:, :1]).all(axis=1)
        agreed = Counter(stacked[same, 0])
        out["all_three_said"] = {k: int(v) for k, v in agreed.most_common()}
        out["models_disagree_with_each_other"] = int((~same).sum())
    return out


def word_line(w: dict) -> str:
    unexplained = f"{w['reviews_unexplained']:>6,} ({100 * w['share_unexplained']:.1f}%)"
    right = f"{w['reviews_right']:>7,} ({100 * w['share_right']:.1f}%)"
    return f"    {w['word']:<16} {w['z']:6.1f}  {unexplained} / {right}"


def main():
    names = list(MODELS)
    validation = load_cache("validation.parquet", extra_columns=["product_id", "rate", "text_seen_in_train"])
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    keep = comparison_mask(validation)
    assert int((~keep).sum()) == SELECTION_ROWS
    predictions, checks = load_predictions(validation, names)
    causes = known_causes(validation)
    any_cause = np.column_stack(list(causes.values())).any(axis=1)
    texts = validation["text"].astype(str).to_numpy()
    rate = validation["rate"].to_numpy(dtype=float)

    result = {"causes_used": list(causes), "checks": checks, "data": data, "heads": {}}
    for head, target in TARGETS.items():
        truth = validation[HEADS[head]["column"]].to_numpy(dtype=object)
        rows = keep & pd.notna(truth)
        is_target = rows & (truth == target)
        right = np.column_stack([predictions[m][head] == truth for m in names])
        right[~rows] = False
        all_wrong = is_target & ~right.any(axis=1)
        all_right = is_target & right.all(axis=1)
        unexplained = all_wrong & ~any_cause
        prior = review_counts(texts[rows])
        a, b = review_counts(texts[unexplained]), review_counts(texts[all_right])
        ranked = distinctive(a, int(unexplained.sum()), b, int(all_right.sum()), prior, int(rows.sum()))
        labels = {m: predictions[m][head] for m in names}
        result["heads"][head] = {
            "class": target,
            "all_wrong": int(all_wrong.sum()),
            "unexplained": describe(unexplained, texts, rate, labels),
            "all_wrong_with_a_known_cause": describe(all_wrong & any_cause, texts, rate, labels),
            "all_right": describe(all_right, texts, rate, None),
            "words_typical_of_unexplained": ranked[:TOP],
            "words_typical_of_right": ranked[::-1][:TOP],
        }

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/reading_models_plan.md",
            **result,
            "code": git_commit(),
        },
    )
    for head, e in result["heads"].items():
        u, r = e["unexplained"], e["all_right"]
        print(f"\n=== {head}, class {e['class']}: all wrong {e['all_wrong']:,}")
        print(f"  of them with no known cause: {u['reviews']:,}")
        print(f"  all three said {u['all_three_said']}")
        print(f"  models disagree with each other: {u['models_disagree_with_each_other']:,}")
        print(f"  words per review, median: unexplained {u['words_median']}, all right {r['words_median']}")
        print(f"  rates of the unexplained: {u['rate_counts']}")
        print(f"  rates of the all right:   {r['rate_counts']}")
        print("  words typical of the unexplained (z; reviews unexplained / right):")
        for w in e["words_typical_of_unexplained"][:25]:
            print(word_line(w))
        print("  words typical of the all right:")
        for w in e["words_typical_of_right"][:15]:
            print(word_line(w))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
