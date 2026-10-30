"""What the hardest training reviews are, from the out-of-fold difficulty of score_difficulty.py.

For each head and for the hardest 5% and 10% of its labeled reviews (lowest probability of
their own label), the share of each class, review length and text that comes more than once
in all the training reviews, and the share of texts whose copies carry more than one label,
each next to the same share among all labeled reviews of the sample. Nothing is trained.

Writes reports/difficulty/summary.json. The plan is in reports/difficulty_plan.md.

Usage:
    uv run python scripts/check_difficulty.py
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.analysis import length_bucket, word_count
from digikala_reviews.config import CACHE_DIR, REPORTS_DIR
from digikala_reviews.difficulty import hardest
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS
from digikala_reviews.tuning import save_json

FRACTIONS = [0.05, 0.10]
ORDER = {"length_in_words": ["1-1", "2-3", "4-7", "8-15", "16-31", "32+"]}


def shares(labels, order):
    counts = pd.Series(labels).value_counts(normalize=True)
    return {g: float(counts.get(g, 0.0)) for g in order}


def main():
    found = pd.read_parquet(REPORTS_DIR / "difficulty" / "n500000.parquet")
    train = pd.read_parquet(
        CACHE_DIR / "train_supervised.parquet", columns=["id", "text", "recommendation_status", "rating_band"]
    )
    copies = train["text"].map(train["text"].value_counts())
    labels_per_text = train.groupby("text")["recommendation_status"].nunique()
    mixed_texts = set(labels_per_text[labels_per_text > 1].index)
    sample = train.set_index("id").loc[found["id"]].reset_index()
    sample_copies = copies.set_axis(train["id"]).loc[found["id"]].to_numpy()
    print(f"{len(found):,} reviews with an out-of-fold difficulty", flush=True)

    groups = {
        "length_in_words": length_bucket(word_count(sample["text"])),
        "text_repeated_in_training": np.where(sample_copies > 1, "repeated", "once"),
        "text_with_more_than_one_recommendation_label": np.where(
            sample["text"].isin(mixed_texts), "mixed", "single"
        ),
    }
    result = {}
    for head, h in HEADS.items():
        p_true = found[f"{head}_p_true"].to_numpy()
        labeled = ~np.isnan(p_true)
        classes = sample[h["column"]].to_numpy()
        result[head] = {"mean_p_true": float(np.nanmean(p_true)), "all_labeled": {}, "hardest": {}}
        named = {"true_class": (classes, h["labels"])} | {
            name: (values, ORDER.get(name) or sorted(set(values))) for name, values in groups.items()
        }
        for name, (values, order) in named.items():
            result[head]["all_labeled"][name] = shares(values[labeled], order)
        for fraction in FRACTIONS:
            chosen = hardest(p_true, fraction)
            result[head]["hardest"][f"{fraction:g}"] = {
                "reviews": int(chosen.sum()),
                "p_true_below": float(p_true[chosen].max()),
                **{name: shares(values[chosen], order) for name, (values, order) in named.items()},
            }

    save_json(
        REPORTS_DIR / "difficulty" / "summary.json",
        {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **result, "code": git_commit()},
    )
    for head, r in result.items():
        print(f"\n{head}: mean p of the true label {r['mean_p_true']:.3f}")
        for name, everyone in r["all_labeled"].items():
            print(f"  {name} (all / hardest {' / hardest '.join(f'{f:.0%}' for f in FRACTIONS)}):")
            for group, share in everyone.items():
                hard = " / ".join(f"{r['hardest'][f'{f:g}'][name][group]:.1%}" for f in FRACTIONS)
                print(f"    {group}: {share:.1%} / {hard}")
        for f in FRACTIONS:
            h = r["hardest"][f"{f:g}"]
            print(
                f"  hardest {f:.0%}: {h['reviews']:,} reviews, "
                f"p of the true label below {h['p_true_below']:.3f}"
            )


if __name__ == "__main__":
    main()
