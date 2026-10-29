"""What merging the training reviews by text would change, before any GPU run.

Counts the different texts, the training steps one epoch would take, how many texts carry more
than one label, and the share of each class when a text counts as often as its copies (as now)
or once. Writes reports/merged_texts_check.json. Nothing is trained; only the training cache
is read. The plan and the expected results are in reports/merged_texts_plan.md.

Usage:
    uv run python scripts/check_merged_texts.py
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR
from digikala_reviews.merge_texts import class_shares, merge, mixed
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS
from digikala_reviews.tuning import save_json

BATCH_SIZE = 64  # as in scripts/train_transformer.py
COPY_BUCKETS = [1, 2, 10, 100, 1000]


def main():
    train = pd.read_parquet(
        CACHE_DIR / "train_supervised.parquet", columns=["text", "recommendation_status", "rating_band"]
    )
    print(f"Read {len(train):,} training reviews", flush=True)
    texts, copies, counts = merge(
        train["text"], {head: (train[h["column"]], h["labels"]) for head, h in HEADS.items()}
    )
    for head, h in HEADS.items():
        # the merged counts must hold every label of the separate rows
        assert counts[head].sum() == train[h["column"]].notna().sum(), head

    buckets = {}
    edges = [*COPY_BUCKETS, np.inf]
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (copies >= low) & (copies < high)
        if high == low + 1:
            name = f"{low}"
        else:
            name = f"{low}+" if high == np.inf else f"{low}-{int(high) - 1}"
        buckets[name] = {
            "share_of_texts": float(inside.mean()),
            "share_of_rows": float(copies[inside].sum() / len(train)),
        }
    top = np.argsort(-copies)[:10]

    result = {
        "rows": int(len(train)),
        "different_texts": int(len(texts)),
        "steps_per_epoch": {
            "now": int(np.ceil(len(train) / BATCH_SIZE)),
            "merged": int(np.ceil(len(texts) / BATCH_SIZE)),
        },
        "copies_per_text": buckets,
        "most_copied_texts": [{"text": str(texts[i]), "copies": int(copies[i])} for i in top],
        "heads": {
            head: {
                "mixed_labels": mixed(counts[head]),
                "class_shares": class_shares(counts[head], h["labels"]),
            }
            for head, h in HEADS.items()
        },
    }
    save_json(
        REPORTS_DIR / "merged_texts_check.json",
        {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **result, "code": git_commit()},
    )

    steps = result["steps_per_epoch"]
    print(
        f"{result['rows']:,} rows -> {result['different_texts']:,} different texts "
        f"({result['different_texts'] / result['rows']:.1%}); "
        f"steps per epoch {steps['now']:,} -> {steps['merged']:,}"
    )
    print("copies per text:")
    for name, b in buckets.items():
        print(f"  {name}: {b['share_of_texts']:.2%} of texts, {b['share_of_rows']:.1%} of rows")
    print("most copied texts:")
    for t in result["most_copied_texts"]:
        print(f"  {t['copies']:,}  {t['text'][:40]}")
    for head, r in result["heads"].items():
        m = r["mixed_labels"]
        print(
            f"\n{head}: {m['texts_with_a_label']:,} texts with a label; more than one label: "
            f"{m['share_of_texts']:.2%} of texts, {m['share_of_rows']:.1%} of rows"
        )
        for label, s in r["class_shares"].items():
            print(
                f"  {label}: {s['copies_counted']:.1%} as now, "
                f"{s['each_text_once']:.1%} if each text counts once"
            )


if __name__ == "__main__":
    main()
