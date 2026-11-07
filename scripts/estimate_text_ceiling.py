"""A ceiling for any text-only model, from reviews whose exact text repeats (plan:
reports/text_ceiling_plan.md).

On the 872,795 validation reviews of the reading-model comparison, for texts seen at least
k times (k = 2, 5, 20): the exact accuracy ceiling (each text gets its most frequent label),
the Macro F1 of that answer, the train-majority answer, and FaBERT, TF-IDF and the BiLSTM on
the same reviews; then the 20 most frequent texts. Report only. The test set is not read.

Writes reports/text_ceiling.json and prints a summary.

Usage:
    uv run python scripts/estimate_text_ceiling.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_reading_models import MODELS, SELECTION_ROWS, comparison_mask, load_predictions  # noqa: E402

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR  # noqa: E402
from digikala_reviews.metrics import classification_report  # noqa: E402
from digikala_reviews.provenance import check_cache, git_commit  # noqa: E402
from digikala_reviews.tfidf import HEADS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

OUTPUT = REPORTS_DIR / "text_ceiling.json"
MIN_COUNTS = [2, 5, 20]
TOP_TEXTS = 20


def majority(counts: pd.DataFrame, labels: list[str]) -> pd.Series:
    """Most frequent label of each text; a tie goes to the label listed first."""
    return counts[labels].idxmax(axis=1)


def scores(y_true, y_pred, labels) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)
    report = classification_report(y_true, y_pred, labels)
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": report["macro_f1"],
        "recall": {c: report["per_class"][c]["recall"] for c in labels},
    }


def main():
    names = list(MODELS)
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    keep = comparison_mask(validation)
    assert int((~keep).sum()) == SELECTION_ROWS
    predictions, checks = load_predictions(validation, names)
    train = load_cache("train_supervised.parquet")
    data = check_cache(CACHE_DIR, ["train_supervised.parquet", "validation.parquet"])

    result = {"checks": checks, "data": data, "heads": {}}
    for head, spec in HEADS.items():
        labels = spec["labels"]
        truth = validation[spec["column"]].to_numpy(dtype=object)
        rows = keep & pd.notna(truth)
        table = pd.DataFrame({"text": validation["text"].to_numpy()[rows], "y": truth[rows].astype(str)})
        for m in names:
            table[m] = predictions[m][head][rows].astype(str)
        counts = (
            table.groupby(["text", "y"]).size().unstack(fill_value=0).reindex(columns=labels, fill_value=0)
        )
        per_text = counts.sum(axis=1)

        labeled_train = train[train[spec["column"]].notna()]
        train_counts = (
            labeled_train.groupby(["text", spec["column"]])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=labels, fill_value=0)
        )
        train_majority = majority(train_counts, labels)

        entry = {"reviews": int(len(table)), "by_min_count": {}}
        for k in MIN_COUNTS:
            texts = per_text.index[per_text >= k]
            sub = table[table["text"].isin(texts)]
            best = majority(counts.loc[texts], labels)
            best_answer = sub["text"].map(best)
            in_train = sub["text"].isin(train_majority.index)
            from_train = sub["text"].map(train_majority)
            entry["by_min_count"][str(k)] = {
                "distinct_texts": int(len(texts)),
                "reviews": int(len(sub)),
                "share_of_compared_reviews": round(len(sub) / len(table), 4),
                "true_label_counts": {c: int((sub["y"] == c).sum()) for c in labels},
                "ceiling_accuracy": float(counts.loc[texts].max(axis=1).sum() / len(sub)),
                "most_frequent_label_per_text": scores(sub["y"], best_answer, labels),
                "most_frequent_label_gives": {c: int((best_answer == c).sum()) for c in labels},
                "train_majority": {
                    "reviews_with_text_in_train": int(in_train.sum()),
                    **scores(sub["y"][in_train], from_train[in_train], labels),
                },
                "models": {m: scores(sub["y"], sub[m], labels) for m in names},
                "models_with_two_answers_for_one_text": {
                    m: int((sub.groupby("text")[m].nunique() > 1).sum()) for m in names
                },
            }
        top = per_text.sort_values(ascending=False).head(TOP_TEXTS).index
        entry["most_frequent_texts"] = [
            {
                "text": t,
                "reviews": int(per_text[t]),
                "labels": {c: int(counts.loc[t, c]) for c in labels},
                "models_say_most_often": {m: table.loc[table["text"] == t, m].mode().iloc[0] for m in names},
            }
            for t in top
        ]
        result["heads"][head] = entry

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/text_ceiling_plan.md",
            **result,
            "code": git_commit(),
        },
    )
    summary(result, names)
    print(f"saved {OUTPUT}")


def summary(result, names):
    for head, e in result["heads"].items():
        print(f"\n=== {head}: {e['reviews']:,} compared reviews")
        for k, x in e["by_min_count"].items():
            print(
                f"  texts seen at least {k} times: {x['distinct_texts']:,} texts, {x['reviews']:,} reviews "
                f"({100 * x['share_of_compared_reviews']:.1f}%); true labels {x['true_label_counts']}"
            )
            best = x["most_frequent_label_per_text"]
            print(
                f"    CEILING accuracy {x['ceiling_accuracy']:.4f}; most frequent label per text: "
                f"Macro F1 {best['macro_f1']:.4f}, gives {x['most_frequent_label_gives']}"
            )
            t = x["train_majority"]
            print(
                f"    train majority ({t['reviews_with_text_in_train']:,} reviews): "
                f"accuracy {t['accuracy']:.4f}, Macro F1 {t['macro_f1']:.4f}"
            )
            for m in names:
                s = x["models"][m]
                recall = ", ".join(f"{c} {v:.3f}" for c, v in s["recall"].items())
                print(
                    f"    {m:<7} accuracy {s['accuracy']:.4f}, Macro F1 {s['macro_f1']:.4f}; recall {recall}"
                )
            print(f"    texts given two answers by a model: {x['models_with_two_answers_for_one_text']}")
        print("  most frequent texts (reviews; labels; FaBERT / TF-IDF / BiLSTM say):")
        for t in e["most_frequent_texts"]:
            said = " / ".join(t["models_say_most_often"][m] for m in names)
            print(f"    {t['text'][:30]:<30} {t['reviews']:>7,}  {t['labels']}  {said}")


if __name__ == "__main__":
    main()
