"""Measures that explain the scores of a model rather than change it.

Used on the saved predictions of the final TF-IDF model, and later on the Transformer's,
so both are looked at the same way.
"""

import numpy as np
import pandas as pd

from digikala_reviews.metrics import classification_report, macro_f1_from_codes

LENGTH_BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, None)]  # words, both ends included
CONFIDENCE_BINS = np.linspace(0.0, 1.0, 11)


def word_count(texts) -> np.ndarray:
    return np.array([len(t.split()) for t in texts])


def length_bucket(counts: np.ndarray) -> np.ndarray:
    names = np.empty(len(counts), dtype=object)
    for low, high in LENGTH_BUCKETS:
        inside = (counts >= low) & (counts <= high) if high is not None else counts >= low
        names[inside] = f"{low}-{high}" if high is not None else f"{low}+"
    return names


def scores_by_group(y_true, y_pred, groups, labels, min_rows: int = 1) -> dict:
    """Macro F1 and per-class recall inside each group, with the group's share of the rows."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    result = {}
    for group in pd.unique(groups):
        inside = groups == group
        if inside.sum() < min_rows:
            continue
        report = classification_report(y_true[inside], y_pred[inside], labels)
        result[str(group)] = {
            "rows": int(inside.sum()),
            "share": float(inside.mean()),
            "macro_f1": report["macro_f1"],
            "recall": {k: v["recall"] for k, v in report["per_class"].items()},
            "class_share": {k: v["support"] / report["rows"] for k, v in report["per_class"].items()},
        }
    return result


def label_ceiling(texts, y_true, labels) -> dict:
    """Best possible scores on reviews whose exact text appears more than once.

    A model that sees only the text must give every copy of a text the same answer. The best
    single answer is the most common label of that text, so these scores are an upper
    bound for any text-only model on those reviews.
    """
    table = pd.DataFrame({"text": np.asarray(texts), "label": np.asarray(y_true)})
    copies = table.groupby("text")["label"].transform("size")
    repeated = table[copies > 1]
    counts = repeated.groupby(["text", "label"]).size().rename("n").reset_index()
    # most common label of each text; ties go to the label listed first, the same for every copy
    counts["order"] = counts["label"].map({label: i for i, label in enumerate(labels)})
    best = counts.sort_values(["text", "n", "order"], ascending=[True, False, True]).drop_duplicates("text")
    answer = repeated["text"].map(best.set_index("text")["label"])
    report = classification_report(repeated["label"].to_numpy(), answer.to_numpy(), labels)
    conflicting = repeated.groupby("text")["label"].nunique() > 1
    return {
        "repeated_rows": int(len(repeated)),
        "repeated_share": float(len(repeated) / len(table)),
        "distinct_repeated_texts": int(repeated["text"].nunique()),
        "texts_with_conflicting_labels": int(conflicting.sum()),
        "ceiling": {
            "macro_f1": report["macro_f1"],
            "accuracy": report["accuracy"],
            "recall": {k: v["recall"] for k, v in report["per_class"].items()},
        },
        "rows_mask": (copies > 1).to_numpy(),
    }


def sample_spread(y_true, y_pred, labels, sizes, draws: int = 500, seed: int = 2026) -> dict:
    """How much Macro F1 moves when only a random sample of validation is scored."""
    code = {label: i for i, label in enumerate(labels)}
    true_codes = np.array([code[v] for v in y_true])
    predicted = np.array([code[v] for v in y_pred])
    full = macro_f1_from_codes(true_codes, predicted, len(labels))
    rng = np.random.default_rng(seed)
    result = {"full": full}
    for size in sizes:
        scores = np.array(
            [
                macro_f1_from_codes(true_codes[rows], predicted[rows], len(labels))
                for rows in (rng.choice(len(true_codes), size, replace=False) for _ in range(draws))
            ]
        )
        result[str(size)] = {
            "std": float(scores.std()),
            "p05": float(np.percentile(scores, 5)),
            "p95": float(np.percentile(scores, 95)),
            "largest_gap_to_full": float(np.abs(scores - full).max()),
        }
    return result


def calibration(confidence, correct) -> dict:
    """Mean confidence next to the share of correct answers, in ten equal confidence bins."""
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=float)
    bins = np.clip(np.digitize(confidence, CONFIDENCE_BINS[1:-1]), 0, 9)
    table, gap = [], 0.0
    for b in range(10):
        inside = bins == b
        if not inside.any():
            continue
        mean_conf, accuracy = float(confidence[inside].mean()), float(correct[inside].mean())
        gap += inside.mean() * abs(mean_conf - accuracy)
        table.append(
            {
                "bin": f"{CONFIDENCE_BINS[b]:.1f}-{CONFIDENCE_BINS[b + 1]:.1f}",
                "rows": int(inside.sum()),
                "mean_confidence": mean_conf,
                "accuracy": accuracy,
            }
        )
    return {"expected_calibration_error": float(gap), "bins": table}


def confident_share(confidence, predicted, thresholds, correct=None) -> dict:
    """Reviews kept at each threshold, the classes they get and, when known, the share that is right."""
    confidence, predicted = np.asarray(confidence), np.asarray(predicted)
    result = {}
    for threshold in thresholds:
        kept = confidence >= threshold
        entry = {
            "kept": int(kept.sum()),
            "share": float(kept.mean()),
            "predicted_classes": {
                str(k): float(v) for k, v in pd.Series(predicted[kept]).value_counts(normalize=True).items()
            },
        }
        if correct is not None and kept.any():
            entry["accuracy"] = float(np.asarray(correct)[kept].mean())
        result[str(threshold)] = entry
    return result
