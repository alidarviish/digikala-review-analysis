"""Compare the fine-tuned Transformer with the final TF-IDF model on the whole validation set.

Both models saved their validation predictions (labels with cross-fitted offsets, and
log-probabilities). For each head this writes:

- the paired difference in Macro F1 with its 95% bootstrap interval
- per-class recall, precision and F1 of both models
- the validation loss (mean cross-entropy of the true label) of both models, and next to it
  the Transformer's training loss around the step that was kept, read from its log

The Transformer is the kept FaBERT run (FINAL_TRANSFORMER_* in config.py).

Writes reports/transformer_final/vs_tfidf.json and prints a summary. Nothing is trained and the
test set is not read.

Usage:
    uv run python scripts/compare_final_models.py
"""

import json
import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import (
    ARTIFACTS_DIR,
    FINAL_TRANSFORMER_DIR,
    FINAL_TRANSFORMER_LOG,
    FINAL_TRANSFORMER_RESULT,
    REPORTS_DIR,
)
from digikala_reviews.experiments import compare
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

TRAIN_LOG = REPORTS_DIR / FINAL_TRANSFORMER_LOG
STEP_LINE = re.compile(r"step ([\d,]+)/[\d,]+, loss ([\d.]+)")


def validation_loss(table, head, y_true):
    labels = HEADS[head]["labels"]
    log_proba = table[[f"{head}__log_proba__{label}" for label in labels]].to_numpy()
    codes = np.array([labels.index(v) for v in y_true])
    return float(-log_proba[np.arange(len(codes)), codes].mean())


def training_loss_near(step, window=2_500):
    """Mean logged training loss (sum over the two heads) within `window` steps before `step`.

    Only the last training run in the log counts, so a smoke run earlier in the file is ignored.
    """
    text = TRAIN_LOG.read_text(encoding="utf-8")
    last_run = text.split("training reviews,")[-1]
    points = [(int(s.replace(",", "")), float(v)) for s, v in STEP_LINE.findall(last_run)]
    near = [v for s, v in points if step - window < s <= step]
    return float(np.mean(near)) if near else None


def main():
    validation = load_cache("validation.parquet")
    transformer = pd.read_parquet(ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "validation_predictions.parquet")
    tfidf = pd.read_parquet(ARTIFACTS_DIR / "tfidf_final" / "validation_predictions.parquet")
    for table in (transformer, tfidf):
        assert (table["id"].to_numpy() == validation["id"].to_numpy()).all(), "rows are not in the same order"
    run = json.loads((REPORTS_DIR / FINAL_TRANSFORMER_RESULT).read_text(encoding="utf-8"))

    heads, losses = {}, {}
    for head in HEADS:
        column = HEADS[head]["column"]
        rows = validation[column].notna().to_numpy()
        y_true = validation[column].to_numpy()[rows]
        found = compare(
            head,
            y_true,
            transformer[f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str),
            tfidf[f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str),
        )
        heads[head] = {
            "transformer": found["variant_scores"],
            "tfidf": found["reference_scores"],
            "macro_f1_difference": found["macro_f1_difference"],
            "difference_interval_95": found["difference_interval_95"],
            "direction": found["direction"],
        }
        losses[head] = {
            "transformer_validation": validation_loss(transformer[rows], head, y_true),
            "tfidf_validation": validation_loss(tfidf[rows], head, y_true),
        }
    kept = run["kept_step"]
    losses["transformer_training_both_heads_near_kept_step"] = training_loss_near(kept)
    losses["transformer_validation_both_heads"] = sum(losses[h]["transformer_validation"] for h in HEADS)

    save_json(
        REPORTS_DIR / "transformer_final" / "vs_tfidf.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "validation_reviews": int(len(validation)),
            "transformer_run": FINAL_TRANSFORMER_RESULT,
            "kept_step": kept,
            "heads": heads,
            "loss": losses,
            "note": "validation loss is unweighted; the training loss is the logged mean over steps",
            "code": git_commit(),
        },
    )
    for head, h in heads.items():
        low, high = h["difference_interval_95"]
        print(
            f"{head}: Transformer {h['transformer']['macro_f1']:.4f}, TF-IDF {h['tfidf']['macro_f1']:.4f}, "
            f"difference {h['macro_f1_difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({h['direction']})"
        )
        for label in HEADS[head]["labels"]:
            a, b = h["transformer"]["per_class"][label], h["tfidf"]["per_class"][label]
            print(
                f"    {label}: recall {a['recall']:.3f} vs {b['recall']:.3f}, "
                f"precision {a['precision']:.3f} vs {b['precision']:.3f}, F1 {a['f1']:.3f} vs {b['f1']:.3f}"
            )
        print(
            f"    validation loss: Transformer {losses[head]['transformer_validation']:.4f}, "
            f"TF-IDF {losses[head]['tfidf_validation']:.4f}"
        )
    training = losses["transformer_training_both_heads_near_kept_step"]
    print(
        f"Transformer loss, both heads: training {training:.4f} "
        f"(steps before {kept:,}), validation {losses['transformer_validation_both_heads']:.4f}"
    )


if __name__ == "__main__":
    main()
