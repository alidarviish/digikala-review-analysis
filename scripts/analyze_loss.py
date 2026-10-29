"""Where the final FaBERT model's validation loss comes from, and how much could be removed.

Three parts, each written to reports/loss_analysis.json:

  floor     for validation texts with many copies: the loss no text-only model could go
            below (the entropy of the text's labels) next to the model's loss on the same rows
  groups    the model's loss by true class, by review length, and by whether the text was
            in the training reviews exactly, nearly (same letters once spacing, punctuation,
            digits and stretched letters are removed) or not at all
  train     how many training reviews repeat a text, exactly or nearly, and how many
            belong to a text that comes with more than one label

The model is the one named in config.py (FINAL_TRANSFORMER_*). Nothing is trained and the
test set is not read. The plan and the expected results are in reports/loss_analysis_plan.md.

Usage:
    uv run python scripts/analyze_loss.py
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.analysis import length_bucket, word_count
from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, FINAL_TRANSFORMER_DIR, REPORTS_DIR
from digikala_reviews.loss_analysis import label_entropy_floor, loss_by_group, near_key, repeats, row_loss
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

MIN_COPIES = [2, 20, 100]


def main():
    validation = load_cache("validation.parquet", extra_columns=["text_seen_in_train"])
    predictions = pd.read_parquet(ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "validation_predictions.parquet")
    same_order = (predictions["id"].to_numpy() == validation["id"].to_numpy()).all()
    assert same_order, "rows are not in the same order"
    train = pd.read_parquet(
        CACHE_DIR / "train_supervised.parquet", columns=["text", "recommendation_status", "rating_band"]
    )
    print(f"Read {len(validation):,} validation and {len(train):,} training reviews", flush=True)

    validation_near = validation["text"].map(near_key)
    train_near = train["text"].map(near_key)
    exact_seen = validation["text_seen_in_train"].to_numpy()
    near_seen = validation_near.isin(set(train_near)).to_numpy()
    seen = np.where(exact_seen, "exact text in train", np.where(near_seen, "nearly the same in train", "new"))
    lengths = length_bucket(word_count(validation["text"]))

    result = {"floor": {}, "groups": {}}
    for head in HEADS:
        column, labels = HEADS[head]["column"], HEADS[head]["labels"]
        rows = validation[column].notna().to_numpy()
        codes = np.array([labels.index(v) for v in validation[column].to_numpy()[rows]])
        log_proba = predictions[[f"{head}__log_proba__{label}" for label in labels]].to_numpy()[rows]
        loss = row_loss(log_proba, codes)
        texts = validation["text"].to_numpy()[rows]

        result["floor"][head] = {"model_loss_all_rows": float(loss.mean())}
        for n in MIN_COPIES:
            floor = label_entropy_floor(texts, codes, len(labels), n)
            inside = ~np.isnan(floor)
            result["floor"][head][f"texts_with_{n}_or_more_copies"] = {
                "share_of_rows": float(inside.mean()),
                "floor": float(floor[inside].mean()),
                "model_loss": float(loss[inside].mean()),
            }
        result["groups"][head] = {
            "true_class": loss_by_group(loss, validation[column].to_numpy()[rows]),
            "length_in_words": loss_by_group(loss, lengths[rows]),
            "text_in_training_reviews": loss_by_group(loss, seen[rows]),
        }
        print(f"{head}: done", flush=True)

    result["train"] = {
        # reviews without a recommendation do not count as a second label
        "exact": repeats(train["text"], train["recommendation_status"]),
        "nearly_the_same": repeats(train_near, train["recommendation_status"]),
    }
    save_json(
        REPORTS_DIR / "loss_analysis.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **result,
            "model": FINAL_TRANSFORMER_DIR,
            "code": git_commit(),
        },
    )

    for head in HEADS:
        f = result["floor"][head]
        print(f"\n{head}: model loss {f['model_loss_all_rows']:.4f} on all rows")
        for n in MIN_COPIES:
            g = f[f"texts_with_{n}_or_more_copies"]
            print(
                f"  texts with {n}+ copies ({g['share_of_rows']:.1%} of rows): "
                f"floor {g['floor']:.4f}, model {g['model_loss']:.4f}"
            )
        for name, groups in result["groups"][head].items():
            print(f"  by {name}:")
            for group, g in sorted(groups.items(), key=lambda kv: -kv[1]["share_of_loss"]):
                print(
                    f"    {group}: {g['share_of_rows']:.1%} of rows, {g['share_of_loss']:.1%} of loss, "
                    f"mean {g['mean_loss']:.3f}"
                )
    for kind, r in result["train"].items():
        print(
            f"\ntraining reviews, {kind}: {r['rows']:,} rows, {r['different_texts']:,} different texts, "
            f"{r['rows_repeating_an_earlier_text']:.1%} repeat an earlier text, "
            f"{r['rows_of_texts_with_more_than_one_label']:.1%} belong to a text with more than one label"
        )


if __name__ == "__main__":
    main()
