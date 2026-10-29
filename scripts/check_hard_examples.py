"""Where focal loss would move the training weight, before any GPU run.

Focal loss multiplies the loss of each review by (1 - p)^gamma, p the probability the model
gives to the true class. This reads the validation probabilities of the three FaBERT pilot runs
on 500,000 training reviews (step 35) and reports, for gamma 0, 1 and 2, the share of the total
weight that each class, review length and seen / new text would get, and how much of the weight
of the hardest reviews (p < 0.2) falls on texts whose copies carry more than one label. The
three seeds are pooled. Validation stands in for training: the training probabilities were not
saved. Nothing is trained and the test set is not read.

Writes reports/hard_examples_check.json. The plan and the expected results are in
reports/hard_examples_plan.md.

Usage:
    uv run python scripts/check_hard_examples.py
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.analysis import length_bucket, word_count
from digikala_reviews.config import ARTIFACTS_DIR, REPORTS_DIR
from digikala_reviews.finetune import focal_weights
from digikala_reviews.loss_analysis import label_entropy_floor, loss_by_group
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

RUNS = [f"sbunlp__fabert__seed{s}__n500000" for s in (2026, 2027, 2028)]
GAMMAS = [0, 1, 2]
HARD = 0.2  # a review is among the hardest when p < HARD
ORDER = {
    "length_in_words": ["1-1", "2-3", "4-7", "8-15", "16-31", "32+"],
    "text_in_training_reviews": ["seen in train", "new text"],
}


def main():
    validation = load_cache("validation.parquet", extra_columns=["text_seen_in_train"])
    position = pd.Series(np.arange(len(validation)), index=validation["id"].to_numpy())
    lengths = length_bucket(word_count(validation["text"]))
    seen = np.where(validation["text_seen_in_train"].to_numpy(), "seen in train", "new text")
    arrays = [np.load(ARTIFACTS_DIR / "transformer_pilot_whole" / f"{run}.npz") for run in RUNS]
    rows = position.loc[arrays[0]["ids"]].to_numpy()
    for found in arrays[1:]:
        assert (found["ids"] == arrays[0]["ids"]).all(), "the runs must score the same reviews"
    print(f"{len(rows):,} validation reviews, {len(RUNS)} runs pooled", flush=True)

    result = {}
    for head, h in HEADS.items():
        values = validation[h["column"]].to_numpy()[rows]
        labeled = pd.notna(values)
        codes = np.array([h["labels"].index(v) for v in values[labeled]])
        picked = rows[labeled]
        # p of the true class in every run, the runs one after the other
        p_true = np.concatenate([np.exp(a[head][labeled][np.arange(len(codes)), codes]) for a in arrays])
        tile = len(arrays)
        groups = {
            "true_class": np.tile(values[labeled], tile),
            "length_in_words": np.tile(lengths[picked], tile),
            "text_in_training_reviews": np.tile(seen[picked], tile),
        }
        floor = label_entropy_floor(validation["text"].to_numpy()[picked], codes, len(h["labels"]), 2)
        mixed = np.tile(np.nan_to_num(floor) > 0, tile)  # a text of 2+ copies with more than one label
        hard = p_true < HARD
        result[head] = {"weights": {}, "hardest": {"p_below": HARD, "share_of_reviews": float(hard.mean())}}
        for gamma in GAMMAS:
            weights = focal_weights(p_true, gamma)
            result[head]["weights"][f"gamma_{gamma}"] = {
                name: {g: v["share_of_loss"] for g, v in loss_by_group(weights, labels).items()}
                for name, labels in groups.items()
            }
            if gamma == 2:
                result[head]["hardest"]["share_of_weight_gamma_2"] = float(
                    weights[hard].sum() / weights.sum()
                )
        result[head]["hardest"]["share_on_texts_with_more_than_one_label"] = float(mixed[hard].mean())
        result[head]["all_reviews_on_texts_with_more_than_one_label"] = float(mixed.mean())

    save_json(
        REPORTS_DIR / "hard_examples_check.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "runs": RUNS,
            **result,
            "code": git_commit(),
        },
    )
    for head, r in result.items():
        print(f"\n{head}:")
        for name in ("true_class", "length_in_words", "text_in_training_reviews"):
            print(f"  share of the weight by {name} (gamma {' / '.join(map(str, GAMMAS))}):")
            for group in ORDER.get(name) or HEADS[head]["labels"]:
                shares = " / ".join(f"{r['weights'][f'gamma_{g}'][name][group]:.1%}" for g in GAMMAS)
                print(f"    {group}: {shares}")
        hardest = r["hardest"]
        print(
            f"  hardest reviews (p < {HARD}): {hardest['share_of_reviews']:.1%} of reviews, "
            f"{hardest['share_of_weight_gamma_2']:.1%} of the weight at gamma 2; "
            f"{hardest['share_on_texts_with_more_than_one_label']:.1%} of them on texts with more than one "
            "label "
            f"(all reviews: {r['all_reviews_on_texts_with_more_than_one_label']:.1%})"
        )


if __name__ == "__main__":
    main()
