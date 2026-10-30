"""FaBERT with more weight on hard reviews against FaBERT trained as usual (steps 41 to 45).

All runs use the same 500,000 training reviews and seeds 2026 to 2028, are scored on all of
validation and measured 10 times during training. The reference is the usual training run again
with those measurements; it should score as in step 35, which is checked here too. For each
variant (focal loss with gamma 0.5, 1 and 2, class weights) and each seed, both runs get labels
with offsets (tuned on the other half, by product) and are compared on exactly the same reviews
(paired bootstrap, per head). The mean over the seeds then decides as in step 36: a lead smaller
than the spread between seeds is a tie, and a tie goes to the reference: it is as fast and simpler.

Writes reports/transformer_pilot_whole/hard_examples_n500000.json. Nothing is trained. The plan
and the expected results are in reports/hard_examples_plan.md.

With --set difficulty the same is done for the runs that use the difficulty of each training
review (steps 49 to 51: the hardest labels left out, curriculum; reports/difficulty_plan.md),
written to reports/transformer_pilot_whole/difficulty_n500000.json, and with --set gradual for
the gradual curriculum (step 53), written to reports/transformer_pilot_whole/gradual_n500000.json,
and with --set learning_rate for other learning rates and layer-wise decay (steps 56 to 58),
written to reports/transformer_pilot_whole/learning_rate_n500000.json.

Usage:
    uv run python scripts/compare_hard_examples.py
    uv run python scripts/compare_hard_examples.py --set difficulty
    uv run python scripts/compare_hard_examples.py --set gradual
    uv run python scripts/compare_hard_examples.py --set learning_rate
"""

import argparse
from datetime import UTC, datetime

import numpy as np
from compare_pilot_models import FOLDER, SEEDS, TRAIN_ROWS, labels_with_offsets

from digikala_reviews.config import REPORTS_DIR
from digikala_reviews.experiments import compare
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

MODEL = "sbunlp__fabert"
CHECKS = "__checks10"
VARIANT_SETS = {
    "hard_examples": {
        "focal 0.5": "__focal0.5",
        "focal 1": "__focal1",
        "focal 2": "__focal2",
        "class weights": "__classweights",
    },
    "difficulty": {
        "leave out 5%": "__leaveout0.05",
        "leave out 10%": "__leaveout0.1",
        "curriculum": "__curriculum",
    },
    "gradual": {"gradual curriculum": "__gradual"},
    "learning_rate": {
        "learning rate 2e-5": "__lr2e-05",
        "learning rate 5e-5": "__lr5e-05",
        "layer decay 0.9": "__llrd0.9",
    },
}


def tag(seed, suffix=""):
    return f"{MODEL}__seed{seed}__n{TRAIN_ROWS}{suffix}"


def mean_of_heads(run):
    return float(np.mean([run[head]["saved_macro_f1"] for head in HEADS]))


def compare_setups(reference, variant):
    """Per seed and head the paired difference, then the rule over the seeds."""
    per_seed, means, minutes = {}, {"variant": [], "reference": []}, {"variant": [], "reference": []}
    for seed in SEEDS:
        runs = {"variant": variant[seed], "reference": reference[seed]}
        per_seed[seed] = {}
        for head in HEADS:
            var, ref = runs["variant"][head], runs["reference"][head]
            assert (var["ids"] == ref["ids"]).all(), "the two runs must score the same reviews"
            found = compare(head, var["y_true"], var["predicted"], ref["predicted"])
            # the labels rebuilt here must score exactly as in the runs themselves
            assert abs(found["variant_scores"]["macro_f1"] - var["saved_macro_f1"]) < 1e-9, head
            assert abs(found["reference_scores"]["macro_f1"] - ref["saved_macro_f1"]) < 1e-9, head
            per_seed[seed][head] = {
                "variant_macro_f1": found["variant_scores"]["macro_f1"],
                "reference_macro_f1": found["reference_scores"]["macro_f1"],
                "difference": found["macro_f1_difference"],
                "difference_interval_95": found["difference_interval_95"],
                "direction": found["direction"],
            }
        for name, run in runs.items():
            means[name].append(mean_of_heads(run))
            minutes[name].append(run["minutes"])
    lead = float(np.mean(means["variant"]) - np.mean(means["reference"]))
    spread = float(max(max(v) - min(v) for v in means.values()))
    if abs(lead) < spread:
        verdict, chosen = "tie", "reference"  # as fast, and simpler
    else:
        verdict, chosen = "lead", "variant" if lead > 0 else "reference"
    return {
        "per_seed": per_seed,
        "mean_of_heads_per_seed": means,
        "mean_over_seeds": {name: float(np.mean(v)) for name, v in means.items()},
        "minutes_per_seed": minutes,
        "lead_of_variant": lead,
        "spread_between_seeds": spread,
        "verdict": verdict,
        "chosen": chosen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["hard", "difficulty", "gradual", "learning_rate"], default="hard")
    args = parser.parse_args()
    set_name = "hard_examples" if args.set == "hard" else args.set
    validation = load_cache("validation.parquet", extra_columns=["product_id"]).set_index("id")
    reference = {seed: labels_with_offsets(tag(seed, CHECKS), validation) for seed in SEEDS}
    first = {seed: labels_with_offsets(tag(seed), validation) for seed in SEEDS}
    # measuring during training must not change the result: the reference against step 35
    same_as_step_35 = {seed: mean_of_heads(reference[seed]) - mean_of_heads(first[seed]) for seed in SEEDS}
    result = {}
    for name, suffix in VARIANT_SETS[set_name].items():
        variant = {seed: labels_with_offsets(tag(seed, suffix + CHECKS), validation) for seed in SEEDS}
        result[name] = compare_setups(reference, variant)

    save_json(
        REPORTS_DIR / FOLDER / f"{set_name}_n{TRAIN_ROWS}.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": MODEL,
            "train_rows": TRAIN_ROWS,
            "reference_minus_step_35_mean_of_heads": same_as_step_35,
            "variants": result,
            "rule": "a lead smaller than the spread between seeds is a tie; a tie goes to the reference",
            "code": git_commit(),
        },
    )
    print(
        "reference with checks minus step 35 (mean of heads): "
        + ", ".join(f"seed {s} {d:+.5f}" for s, d in same_as_step_35.items())
    )
    for name, r in result.items():
        print(f"\n{name}:")
        for seed, heads in r["per_seed"].items():
            cells = []
            for head, h in heads.items():
                low, high = h["difference_interval_95"]
                cells.append(f"{head} {h['difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({h['direction']})")
            print(f"  seed {seed}, variant minus reference: " + "; ".join(cells))
        for side, values in r["mean_of_heads_per_seed"].items():
            print(
                f"  {side}: mean {np.mean(values):.4f}, seeds {min(values):.4f} to {max(values):.4f}, "
                f"{np.mean(r['minutes_per_seed'][side]):.1f} min"
            )
        print(
            f"  lead of the variant {r['lead_of_variant']:+.4f}, spread between seeds "
            f"{r['spread_between_seeds']:.4f}: {r['verdict']}, chosen {r['chosen']}"
        )


if __name__ == "__main__":
    main()
