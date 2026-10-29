"""FaBERT trained on reviews merged by text against FaBERT trained on the separate reviews.

Both on the same 500,000 training reviews and the same three seeds, scored on all of
validation (artifacts/transformer_pilot_whole/*.npz). For each way of merging (each text
weighted by its copies, or counted once) and each seed, both runs get labels with offsets
(tuned on the other half, by product) and are compared on exactly the same reviews (paired
bootstrap, per head). The mean over the seeds then decides as in step 36: a lead smaller than
the spread between seeds is a tie, and a tie goes to the faster setup.

Writes reports/transformer_pilot_whole/merged_vs_separate_n500000.json. Nothing is trained.
The plan and the expected results are in reports/merged_texts_plan.md.

Usage:
    uv run python scripts/compare_merged_runs.py
"""

from datetime import UTC, datetime

import numpy as np
from compare_pilot_models import FOLDER, SEEDS, TRAIN_ROWS, labels_with_offsets

from digikala_reviews.config import REPORTS_DIR
from digikala_reviews.experiments import compare
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

MODEL = "sbunlp__fabert"
MODES = ["copies", "once"]


def compare_mode(mode, validation):
    per_seed, means, minutes = {}, {"merged": [], "separate": []}, {"merged": [], "separate": []}
    for seed in SEEDS:
        tag = f"{MODEL}__seed{seed}__n{TRAIN_ROWS}"
        runs = {
            "merged": labels_with_offsets(f"{tag}__merge-{mode}", validation),
            "separate": labels_with_offsets(tag, validation),
        }
        per_seed[seed] = {}
        for head in HEADS:
            merged, separate = runs["merged"][head], runs["separate"][head]
            assert (merged["ids"] == separate["ids"]).all(), "the two runs must score the same reviews"
            found = compare(head, merged["y_true"], merged["predicted"], separate["predicted"])
            # the labels rebuilt here must score exactly as in the runs themselves
            assert abs(found["variant_scores"]["macro_f1"] - merged["saved_macro_f1"]) < 1e-9, head
            assert abs(found["reference_scores"]["macro_f1"] - separate["saved_macro_f1"]) < 1e-9, head
            per_seed[seed][head] = {
                "merged_macro_f1": found["variant_scores"]["macro_f1"],
                "separate_macro_f1": found["reference_scores"]["macro_f1"],
                "difference": found["macro_f1_difference"],
                "difference_interval_95": found["difference_interval_95"],
                "direction": found["direction"],
            }
        for name in means:
            means[name].append(float(np.mean([per_seed[seed][h][f"{name}_macro_f1"] for h in HEADS])))
            minutes[name].append(runs[name]["minutes"])

    lead = float(np.mean(means["merged"]) - np.mean(means["separate"]))
    spread = float(max(max(v) - min(v) for v in means.values()))
    if abs(lead) < spread:
        verdict = "tie"
        chosen = min(means, key=lambda name: np.mean(minutes[name]))
    else:
        verdict = "lead"
        chosen = "merged" if lead > 0 else "separate"
    return {
        "per_seed": per_seed,
        "mean_of_heads_per_seed": means,
        "mean_over_seeds": {name: float(np.mean(v)) for name, v in means.items()},
        "minutes_per_seed": minutes,
        "lead_of_merged": lead,
        "spread_between_seeds": spread,
        "verdict": verdict,
        "chosen": chosen,
    }


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"]).set_index("id")
    result = {mode: compare_mode(mode, validation) for mode in MODES}
    save_json(
        REPORTS_DIR / FOLDER / f"merged_vs_separate_n{TRAIN_ROWS}.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": MODEL,
            "train_rows": TRAIN_ROWS,
            **result,
            "rule": "a lead smaller than the spread between seeds is a tie; a tie goes to the faster setup",
            "code": git_commit(),
        },
    )
    for mode, r in result.items():
        print(f"\nmerged, each text {'weighted by its copies' if mode == 'copies' else 'counted once'}:")
        for seed, heads in r["per_seed"].items():
            cells = []
            for head, h in heads.items():
                low, high = h["difference_interval_95"]
                cells.append(f"{head} {h['difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({h['direction']})")
            print(f"  seed {seed}, merged minus separate: " + "; ".join(cells))
        for name, values in r["mean_of_heads_per_seed"].items():
            print(
                f"  {name}: mean {np.mean(values):.4f}, seeds {min(values):.4f} to {max(values):.4f}, "
                f"{np.mean(r['minutes_per_seed'][name]):.1f} min"
            )
        print(
            f"  lead of merged {r['lead_of_merged']:+.4f}, spread between seeds "
            f"{r['spread_between_seeds']:.4f}: {r['verdict']}, chosen {r['chosen']}"
        )


if __name__ == "__main__":
    main()
