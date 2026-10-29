"""Compare TookaBERT-Large with FaBERT directly, both trained on 500,000 reviews, on all of validation.

Reads the validation probabilities of the pilot runs scored on the whole validation set
(artifacts/transformer_pilot_whole/*.npz). For each seed both runs get labels with offsets
(tuned on the other half, by product, as in the runs themselves), and the two models are
compared on exactly the same reviews (paired bootstrap, per head). The mean over the seeds
then decides with the rule of reports/transformer_model_choice.md: a lead smaller than the
spread between seeds is a tie, and a tie goes to the faster model.

Writes reports/transformer_pilot_whole/large_vs_fabert_n500000.json. Nothing is trained.

Usage:
    uv run python scripts/compare_pilot_models.py
"""

import json
from datetime import UTC, datetime

import numpy as np

from digikala_reviews.config import ARTIFACTS_DIR, REPORTS_DIR
from digikala_reviews.experiments import compare, cross_fitted_offsets, product_half
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

FOLDER = "transformer_pilot_whole"
MODELS = {"large": "PartAI__TookaBERT-Large", "fabert": "sbunlp__fabert"}
SEEDS = [2026, 2027, 2028]
TRAIN_ROWS = 500_000


def labels_with_offsets(tag, validation):
    """Labels with cross-fitted offsets for every head, with the score the run saved itself."""
    arrays = np.load(ARTIFACTS_DIR / FOLDER / f"{tag}.npz")
    record = json.loads((REPORTS_DIR / FOLDER / f"{tag}.json").read_text(encoding="utf-8"))
    rows = validation.loc[arrays["ids"]]
    found = {"minutes": record["train_minutes"]}
    for head in HEADS:
        column, labels = HEADS[head]["column"], HEADS[head]["labels"]
        labeled = rows[column].notna().to_numpy()
        y_true = rows[column].to_numpy()[labeled]
        code = {label: i for i, label in enumerate(labels)}
        predicted, _ = cross_fitted_offsets(
            arrays[head][labeled],
            np.array([code[v] for v in y_true]),
            product_half(rows["product_id"].to_numpy()[labeled]),
        )
        found[head] = {
            "ids": arrays["ids"][labeled],
            "y_true": y_true,
            "predicted": np.array(labels)[predicted],
            "saved_macro_f1": record["heads"][head]["with_offsets"]["macro_f1"],
        }
    return found


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"]).set_index("id")
    per_seed, means = {}, {name: [] for name in MODELS}
    minutes = {name: [] for name in MODELS}
    for seed in SEEDS:
        runs = {}
        for name, model in MODELS.items():
            runs[name] = labels_with_offsets(f"{model}__seed{seed}__n{TRAIN_ROWS}", validation)
            minutes[name].append(runs[name]["minutes"])
        per_seed[seed] = {}
        for head in HEADS:
            large, fabert = runs["large"][head], runs["fabert"][head]
            assert (large["ids"] == fabert["ids"]).all(), "the two runs must score the same reviews"
            found = compare(head, large["y_true"], large["predicted"], fabert["predicted"])
            # the labels rebuilt here must score exactly as in the runs themselves
            assert abs(found["variant_scores"]["macro_f1"] - large["saved_macro_f1"]) < 1e-9, head
            assert abs(found["reference_scores"]["macro_f1"] - fabert["saved_macro_f1"]) < 1e-9, head
            per_seed[seed][head] = {
                "large_macro_f1": found["variant_scores"]["macro_f1"],
                "fabert_macro_f1": found["reference_scores"]["macro_f1"],
                "difference": found["macro_f1_difference"],
                "difference_interval_95": found["difference_interval_95"],
                "direction": found["direction"],
            }
        for name in MODELS:
            means[name].append(float(np.mean([per_seed[seed][h][f"{name}_macro_f1"] for h in HEADS])))

    lead = float(np.mean(means["large"]) - np.mean(means["fabert"]))
    spread = float(max(max(v) - min(v) for v in means.values()))
    faster = min(MODELS, key=lambda name: np.mean(minutes[name]))
    if abs(lead) < spread:
        verdict, chosen = "tie", faster
    else:
        verdict, chosen = "lead", "large" if lead > 0 else "fabert"

    save_json(
        REPORTS_DIR / FOLDER / f"large_vs_fabert_n{TRAIN_ROWS}.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "train_rows": TRAIN_ROWS,
            "per_seed": per_seed,
            "mean_of_heads_per_seed": means,
            "mean_over_seeds": {name: float(np.mean(v)) for name, v in means.items()},
            "minutes_per_seed": minutes,
            "lead_of_large": lead,
            "spread_between_seeds": spread,
            "verdict": verdict,
            "chosen": MODELS[chosen],
            "rule": "a lead smaller than the spread between seeds is a tie; a tie goes to the faster model",
            "code": git_commit(),
        },
    )
    for seed, heads in per_seed.items():
        cells = []
        for head, h in heads.items():
            low, high = h["difference_interval_95"]
            cells.append(f"{head} {h['difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({h['direction']})")
        print(f"seed {seed}, Large minus FaBERT: " + "; ".join(cells))
    for name, values in means.items():
        print(
            f"{MODELS[name]}: mean {np.mean(values):.4f}, seeds {min(values):.4f} to {max(values):.4f}, "
            f"{np.mean(minutes[name]):.1f} min"
        )
    print(f"Lead of Large {lead:+.4f}, spread between seeds {spread:.4f}: {verdict}, chosen {MODELS[chosen]}")


if __name__ == "__main__":
    main()
