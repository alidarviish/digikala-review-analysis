"""Compare FaBERT trained for every epoch with the first FaBERT run, on the whole validation set.

Both runs saved their validation predictions with cross-fitted offsets. For each head this
writes the paired difference in Macro F1 with its 95% bootstrap interval, the per-class
scores of both runs and the validation loss of both. The longer run replaces the first one
only if the whole interval is above zero, judged for each head separately
(reports/transformer_model_choice.md). Nothing is trained and the test set is not read.

Writes reports/transformer_final/no_early_stop_vs_first.json and prints a summary.

With --large the same is done for TookaBERT-Large trained the same way against the kept FaBERT
run (reports/large_final_plan.md), written to reports/transformer_final/large_vs_fabert.json.

With --adapted the same is done for FaBERT trained from the further pretrained weights against
the kept FaBERT run (reports/domain_adaptation_plan.md), written to
reports/transformer_final/adapted_vs_fabert.json; and each of the two is also combined with the
final TF-IDF model as in step 60 (weight and offsets tuned on the other half by product), and
the two combinations are compared on the same reviews.

Usage:
    uv run python scripts/compare_fabert_runs.py
    uv run python scripts/compare_fabert_runs.py --large
    uv run python scripts/compare_fabert_runs.py --adapted
"""

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, REPORTS_DIR
from digikala_reviews.ensemble import cross_fitted_blend
from digikala_reviews.experiments import compare, product_half
from digikala_reviews.finetune import mean_log_loss
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

# for each comparison: the variant, the reference, where each run is, the output and its verdict key
COMPARISONS = {
    "epochs": {
        "variant": "no_early_stop",
        "reference": "first",
        "runs": {
            "first": ("transformer_final", "fabert.json"),
            "no_early_stop": ("transformer_no_early_stop", "fabert_no_early_stop.json"),
        },
        "output": "no_early_stop_vs_first.json",
        "verdict": "replaces_first_run",
    },
    "large": {
        "variant": "large",
        "reference": "fabert",
        "runs": {
            "fabert": ("transformer_no_early_stop", "fabert_no_early_stop.json"),
            "large": ("transformer_large", "large.json"),
        },
        "output": "large_vs_fabert.json",
        "verdict": "large_measurably_better",
    },
    "adapted": {
        "variant": "adapted",
        "reference": "fabert",
        "runs": {
            "fabert": ("transformer_no_early_stop", "fabert_no_early_stop.json"),
            "adapted": ("transformer_adapted", "adapted.json"),
        },
        "output": "adapted_vs_fabert.json",
        "verdict": "adapted_measurably_better",
    },
}


def log_proba(table, head):
    return table[[f"{head}__log_proba__{label}" for label in HEADS[head]["labels"]]].to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true", help="TookaBERT-Large against the kept FaBERT run")
    parser.add_argument(
        "--adapted", action="store_true", help="further pretrained FaBERT against the kept run"
    )
    args = parser.parse_args()
    assert not (args.large and args.adapted), "one comparison at a time"
    c = COMPARISONS["large" if args.large else "adapted" if args.adapted else "epochs"]
    variant, reference = c["variant"], c["reference"]
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    tfidf = None
    if args.adapted:  # both runs are also combined with the final TF-IDF model, as in step 60
        tfidf = pd.read_parquet(ARTIFACTS_DIR / "tfidf_final" / "validation_predictions.parquet")
        assert (tfidf["id"].to_numpy() == validation["id"].to_numpy()).all(), "tfidf"
    tables, runs = {}, {}
    for name, (folder, result) in c["runs"].items():
        tables[name] = pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet")
        assert (tables[name]["id"].to_numpy() == validation["id"].to_numpy()).all(), name
        runs[name] = json.loads((REPORTS_DIR / "transformer_final" / result).read_text(encoding="utf-8"))

    heads = {}
    for head in HEADS:
        column = HEADS[head]["column"]
        rows = validation[column].notna().to_numpy()
        y_true = validation[column].to_numpy()[rows]
        codes = np.array([HEADS[head]["labels"].index(v) for v in y_true])
        found = compare(
            head,
            y_true,
            tables[variant][f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str),
            tables[reference][f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str),
        )
        heads[head] = {
            variant: found["variant_scores"],
            reference: found["reference_scores"],
            "macro_f1_difference": found["macro_f1_difference"],
            "difference_interval_95": found["difference_interval_95"],
            "direction": found["direction"],
            "validation_loss": {
                name: mean_log_loss(log_proba(table[rows], head), codes) for name, table in tables.items()
            },
        }
        if tfidf is not None:
            labels = np.array(HEADS[head]["labels"])
            halves = product_half(validation["product_id"].to_numpy()[rows])
            log_t = log_proba(tfidf[rows], head)
            blended = {
                name: cross_fitted_blend(log_proba(tables[name][rows], head), log_t, codes, halves)
                for name in (variant, reference)
            }
            with_tfidf = compare(head, y_true, labels[blended[variant][0]], labels[blended[reference][0]])
            heads[head]["with_tfidf"] = {
                variant: with_tfidf["variant_scores"],
                reference: with_tfidf["reference_scores"],
                "weights": {name: b[1] for name, b in blended.items()},
                "macro_f1_difference": with_tfidf["macro_f1_difference"],
                "difference_interval_95": with_tfidf["difference_interval_95"],
                "direction": with_tfidf["direction"],
            }

    save_json(
        REPORTS_DIR / "transformer_final" / c["output"],
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "validation_reviews": int(len(validation)),
            "kept_epoch": {name: run["kept_epoch"] for name, run in runs.items()},
            "heads": heads,
            c["verdict"]: {head: h["direction"] == "better" for head, h in heads.items()},
            **(
                {
                    "with_tfidf_better": {
                        head: h["with_tfidf"]["direction"] == "better" for head, h in heads.items()
                    }
                }
                if tfidf is not None
                else {}
            ),
            "code": git_commit(),
        },
    )
    print(
        f"Kept checks: {reference} epoch {runs[reference]['kept_epoch']}, "
        f"{variant} epoch {runs[variant]['kept_epoch']}"
    )
    for head, h in heads.items():
        low, high = h["difference_interval_95"]
        loss = h["validation_loss"]
        print(
            f"{head}: {variant} {h[variant]['macro_f1']:.4f}, "
            f"{reference} {h[reference]['macro_f1']:.4f}, "
            f"difference {h['macro_f1_difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({h['direction']}); "
            f"validation loss {loss[variant]:.4f} vs {loss[reference]:.4f}"
        )
        if "with_tfidf" in h:
            t = h["with_tfidf"]
            low, high = t["difference_interval_95"]
            print(
                f"  with TF-IDF: {variant} {t[variant]['macro_f1']:.4f}, "
                f"{reference} {t[reference]['macro_f1']:.4f}, "
                f"difference {t['macro_f1_difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({t['direction']}); "
                f"FaBERT weights {t['weights']}"
            )


if __name__ == "__main__":
    main()
