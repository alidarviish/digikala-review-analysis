"""How much the BiLSTM and the three-model comparison depend on one training run (plan:
reports/bilstm_plan.md, "Two more seeds").

For the BiLSTM runs with seeds 2026, 2027 and 2028 (same setting, vocabulary, training
reviews and choosing sample): Macro F1 of each on the whole validation set and its spread;
FaBERT minus BiLSTM per seed; and the three-model comparison of
scripts/compare_reading_models.py (all right, all wrong, the same wrong label, no known
cause) redone with each BiLSTM seed on the same 872,795 reviews. Report only. The test set
is not read.

Writes reports/bilstm_seeds.json and prints a summary.

Usage:
    uv run python scripts/check_bilstm_seeds.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_reading_models as crm  # noqa: E402

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR  # noqa: E402
from digikala_reviews.metrics import classification_report  # noqa: E402
from digikala_reviews.provenance import check_cache, git_commit  # noqa: E402
from digikala_reviews.tfidf import HEADS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

OUTPUT = REPORTS_DIR / "bilstm_seeds.json"
SEEDS = [2026, 2027, 2028]


def bilstm_name(seed: int) -> str:
    return "bilstm" if seed == 2026 else f"bilstm_seed{seed}"


def main():
    for seed in SEEDS[1:]:  # each seed's predictions are checked against its own record
        crm.MODELS[bilstm_name(seed)] = (bilstm_name(seed), (f"bilstm/final_seed{seed}.json", None))
    names = ["fabert", "tfidf", *[bilstm_name(s) for s in SEEDS]]
    validation = load_cache("validation.parquet", extra_columns=["product_id", "rate", "text_seen_in_train"])
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    keep = crm.comparison_mask(validation)
    assert int((~keep).sum()) == crm.SELECTION_ROWS
    predictions, checks = crm.load_predictions(validation, names)
    causes = crm.known_causes(validation)
    any_cause = np.column_stack(list(causes.values())).any(axis=1)

    result = {"seeds": SEEDS, "checks": checks, "data": data, "heads": {}}
    for head, spec in HEADS.items():
        whole = {s: checks[bilstm_name(s)][head] for s in SEEDS}
        values = np.array(list(whole.values()))
        truth_all = validation[spec["column"]].to_numpy(dtype=object)
        rows = keep & pd.notna(truth_all)
        truth = truth_all[rows].astype(str)
        per_seed = {}
        for s in SEEDS:
            trio = ["fabert", "tfidf", bilstm_name(s)]
            labels = {m: predictions[m][head][rows].astype(str) for m in trio}
            right = np.column_stack([labels[m] == truth for m in trio])
            all_wrong = ~right.any(axis=1)
            stacked = np.column_stack([labels[m] for m in trio])
            same = all_wrong & (stacked == stacked[:, :1]).all(axis=1)
            per_seed[str(s)] = {
                "bilstm_macro_f1_compared_reviews": classification_report(
                    truth, labels[bilstm_name(s)], spec["labels"]
                )["macro_f1"],
                "all_right": int(right.all(axis=1).sum()),
                "all_wrong": int(all_wrong.sum()),
                "all_wrong_same_label": int(same.sum()),
                "all_wrong_no_known_cause": int((all_wrong & ~any_cause[rows]).sum()),
                "only_bilstm_right": int((right[:, 2] & ~right[:, 0] & ~right[:, 1]).sum()),
            }
        counts = {k: [per_seed[str(s)][k] for s in SEEDS] for k in per_seed[str(SEEDS[0])]}
        result["heads"][head] = {
            "bilstm_macro_f1_whole_validation": {str(s): v for s, v in whole.items()},
            "spread_max_minus_min": float(values.max() - values.min()),
            "mean": float(values.mean()),
            "fabert_minus_bilstm": {str(s): checks["fabert"][head] - v for s, v in whole.items()},
            "three_model_comparison": per_seed,
            "relative_spread_of_counts": {
                k: float((max(v) - min(v)) / max(np.mean(v), 1))
                for k, v in counts.items()
                if k.startswith("all")
            },
            "reviews_compared": int(rows.sum()),
        }

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/bilstm_plan.md",
            **result,
            "code": git_commit(),
        },
    )
    for head, e in result["heads"].items():
        print(f"\n=== {head}")
        print(
            "  BiLSTM Macro F1: "
            + ", ".join(f"seed {s} {v:.4f}" for s, v in e["bilstm_macro_f1_whole_validation"].items())
            + f"; spread {e['spread_max_minus_min']:.4f}, mean {e['mean']:.4f}"
        )
        print(
            "  FaBERT minus BiLSTM: "
            + ", ".join(f"{s} {v:+.4f}" for s, v in e["fabert_minus_bilstm"].items())
        )
        print(f"  three models on {e['reviews_compared']:,} reviews, per BiLSTM seed:")
        for s, x in e["three_model_comparison"].items():
            print(
                f"    seed {s}: all right {x['all_right']:,}, all wrong {x['all_wrong']:,} "
                f"(same label {x['all_wrong_same_label']:,}, "
                f"no known cause {x['all_wrong_no_known_cause']:,}), "
                f"only BiLSTM right {x['only_bilstm_right']:,}"
            )
        spread = ", ".join(f"{k} {100 * v:.2f}%" for k, v in e["relative_spread_of_counts"].items())
        print(f"  relative spread of the counts: {spread}")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
