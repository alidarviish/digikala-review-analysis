"""Compare FaBERT trained for every epoch with the first FaBERT run, on the whole validation set.

Both runs saved their validation predictions with cross-fitted offsets. For each head this
writes the paired difference in Macro F1 with its 95% bootstrap interval, the per-class
scores of both runs and the validation loss of both. The longer run replaces the first one
only if the whole interval is above zero, judged for each head separately
(reports/transformer_model_choice.md). Nothing is trained and the test set is not read.

Writes reports/transformer_final/no_early_stop_vs_first.json and prints a summary.

Usage:
    uv run python scripts/compare_fabert_runs.py
"""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, REPORTS_DIR
from digikala_reviews.experiments import compare
from digikala_reviews.finetune import mean_log_loss
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

RUNS = {
    "first": ("transformer_final", "fabert.json"),
    "no_early_stop": ("transformer_no_early_stop", "fabert_no_early_stop.json"),
}


def log_proba(table, head):
    return table[[f"{head}__log_proba__{label}" for label in HEADS[head]["labels"]]].to_numpy()


def main():
    validation = load_cache("validation.parquet")
    tables, runs = {}, {}
    for name, (folder, result) in RUNS.items():
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
            tables["no_early_stop"][f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str),
            tables["first"][f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str),
        )
        heads[head] = {
            "no_early_stop": found["variant_scores"],
            "first": found["reference_scores"],
            "macro_f1_difference": found["macro_f1_difference"],
            "difference_interval_95": found["difference_interval_95"],
            "direction": found["direction"],
            "validation_loss": {
                name: mean_log_loss(log_proba(table[rows], head), codes) for name, table in tables.items()
            },
        }

    save_json(
        REPORTS_DIR / "transformer_final" / "no_early_stop_vs_first.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "validation_reviews": int(len(validation)),
            "kept_epoch": {name: run["kept_epoch"] for name, run in runs.items()},
            "heads": heads,
            "replaces_first_run": {head: h["direction"] == "better" for head, h in heads.items()},
            "code": git_commit(),
        },
    )
    print(
        f"Kept checks: first run epoch {runs['first']['kept_epoch']}, "
        f"no early stop epoch {runs['no_early_stop']['kept_epoch']}"
    )
    for head, h in heads.items():
        low, high = h["difference_interval_95"]
        loss = h["validation_loss"]
        print(
            f"{head}: no early stop {h['no_early_stop']['macro_f1']:.4f}, "
            f"first {h['first']['macro_f1']:.4f}, "
            f"difference {h['macro_f1_difference']:+.4f} [{low:+.4f}, {high:+.4f}] ({h['direction']}); "
            f"validation loss {loss['no_early_stop']:.4f} vs {loss['first']:.4f}"
        )


if __name__ == "__main__":
    main()
