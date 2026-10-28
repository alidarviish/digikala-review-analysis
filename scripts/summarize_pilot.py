"""Put every fine-tuning comparison run next to the final TF-IDF model, on the same reviews.

Reads the validation probabilities saved by scripts/pilot_transformers.py
(artifacts/transformer_pilot/*.npz) and the final TF-IDF predictions
(artifacts/tfidf_final/validation_predictions.parquet). For each run and head it scores the
labels with offsets (tuned on the other half of the sample, by product, as in the runs
themselves) and compares them with TF-IDF on exactly the same reviews (paired bootstrap).
Runs of the same model and sample size are also averaged over their seeds.

Writes reports/transformer_pilot/summary.json and prints a table. Nothing is trained.

With --whole-validation it reads and writes reports/transformer_pilot_whole/ instead.

Usage:
    uv run python scripts/summarize_pilot.py [--whole-validation]
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, REPORTS_DIR
from digikala_reviews.experiments import compare, cross_fitted_offsets, product_half
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

NAME = re.compile(r"^(?P<model>.+?)(?:__seed(?P<seed>\d+)__n(?P<rows>\d+))?$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--whole-validation", action="store_true", help="the runs scored on all of validation"
    )
    args = parser.parse_args()
    folder = "transformer_pilot_whole" if args.whole_validation else "transformer_pilot"
    runs_dir, arrays_dir = REPORTS_DIR / folder, ARTIFACTS_DIR / folder
    validation = load_cache("validation.parquet", extra_columns=["product_id"]).set_index("id")
    tfidf = pd.read_parquet(ARTIFACTS_DIR / "tfidf_final" / "validation_predictions.parquet").set_index("id")
    runs, groups = {}, defaultdict(list)
    for path in sorted(arrays_dir.glob("*.npz")):
        record = json.loads((runs_dir / f"{path.stem}.json").read_text(encoding="utf-8"))
        arrays = np.load(path)
        rows = validation.loc[arrays["ids"]]
        result = {"model": record["model"], "seed": record["fingerprint"]["settings"]["seed"]}
        result["train_rows"] = record["fingerprint"]["settings"]["train_rows"]
        result["train_minutes"] = record["train_minutes"]
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
            tfidf_labels = tfidf.loc[arrays["ids"][labeled], f"{head}__predicted_with_offsets"].to_numpy()
            found = compare(head, y_true, np.array(labels)[predicted], tfidf_labels.astype(str))
            result[head] = {
                "macro_f1": found["variant_scores"]["macro_f1"],
                "tfidf_macro_f1": found["reference_scores"]["macro_f1"],
                "difference": found["macro_f1_difference"],
                "difference_interval_95": found["difference_interval_95"],
                "direction": found["direction"],
            }
        result["mean_macro_f1"] = float(np.mean([result[h]["macro_f1"] for h in HEADS]))
        runs[path.stem] = result
        groups[(record["model"], result["train_rows"])].append(result)

    averaged = []
    for (model, train_rows), members in sorted(groups.items()):
        means = [m["mean_macro_f1"] for m in members]
        averaged.append(
            {
                "model": model,
                "train_rows": train_rows,
                "seeds": sorted(m["seed"] for m in members),
                **{f"{h}_macro_f1": float(np.mean([m[h]["macro_f1"] for m in members])) for h in HEADS},
                "mean_macro_f1": float(np.mean(means)),
                "lowest_seed": float(min(means)),
                "highest_seed": float(max(means)),
                "minutes": float(np.mean([m["train_minutes"] for m in members])),
            }
        )
    averaged.sort(key=lambda a: (a["train_rows"], -a["mean_macro_f1"]))

    save_json(
        runs_dir / "summary.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "runs": runs,
            "averaged_over_seeds": averaged,
            "code": git_commit(),
        },
    )
    print("run | rec (vs TF-IDF) | band (vs TF-IDF)")
    for name, r in runs.items():
        cells = []
        for head in HEADS:
            low, high = r[head]["difference_interval_95"]
            cells.append(
                f"{r[head]['macro_f1']:.4f} ({r[head]['difference']:+.4f} [{low:+.4f}, {high:+.4f}])"
            )
        print(f"{name} | " + " | ".join(cells))
    print("\nmodel | rows | seeds | rec | band | mean | seed range | minutes")
    for a in averaged:
        print(
            f"{a['model']} | {a['train_rows']:,} | {len(a['seeds'])} | {a['recommendation_macro_f1']:.4f} | "
            f"{a['rating_band_macro_f1']:.4f} | {a['mean_macro_f1']:.4f} | "
            f"{a['lowest_seed']:.4f}-{a['highest_seed']:.4f} | {a['minutes']:.1f}"
        )


if __name__ == "__main__":
    main()
