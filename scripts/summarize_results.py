"""Write every result of the TF-IDF and Transformer stages into one table, from the saved files.

Nothing is trained or scored again; each number is read from a JSON file in reports/ and
printed with the commit it was made with. Scores are Macro F1 on the validation set. A
difference is shown with its 95% bootstrap interval: [low, high].

Writes reports/results_summary.md.

Usage:
    uv run python scripts/summarize_results.py
"""

import json

from digikala_reviews.config import REPORTS_DIR

OUT = REPORTS_DIR / "results_summary.md"

# Measurements that are missing on purpose or by accident, with the reason. Kept by hand.
NOT_MEASURED = [
    (
        "E6 linear SVM with offsets against logistic regression with offsets",
        "never run: E6 compared the SVM without offsets against the reference without offsets",
    ),
    (
        "E6 rating band: SVM C=1, Complement Naive Bayes alpha 0.1 and 1",
        "run on molab, lost when the notebook was relaunched before the download",
    ),
    (
        "Transformer pilot runs on the whole validation set",
        "by design they were scored on a 50,000-review sample to save time",
    ),
    (
        "FaBERT validation loss at each check",
        "not logged during training; only the kept model's validation loss is known",
    ),
    ("FaBERT trained past epoch 1.75", "early stopping ended the run after two checks without gain"),
    ("Test set", "kept untouched until the end; scored once for each final model"),
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def short(code):
    return code["commit"][:7] + (" (uncommitted)" if code["uncommitted_changes"] else "")


def interval(values):
    return f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


def tfidf_tuning():
    rows = []
    for path in sorted((REPORTS_DIR / "tfidf_runs").glob("*.json")):
        run = load(path)
        rows.append(
            [
                run["head"],
                run["stage"],
                run["setup"],
                run["class_weight"] or "none",
                run["C"],
                "yes" if run["converged"] else "**no**",
                f"{run['validation']['macro_f1']:.4f}",
                short(run["code"]),
            ]
        )
    header = ["Head", "Stage", "Features", "Class weight", "C", "Converged", "Macro F1", "Commit"]
    return ["## TF-IDF stages 1 and 2 (no offsets)", ""] + table(header, rows)


def tfidf_experiments():
    rows = []
    for folder in ("tfidf_experiments", "tfidf_extra"):
        for path in sorted((REPORTS_DIR / folder).glob("*.json")):
            run = load(path)
            if "macro_f1_difference" in run:
                result = f"{run['macro_f1_difference']:+.4f} {interval(run['difference_interval_95'])}"
                verdict = run["direction"]
            else:  # report only: its own score, sometimes on a different validation set
                low, high = run["macro_f1_interval_95"]
                result = f"score {run['scores']['macro_f1']:.4f} [{low:.4f}, {high:.4f}]"
                verdict = "report only"
            converged = run["converged"]
            if isinstance(converged, dict):
                converged = all(converged.values())
            rows.append(
                [
                    run["experiment"],
                    run["head"],
                    run["variant"],
                    result,
                    verdict,
                    "yes" if converged else "**no**",
                    short(run["code"]),
                ]
            )
    # E1, E2, ..., E12, then the reference
    rows.sort(key=lambda row: (int(row[0][1:]) if row[0][1:].isdigit() else 99, row[1], row[2]))
    header = [
        "Experiment",
        "Head",
        "Variant",
        "Difference to reference [95%]",
        "Verdict",
        "Converged",
        "Commit",
    ]
    return ["## TF-IDF experiments (each against the reference, no offsets unless named)", ""] + table(
        header, rows
    )


def tfidf_final():
    rows = []
    for path in sorted((REPORTS_DIR / "tfidf_final").glob("final__*.json")):
        run = load(path)
        for name in (
            "final_vs_reference",
            "final_vs_reference_with_offsets",
            "final_without_offsets_vs_reference",
        ):
            found = run[name]
            rows.append(
                [
                    run["head"],
                    name.replace("_", " "),
                    f"{found['variant_scores']['macro_f1']:.4f}",
                    f"{found['reference_scores']['macro_f1']:.4f}",
                    f"{found['macro_f1_difference']:+.4f} {interval(found['difference_interval_95'])}",
                    short(run["code"]),
                ]
            )
    chosen = load(REPORTS_DIR / "tfidf_final" / "chosen_model.json")
    kept = []
    for head, found in chosen["heads"].items():
        kept.append(
            [
                head,
                f"{found['validation_without_offsets']['macro_f1']:.4f}",
                f"{found['validation_with_offsets']['macro_f1']:.4f}",
                short(chosen["code"]),
            ]
        )
    return (
        ["## TF-IDF combined run (final_tfidf.py)", ""]
        + table(["Head", "Comparison", "Variant", "Reference", "Difference [95%]", "Commit"], rows)
        + ["## TF-IDF model kept: reference with offsets (fit_final_tfidf.py)", ""]
        + table(["Head", "Without offsets", "With offsets", "Commit"], kept)
    )


def transformer_pilot():
    summary = load(REPORTS_DIR / "transformer_pilot" / "summary.json")
    rows = []
    for run in summary["runs"].values():
        row = [run["model"], run["seed"], f"{run['train_rows']:,}", run["train_minutes"]]
        for head in ("recommendation", "rating_band"):
            found = run[head]
            difference = f"{found['difference']:+.4f} {interval(found['difference_interval_95'])}"
            row.append(f"{found['macro_f1']:.4f} ({difference})")
        row.append(f"{run['mean_macro_f1']:.4f}")
        rows.append(row)
    rows.sort(key=lambda row: (row[0], row[2], row[1]))
    header = [
        "Model",
        "Seed",
        "Training reviews",
        "Minutes",
        "Recommendation (vs TF-IDF [95%])",
        "Rating band (vs TF-IDF [95%])",
        "Mean",
    ]
    note = "Scored with offsets on the same 50,000 validation reviews, as is TF-IDF (the kept model)."
    return (
        ["## Transformer pilot (1 epoch)", "", note, ""]
        + table(header, rows)
        + [f"Commit {short(summary['code'])}", ""]
    )


def transformer_final():
    run = load(REPORTS_DIR / "transformer_final" / "fabert.json")
    comparison = load(REPORTS_DIR / "transformer_final" / "vs_tfidf.json")
    checks = [
        [
            c["epoch"],
            f"{c['step']:,}",
            f"{c['score']:.4f}",
            f"{c['recommendation_macro_f1']:.4f}",
            f"{c['rating_band_macro_f1']:.4f}",
            "kept" if c["kept"] else "",
        ]
        for c in run["checks"]
    ]
    heads = []
    for head, found in comparison["heads"].items():
        heads.append(
            [
                head,
                f"{run['heads'][head]['without_offsets']['macro_f1']:.4f}",
                f"{found['transformer']['macro_f1']:.4f}",
                f"{found['tfidf']['macro_f1']:.4f}",
                f"{found['macro_f1_difference']:+.4f} {interval(found['difference_interval_95'])}",
            ]
        )
    loss = comparison["loss"]
    losses = [
        [head, f"{loss[head]['transformer_validation']:.4f}", f"{loss[head]['tfidf_validation']:.4f}"]
        for head in comparison["heads"]
    ]
    training = loss["transformer_training_both_heads_near_kept_step"]
    return (
        [
            "## FaBERT on all training reviews",
            "",
            "Checks on the 50,000-review validation sample, with offsets:",
            "",
        ]
        + table(["Epoch", "Step", "Mean", "Recommendation", "Rating band", ""], checks)
        + [f"Commit {short(run['code'])}. Whole validation set:", ""]
        + table(
            [
                "Head",
                "FaBERT without offsets",
                "FaBERT with offsets",
                "TF-IDF with offsets",
                "Difference [95%]",
            ],
            heads,
        )
        + table(["Head", "FaBERT validation loss", "TF-IDF validation loss"], losses)
        + [
            f"FaBERT loss of both heads: training {training:.4f} (logged steps before the kept one), "
            f"validation {loss['transformer_validation_both_heads']:.4f}.",
            "",
        ]
    )


def main():
    lines = [
        "# All results",
        "",
        "Written by `scripts/summarize_results.py` from the files in `reports/`; do not edit by hand.",
        "Macro F1 on the validation set. A difference is shown with its 95% bootstrap interval",
        "[low, high]: a change counts only when the whole interval is above zero.",
        "",
    ]
    lines += tfidf_tuning() + tfidf_experiments() + tfidf_final() + transformer_pilot() + transformer_final()
    lines += ["## Not measured", ""] + table(["What", "Why"], NOT_MEASURED)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
