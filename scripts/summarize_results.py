"""Write every result of the TF-IDF and Transformer stages into one table, from the saved files.

Nothing is trained or scored again; each number is read from a JSON file in reports/ and
printed with the commit it was made with. Scores are Macro F1 on the validation set. A
difference is shown with its 95% bootstrap interval: [low, high].

Writes reports/results_summary.md.

Usage:
    uv run python scripts/summarize_results.py
"""

import json

from digikala_reviews.config import FINAL_TRANSFORMER_RESULT, REPORTS_DIR
from digikala_reviews.provenance import machine_kind

OUT = REPORTS_DIR / "results_summary.md"

# Measurements that are missing on purpose or by accident, with the reason. Kept by hand.
NOT_MEASURED = [
    ("Test set", "kept untouched until the end; scored once for each final model"),
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def short(code):
    return code["commit"][:7] + (" (uncommitted)" if code["uncommitted_changes"] else "")


def source(run):
    """The commit a result was made with, and on which machine."""
    return f"{short(run['code'])} · {machine_kind(run.get('environment'))}"


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
                source(run),
            ]
        )
    header = ["Head", "Stage", "Features", "Class weight", "C", "Converged", "Macro F1", "Made with"]
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
                    source(run),
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
        "Made with",
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
                    source(run),
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
                source(chosen),
            ]
        )
    return (
        ["## TF-IDF combined run (final_tfidf.py)", ""]
        + table(["Head", "Comparison", "Variant", "Reference", "Difference [95%]", "Made with"], rows)
        + ["## TF-IDF model kept: reference with offsets (fit_final_tfidf.py)", ""]
        + table(["Head", "Without offsets", "With offsets", "Made with"], kept)
    )


def transformer_pilot():
    summary = load(REPORTS_DIR / "transformer_pilot" / "summary.json")
    rows = []
    for tag, run in summary["runs"].items():
        machine = machine_kind(load(REPORTS_DIR / "transformer_pilot" / f"{tag}.json").get("environment"))
        row = [run["model"], run["seed"], f"{run['train_rows']:,}", run["train_minutes"], machine]
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
        "Machine",
        "Recommendation (vs TF-IDF [95%])",
        "Rating band (vs TF-IDF [95%])",
        "Mean",
    ]
    note = "Scored with offsets on the same 50,000 validation reviews, as is TF-IDF (the kept model)."
    return (
        ["## Transformer pilot (1 epoch)", "", note, ""]
        + table(header, rows)
        + [f"Summary made with {source(summary)}", ""]
    )


def transformer_pilot_whole():
    folder = REPORTS_DIR / "transformer_pilot_whole"
    summary = load(folder / "summary.json")
    rows = [
        [
            a["model"],
            f"{a['train_rows']:,}",
            len(a["seeds"]),
            f"{a['recommendation_macro_f1']:.4f}",
            f"{a['rating_band_macro_f1']:.4f}",
            f"{a['mean_macro_f1']:.4f}",
            f"{a['lowest_seed']:.4f} to {a['highest_seed']:.4f}",
            f"{a['minutes']:.1f}",
        ]
        for a in summary["averaged_over_seeds"]
    ]
    header = [
        "Model",
        "Training reviews",
        "Seeds",
        "Recommendation",
        "Rating band",
        "Mean",
        "Seeds range",
        "Minutes",
    ]
    choice = load(folder / "large_vs_fabert_n500000.json")
    per_seed = []
    for seed, heads in choice["per_seed"].items():
        per_seed.append(
            [seed]
            + [f"{h['difference']:+.4f} {interval(h['difference_interval_95'])}" for h in heads.values()]
        )
    return (
        [
            "## Transformer pilot on the whole validation set (1 epoch, mean over seeds, with offsets)",
            "",
        ]
        + table(header, rows)
        + [f"Summary made with {source(summary)}", ""]
        + ["TookaBERT-Large minus FaBERT, both on 500,000 reviews, same reviews per seed:", ""]
        + table(["Seed", "Recommendation [95%]", "Rating band [95%]"], per_seed)
        + [
            f"Lead of TookaBERT-Large {choice['lead_of_large']:+.5f}, spread between seeds "
            f"{choice['spread_between_seeds']:.5f}: {choice['verdict']}, chosen {choice['chosen']} "
            f"({choice['rule']}). Made with {short(choice['code'])}.",
            "",
        ]
    )


def transformer_final():
    run = load(REPORTS_DIR / FINAL_TRANSFORMER_RESULT)
    comparison = load(REPORTS_DIR / "transformer_final" / "vs_tfidf.json")
    checks = [
        [
            c["epoch"],
            f"{c['step']:,}",
            f"{c['score']:.4f}",
            f"{c['recommendation_macro_f1']:.4f}",
            f"{c['rating_band_macro_f1']:.4f}",
            f"{c['validation_loss_both_heads']:.4f}",
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
            "## FaBERT on all training reviews (the kept run: every epoch, best check kept)",
            "",
            "Checks on the 50,000-review validation sample, with offsets:",
            "",
        ]
        + table(["Epoch", "Step", "Mean", "Recommendation", "Rating band", "Validation loss", ""], checks)
        + [f"Trained with {source(run)}. Whole validation set (compared with {source(comparison)}):", ""]
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
        + kept_against_first()
    )


def kept_against_first():
    """The kept run (every epoch) against the first run (stopped early), on the whole validation set."""
    found = load(REPORTS_DIR / "transformer_final" / "no_early_stop_vs_first.json")
    rows = [
        [
            head,
            f"{h['no_early_stop']['macro_f1']:.4f}",
            f"{h['first']['macro_f1']:.4f}",
            f"{h['macro_f1_difference']:+.4f} {interval(h['difference_interval_95'])}",
            h["direction"],
        ]
        for head, h in found["heads"].items()
    ]
    epochs = found["kept_epoch"]
    return [
        f"Kept run (best check at epoch {epochs['no_early_stop']}) against the first run, stopped early "
        f"(best check at epoch {epochs['first']}), both with offsets on the whole validation set "
        f"(made with {short(found['code'])}):",
        "",
    ] + table(["Head", "Kept run", "First run", "Difference [95%]", "Verdict"], rows)


def main():
    lines = [
        "# All results",
        "",
        "Written by `scripts/summarize_results.py` from the files in `reports/`; do not edit by hand.",
        "Macro F1 on the validation set. A difference is shown with its 95% bootstrap interval",
        "[low, high]: a change counts only when the whole interval is above zero.",
        "",
    ]
    lines += tfidf_tuning() + tfidf_experiments() + tfidf_final() + transformer_pilot()
    lines += transformer_pilot_whole() + transformer_final()
    lines += ["## Not measured", ""] + table(["What", "Why"], NOT_MEASURED)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
