"""Write every result of the project into one file, from the saved files.

Nothing is trained or scored again; each number is read from a JSON file in reports/ and
printed with the commit it was made with. Scores are Macro F1 on the validation set, except in
the sections on the test set. A difference is shown with its 95% bootstrap interval: [low, high].

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
    (
        "Class-balanced pseudo-labels",
        "not part of the project's protocol or of the extensions plan (reports/extensions_plan.md)",
    ),
    (
        "The quality of the naturally unlabeled reviews' pseudo-labels",
        "those reviews have no label to check them against",
    ),
    ("The service on a GPU", "it is built and checked for the CPU only (reports/docker_check.json)"),
    (
        "The test set for any model outside the two fixed lists",
        "read once for the four official models and once for the extensions (D73, D74)",
    ),
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


def final_per_class():
    """Precision, recall and F1 of every class for both final models, with offsets."""
    comparison = load(REPORTS_DIR / "transformer_final" / "vs_tfidf.json")
    rows, overall = [], []
    for head, found in comparison["heads"].items():
        fabert, tfidf = found["transformer"], found["tfidf"]
        for label, f in fabert["per_class"].items():
            t = tfidf["per_class"][label]
            rows.append(
                [
                    head,
                    label,
                    f"{f['support']:,}",
                    f"{f['precision']:.3f} / {f['recall']:.3f} / {f['f1']:.3f}",
                    f"{t['precision']:.3f} / {t['recall']:.3f} / {t['f1']:.3f}",
                ]
            )
        overall.append(
            [
                head,
                f"{fabert['macro_f1']:.4f} / {fabert['balanced_accuracy']:.4f} / {fabert['accuracy']:.4f}",
                f"{tfidf['macro_f1']:.4f} / {tfidf['balanced_accuracy']:.4f} / {tfidf['accuracy']:.4f}",
            ]
        )
    return (
        [
            "## Final models per class (whole validation set, with offsets)",
            "",
            "Balanced accuracy is the mean recall over the classes, so the large classes do not",
            "weigh more; plain accuracy does, and is shown only next to it.",
            "",
        ]
        + table(
            ["Head", "Class", "Reviews", "FaBERT precision / recall / F1", "TF-IDF precision / recall / F1"],
            rows,
        )
        + table(
            [
                "Head",
                "FaBERT Macro F1 / balanced accuracy / accuracy",
                "TF-IDF Macro F1 / balanced accuracy / accuracy",
            ],
            overall,
        )
    )


TRAINING_VARIANTS = [
    ("merged_vs_separate_n500000.json", "Texts merged (reports/merged_texts_plan.md)"),
    ("hard_examples_n500000.json", "Hard reviews weighted up (reports/hard_examples_plan.md)"),
    ("difficulty_n500000.json", "Difficulty used (reports/difficulty_plan.md)"),
    ("gradual_n500000.json", "Gradual curriculum (reports/difficulty_plan.md)"),
    ("learning_rate_n500000.json", "Learning rate (reports/learning_rate_plan.md)"),
    ("pseudo_n500000.json", "Pseudo-labels (reports/semi_supervised_plan.md)"),
    ("adapted_n500000.json", "Domain-adapted FaBERT (reports/domain_adaptation_plan.md)"),
]


def training_variants():
    """FaBERT pilot runs on 500,000 reviews with a changed training, against the usual training."""
    rows = []
    for name, title in TRAINING_VARIANTS:
        path = REPORTS_DIR / "transformer_pilot_whole" / name
        if not path.exists():
            rows.append([title, "", "", "", "", "", "not run yet", ""])
            continue
        found = load(path)
        variants = found.get("variants") or {
            k: v for k, v in found.items() if isinstance(v, dict) and "verdict" in v
        }
        for variant, r in variants.items():
            means = r["mean_over_seeds"]
            new, old = (
                (means["merged"], means["separate"])
                if "merged" in means
                else (means["variant"], means["reference"])
            )
            lead = r.get("lead_of_variant", r.get("lead_of_merged"))
            rows.append(
                [
                    title,
                    variant,
                    f"{new:.4f}",
                    f"{old:.4f}",
                    f"{lead:+.4f}",
                    f"{r['spread_between_seeds']:.4f}",
                    r["verdict"] if r["verdict"] == "tie" else ("better" if lead > 0 else "worse"),
                    short(found["code"]),
                ]
            )
    return [
        "## Training changes on the FaBERT pilot (500,000 reviews, 1 epoch, 3 seeds, with offsets)",
        "",
        "Mean Macro F1 of the two heads over the seeds. A lead smaller than the spread between",
        "seeds is a tie.",
        "",
    ] + table(
        ["Experiment", "Variant", "Score", "Usual training", "Lead", "Spread", "Verdict", "Made with"], rows
    )


def calibration():
    path = REPORTS_DIR / "calibration.json"
    if not path.exists():
        return []
    found = load(path)
    rows = []
    for head, r in found["heads"].items():
        b, a = r["before"], r["after"]
        sure_b, sure_a = b["at_least_threshold"], a["at_least_threshold"]
        rows.append(
            [
                head,
                f"{r['temperature_all_validation']:.3f}",
                f"{b['reliability']['ece']:.4f} -> {a['reliability']['ece']:.4f}",
                f"{b['mean_loss']:.4f} -> {a['mean_loss']:.4f}",
                f"{sure_b['share']:.1%}, {sure_b['accuracy']:.1%} right -> "
                f"{sure_a['share']:.1%}, {sure_a['accuracy']:.1%} right",
                f"{b['macro_f1_with_offsets']:.4f} -> {a['macro_f1_with_offsets']:.4f}",
            ]
        )
    return [
        "## Calibration of the final FaBERT model (reports/calibration_plan.md)",
        "",
        f"Temperature scaling, fitted on one half of validation by product and scored on the other. "
        f"Made with {short(found['code'])}.",
        "",
    ] + table(
        [
            "Head",
            "Temperature",
            "ECE",
            "Mean loss",
            f"Reviews with p >= {found['threshold']}",
            "Macro F1 with offsets",
        ],
        rows,
    )


def label_sample():
    path = REPORTS_DIR / "label_sample.json"
    if not path.exists():
        return []
    found = load(path)
    mine, model = found["my_label_vs_original"], found["fabert_vs_my_label"]
    rows = [
        [label, mine[label]["rows"], f"{mine[label]['agree']:.0%}", f"{model[label]['agree']:.0%}"]
        for label in ("recommended", "not_recommended", "no_idea")
    ]
    return [
        "## Hand-labeled validation sample (reports/label_sample_plan.md)",
        "",
        f"{found['reviews']} reviews labeled again from the text alone. Made with {short(found['code'])}.",
        "",
    ] + table(["Original label", "Reviews", "Hand label agrees", "FaBERT agrees with the hand label"], rows)


def required_metrics():
    """Every required score of every final model (scripts/score_required_metrics.py)."""
    path = REPORTS_DIR / "required_metrics.json"
    if not path.exists():
        return []
    found = load(path)
    lines = [
        "## Every required score of the final models (whole validation set, with offsets)",
        "",
        f"Made with {short(found['code'])}. Macro precision and recall are plain means over the classes; "
        "the ordinal error is how many bands a prediction is off on average, and the share off by two.",
        "",
    ]
    for head, models in found["heads"].items():
        rows = []
        for name, e in models.items():
            row = [
                name,
                f"{e['accuracy']:.4f}",
                f"{e['macro_f1']:.4f}",
                f"{e['macro_precision']:.4f}",
                f"{e['macro_recall']:.4f}",
                f"{e['balanced_accuracy']:.4f}",
            ]
            if "ordinal_distance_error" in e:
                o = e["ordinal_distance_error"]
                row += [f"{o['mean_distance']:.4f}", f"{o['share_off_by_two']:.2%}"]
            rows.append(row)
        header = ["Model", "Accuracy", "Macro F1", "Macro precision", "Macro recall", "Balanced accuracy"]
        if head == "rating_band":
            header += ["Ordinal error", "Off by two"]
        lines += [f"{head} ({next(iter(models.values()))['rows']:,} reviews):", ""] + table(header, rows)
    return lines


def decision_recheck():
    """The main decisions with intervals that draw products (reports/decision_recheck_plan.md)."""
    path = REPORTS_DIR / "decision_recheck.json"
    if not path.exists():
        return []
    found = load(path)
    heads = list(found["heads"])
    first = found["heads"][heads[0]]["comparisons"]
    rows = []
    for i, c in enumerate(first):
        row = [c["decision"], f"{c['variant']} against {c['reference']}"]
        for head in heads:
            d = found["heads"][head]["comparisons"][i]
            row.append(f"{d['difference']:+.4f} {interval(d['interval_95_products'])}")
        rows.append(row)
    return [
        "## The main decisions, with product-level intervals (reports/decision_recheck_plan.md)",
        "",
        f"Made with {short(found['code'])}.",
        "",
    ] + table(["Decision", "Comparison", *heads], rows)


def full_seeds():
    """The full runs trained again with other seeds (reports/full_seeds_plan.md)."""
    path = REPORTS_DIR / "full_seeds.json"
    if not path.exists():
        return []
    found = load(path)
    seeds = [str(s) for s in found["seeds"]]
    rows, d58 = [], []
    for head, e in found["heads"].items():
        f = e["macro_f1"]
        for model, label in [("fabert", "FaBERT"), ("bilstm", "BiLSTM, every epoch")]:
            values = [f[f"{model}_{s}"] for s in seeds]
            rows.append([label, head, *[f"{v:.4f}" for v in values], f"{max(values) - min(values):.4f}"])
        d58.append(
            [head]
            + [
                f"{e['d58_with_each_fabert_seed'][s]['difference']:+.4f} "
                f"{interval(e['d58_with_each_fabert_seed'][s]['interval_95_products'])}"
                for s in seeds
            ]
        )
    return (
        [
            "## The full runs with three training seeds (reports/full_seeds_plan.md)",
            "",
            f"Made with {short(found['code'])}. The official model is the seed 2026 run.",
            "",
        ]
        + table(["Model", "Head", *[f"Seed {s}" for s in seeds], "Spread"], rows)
        + ["D58 (FaBERT + TF-IDF minus FaBERT) with each FaBERT seed:", ""]
        + table(["Head", *[f"Seed {s}" for s in seeds]], d58)
    )


def error_analyses():
    """What the mistakes are made of: the reading models, the text ceiling, the independent reader."""
    lines = []
    path = REPORTS_DIR / "reading_models.json"
    if path.exists():
        found = load(path)
        rows = []
        for head, e in found["heads"].items():
            g = e["overall"]["groups"]
            no_cause = e["causes"]["all wrong"]["none_of_these"]["share"]
            rows.append(
                [
                    head,
                    f"{e['reviews']:,}",
                    f"{g['all right']['share']:.1%}",
                    f"{g['all wrong']['share']:.1%}",
                    f"{e['all_wrong_same_label']['share']:.1%}",
                    f"{no_cause:.1%}",
                ]
            )
        lines += [
            "## Three models that read differently (reports/reading_models_plan.md)",
            "",
            f"FaBERT, TF-IDF and the BiLSTM on 872,795 validation reviews. Made with {short(found['code'])}.",
            "",
        ] + table(
            [
                "Head",
                "Reviews",
                "All right",
                "All wrong",
                "All wrong, same label",
                "All wrong, no known cause",
            ],
            rows,
        )
    path = REPORTS_DIR / "text_ceiling.json"
    if path.exists():
        found = load(path)
        rows = []
        for head, e in found["heads"].items():
            r = e["by_min_count"]["5"]
            m = r["models"]
            rows.append(
                [
                    head,
                    f"{r['reviews']:,} ({r['share_of_compared_reviews']:.1%})",
                    f"{r['ceiling_accuracy']:.4f}",
                    f"{m['fabert']['accuracy']:.4f}",
                    f"{m['bilstm']['accuracy']:.4f}",
                    f"{m['tfidf']['accuracy']:.4f}",
                ]
            )
        lines += [
            "## The accuracy ceiling of any text-only model on repeated texts (reports/text_ceiling_plan.md)",
            "",
            f"Texts seen at least 5 times. Made with {short(found['code'])}.",
            "",
        ] + table(["Head", "Reviews", "Ceiling", "FaBERT", "BiLSTM", "TF-IDF"], rows)
    path = REPORTS_DIR / "independent_reader.json"
    if path.exists():
        found = load(path)
        rows = []
        for head, e in found["heads"].items():
            d = e["decision"]
            rows.append(
                [
                    head,
                    f"{d['control']:.1%}",
                    f"{d['with_label']:.1%}",
                    f"{d['with_models']:.1%}",
                    d["verdict"],
                ]
            )
        lines += [
            "## A reader that never saw the labels (reports/independent_reader_plan.md)",
            "",
            f"{found['reader']}, zero-shot. Made with {short(found['code'])}.",
            "",
        ] + table(
            [
                "Head",
                "All right: agrees with the label",
                "All wrong, same label: with the label",
                "with the models",
                "Verdict",
            ],
            rows,
        )
    return lines


def p7_conflicts():
    path = REPORTS_DIR / "p7_conflicts.json"
    if not path.exists():
        return []
    found = load(path)
    rows = [[name, count] for name, count in found["categories"].items()]
    return [
        "## Ids removed by cleaning rule P7 (report only)",
        "",
        f"{found['ids']} ids, {found['rows']} rows; every pair shares a product and a creation time. "
        f"Made with {short(found['code'])}.",
        "",
    ] + table(["What differs", "Ids"], rows)


def final_evaluation():
    """The four official models on the test set, read once (reports/final_evaluation_plan.md)."""
    path = REPORTS_DIR / "final_evaluation.json"
    if not path.exists():
        return []
    found = load(path)
    lines = [
        "## The test set: the four official models (reports/final_evaluation_plan.md)",
        "",
        f"Made with {short(found['code'])}. Test Macro F1 with offsets tuned on all of validation, its 95% "
        "interval drawing products, and the validation Macro F1 beside it.",
        "",
    ]
    for head, entry in found["heads"].items():
        rows = []
        for name, m in entry["models"].items():
            w = m["with_offsets"]
            rows.append(
                [
                    name,
                    f"{w['accuracy']:.4f}",
                    f"{w['macro_f1']:.4f} [{m['macro_f1_interval_95_products'][0]:.4f}, "
                    f"{m['macro_f1_interval_95_products'][1]:.4f}]",
                    f"{w['balanced_accuracy']:.4f}",
                    f"{m['validation']['macro_f1']:.4f}",
                ]
            )
        header = ["Model", "Accuracy", "Macro F1 [95%]", "Balanced accuracy", "Validation Macro F1"]
        lines += [f"{head} ({w['rows']:,} reviews):", ""] + table(header, rows)
        u = entry["models"]["final system"]["uncertain"]
        answered = u["answered_reviews"]
        lines += [
            f'The final system answers "uncertain" on {u["share"]:.1%} of them; on the others accuracy '
            f"{answered['accuracy']:.4f}.",
            "",
        ]
    return lines


def semi_supervised_official():
    """The semi-supervised experiment as the project defines it (reports/semi_supervised_official_plan.md)."""
    path = REPORTS_DIR / "semi_supervised.json"
    if not path.exists():
        return []
    found = load(path)
    accepted = found["pseudo_labels"][str(found["threshold"])]
    kept = ", ".join(f"{name} {epoch}" for name, epoch in found["kept_check"].items())
    rows = [
        [
            name,
            f"{m['with_offsets']['macro_f1']:.4f}",
            f"{found['heads']['rating_band']['models'][name]['with_offsets']['macro_f1']:.4f}",
        ]
        for name, m in found["heads"]["recommendation"]["models"].items()
    ]
    differences = found["heads"]["recommendation"]["differences_with_offsets"]
    rows_differences = [
        [name, f"{d['difference']:+.4f} {interval(d['interval_95_products'])}"]
        for name, d in differences.items()
    ]
    return (
        [
            "## Semi-supervised learning, the project's protocol (reports/semi_supervised_official_plan.md)",
            "",
            f"Made with {short(found['code'])}. Threshold {found['threshold']}: {found['accepted']:,} of "
            f"{found['hidden_reviews']:,} hidden reviews accepted, {accepted['accuracy']:.1%} of them right; "
            f"the rule (more than {found['rule']['min_lead']:.4f} over the control, interval above zero): "
            f"pseudo-labels help = {found['rule']['pseudo_labels_help']}. Checks kept (epochs): "
            f"{kept}; the student and the control kept their start, the teacher, because "
            "no later check was better, so the comparison uses both at the end of their epoch.",
            "",
        ]
        + table(["Model (validation)", "recommendation", "rating_band"], rows)
        + table(["Recommendation difference", "Macro F1 [95%]"], rows_differences)
    )


def extensions():
    """The optional extensions on validation and on the test set (reports/extensions_plan.md)."""
    path, test_path = (
        REPORTS_DIR / "extensions_validation.json",
        REPORTS_DIR / "final_evaluation_extensions.json",
    )
    if not path.exists() or not test_path.exists():
        return []
    found, test = load(path), load(test_path)
    rate_rows = []
    for name, r in found["rate"].items():
        row = [name]
        for scores in (r["scores"], test["rate"][name]["scores"]):
            spearman = "–" if scores["spearman"] is None else f"{scores['spearman']:.3f}"
            r2 = round(scores["r2"], 3) + 0.0  # no "-0.000" for the constant
            row += [f"{scores['mae']:.3f}", f"{scores['rmse']:.3f}", f"{r2:.3f}", spearman]
        rate_rows.append(row)
    lines = [
        "## The extensions (reports/extensions_plan.md)",
        "",
        f"Validation made with {short(found['code'])}; the test set read a second and last time with "
        f"{short(test['code'])} (D74). The exact rate is scored on the reviews with a valid rate (D75).",
        "",
        "Exact rate:",
        "",
    ]
    header = ["Model", "Val MAE", "Val RMSE", "Val R2", "Val Spearman"]
    header += ["Test MAE", "Test RMSE", "Test R2", "Test Spearman"]
    lines += table(header, rate_rows)
    for head in found["heads"]:
        rows = []
        for name, m in test["heads"][head]["models"].items():
            rows.append(
                [name, f"{found['heads'][head]['models'][name]['macro_f1']:.4f}", f"{m['macro_f1']:.4f}"]
            )
        differences = [
            [name, f"{d['difference']:+.4f} {interval(d['interval_95_products'])}"]
            for name, d in test["heads"][head]["differences"].items()
        ]
        lines += [f"{head}, Macro F1 with offsets:", ""] + table(["Model", "Validation", "Test"], rows)
        lines += table(["Test difference", "Macro F1 [95%]"], differences)
    rule = ", ".join(f"{name} {'yes' if v else 'no'}" for name, v in found["rule_pseudo_labels_help"].items())
    band = found["band_guesses"]["quality"]["0.9"]
    lines += [
        f"Pseudo-labels help by the rule (on validation): {rule}. Band guesses at 0.9: "
        f"{band['share_kept']:.1%} kept, {band['accuracy']:.1%} right. Naturally unlabeled reviews at 0.9: "
        f"{found['natural_guesses']['share_at_least_0.9']:.1%}.",
        "",
    ]
    return lines


def service():
    """Memory of the predictor and the service inside Docker (reports/extensions_plan.md, 6 to 8)."""
    memory_path, docker_path = REPORTS_DIR / "predictor_memory.json", REPORTS_DIR / "docker_check.json"
    check_path = REPORTS_DIR / "predictor_check.json"
    if not memory_path.exists() or not docker_path.exists() or not check_path.exists():
        return []
    memory, docker, check = load(memory_path), load(docker_path), load(check_path)
    m, latency = memory["memory_gb"], memory["latency_ms_one_review"]
    rate = check.get("rate") or {}
    rows = [
        [
            "Predictor against the saved predictions: passed; predicted rate, mean difference",
            f"{check['passed']}; {rate.get('mean_difference', float('nan')):.4f}",
            short(check["code"]),
        ],
        ["Models in memory", f"{m['models_alone']} GB", short(memory["code"])],
        [
            "Process after loading, peak while loading",
            f"{m['after_loading']['now']} GB, {m['after_loading']['peak']} GB",
            short(memory["code"]),
        ],
        [
            "One review (median, 95th percentile)",
            f"{latency['median']:.0f} ms, {latency['p95']:.0f} ms",
            short(memory["code"]),
        ],
        ["Docker image", f"{docker['image_size_gb']} GB", short(docker["code"])],
        [
            "Container ready after (models loaded in)",
            f"{docker['seconds_until_ready']} s ({docker['models_loaded_seconds']} s)",
            short(docker["code"]),
        ],
        [
            "Health check; /health and /ready from inside; GPU inside",
            f"{docker['health_check']}; {docker.get('from_inside')}; {docker['gpu_inside']}",
            short(docker["code"]),
        ],
        [
            "Requests answered as expected; review text in the log",
            f"{docker['all_as_expected']}; {docker['review_text_in_log']}",
            short(docker["code"]),
        ],
    ]
    return ["## The service (CPU)", ""] + table(["What", "Measured", "Made with"], rows)


def main():
    lines = [
        "# All results",
        "",
        "Written by `scripts/summarize_results.py` from the files in `reports/`; do not edit by hand.",
        "Macro F1 on the validation set, except where a section says test. A difference is shown with",
        "its 95% bootstrap interval [low, high]: a change counts only when the whole interval is above zero.",
        "",
    ]
    lines += tfidf_tuning() + tfidf_experiments() + tfidf_final() + transformer_pilot()
    lines += transformer_pilot_whole() + transformer_final() + final_per_class()
    lines += label_sample() + training_variants() + calibration()
    lines += required_metrics() + decision_recheck() + full_seeds() + error_analyses() + p7_conflicts()
    lines += final_evaluation() + semi_supervised_official() + extensions() + service()
    lines += ["## Not measured", ""] + table(["What", "Why"], NOT_MEASURED)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
