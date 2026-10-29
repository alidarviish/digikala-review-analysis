"""One card per experiment: where its predictions differ from the reference.

Each card is a folder in reports/figures/experiments/ with up to four figures:

  confusion.png  the reference's and the experiment's confusion matrices, and their difference
  per_class.png  the change in F1 of every class
  where.png      the change in Macro F1 by review length, and on texts seen in training or new
  loss.png       the change in validation loss, and how training went when the runs compare

Transformer runs saved the probabilities of every validation review, so their cards have all
four figures, with paired 95% intervals (digikala_reviews.cards). With several seeds the seeds
are pooled. TF-IDF experiments saved only their confusion matrices, so their cards have the
first two figures and no intervals. The numbers of every card are also written to
reports/experiment_cards.json. Cards whose inputs are not on this machine are skipped.
Nothing is trained and the test set is not read.

Usage:
    uv run python scripts/plot_experiment_cards.py
    uv run python scripts/plot_experiment_cards.py --only merged_copies
"""

import argparse
import json
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from digikala_reviews import plotting
from digikala_reviews.analysis import length_bucket, word_count
from digikala_reviews.cards import f1_per_class, loss_comparison, pooled_comparison, row_shares
from digikala_reviews.config import ARTIFACTS_DIR, FIGURES_DIR, FINAL_TRANSFORMER_DIR, REPORTS_DIR
from digikala_reviews.experiments import cross_fitted_offsets, product_half
from digikala_reviews.metrics import classification_report
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

OUT = FIGURES_DIR / "experiments"
HEAD_TITLES = {"recommendation": "Recommendation", "rating_band": "Rating band"}
LENGTH_ORDER = ["1-1", "2-3", "4-7", "8-15", "16-31", "32+"]
LENGTH_NAMES = {"1-1": "1"}
GREEN, RED = "#1baf7a", "#e34948"
PILOT = "transformer_pilot_whole"
FABERT = "sbunlp__fabert"
SEEDS = [2026, 2027, 2028]


def pilot_tag(model, seed, rows):
    """Name of a pilot run, as scripts/pilot_transformers.py saves it."""
    if seed == 2026 and rows == 100_000:
        return model
    return f"{model}__seed{seed}__n{rows}"


def pilot_card(name, model, short, rows, seeds):
    return {
        "name": name,
        "kind": "pilot",
        "title": f"{short} against FaBERT",
        "reference": ("FaBERT", [pilot_tag(FABERT, s, rows) for s in seeds]),
        "variant": (short, [pilot_tag(model, s, rows) for s in seeds]),
        "subtitle": f"{rows:,} training reviews · 1 epoch · "
        + ("1 seed" if len(seeds) == 1 else f"{len(seeds)} seeds pooled")
        + " · whole validation set",
        "training_curve": True,  # same reviews in the same order; only the model differs
    }


TRANSFORMER_CARDS = [
    {
        "name": "final_fabert_vs_tfidf",
        "kind": "final",
        "title": "Final FaBERT against final TF-IDF",
        "reference": ("TF-IDF", "tfidf_final"),
        "variant": ("FaBERT", FINAL_TRANSFORMER_DIR),
        "subtitle": "both trained on all 3.55 million training reviews · whole validation set",
        "training_curve": False,
        "no_curve_reason": (
            "Not drawn: TF-IDF is fitted in one go, not in training steps, so there is no curve to compare."
        ),
    },
    {
        "name": "fabert_every_epoch_vs_first",
        "kind": "final",
        "title": "FaBERT trained for every epoch against the first run",
        "reference": ("first run", "transformer_final"),
        "variant": ("every epoch", "transformer_no_early_stop"),
        "subtitle": "all 3.55 million training reviews · kept at epoch 1.75 and 2.75 · whole validation set",
        "training_curve": "checks",
        "checks": ("transformer_final/fabert.json", "transformer_final/fabert_no_early_stop.json"),
    },
    pilot_card("pilot_tookabert_base", "PartAI__TookaBERT-Base", "TookaBERT-Base", 100_000, SEEDS),
    pilot_card("pilot_tookabert_large", "PartAI__TookaBERT-Large", "TookaBERT-Large", 100_000, SEEDS),
    pilot_card("pilot_xlm_roberta", "FacebookAI__xlm-roberta-base", "XLM-R base", 100_000, [2026]),
    pilot_card("pilot_mmbert", "jhu-clsp__mmBERT-base", "mmBERT base", 100_000, [2026]),
    pilot_card("pilot_parsbert", "HooshvareLab__bert-fa-zwnj-base", "ParsBERT", 100_000, [2026]),
    pilot_card("pilot_tookabert_large_500k", "PartAI__TookaBERT-Large", "TookaBERT-Large", 500_000, SEEDS),
    *[
        {
            "name": f"merged_{mode}",
            "kind": "pilot",
            "title": f"Merged by text ({meaning})",
            "reference": ("separate", [pilot_tag(FABERT, s, 500_000) for s in SEEDS]),
            "variant": ("merged", [f"{pilot_tag(FABERT, s, 500_000)}__merge-{mode}" for s in SEEDS]),
            "subtitle": "FaBERT · 500,000 training reviews · 1 epoch · 3 seeds pooled · whole validation set",
            "training_curve": False,
            "no_curve_reason": (
                "Not drawn: merging builds its batches differently, so the logged training losses "
                "are not on the same scale. Over the whole sample the two losses are the same; "
                "the validation loss on the left is the comparable measure."
            ),
        }
        for mode, meaning in [("copies", "weight = copies"), ("once", "each text once")]
    ],
]

HARD_EXAMPLES = [
    ("hard_focal_0.5", "__focal0.5", "focal 0.5", "Focal loss, gamma 0.5"),
    ("hard_focal_1", "__focal1", "focal 1", "Focal loss, gamma 1"),
    ("hard_focal_2", "__focal2", "focal 2", "Focal loss, gamma 2"),
    ("hard_class_weights", "__classweights", "class weights", "Class weights"),
]
TRANSFORMER_CARDS += [
    {
        "name": name,
        "kind": "pilot",
        "title": title,
        "reference": ("usual", [f"{pilot_tag(FABERT, s, 500_000)}__checks10" for s in SEEDS]),
        "variant": (short, [f"{pilot_tag(FABERT, s, 500_000)}{suffix}__checks10" for s in SEEDS]),
        "subtitle": "FaBERT · 500,000 training reviews · 1 epoch · 3 seeds pooled · whole validation set",
        "training_curve": False,  # the training losses differ by design; the checks are drawn instead
        "no_curve_reason": "Not drawn: the runs were not measured during training.",
    }
    for name, suffix, short, title in HARD_EXAMPLES
]

TFIDF_NAMES = {
    "E1": "per-class offsets",
    "E2": "character n-grams added",
    "E3": "100 frequent words removed",
    "E4": "ordinal model",
    "E5": "plain term counts",
    "E6": "other classifier",
    "E7": "rating 0 as low",
    "E8": "spelling fixes",
    "E11": "conflicting labels removed",
    "E13": "linear SVM, both with offsets",
}


# ---------- reading the predictions ----------


class Validation:
    """Validation labels and the groups a card splits them by, read once."""

    def __init__(self):
        table = load_cache("validation.parquet", extra_columns=["product_id", "text_seen_in_train"])
        self.table = table
        self.position = pd.Series(np.arange(len(table)), index=table["id"].to_numpy())
        self.lengths = length_bucket(word_count(table["text"]))
        self.seen = np.where(table["text_seen_in_train"].to_numpy(), "seen in train", "new text")


def final_run(folder, validation):
    """Predictions with offsets and loss of every labeled review, from a final run's table."""
    table = pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet")
    assert (table["id"].to_numpy() == validation.table["id"].to_numpy()).all(), folder
    found = {}
    for head, h in HEADS.items():
        rows = np.flatnonzero(validation.table[h["column"]].notna().to_numpy())
        true = np.array([h["labels"].index(v) for v in validation.table[h["column"]].to_numpy()[rows]])
        predicted = np.array(
            [h["labels"].index(v) for v in table[f"{head}__predicted_with_offsets"].to_numpy()[rows]]
        )
        log_proba = table[[f"{head}__log_proba__{label}" for label in h["labels"]]].to_numpy()[rows]
        found[head] = {
            "rows": rows,
            "true": true,
            "predicted": predicted,
            "loss": -log_proba[np.arange(len(rows)), true],
        }
    return found


def pilot_run(tag, validation):
    """Predictions with cross-fitted offsets and loss of every labeled review, from a pilot run."""
    arrays = np.load(ARTIFACTS_DIR / PILOT / f"{tag}.npz")
    record = json.loads((REPORTS_DIR / PILOT / f"{tag}.json").read_text(encoding="utf-8"))
    positions = validation.position.loc[arrays["ids"]].to_numpy()
    found = {
        "losses": record.get("losses", []),
        "steps": record.get("steps"),
        "checks": record.get("checks", []),
    }
    for head, h in HEADS.items():
        values = validation.table[h["column"]].to_numpy()[positions]
        labeled = pd.notna(values)
        true = np.array([h["labels"].index(v) for v in values[labeled]])
        halves = product_half(validation.table["product_id"].to_numpy()[positions][labeled])
        predicted, _ = cross_fitted_offsets(arrays[head][labeled], true, halves)
        rebuilt = classification_report(values[labeled], np.array(h["labels"])[predicted], h["labels"])[
            "macro_f1"
        ]
        saved = record["heads"][head]["with_offsets"]["macro_f1"]
        assert abs(rebuilt - saved) < 1e-9, (tag, head, rebuilt, saved)  # the run's own score, rebuilt
        log_proba = arrays[head][labeled]
        found[head] = {
            "rows": positions[labeled],
            "true": true,
            "predicted": predicted,
            "loss": -log_proba[np.arange(len(true)), true],
        }
    return found


def inputs_exist(card):
    if card["kind"] == "final":
        return all(
            (ARTIFACTS_DIR / folder / "validation_predictions.parquet").exists()
            for _, folder in (card["reference"], card["variant"])
        )
    tags = card["reference"][1] + card["variant"][1]
    return all((ARTIFACTS_DIR / PILOT / f"{t}.npz").exists() for t in tags)


# ---------- figures ----------


def heading(fig, title, subtitle, note):
    height = fig.get_figheight()  # the lines sit a fixed distance from the top, in inches
    fig.text(0.01, 1 - 0.3 / height, title, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(
        0.01, 1 - 0.62 / height, subtitle, ha="left", va="top", fontsize=10, color=plotting.INK_SECONDARY
    )
    fig.text(0.01, 0.12 / height, note, fontsize=8.5, color=plotting.INK_SECONDARY)


def matrix(ax, shares, counts, labels, title, diff=False, first=True):
    names = [x.replace("_", " ") for x in labels]
    if diff:
        good = shares * (2 * np.eye(len(labels)) - 1)  # green: more right answers, fewer mistakes
        limit = max(1.0, np.abs(good).max())
        ax.imshow(good, cmap="RdYlGn", norm=TwoSlopeNorm(0, -limit, limit))
    else:
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=100)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if diff:
                text, color = f"{shares[i, j]:+.2f}", plotting.INK
            else:
                text = f"{shares[i, j]:.1f}%\n{plotting.compact(counts[i, j])}"
                color = "white" if shares[i, j] > 55 else plotting.INK
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(labels)), names, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), names if first else [])
    ax.set_xlabel("Predicted")
    if first:
        ax.set_ylabel("True")
    ax.set_title(title, fontsize=10.5)


def confusion_figure(card, matrices, seeds, folder):
    """matrices: head -> (reference counts, variant counts), summed over the seeds."""
    ref_name, var_name = card["reference"][0], card["variant"][0]
    heads = list(matrices)
    fig, axes = plt.subplots(len(heads), 3, figsize=(13.5, 4.8 * len(heads)), squeeze=False)
    for r, head in enumerate(heads):
        labels = HEADS[head]["labels"]
        ref, var = matrices[head]
        ref_s, var_s = row_shares(ref), row_shares(var)
        matrix(
            axes[r][0],
            ref_s,
            ref / seeds,
            labels,
            f"{HEAD_TITLES[head]} · {ref_name}\nMacro F1 {f1_per_class(ref).mean():.4f}",
        )
        matrix(
            axes[r][1],
            var_s,
            var / seeds,
            labels,
            f"{HEAD_TITLES[head]} · {var_name}\nMacro F1 {f1_per_class(var).mean():.4f}",
            first=False,
        )
        matrix(
            axes[r][2],
            var_s - ref_s,
            None,
            labels,
            f"{HEAD_TITLES[head]}\n{var_name} minus {ref_name} (points)",
            diff=True,
            first=False,
        )
    fig.subplots_adjust(
        top=1 - 1.45 / fig.get_figheight(), bottom=0.9 / fig.get_figheight(), wspace=0.12, hspace=0.55
    )
    counts = "counts are per seed" if seeds > 1 else "counts of reviews"
    heading(
        fig,
        f"{card['title']}: confusion matrices",
        card["subtitle"],
        f"Rows sum to 100%; {counts}. Difference: points of each row; green = better (more right "
        "answers on the diagonal, fewer mistakes elsewhere), red = worse.",
    )
    fig.savefig(folder / "confusion.png")
    plt.close(fig)


def per_class_figure(card, changes, folder):
    """changes: head -> (change per class, interval per class or None), in F1 points."""
    ref_name, var_name = card["reference"][0], card["variant"][0]
    heads = list(changes)
    fig, axes = plt.subplots(1, len(heads), figsize=(6 * len(heads), 4.4), squeeze=False)
    for ax, head in zip(axes[0], heads, strict=True):
        labels = HEADS[head]["labels"]
        change, interval = changes[head]
        pos = np.arange(len(change))
        if interval is None:
            ax.barh(pos, change, color=plotting.SERIES[0], height=0.55)
            ends = np.column_stack([change, change])
        else:
            colors = [GREEN if lo > 0 else RED if hi < 0 else plotting.NEUTRAL for lo, hi in interval]
            ax.barh(pos, change, color=colors, height=0.55)
            ax.errorbar(
                change,
                pos,
                xerr=[change - interval[:, 0], interval[:, 1] - change],
                fmt="none",
                ecolor=plotting.INK,
                capsize=3,
                lw=1,
            )
            ends = interval
        ax.axvline(0, color=plotting.AXIS, lw=1)
        ax.set_yticks(pos, [x.replace("_", " ") for x in labels])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        lim = max(0.5, np.abs(ends).max() * 1.4)
        for p, c, (lo, hi) in zip(pos, change, ends, strict=True):
            text = f"{c:+.2f}" if interval is None else f"{c:+.2f}  [{lo:+.2f}, {hi:+.2f}]"
            ax.text(max(hi, 0) + lim * 0.04, p, text, va="center", fontsize=8.5, color=plotting.INK_SECONDARY)
        ax.set_xlim(-lim, lim * 1.6)
        ax.set_xlabel("Change in F1 (points" + (")" if interval is None else ", 95% interval)"))
        ax.set_title(HEAD_TITLES[head], fontsize=10.5)
    fig.subplots_adjust(top=0.74, bottom=0.2, wspace=0.35)
    note = (
        "Only the confusion matrices of this experiment are saved, so there is no interval per class."
        if changes[heads[0]][1] is None
        else "Green / red: the whole 95% interval is above / below zero; gray: no measurable difference."
    )
    heading(fig, f"{card['title']}: F1 per class, {var_name} minus {ref_name}", card["subtitle"], note)
    fig.savefig(folder / "per_class.png")
    plt.close(fig)


def where_figure(card, groups, folder):
    """groups: head -> {"length": {...}, "seen": {...}} from pooled_comparison."""
    ref_name, var_name = card["reference"][0], card["variant"][0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.4, 1]})
    parts = [
        ("length", LENGTH_ORDER, "By length in words (share of reviews)"),
        ("seen", ["seen in train", "new text"], "Seen in training or new"),
    ]
    for ax, (name, order, title) in zip(axes, parts, strict=True):
        width = 0.38
        for k, head in enumerate(HEADS):
            found = groups[head][name]
            values = np.array([found[g]["change"] for g in order]) * 100
            intervals = np.array([found[g]["interval"] for g in order]) * 100
            x = np.arange(len(order)) + (k - 0.5) * width
            ax.bar(x, values, width=width * 0.92, color=plotting.SERIES[k * 3], label=HEAD_TITLES[head])
            ax.errorbar(
                x,
                values,
                yerr=[values - intervals[:, 0], intervals[:, 1] - values],
                fmt="none",
                ecolor=plotting.INK,
                capsize=2.5,
                lw=1,
            )
        shares = [groups["recommendation"][name][g]["share"] for g in order]
        ax.set_xticks(
            np.arange(len(order)),
            [f"{LENGTH_NAMES.get(g, g)}\n{s:.0%}" for g, s in zip(order, shares, strict=True)],
        )
        ax.axhline(0, color=plotting.AXIS, lw=1)
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("Change in Macro F1 (points, 95% interval)")
    axes[0].legend(fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.08), ncols=2)
    fig.subplots_adjust(top=0.72, bottom=0.2, wspace=0.18)
    heading(
        fig,
        f"{card['title']}: where the score changes, {var_name} minus {ref_name}",
        card["subtitle"],
        "A bar whose interval crosses zero is no measurable difference.",
    )
    fig.savefig(folder / "where.png")
    plt.close(fig)


def loss_figure(card, losses, curve, folder):
    """losses: head -> loss_comparison result; curve: None, or (x label, y label, {name: (x, y)})."""
    ref_name, var_name = card["reference"][0], card["variant"][0]
    tall = curve is not None and len(curve[2]) > 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2 if tall else 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    heads = list(losses)
    pos = np.arange(len(heads))
    change = np.array([losses[h]["change"] for h in heads])
    interval = np.array([losses[h]["interval"] for h in heads])
    colors = [
        GREEN if hi < 0 else RED if lo > 0 else plotting.NEUTRAL for lo, hi in interval
    ]  # lower is better
    ax.barh(pos, change, color=colors, height=0.5)
    ax.errorbar(
        change,
        pos,
        xerr=[change - interval[:, 0], interval[:, 1] - change],
        fmt="none",
        ecolor=plotting.INK,
        capsize=3,
        lw=1,
    )
    ax.axvline(0, color=plotting.AXIS, lw=1)
    ax.set_yticks(pos, [HEAD_TITLES[h] for h in heads])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    lim = max(np.abs(interval).max() * 1.5, 1e-4)  # not 0 when the two setups agree everywhere
    for p, h, c, (lo, hi) in zip(pos, heads, change, interval, strict=True):
        ax.text(
            max(hi, 0) + lim * 0.05,
            p,
            f"{losses[h]['reference']:.4f} -> {losses[h]['variant']:.4f}"
            f"   ({c:+.4f}  [{lo:+.4f}, {hi:+.4f}])",
            va="center",
            fontsize=8.5,
            color=plotting.INK_SECONDARY,
        )
    ax.set_xlim(-lim, lim * 3.2)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("Change in validation loss (95% interval; lower is better)")
    ax.set_title(f"Validation loss, {var_name} minus {ref_name}", fontsize=10.5)

    ax = axes[1]
    if curve is None:
        ax.axis("off")
        ax.set_title("During training", fontsize=10.5)
        reason = card.get("no_curve_reason", "Not drawn: the two setups were not logged on the same scale.")
        ax.text(
            0,
            0.85,
            reason,
            va="top",
            wrap=True,
            fontsize=9.5,
            color=plotting.INK_SECONDARY,
            transform=ax.transAxes,
        )
    else:
        xlabel, ylabel, lines = curve
        for label, x, y, color, style in lines:
            ax.plot(
                x, y, color=color, ls=style, lw=1.6, label=label, marker="o" if len(x) < 30 else None, ms=4
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("During training", fontsize=10.5)
        if len(lines) > 2:  # below the chart, so it does not cover the lines
            ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.22), ncols=2)
        else:
            ax.legend(fontsize=8.5)
    fig.subplots_adjust(top=0.75, bottom=0.34 if tall else 0.2, wspace=0.25, left=0.12)
    heading(
        fig,
        f"{card['title']}: loss",
        card["subtitle"],
        "Validation loss: mean of -log p(true class) over the same reviews, averaged over the seeds; "
        "green / red: the whole interval is below / above zero.",
    )
    fig.savefig(folder / "loss.png")
    plt.close(fig)


# ---------- cards ----------


def training_curve(card, ref_runs, var_runs):
    """What the right half of loss.png draws, or None.

    Runs trained with checks (pilot_transformers.py --checks) have the validation loss measured
    during training, which compares in every case. Otherwise the training loss is drawn when
    both setups build their batches the same way, and the checks of the final runs are drawn
    as their Macro F1.
    """
    setups = (
        (card["reference"][0], ref_runs, plotting.MUTED),
        (card["variant"][0], var_runs, plotting.SERIES[0]),
    )
    if card["kind"] == "pilot" and all(r["checks"] for r in ref_runs + var_runs):
        lines = []
        for name, runs, color in setups:
            for head, style in zip(HEADS, ["-", "--"], strict=True):
                x = [c["epoch"] for c in runs[0]["checks"]]
                y = np.mean([[c[f"{head}_validation_loss"] for c in r["checks"]] for r in runs], axis=0)
                lines.append((f"{name} · {HEAD_TITLES[head]}", x, y, color, style))
        return "Epoch", "Validation loss on 50,000 reviews (mean of seeds)", lines
    if card["training_curve"] is True:
        lines = []
        for name, runs, color in setups:
            x = np.array([p["step"] for p in runs[0]["losses"]]) / runs[0]["steps"]
            y = np.mean([[p["mean_loss"] for p in r["losses"]] for r in runs], axis=0)
            lines.append((name, x, y, color, "-"))
        return "Share of the epoch", "Training loss (mean of seeds)", lines
    if card["training_curve"] == "checks":
        lines = []
        for (name, _, color), path in zip(setups, card["checks"], strict=True):
            checks = json.loads((REPORTS_DIR / path).read_text(encoding="utf-8"))["checks"]
            lines.append((name, [c["epoch"] for c in checks], [c["score"] for c in checks], color, "-"))
        return "Epoch", "Mean Macro F1 on the 50,000-review sample", lines
    return None


def transformer_card(card, validation):
    load = (
        (lambda f: final_run(f, validation))
        if card["kind"] == "final"
        else (lambda t: pilot_run(t, validation))
    )
    sources = [card["reference"][1]] if card["kind"] == "final" else card["reference"][1]
    ref_runs = [load(s) for s in sources]
    sources = [card["variant"][1]] if card["kind"] == "final" else card["variant"][1]
    var_runs = [load(s) for s in sources]
    seeds = len(ref_runs)

    matrices, changes, groups, losses, numbers = {}, {}, {}, {}, {}
    for head, h in HEADS.items():
        rows = ref_runs[0][head]["rows"]
        for run in ref_runs + var_runs:  # every run must score the same reviews in the same order
            assert (run[head]["rows"] == rows).all() and (
                run[head]["true"] == ref_runs[0][head]["true"]
            ).all()
        found = pooled_comparison(
            ref_runs[0][head]["true"],
            np.array([r[head]["predicted"] for r in ref_runs]),
            np.array([r[head]["predicted"] for r in var_runs]),
            {"length": validation.lengths[rows], "seen": validation.seen[rows]},
            len(h["labels"]),
        )
        losses[head] = loss_comparison(
            np.array([r[head]["loss"] for r in ref_runs]), np.array([r[head]["loss"] for r in var_runs])
        )
        matrices[head] = (found["reference_matrix"], found["variant_matrix"])
        changes[head] = (found["f1_change"] * 100, found["f1_change_interval"] * 100)
        groups[head] = found["groups"]
        numbers[head] = {
            "reference_macro_f1": float(f1_per_class(found["reference_matrix"]).mean()),
            "variant_macro_f1": float(f1_per_class(found["variant_matrix"]).mean()),
            "f1_change_per_class": dict(zip(h["labels"], found["f1_change"].tolist(), strict=True)),
            "f1_change_interval_per_class": dict(
                zip(h["labels"], found["f1_change_interval"].tolist(), strict=True)
            ),
            "macro_f1_change_by_group": found["groups"],
            "validation_loss": losses[head],
        }
    folder = OUT / card["name"]
    folder.mkdir(parents=True, exist_ok=True)
    confusion_figure(card, matrices, seeds, folder)
    per_class_figure(card, changes, folder)
    where_figure(card, groups, folder)
    loss_figure(card, losses, training_curve(card, ref_runs, var_runs), folder)
    return {"seeds": seeds, "heads": numbers}


def tfidf_cards():
    """One card per TF-IDF experiment and variant that saved both confusion matrices."""
    found = {}
    for folder in ("tfidf_experiments", "tfidf_extra"):
        for path in sorted((REPORTS_DIR / folder).glob("E*__*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            scores = (record.get("variant_scores", {}), record.get("reference_scores", {}))
            if not all("confusion_matrix" in s for s in scores):
                continue
            code, head, variant = path.stem.split("__", 2)
            found.setdefault((code, variant), {})[head] = record
    cards = []
    for (code, variant), unordered in sorted(found.items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1])):
        heads = {h: unordered[h] for h in HEADS if h in unordered}
        name = TFIDF_NAMES.get(code, code)
        if code == "E6":
            name = f"{name}: {variant.replace('_', ' ')}"
        cards.append(
            {
                "name": f"tfidf_{code}_{variant}",
                "kind": "tfidf",
                "title": f"TF-IDF {code}, {name}",
                "reference": ("reference", None),
                "variant": (code, None),
                "subtitle": "TF-IDF on all training reviews · whole validation set · "
                + " · ".join(
                    f"{HEAD_TITLES[h]} Macro F1 {r['macro_f1_difference'] * 100:+.2f} points "
                    f"[{r['difference_interval_95'][0] * 100:+.2f}, "
                    f"{r['difference_interval_95'][1] * 100:+.2f}]"
                    for h, r in heads.items()
                ),
                "records": heads,
            }
        )
    return cards


def tfidf_card(card):
    matrices, changes, numbers = {}, {}, {}
    for head in HEADS:
        if head not in card["records"]:
            continue
        record = card["records"][head]
        ref = np.array(record["reference_scores"]["confusion_matrix"]["rows_are_true_class"], dtype=float)
        var = np.array(record["variant_scores"]["confusion_matrix"]["rows_are_true_class"], dtype=float)
        assert record["reference_scores"]["confusion_matrix"]["labels"] == HEADS[head]["labels"]
        matrices[head] = (ref, var)
        changes[head] = ((f1_per_class(var) - f1_per_class(ref)) * 100, None)
        numbers[head] = {
            "reference_macro_f1": float(f1_per_class(ref).mean()),
            "variant_macro_f1": float(f1_per_class(var).mean()),
            "f1_change_per_class": dict(
                zip(HEADS[head]["labels"], (f1_per_class(var) - f1_per_class(ref)).tolist(), strict=True)
            ),
            "macro_f1_difference": record["macro_f1_difference"],
            "difference_interval_95": record["difference_interval_95"],
        }
    folder = OUT / card["name"]
    folder.mkdir(parents=True, exist_ok=True)
    confusion_figure(card, matrices, 1, folder)
    per_class_figure(card, changes, folder)
    return {"seeds": 1, "heads": numbers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated card names")
    args = parser.parse_args()
    plotting.apply_style()
    cards = [*TRANSFORMER_CARDS, *tfidf_cards()]
    if args.only:
        wanted = set(args.only.split(","))
        cards = [c for c in cards if c["name"] in wanted]
        assert len(cards) == len(wanted), f"unknown card in {sorted(wanted)}"

    path = REPORTS_DIR / "experiment_cards.json"
    numbers = json.loads(path.read_text(encoding="utf-8"))["cards"] if path.exists() else {}
    validation = None
    for card in cards:
        if card["kind"] != "tfidf" and not inputs_exist(card):
            print(f"{card['name']}: skipped, its predictions are not on this machine", flush=True)
            continue
        if card["kind"] == "tfidf":
            numbers[card["name"]] = tfidf_card(card)
        else:
            validation = validation or Validation()
            numbers[card["name"]] = transformer_card(card, validation)
        print(f"{card['name']}: done", flush=True)
    save_json(
        path,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "cards": numbers,
            "code": git_commit(),
        },
    )
    print(f"{len(numbers)} cards in {OUT}")


if __name__ == "__main__":
    main()
