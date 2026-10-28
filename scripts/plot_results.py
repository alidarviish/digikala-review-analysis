"""Draw the result charts of the TF-IDF and Transformer stages from the saved result files.

Every number comes from a JSON file in reports/; nothing is trained or scored again.
Charts are written to reports/figures/results_*.png:

  results_confusion_matrices   both final models on the whole validation set, per head
  results_per_class            F1 of every class, both final models
  results_tfidf_experiments    each TF-IDF experiment against the reference, with its interval
  results_learning_curve       Macro F1 by number of training reviews, TF-IDF and FaBERT
  results_model_comparison     the six Transformer candidates and TF-IDF on the same reviews
  results_training_curve       FaBERT's checks and training loss during the full training
  results_where_it_helps       both final models by review length and by seen / new text

Usage:
    uv run python scripts/plot_results.py
"""

import json
import re

import matplotlib.pyplot as plt
import numpy as np

from digikala_reviews import plotting
from digikala_reviews.config import REPORTS_DIR
from digikala_reviews.tfidf import HEADS

HEAD_TITLES = {"recommendation": "Recommendation", "rating_band": "Rating band"}
MODEL_COLORS = {"TF-IDF": plotting.NEUTRAL, "FaBERT": plotting.SERIES[0]}
SHORT_NAMES = {
    "PartAI/TookaBERT-Large": "TookaBERT-Large",
    "sbunlp/fabert": "FaBERT",
    "PartAI/TookaBERT-Base": "TookaBERT-Base",
    "FacebookAI/xlm-roberta-base": "XLM-R base",
    "jhu-clsp/mmBERT-base": "mmBERT base",
    "HooshvareLab/bert-fa-zwnj-base": "ParsBERT",
}
LENGTH_ORDER = ["1-1", "2-3", "4-7", "8-15", "16-31", "32+"]


def load(relative):
    return json.loads((REPORTS_DIR / relative).read_text(encoding="utf-8"))


def confusion_matrices(comparison):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2))
    for row, head in enumerate(HEADS):
        for col, (model, key) in enumerate([("TF-IDF", "tfidf"), ("FaBERT", "transformer")]):
            ax = axes[row][col]
            scores = comparison["heads"][head][key]
            labels = scores["confusion_matrix"]["labels"]
            counts = np.array(scores["confusion_matrix"]["rows_are_true_class"], dtype=float)
            shares = counts / counts.sum(axis=1, keepdims=True) * 100
            ax.imshow(shares, cmap="Blues", vmin=0, vmax=100)
            ax.grid(False)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    color = "white" if shares[i, j] > 55 else plotting.INK
                    ax.text(
                        j,
                        i,
                        f"{shares[i, j]:.1f}%\n{plotting.compact(counts[i, j])}",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=color,
                    )
            names = [label.replace("_", " ") for label in labels]
            ax.set_xticks(range(len(labels)), names, rotation=20, ha="right")
            ax.set_yticks(range(len(labels)), names)
            ax.set_xlabel("Predicted")
            if col == 0:
                ax.set_ylabel("True")
            ax.set_title(f"{HEAD_TITLES[head]} · {model} · Macro F1 {scores['macro_f1']:.3f}", fontsize=10.5)
    fig.suptitle(
        "Confusion matrices on the whole validation set (rows sum to 100%, offsets applied)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plotting.save(fig, "results_confusion_matrices")


def per_class(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, head in zip(axes, HEADS, strict=True):
        labels = HEADS[head]["labels"]
        x = np.arange(len(labels))
        for k, (model, key) in enumerate([("TF-IDF", "tfidf"), ("FaBERT", "transformer")]):
            values = [comparison["heads"][head][key]["per_class"][label]["f1"] for label in labels]
            bars = ax.bar(x + (k - 0.5) * 0.36, values, width=0.34, color=MODEL_COLORS[model], label=model)
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    fontsize=8,
                    color=plotting.INK_SECONDARY,
                )
        ax.set_xticks(x, [label.replace("_", " ") for label in labels])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1")
        ax.grid(axis="x", visible=False)
        ax.set_title(HEAD_TITLES[head])
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "F1 per class on the whole validation set", x=0.01, ha="left", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    plotting.save(fig, "results_per_class")


def tfidf_experiments():
    names = {
        "E1": "E1 per-class offsets",
        "E2": "E2 + character n-grams",
        "E3": "E3 100 frequent words removed",
        "E4": "E4 ordinal model",
        "E5": "E5 plain term counts",
        "E8": "E8 spelling fixes",
        "E11": "E11 conflicting labels removed",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, head in zip(axes, HEADS, strict=True):
        rows = []
        for code, name in names.items():
            folder = "tfidf_extra" if code in ("E8", "E11") else "tfidf_experiments"
            matches = sorted((REPORTS_DIR / folder).glob(f"{code}__{head}__*.json"))
            if matches:
                found = json.loads(matches[0].read_text(encoding="utf-8"))
                rows.append(
                    (
                        name,
                        found["macro_f1_difference"] * 100,
                        [v * 100 for v in found["difference_interval_95"]],
                        found["direction"],
                    )
                )
            else:
                rows.append((name, None, None, "not run for this head"))
        y = np.arange(len(rows))
        for i, (_, value, interval, direction) in enumerate(rows):
            if value is None:
                ax.text(0.05, i, "not run for this head", va="center", fontsize=8, color=plotting.MUTED)
                continue
            color = {"better": plotting.SERIES[2], "worse": plotting.SERIES[1]}.get(
                direction, plotting.NEUTRAL
            )
            ax.barh(i, value, height=0.55, color=color)
            ax.plot(interval, [i, i], color=plotting.INK, linewidth=1.2)
            edge = interval[1] if value >= 0 else interval[0]
            ax.text(
                edge + (0.2 if value >= 0 else -0.2),
                i,
                f"{value:+.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
                color=plotting.INK_SECONDARY,
            )
        ax.axvline(0, color=plotting.AXIS, linewidth=1)
        ax.set_yticks(y, [r[0] for r in rows])
        ax.set_xlim(-9.5, 4)
        ax.set_xlabel("Macro F1 change against the reference (points, 95% interval)")
        ax.grid(axis="y", visible=False)
        ax.set_title(HEAD_TITLES[head])
    axes[0].invert_yaxis()  # the axes share y, so inverting both would undo it
    fig.suptitle(
        "TF-IDF experiments, each on its own against the reference (green: whole interval above zero)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plotting.save(fig, "results_tfidf_experiments")


def learning_curve(final):
    chosen = load("tfidf_final/chosen_model.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, head in zip(axes, HEADS, strict=True):
        sizes, values = [], []
        for n in (10_000, 100_000, 1_000_000):
            found = load(f"tfidf_extra/E12__{head}__n{n}.json")
            sizes.append(found["training_reviews"])
            values.append(found["scores"]["macro_f1"])
        sizes.append(chosen["heads"][head]["training_rows"])
        values.append(chosen["heads"][head]["validation_without_offsets"]["macro_f1"])
        ax.plot(sizes, values, marker="o", color=plotting.MUTED, label="TF-IDF (whole validation)")

        fabert_sizes, fabert_values = [], []
        for name in ("sbunlp__fabert", "sbunlp__fabert__seed2026__n500000"):
            run = load(f"transformer_pilot/{name}.json")
            fabert_sizes.append(run["fingerprint"]["settings"]["train_rows"])
            fabert_values.append(run["heads"][head]["without_offsets"]["macro_f1"])
        fabert_sizes.append(final["training_reviews"])
        fabert_values.append(final["heads"][head]["without_offsets"]["macro_f1"])
        ax.plot(fabert_sizes, fabert_values, marker="o", color=plotting.SERIES[0], label="FaBERT")
        ax.plot(
            fabert_sizes[:2], fabert_values[:2], "o", color="white", markersize=3.5
        )  # 50,000-review sample
        for x, v in zip(sizes + fabert_sizes, values + fabert_values, strict=True):
            ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5, color=plotting.INK_SECONDARY)
        ax.set_xscale("log")
        ax.set_xlabel("Training reviews (log scale)")
        ax.set_title(HEAD_TITLES[head])
    axes[0].set_ylabel("Macro F1 without offsets")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(
        "More training data: TF-IDF against FaBERT (open dots: scored on the 50,000-review sample)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plotting.save(fig, "results_learning_curve")


def model_comparison():
    summary = load("transformer_pilot/summary.json")
    rows = [a for a in summary["averaged_over_seeds"] if a["train_rows"] == 100_000]
    rows.sort(key=lambda a: -a["mean_macro_f1"])
    tfidf = np.mean([np.mean([r[h]["tfidf_macro_f1"] for h in HEADS]) for r in summary["runs"].values()])
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = np.arange(len(rows))
    values = [a["mean_macro_f1"] for a in rows]
    colors = [
        plotting.SERIES[0] if a["model"] == "sbunlp/fabert" else plotting.SERIES[0] + "88" for a in rows
    ]
    ax.barh(y, values, height=0.55, color=colors)
    for i, a in enumerate(rows):
        if len(a["seeds"]) > 1:
            ax.plot([a["lowest_seed"], a["highest_seed"]], [i, i], color=plotting.INK, linewidth=1.4)
        seeds = f"{len(a['seeds'])} seeds" if len(a["seeds"]) > 1 else "1 seed"
        ax.text(
            max(a["highest_seed"], a["mean_macro_f1"]) + 0.001,
            i,
            f"{a['mean_macro_f1']:.4f} · {seeds} · {a['minutes']:.1f} min",
            va="center",
            fontsize=8,
            color=plotting.INK_SECONDARY,
        )
    ax.axvline(tfidf, color=plotting.SERIES[1], linewidth=1.3, linestyle="--")
    ax.text(tfidf, -0.55, f" TF-IDF {tfidf:.4f}", color=plotting.SERIES[1], fontsize=8.5)
    ax.set_yticks(y, [SHORT_NAMES[a["model"]] for a in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.665, 0.705)
    ax.set_xlabel("Mean Macro F1 of the two heads, with offsets (same 50,000 validation reviews)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Transformer candidates after 1 epoch on 100,000 reviews (lines: range over seeds)")
    fig.tight_layout()
    plotting.save(fig, "results_model_comparison")


def training_curve(final):
    text = (REPORTS_DIR / "logs" / "train_molab.log").read_text(encoding="utf-8")
    last = text.split("training reviews,")[-1]  # the full run, not the smoke test before it
    points = [
        (int(s.replace(",", "")), float(v))
        for s, v in re.findall(r"step ([\d,]+)/[\d,]+, loss ([\d.]+)", last)
    ]
    steps_per_epoch = final["steps_per_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    epochs = [s / steps_per_epoch for s, _ in points]
    losses = np.array([v for _, v in points])
    smooth = np.convolve(losses, np.ones(10) / 10, mode="valid")
    axes[0].plot(epochs, losses, color=plotting.SERIES[0] + "44", linewidth=1)
    axes[0].plot(epochs[9:], smooth, color=plotting.SERIES[0], label="training loss (mean of 10 logs)")
    axes[0].axhline(
        final_loss := load("transformer_final/vs_tfidf.json")["loss"]["transformer_validation_both_heads"],
        color=plotting.SERIES[1],
        linestyle="--",
        linewidth=1.3,
        label=f"validation loss of the kept model ({final_loss:.3f})",
    )
    axes[0].set_ylim(0.7, 1.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (sum of the two heads)")
    axes[0].legend(fontsize=8.5)
    axes[0].set_title("Loss")

    checks = final["checks"]
    x = [c["epoch"] for c in checks]
    axes[1].plot(x, [c["score"] for c in checks], marker="o", color=plotting.SERIES[0], label="mean of both")
    axes[1].plot(
        x,
        [c["recommendation_macro_f1"] for c in checks],
        marker=".",
        color=plotting.SERIES[2],
        linewidth=1.2,
        label="recommendation",
    )
    axes[1].plot(
        x,
        [c["rating_band_macro_f1"] for c in checks],
        marker=".",
        color=plotting.SERIES[3],
        linewidth=1.2,
        label="rating band",
    )
    kept = final["kept_epoch"]
    axes[1].axvline(kept, color=plotting.MUTED, linestyle=":", linewidth=1.2)
    axes[1].text(
        kept, axes[1].get_ylim()[1], f" kept: epoch {kept}", va="top", fontsize=8, color=plotting.MUTED
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1 with offsets")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_title("Checks on the 50,000-review validation sample")
    fig.suptitle(
        "FaBERT on all 3.55 million training reviews (stopped after two checks without gain)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plotting.save(fig, "results_training_curve")


def where_it_helps():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), gridspec_kw={"width_ratios": [3, 1.4]})
    for row, head in enumerate(HEADS):
        for col, (name, order) in enumerate([("length", LENGTH_ORDER), ("seen", ["seen_in_train", "new"])]):
            ax = axes[row][col]
            tfidf = load(f"tfidf_analysis/{name}.json")[head]
            transformer = load(f"transformer_analysis/{name}.json")[head]
            x = np.arange(len(order))
            for k, (model, source) in enumerate([("TF-IDF", tfidf), ("FaBERT", transformer)]):
                values = [source[g]["macro_f1"] for g in order]
                ax.bar(x + (k - 0.5) * 0.38, values, width=0.36, color=MODEL_COLORS[model], label=model)
            for i, g in enumerate(order):
                gain = (transformer[g]["macro_f1"] - tfidf[g]["macro_f1"]) * 100
                top = max(transformer[g]["macro_f1"], tfidf[g]["macro_f1"])
                ax.text(i, top + 0.008, f"{gain:+.1f}", ha="center", fontsize=8, color=plotting.INK_SECONDARY)
            labels = [f"{g}\n{tfidf[g]['share'] * 100:.0f}%" for g in order]
            if name == "seen":
                names = {"seen_in_train": "seen in train", "new": "new text"}
                labels = [f"{names[g]}\n{tfidf[g]['share'] * 100:.0f}%" for g in order]
            ax.set_xticks(x, labels, fontsize=8.5)
            ax.set_ylim(0.5, 0.78)
            ax.grid(axis="x", visible=False)
            if col == 0:
                ax.set_ylabel("Macro F1 with offsets")
                ax.set_title(f"{HEAD_TITLES[head]} · by length in words (share of reviews)")
            else:
                ax.set_title(f"{HEAD_TITLES[head]} · seen or new")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(
        "Where FaBERT gains over TF-IDF (numbers: gain in points)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plotting.save(fig, "results_where_it_helps")


def main():
    plotting.apply_style()
    comparison = load("transformer_final/vs_tfidf.json")
    final = load("transformer_final/fabert.json")
    confusion_matrices(comparison)
    per_class(comparison)
    tfidf_experiments()
    learning_curve(final)
    model_comparison()
    training_curve(final)
    where_it_helps()
    print("Charts written to", plotting.FIGURES_DIR)


if __name__ == "__main__":
    main()
