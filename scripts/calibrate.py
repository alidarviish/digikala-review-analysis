"""Temperature scaling for the final FaBERT model: does a probability of 0.9 mean right 9 times
out of 10, and does one number per head fix it?

For each head the temperature is fitted on one half of validation (by product) and applied to
the other half, then the other way round, so no review is scored with a temperature fitted on
it. Before and after, on all labeled validation reviews: the mean loss, the Brier score, the
expected calibration error (ECE, 15 bins of the top probability) with its bins, the share of
reviews with a top probability of at least 0.9 (the pseudo-label threshold of the
semi-supervised step) and how often they are right, and Macro F1 with and without offsets. The
temperature fitted on all of validation is kept for later use, as the offsets are.

Writes reports/calibration.json and reports/figures/results_calibration.png. Nothing is
trained and the test set is not read. The plan and the expected results are in
reports/calibration_plan.md.

Usage:
    uv run python scripts/calibrate.py
"""

from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from digikala_reviews import plotting
from digikala_reviews.calibration import above, brier, fit_temperature, mean_loss, reliability, scale
from digikala_reviews.config import ARTIFACTS_DIR, FINAL_TRANSFORMER_DIR, PSEUDO_LABEL_THRESHOLD, REPORTS_DIR
from digikala_reviews.experiments import cross_fitted_offsets, product_half
from digikala_reviews.metrics import macro_f1_from_codes
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

HEAD_TITLES = {"recommendation": "Recommendation", "rating_band": "Rating band"}


def scores(log_proba, codes, halves):
    n_classes = log_proba.shape[1]
    with_offsets, _ = cross_fitted_offsets(log_proba, codes, halves)
    return {
        "mean_loss": mean_loss(log_proba, codes),
        "brier": brier(log_proba, codes),
        "reliability": reliability(log_proba, codes),
        "at_least_threshold": above(log_proba, codes, PSEUDO_LABEL_THRESHOLD),
        "macro_f1_without_offsets": macro_f1_from_codes(codes, log_proba.argmax(axis=1), n_classes),
        "macro_f1_with_offsets": macro_f1_from_codes(codes, with_offsets, n_classes),
    }


def draw(result):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, (head, r) in zip(axes, result.items(), strict=True):
        ax.plot([0, 1], [0, 1], color=plotting.NEUTRAL, linewidth=1, linestyle="--", label="perfect")
        for key, name, color in [
            ("before", "as trained", plotting.MUTED),
            ("after", "temperature scaled", plotting.SERIES[0]),
        ]:
            bins = [b for b in r[key]["reliability"]["bins"] if b["share"] >= 0.001]
            ax.plot(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                marker="o",
                color=color,
                label=f"{name} (ECE {r[key]['reliability']['ece']:.3f})",
            )
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("Top probability (mean in each bin)")
        ax.set_ylabel("Share right in the bin")
        ax.set_title(
            f"{HEAD_TITLES[head]} · temperature {r['temperature_all_validation']:.2f}", fontsize=10.5
        )
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle(
        "Calibration of the final FaBERT model on validation",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Bins with less than 0.1% of the reviews are not drawn. The temperatures are fitted on one "
        "half of validation by product and scored on the other.",
        fontsize=8.5,
        color=plotting.INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plotting.save(fig, "results_calibration")


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    predictions = pd.read_parquet(ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "validation_predictions.parquet")
    assert (predictions["id"].to_numpy() == validation["id"].to_numpy()).all(), (
        "rows are not in the same order"
    )

    result = {}
    for head, h in HEADS.items():
        labeled = validation[h["column"]].notna().to_numpy()
        codes = np.array([h["labels"].index(v) for v in validation[h["column"]].to_numpy()[labeled]])
        halves = product_half(validation["product_id"].to_numpy()[labeled])
        log_proba = predictions[[f"{head}__log_proba__{label}" for label in h["labels"]]].to_numpy()[labeled]
        log_proba = log_proba.astype(float)

        calibrated = np.empty_like(log_proba)
        per_half = {}
        for name, fit_on in [("fitted_on_a", halves), ("fitted_on_b", ~halves)]:
            t = fit_temperature(log_proba[fit_on], codes[fit_on])
            calibrated[~fit_on] = scale(log_proba[~fit_on], t)
            per_half[name] = t
        result[head] = {
            "reviews": int(labeled.sum()),
            "temperature_per_half": per_half,
            "temperature_all_validation": fit_temperature(log_proba, codes),
            "before": scores(log_proba, codes, halves),
            "after": scores(calibrated, codes, halves),
        }

    save_json(
        REPORTS_DIR / "calibration.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": FINAL_TRANSFORMER_DIR,
            "threshold": PSEUDO_LABEL_THRESHOLD,
            "heads": result,
            "code": git_commit(),
        },
    )
    plotting.apply_style()
    draw(result)
    for head, r in result.items():
        print(
            f"\n{head}: temperature {r['temperature_all_validation']:.3f} on all of validation "
            f"(halves {r['temperature_per_half']['fitted_on_a']:.3f}, "
            f"{r['temperature_per_half']['fitted_on_b']:.3f})"
        )
        for key in ("before", "after"):
            s = r[key]
            sure = s["at_least_threshold"]
            print(
                f"  {key:6}: loss {s['mean_loss']:.4f}, Brier {s['brier']:.4f}, "
                f"ECE {s['reliability']['ece']:.4f}, p >= {PSEUDO_LABEL_THRESHOLD}: {sure['share']:.1%} of "
                f"reviews, {sure['accuracy']:.1%} right (mean p {sure['mean_probability']:.1%}); "
                f"Macro F1 {s['macro_f1_without_offsets']:.4f} without offsets, "
                f"{s['macro_f1_with_offsets']:.4f} with"
            )


if __name__ == "__main__":
    main()
