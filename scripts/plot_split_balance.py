"""Draw the class shares of both targets in train, validation, test and all data.

Reads reports/split_report.json (made by build_splits.py) and writes
reports/figures/split_balance.png. Takes a few seconds, so the chart can be changed
without running the split again.

Usage:
    uv run python scripts/plot_split_balance.py
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from digikala_reviews import plotting
from digikala_reviews.config import REPORTS_DIR

PARTS = ["train", "validation", "test", "all"]
PANELS = [
    ("recommendation", ["recommended", "no_idea", "not_recommended", "missing"], "Recommendation"),
    ("rating_band", ["low", "medium", "high", "missing"], "Rating band"),
]
COLORS = {**plotting.RECOMMENDATION_COLORS, **plotting.RATING_BAND_COLORS}


def main():
    report = json.loads((REPORTS_DIR / "split_report.json").read_text(encoding="utf-8"))
    sizes = report["sizes"]
    labels = [f"{part} ({sizes[part]['reviews_pct']:.0f}%)" for part in PARTS[:-1]] + ["all data"]

    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, (target, classes, title) in zip(axes, PANELS, strict=True):
        shares = pd.DataFrame(report["shares_pct"][target]).loc[PARTS, classes]
        shares.index = labels
        plotting.stacked_hbar(ax, shares, COLORS)
        gap = report["max_gap_pp"][target]
        gap_text = f"{gap:.2f} pp" if gap >= 0.01 else "under 0.01 pp"
        ax.set_title(f"{title}  (largest gap to all data: {gap_text})", pad=30)
    fig.suptitle(
        "Class shares after the product-grouped split",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    plotting.save(fig, "split_balance")
    print("Saved reports/figures/split_balance.png")


if __name__ == "__main__":
    main()
