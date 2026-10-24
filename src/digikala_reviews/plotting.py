"""Shared chart style, so every figure in the project looks the same.

Colors follow a fixed role:
- recommendation classes are categorical (one hue per class, same hue everywhere)
- rating bands are ordered, so they use one blue ramp from light (low) to dark (high)
- anything de-emphasized or missing is neutral gray
"""

import matplotlib as mpl
from matplotlib import font_manager

from digikala_reviews.config import FIGURES_DIR, PROJECT_ROOT

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"

# Categorical slots, always used in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

RECOMMENDATION_COLORS = {
    "recommended": SERIES[0],
    "not_recommended": SERIES[1],
    "no_idea": SERIES[2],
    "missing": NEUTRAL,
}
RATING_BAND_COLORS = {"low": "#86b6ef", "medium": "#3987e5", "high": "#1c5cab"}

PERSIAN_FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "Vazirmatn-Regular.ttf"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def compact(n: float) -> str:
    """1234 -> '1.2K', 6156289 -> '6.16M'."""
    n = float(n)
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:,.0f}"


def hbar(ax, labels, values, colors=None, value_labels=None, xlabel=None):
    """Horizontal bars, first item on top, value written at the tip of each bar."""
    positions = range(len(labels))
    colors = colors or SERIES[0]
    bars = ax.barh(positions, values, height=0.55, color=colors)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if xlabel:
        ax.set_xlabel(xlabel)

    value_labels = value_labels or [compact(v) for v in values]
    pad = max(values) * 0.01 if len(values) else 0
    for bar, text in zip(bars, value_labels, strict=True):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    ax.set_xlim(0, max(values) * 1.12 if len(values) else 1)
    return bars


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")


def stacked_hbar(ax, shares, colors, xlabel="Share of reviews (%)"):
    """100% stacked horizontal bars. `shares` is a DataFrame of percentages, one row per bar."""
    left = [0.0] * len(shares)
    positions = range(len(shares))
    for column in shares.columns:
        ax.barh(
            positions,
            shares[column],
            left=left,
            height=0.6,
            color=colors[column],
            edgecolor=SURFACE,
            linewidth=2,  # thin gap between segments instead of an outline
            label=column,
        )
        left = [a + b for a, b in zip(left, shares[column], strict=True)]
    ax.set_yticks(list(positions), shares.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncols=len(shares.columns), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)


def use_persian_ticks(ax, axis: str = "y") -> None:
    """Draw the tick labels of one axis with a Persian font.

    matplotlib 3.11 joins Persian letters and lays them out right to left by itself,
    so the text needs no preprocessing; it only needs a font with good Persian glyphs.
    """
    font = font_manager.FontProperties(fname=PERSIAN_FONT_PATH, size=10)
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for label in labels:
        label.set_fontproperties(font)
