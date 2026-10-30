"""Temperature scaling: one number per head that makes the model's probabilities match how often
it is right, without changing which class it ranks first (reports/calibration_plan.md)."""

import numpy as np

TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 401))  # 0.25 to 4, evenly in log
BINS = 15


def scale(log_proba: np.ndarray, temperature: float) -> np.ndarray:
    """Log probabilities after dividing the log probabilities by the temperature and normalizing
    again: above 1 the probabilities move towards each other, below 1 apart."""
    scaled = np.asarray(log_proba, dtype=float) / temperature
    return scaled - np.logaddexp.reduce(scaled, axis=1, keepdims=True)


def mean_loss(log_proba: np.ndarray, codes: np.ndarray) -> float:
    """Mean of -log p of the true class."""
    return float(-np.mean(log_proba[np.arange(len(codes)), codes]))


def fit_temperature(log_proba: np.ndarray, codes: np.ndarray) -> float:
    """The temperature with the lowest mean loss, searched on a grid and then refined around the
    best grid point (the loss has a single minimum in the temperature)."""
    losses = [mean_loss(scale(log_proba, t), codes) for t in TEMPERATURE_GRID]
    best = int(np.argmin(losses))
    low, high = TEMPERATURE_GRID[max(best - 1, 0)], TEMPERATURE_GRID[min(best + 1, len(TEMPERATURE_GRID) - 1)]
    fine = np.linspace(low, high, 41)
    losses = [mean_loss(scale(log_proba, t), codes) for t in fine]
    return float(fine[int(np.argmin(losses))])


def brier(log_proba: np.ndarray, codes: np.ndarray) -> float:
    """Mean squared distance between the probabilities and the true class (one-hot)."""
    p = np.exp(log_proba)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(codes)), codes] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def reliability(log_proba: np.ndarray, codes: np.ndarray, bins: int = BINS) -> dict:
    """The top class's probability against how often the top class is right, in equal-width
    bins of that probability. The expected calibration error (ECE) is the mean gap, weighted by
    the share of reviews in each bin."""
    p = np.exp(log_proba)
    confidence = p.max(axis=1)
    right = p.argmax(axis=1) == codes
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        inside = which == b
        if not inside.any():
            continue
        mean_confidence, accuracy = float(confidence[inside].mean()), float(right[inside].mean())
        share = float(inside.mean())
        ece += share * abs(mean_confidence - accuracy)
        rows.append(
            {
                "low": float(edges[b]),
                "high": float(edges[b + 1]),
                "share": share,
                "confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"ece": float(ece), "bins": rows}


def above(log_proba: np.ndarray, codes: np.ndarray, threshold: float) -> dict:
    """The share of reviews whose top class has at least this probability, how often that top
    class is right among them, and the mean of that top probability among them."""
    p = np.exp(log_proba)
    top = p.max(axis=1)
    chosen = top >= threshold
    right = p.argmax(axis=1) == codes
    return {
        "share": float(chosen.mean()),
        "accuracy": float(right[chosen].mean()) if chosen.any() else None,
        "mean_probability": float(top[chosen].mean()) if chosen.any() else None,
    }
