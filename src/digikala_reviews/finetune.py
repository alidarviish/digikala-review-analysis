"""Pieces of Transformer fine-tuning that do not need PyTorch, so they can be tested anywhere."""

import numpy as np

IGNORE = -100  # label of a review that has no label for a head; the loss skips it


def fixed_sample(n_rows: int, size: int, seed: int) -> np.ndarray:
    """Positions of a random sample, in their original order, the same on every machine."""
    if size >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size, replace=False))


def label_codes(values, labels: list[str]) -> np.ndarray:
    """Class number of each label, or IGNORE where the label is missing."""
    code = {label: i for i, label in enumerate(labels)}
    result = np.full(len(values), IGNORE, dtype=np.int64)
    for i, value in enumerate(values):
        if value is not None and value == value:  # value == value is False for NaN
            result[i] = code[value]
    return result


def linear_warmup(step: int, total: int, warmup: int) -> float:
    """Learning-rate factor: rises linearly over the warmup steps, then falls linearly to 0."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))
