"""Training reviews merged by text: one row per different text, with the share of each label.

A text that comes several times (for example "عالی بود") becomes one row. Its target for each
head is the share of each label among its copies, and its number of copies is kept, so the
training loss can either count the text as often as before (weight = copies, the same loss as
on the separate rows) or once (weight = 1).
"""

import numpy as np
import pandas as pd


def label_counts(text_codes: np.ndarray, label_codes: np.ndarray, n_texts: int, n_classes: int) -> np.ndarray:
    """How many copies of each text have each label. Rows with label code -1 (no label) are skipped."""
    text_codes, label_codes = np.asarray(text_codes), np.asarray(label_codes)
    has_label = label_codes >= 0
    flat = text_codes[has_label] * n_classes + label_codes[has_label]
    return np.bincount(flat, minlength=n_texts * n_classes).reshape(n_texts, n_classes)


def merge(texts, heads: dict) -> tuple[pd.Index, np.ndarray, dict]:
    """Different texts, copies of each, and per head the label counts of each text.

    heads maps a head name to (labels of every row, list of the head's classes); a missing
    label is None or NaN.
    """
    text_codes, unique_texts = pd.factorize(pd.Series(np.asarray(texts)), sort=False)
    copies = np.bincount(text_codes, minlength=len(unique_texts))
    counts = {}
    for head, (values, classes) in heads.items():
        code = {label: i for i, label in enumerate(classes)}
        label_codes = pd.Series(np.asarray(values, dtype=object)).map(code).fillna(-1).astype(int).to_numpy()
        counts[head] = label_counts(text_codes, label_codes, len(unique_texts), len(classes))
    return unique_texts, copies, counts


def class_shares(counts: np.ndarray, classes: list[str]) -> dict:
    """Share of each class when every text counts as often as its copies, and when it counts once."""
    labeled = counts.sum(axis=1) > 0
    as_rows = counts.sum(axis=0) / counts.sum()
    per_text = counts[labeled] / counts[labeled].sum(axis=1, keepdims=True)
    once = per_text.mean(axis=0)
    return {
        label: {"copies_counted": float(as_rows[i]), "each_text_once": float(once[i])}
        for i, label in enumerate(classes)
    }


def mixed(counts: np.ndarray) -> dict:
    """Texts whose copies carry more than one label, as a share of texts and of labeled rows."""
    labeled = counts.sum(axis=1) > 0
    is_mixed = (counts > 0).sum(axis=1) > 1
    return {
        "texts_with_a_label": int(labeled.sum()),
        "share_of_texts": float(is_mixed[labeled].mean()),
        "share_of_rows": float(counts[is_mixed].sum() / counts.sum()),
    }


def soft_targets(counts: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Training target and weight of each merged text for one head.

    The target is the share of each label among the copies. The weight is the number of
    labeled copies (mode "copies": the same loss as on the separate rows) or 1 for every
    labeled text (mode "once"). A text without a label for this head gets weight 0.
    """
    assert mode in {"copies", "once"}, mode
    total = counts.sum(axis=1)
    shares = counts / np.maximum(total, 1)[:, None]
    weights = total if mode == "copies" else (total > 0)
    return shares.astype(np.float32), weights.astype(np.float32)


def weighted_soft_loss(log_proba, shares, weights):
    """Weighted mean over rows of the cross-entropy against the label shares.

    Works on NumPy arrays and on PyTorch tensors alike. With one-hot shares and weights of 1
    it is the usual mean cross-entropy; the weights must not all be 0.
    """
    per_row = -(shares * log_proba).sum(-1)
    return (weights * per_row).sum() / weights.sum()
