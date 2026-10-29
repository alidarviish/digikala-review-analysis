"""Experiment cards: where an experiment's predictions differ from the reference.

A card compares two setups on the same validation reviews. With several seeds, the
predictions of all seeds are pooled (the same reviews once per seed), so a matrix holds the
reviews of every seed.

Intervals come from a paired bootstrap over reviews: each round draws a weight for every
review (Poisson, mean 1) and gives it to that review in every seed and in both setups. Reviews
that agree on everything that matters (true class, every prediction, every group) are counted
together: the sum of n Poisson(1) weights is one Poisson(n) weight, so drawing one weight per
such combination gives exactly the same bootstrap, with a few thousand draws per round instead
of a million.
"""

import numpy as np
import pandas as pd

ROUNDS = 1000


def confusion(true_codes, predicted_codes, n_classes: int, weights=None) -> np.ndarray:
    """Counts with rows the true class and columns the predicted class."""
    flat = np.asarray(true_codes) * n_classes + np.asarray(predicted_codes)
    return np.bincount(flat, weights=weights, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def f1_per_class(matrix: np.ndarray) -> np.ndarray:
    """F1 of every class; matrix is (..., classes, classes) with rows the true class."""
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    predicted, actual = matrix.sum(axis=-2), matrix.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(2 * tp / (predicted + actual))


def row_shares(matrix: np.ndarray) -> np.ndarray:
    """Each row as percentages of its true class."""
    return matrix / matrix.sum(axis=1, keepdims=True) * 100


def pooled_comparison(true_codes, reference, variant, groups, n_classes: int, seed: int = 2026) -> dict:
    """Matrices, F1 per class and Macro F1 per group for the two setups, with paired intervals.

    true_codes: (reviews,) true class; reference, variant: (seeds, reviews) predicted class;
    groups: dict of name -> (reviews,) group label (for example the length bucket).
    """
    true_codes = np.asarray(true_codes)
    reference, variant = np.atleast_2d(reference), np.atleast_2d(variant)
    seeds = reference.shape[0]

    table = pd.DataFrame({"true": true_codes})
    for s in range(seeds):
        table[f"ref{s}"], table[f"var{s}"] = reference[s], variant[s]
    for name, labels in groups.items():
        table[f"group_{name}"] = np.asarray(labels)
    combos = table.value_counts(sort=False).reset_index(name="n")
    n = combos["n"].to_numpy()
    true_c = combos["true"].to_numpy()

    def onehot(predicted):  # (combos, classes * classes): the matrix cell of every combination
        cells = np.zeros((len(combos), n_classes * n_classes))
        cells[np.arange(len(combos)), true_c * n_classes + predicted] = 1
        return cells

    ref_cells = sum(onehot(combos[f"ref{s}"].to_numpy()) for s in range(seeds))
    var_cells = sum(onehot(combos[f"var{s}"].to_numpy()) for s in range(seeds))
    rng = np.random.default_rng(seed)
    draws = rng.poisson(n, size=(ROUNDS, len(combos))).astype(float)

    def matrices(weights, inside=slice(None)):  # weights: (rounds, combos) or (combos,)
        shape = (*np.shape(weights)[:-1], n_classes, n_classes)
        ref = (weights[..., inside] @ ref_cells[inside]).reshape(shape)
        var = (weights[..., inside] @ var_cells[inside]).reshape(shape)
        return ref, var

    ref_matrix, var_matrix = matrices(n.astype(float))
    ref_b, var_b = matrices(draws)
    f1_changes = f1_per_class(var_b) - f1_per_class(ref_b)

    found_groups = {}
    for name in groups:
        labels = combos[f"group_{name}"].to_numpy()
        found_groups[name] = {}
        for group in pd.unique(labels):
            inside = labels == group
            ref, var = matrices(n.astype(float), inside)
            ref_d, var_d = matrices(draws, inside)
            changes = f1_per_class(var_d).mean(axis=-1) - f1_per_class(ref_d).mean(axis=-1)
            found_groups[name][str(group)] = {
                "share": float(n[inside].sum() / n.sum()),
                "change": float(f1_per_class(var).mean() - f1_per_class(ref).mean()),
                "interval": np.percentile(changes, [2.5, 97.5]).tolist(),
            }
    return {
        "reference_matrix": ref_matrix,
        "variant_matrix": var_matrix,
        "f1_change": f1_per_class(var_matrix) - f1_per_class(ref_matrix),
        "f1_change_interval": np.percentile(f1_changes, [2.5, 97.5], axis=0).T,
        "groups": found_groups,
    }


def loss_comparison(reference_loss, variant_loss, seed: int = 2026, chunk: int = 100) -> dict:
    """Mean validation loss of the two setups and their difference, with a paired interval.

    reference_loss, variant_loss: (seeds, reviews) loss of every review, -log p(true class).
    Each bootstrap round gives every review one Poisson(1) weight, shared by all seeds and
    both setups.
    """
    reference_loss, variant_loss = np.atleast_2d(reference_loss), np.atleast_2d(variant_loss)
    ref, var = reference_loss.mean(axis=0), variant_loss.mean(axis=0)  # per review, mean of seeds
    rng = np.random.default_rng(seed)
    changes = []
    for start in range(0, ROUNDS, chunk):
        weights = rng.poisson(1.0, size=(min(chunk, ROUNDS - start), len(ref))).astype(np.float32)
        changes.append((weights @ (var - ref)) / weights.sum(axis=1))
    return {
        "reference": float(ref.mean()),
        "variant": float(var.mean()),
        "change": float(var.mean() - ref.mean()),
        "interval": np.percentile(np.concatenate(changes), [2.5, 97.5]).tolist(),
    }
