"""Pseudo-labels for the training reviews whose recommendation label was hidden
(reports/semi_supervised_plan.md)."""

import numpy as np
import pandas as pd

from digikala_reviews.finetune import fixed_sample

RECOMMENDATION_LABELS = ["recommended", "not_recommended", "no_idea"]


def guesses(log_proba, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """The guessed class (its position in RECOMMENDATION_LABELS) and its probability, after
    dividing the log probabilities by the temperature and normalizing again."""
    scaled = np.asarray(log_proba, dtype=float) / temperature
    scaled = scaled - np.logaddexp.reduce(scaled, axis=1, keepdims=True)
    return scaled.argmax(axis=1), np.exp(scaled.max(axis=1))


def quality(guessed, p_top, true, thresholds, labels=RECOMMENDATION_LABELS) -> dict:
    """For each threshold: the share of reviews kept, how often their guess is right, the
    share of each class among the kept guesses, and for each true class the share of its
    reviews that were kept with the right label. labels: the classes of the head guessed."""
    guessed, p_top, true = np.asarray(guessed), np.asarray(p_top), np.asarray(true)
    result = {}
    for threshold in thresholds:
        kept = p_top >= threshold
        right = kept & (guessed == true)
        result[f"{threshold:g}"] = {
            "kept": int(kept.sum()),
            "share_kept": float(kept.mean()),
            "accuracy": float(right.sum() / kept.sum()) if kept.any() else None,
            "kept_classes": {
                label: float((guessed[kept] == label).mean()) if kept.any() else 0.0 for label in labels
            },
            "kept_right_per_true_class": {
                label: float(right[true == label].mean()) if (true == label).any() else None
                for label in labels
            },
        }
    return result


def pseudo_rows(masked: pd.DataFrame, guessed: pd.DataFrame, n_rows: int, seed: int, threshold: float):
    """Training rows from a fixed random n_rows of the hidden reviews, keeping those whose
    guess reaches the threshold. `masked` has id and text; `guessed` has id, label and p.
    The rating band stays empty: it was hidden with the rest of the row."""
    sample = masked.iloc[fixed_sample(len(masked), n_rows, seed)]
    found = sample.merge(guessed, on="id", how="left", validate="one_to_one")
    assert found["p"].notna().all(), "every sampled review must have a guess"
    found = found[found["p"] >= threshold]
    return pd.DataFrame(
        {
            "id": found["id"].to_numpy(),
            "text": found["text"].to_numpy(),
            "recommendation_status": found["label"].to_numpy(),
            "rating_band": None,
        }
    )
