import numpy as np
import pandas as pd
import pytest

from digikala_reviews.pseudo import guesses, pseudo_rows, quality


def test_guesses_scale_by_the_temperature():
    log_proba = np.log(np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1]]))
    label, p = guesses(log_proba, 1.0)
    assert label.tolist() == [0, 1]
    assert p == pytest.approx([0.9, 0.7])
    _, softer = guesses(log_proba, 2.0)
    assert (softer < p).all()  # a temperature above 1 makes the model less sure


def test_quality_counts_kept_and_right_guesses():
    guessed = np.array(["recommended", "recommended", "no_idea", "not_recommended"])
    p_top = np.array([0.95, 0.92, 0.5, 0.91])
    true = np.array(["recommended", "no_idea", "no_idea", "not_recommended"])
    found = quality(guessed, p_top, true, [0.9])["0.9"]
    assert found["kept"] == 3
    assert found["accuracy"] == pytest.approx(2 / 3)
    assert found["kept_classes"]["recommended"] == pytest.approx(2 / 3)
    assert found["kept_right_per_true_class"]["no_idea"] == pytest.approx(0.0)
    assert found["kept_right_per_true_class"]["recommended"] == pytest.approx(1.0)


def test_pseudo_rows_keep_only_confident_guesses_of_the_sample():
    masked = pd.DataFrame({"id": [str(i) for i in range(10)], "text": [f"t{i}" for i in range(10)]})
    guessed = pd.DataFrame({"id": masked["id"], "label": ["recommended"] * 10, "p": [0.95, 0.5] * 5})
    rows = pseudo_rows(masked, guessed, n_rows=6, seed=1, threshold=0.9)
    assert len(rows) <= 6
    assert set(rows.columns) == {"id", "text", "recommendation_status", "rating_band"}
    assert rows["rating_band"].isna().all()
    kept_p = guessed.set_index("id").loc[rows["id"], "p"]
    assert (kept_p >= 0.9).all()
    again = pseudo_rows(masked, guessed, n_rows=6, seed=1, threshold=0.9)
    assert rows["id"].tolist() == again["id"].tolist()  # the same rows every time
