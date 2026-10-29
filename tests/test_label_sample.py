import pandas as pd
import pytest

from digikala_reviews.label_sample import PER_CLASS, REPEATS, agreement, choose_sample, read_labels


def fake_validation():
    rows = []
    for label, count in PER_CLASS.items():
        for i in range(count * 3):
            rows.append({"id": len(rows) + 1, "body": f"{label} text {i}", "recommendation_status": label})
    rows.append({"id": len(rows) + 1, "body": "no label", "recommendation_status": None})
    # the same text twice: it may be chosen once at most
    rows.append({"id": len(rows) + 1, "body": "recommended  text 0", "recommendation_status": "recommended"})
    return pd.DataFrame(rows)


def test_sample_has_the_planned_rows_and_repeats():
    table = choose_sample(fake_validation(), seed=1)
    first = table[table["repeat_of"] == 0]
    assert len(table) == sum(PER_CLASS.values()) + REPEATS
    assert first["original_label"].value_counts().to_dict() == PER_CLASS
    assert first["id"].is_unique and first["body"].map(lambda t: " ".join(t.split())).is_unique
    assert list(table["row"]) == list(range(1, len(table) + 1))
    repeats = table[table["repeat_of"] > 0]
    assert len(repeats) == REPEATS
    for _, repeat in repeats.iterrows():
        original = table[table["row"] == repeat["repeat_of"]].iloc[0]
        assert original["id"] == repeat["id"] and original["repeat_of"] == 0


def test_sample_is_the_same_for_the_same_seed():
    validation = fake_validation()
    pd.testing.assert_frame_equal(choose_sample(validation, 7), choose_sample(validation, 7))


def test_labels_are_codes_1_2_3_only():
    assert list(read_labels(pd.Series([" 1", "2", "3"]))) == ["recommended", "not_recommended", "no_idea"]
    with pytest.raises(AssertionError):
        read_labels(pd.Series(["1", ""]))


def test_agreement_per_original_label():
    table = pd.DataFrame(
        {
            "original_label": ["recommended", "recommended", "no_idea", "not_recommended"],
            "my_label": ["recommended", "no_idea", "no_idea", "recommended"],
        }
    )
    found = agreement(table, "original_label")
    assert found["agree"] == 0.5
    assert found["recommended"]["agree"] == 0.5 and found["no_idea"]["agree"] == 1.0
    assert found["not_recommended"]["agree"] == 0.0
