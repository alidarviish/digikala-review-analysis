import pandas as pd

from digikala_reviews.standard_files import read_split_file


def test_texts_that_look_missing_stay_text(tmp_path):
    written = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "product_id": [10, 10, 20, 20],
            "body": ["NA", "null", 'خوب\r\nبود, "عالی"', "None 😍"],
            "rate": [4.5, None, 0.0, 1.0],
            "recommendation_status": ["recommended", None, "no_idea", "not_recommended"],
            "rating_band": ["high", None, None, "low"],
            "text_seen_in_train": [True, False, True, False],
        }
    )
    written.to_csv(tmp_path / "part.csv", index=False, encoding="utf-8")
    back = read_split_file("part.csv", folder=tmp_path)

    assert back["body"].tolist() == written["body"].tolist()
    assert back["recommendation_status"].isna().tolist() == [False, True, False, False]
    assert back["rating_band"].isna().tolist() == [False, True, True, False]
    assert back["rate"].isna().tolist() == [False, True, False, False]
    assert back["text_seen_in_train"].tolist() == [True, False, True, False]
    assert back["id"].dtype == "int64"
