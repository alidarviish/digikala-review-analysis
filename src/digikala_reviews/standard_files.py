"""Read the standard data files in data/splits with the right types.

Plain pd.read_csv turns review texts such as "NA", "null" or "None" into missing values.
Here only an empty label or rate cell counts as missing; a body is always text.
"""

import pandas as pd

from digikala_reviews.config import SPLITS_DIR

OPTIONAL_COLUMNS = ["rate", "recommendation_status", "rating_band"]  # empty cell = missing


def read_split_file(name: str, folder=SPLITS_DIR) -> pd.DataFrame:
    """Read one standard file, for example read_split_file("train_supervised.csv")."""
    table = pd.read_csv(folder / name, dtype=str, keep_default_na=False, encoding="utf-8")
    for column in OPTIONAL_COLUMNS:
        if column in table:
            table[column] = table[column].replace("", None)
    if "rate" in table:
        table["rate"] = pd.to_numeric(table["rate"])
    if "text_seen_in_train" in table:
        table["text_seen_in_train"] = table["text_seen_in_train"].map({"True": True, "False": False})
    for column in ["id", "product_id"]:
        if column in table:
            table[column] = table[column].astype("int64")
    return table
