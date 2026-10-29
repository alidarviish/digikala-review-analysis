"""Make the file for labeling a validation sample by hand, and its key.

Writes two files in data/label_sample/ (not in git):
  to_label.csv  row, text and an empty my_label column; this is the only file to open
  key.csv       row, id, original label and which earlier row a repeat repeats; do not open it
                before labeling is finished

Type 1 (recommended), 2 (not recommended) or 3 (no idea) in my_label for every row, judging
from the text only, then save the file as it is (CSV). The plan and the expected results are
in reports/label_sample_plan.md. The test set is not read.

Usage:
    uv run python scripts/make_label_sample.py
"""

from digikala_reviews.config import DATA_DIR, SPLIT_SEED
from digikala_reviews.label_sample import choose_sample
from digikala_reviews.standard_files import read_split_file

OUT_DIR = DATA_DIR / "label_sample"


def main():
    to_label, key = OUT_DIR / "to_label.csv", OUT_DIR / "key.csv"
    assert not to_label.exists(), f"{to_label} already exists; delete it only if no labels are in it"
    validation = read_split_file("validation.csv")
    table = choose_sample(validation, SPLIT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so that Excel shows the Persian text correctly
    table[["row", "body"]].rename(columns={"body": "text"}).assign(my_label="").to_csv(
        to_label, index=False, encoding="utf-8-sig"
    )
    table[["row", "id", "original_label", "repeat_of"]].to_csv(key, index=False, encoding="utf-8")
    print(f"{len(table)} rows to label in {to_label}")
    print(f"Key (do not open before labeling is done): {key}")


if __name__ == "__main__":
    main()
