"""Mask the training labels and write the standard data files of the project.

Inputs: the cleaned reviews and data/splits/split_assignment.parquet (from build_splits.py).
Outputs in data/splits/:
    train_supervised.csv        train rows with at least one label, masked rows removed
    train_pseudo_unlabeled.csv  masked train rows, id and body only
    natural_unlabeled.csv       train rows without a recommendation label, id and body only
    validation.csv, test.csv    every row of the part, plus text_seen_in_train
    split_manifest.json         seeds, rules, counts and file hashes
    audit/masked_labels_reference.csv
                                true labels of the masked rows, only for checking pseudo-labels later;
                                never used for training

The body column stays raw text in every file; the models normalize it with the same function.

Usage:
    uv run python scripts/build_standard_files.py
"""

import json
import math
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import (
    COMMENTS_CLEAN_PARQUET,
    MASK_RATE,
    MASK_SEED,
    RATE_ZERO_POLICY,
    RATING_BAND_CLASSES,
    RATING_BAND_EDGES,
    RECOMMENDATION_CLASSES,
    REPORTS_DIR,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
    SPLITS_DIR,
)
from digikala_reviews.ingest import file_sha256
from digikala_reviews.mask import select_masked
from digikala_reviews.normalize import NORMALIZER_VERSION, normalize

ASSIGNMENT_PATH = SPLITS_DIR / "split_assignment.parquet"
SPLIT_REPORT_PATH = REPORTS_DIR / "split_report.json"
MANIFEST_PATH = SPLITS_DIR / "split_manifest.json"
AUDIT_DIR = SPLITS_DIR / "audit"
LABELED_COLUMNS = ["id", "product_id", "body", "rate", "recommendation_status", "rating_band"]
CHUNK_ROWS = 250_000


def normalized_text_key(body: pd.Series) -> pd.Series:
    """A number that is the same for two reviews when their normalized texts are the same."""
    keys = []
    for start in range(0, len(body), CHUNK_ROWS):
        chunk = body.iloc[start : start + CHUNK_ROWS].map(normalize)
        keys.append(pd.util.hash_pandas_object(chunk, index=False))
    return pd.concat(keys).set_axis(body.index)


def class_counts(values: pd.Series) -> dict:
    counts = values.astype("string").fillna("missing").value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def band_definition() -> str:
    """Written from the config, so the text cannot drift from the code."""
    low_edge, high_edge = RATING_BAND_EDGES
    lowest = "0 < rate" if RATE_ZERO_POLICY == "invalid" else "0 <= rate"
    return (
        f"low: {lowest} < {low_edge}, medium: {low_edge} <= rate < {high_edge}, "
        f"high: {high_edge} <= rate <= 5"
    )


def main():
    start = time.perf_counter()
    columns = ["id", "product_id", "body", "rate", "recommendation", "rating_band"]
    data = pd.read_parquet(COMMENTS_CLEAN_PARQUET, columns=columns)
    data = data.rename(columns={"recommendation": "recommendation_status"})
    assignment = pd.read_parquet(ASSIGNMENT_PATH, columns=["id", "split"])
    data = data.merge(assignment, on="id", how="inner", validate="one_to_one")
    assert len(data) == len(assignment), "some ids of the split are missing from the cleaned data"
    del assignment
    assert data["body"].notna().all(), "every review must have a body"
    data["rate"] = data["rate"].astype("float64")  # prints the same way on every machine
    data = data.sort_values("id", ignore_index=True)  # fixed row order, so the files are byte-identical
    print(f"Read {len(data):,} reviews ({time.perf_counter() - start:.0f}s)")

    data["masked"] = select_masked(data["id"], data["recommendation_status"], data["split"])
    data["text_key"] = normalized_text_key(data["body"])
    print(f"Masked and normalized ({time.perf_counter() - start:.0f}s)")

    is_train = data["split"] == "train"
    has_label = data["recommendation_status"].notna() | data["rating_band"].notna()
    train_texts = np.unique(data.loc[is_train, "text_key"].to_numpy())
    data["text_seen_in_train"] = data["text_key"].isin(train_texts)
    data = data.drop(columns="text_key")

    # Check the masking against the rule before writing anything
    labeled_train = data.loc[
        is_train & data["recommendation_status"].notna(), ["recommendation_status", "masked"]
    ]
    found = set(labeled_train["recommendation_status"].unique())
    assert found == set(RECOMMENDATION_CLASSES), f"unexpected recommendation values: {found}"
    masked_per_class = {}
    for label in RECOMMENDATION_CLASSES:
        in_class = labeled_train["recommendation_status"] == label
        expected = math.floor(MASK_RATE * in_class.sum())
        actual = int(labeled_train.loc[in_class, "masked"].sum())
        assert actual == expected, f"{label}: masked {actual}, expected {expected}"
        masked_per_class[label] = {
            "train_labeled": int(in_class.sum()),
            "rate_times_count": round(MASK_RATE * int(in_class.sum()), 4),
            "masked": actual,
        }
    assert not data.loc[~is_train, "masked"].any(), "a validation or test row was masked"
    del labeled_train

    # Which rows and columns go into each file. Each table is written and dropped right away,
    # so only one copy of the texts is in memory at a time.
    is_validation, is_test = data["split"] == "validation", data["split"] == "test"
    files = {
        "train_supervised.csv": (is_train & ~data["masked"] & has_label, LABELED_COLUMNS),
        "train_pseudo_unlabeled.csv": (data["masked"], ["id", "body"]),
        "natural_unlabeled.csv": (is_train & data["recommendation_status"].isna(), ["id", "body"]),
        "validation.csv": (is_validation, [*LABELED_COLUMNS, "text_seen_in_train"]),
        "test.csv": (is_test, [*LABELED_COLUMNS, "text_seen_in_train"]),
        "audit/masked_labels_reference.csv": (data["masked"], ["id", "recommendation_status", "rating_band"]),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    row_counts, labels_per_file = {}, {}
    for name, (rows, file_columns) in files.items():
        table = data.loc[rows, file_columns]
        table.to_csv(SPLITS_DIR / name, index=False, encoding="utf-8")
        row_counts[name] = len(table)
        if "rate" in file_columns:
            labels_per_file[name] = {
                "recommendation_status": class_counts(table["recommendation_status"]),
                "rating_band": class_counts(table["rating_band"]),
            }
        print(f"  wrote {name}: {len(table):,} rows")
        del table
    print(f"Files written ({time.perf_counter() - start:.0f}s)")

    split_report = json.loads(SPLIT_REPORT_PATH.read_text(encoding="utf-8"))
    parts = list(SPLIT_FRACTIONS)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seeds": {"SPLIT_SEED": SPLIT_SEED, "MASK_SEED": MASK_SEED},
        # What the files were built from; the same inputs must give byte-identical files
        "inputs": {
            "comments_clean_sha256": file_sha256(COMMENTS_CLEAN_PARQUET),
            "split_assignment_sha256": file_sha256(ASSIGNMENT_PATH),
            "normalizer_version": NORMALIZER_VERSION,
            "mask_rate": MASK_RATE,
            "rating_band_edges": list(RATING_BAND_EDGES),
            "rate_zero_policy": RATE_ZERO_POLICY,
        },
        "split": {
            "strategy": split_report["method"],
            "fractions": SPLIT_FRACTIONS,
            "sklearn_version": split_report["sklearn_version"],
            "products_in_several_parts": split_report["products_in_several_parts"],
            "left_out_empty_after_normalization": split_report["left_out_empty_after_normalization"],
        },
        "masking": {
            "rate": MASK_RATE,
            "rule": "inside each recommendation class of the train part, sort by "
            "stable_hash(id) = int(sha256(f'{MASK_SEED}|{id}').hexdigest()[:16], 16) "
            "and mask the first floor(rate * class_count) rows; ties broken by id",
            "rounding": "floor",
            "applies_to": "train rows with a recommendation label; validation and test are never masked",
            "per_class": masked_per_class,
            "masked_total": int(data["masked"].sum()),
            "masked_rows_rating_band": "masked rows leave supervised training completely, rating band "
            "included, so the hidden label cannot leak through the rating; their true labels are kept "
            "only in audit/masked_labels_reference.csv",
        },
        "rating_band": {
            "edges": list(RATING_BAND_EDGES),
            "definition": band_definition(),
            "encoding": {name: i for i, name in enumerate(RATING_BAND_CLASSES)},
            "rate_zero_policy": RATE_ZERO_POLICY,
            "rate_zero_meaning": "a rate of exactly 0 means no rating was recorded, so there is no band",
        },
        "text": {
            "body": "raw text in every file; normalize with digikala_reviews.normalize before modeling",
            "normalizer_version": NORMALIZER_VERSION,
            "text_seen_in_train": "true when the normalized text is identical to the text of any train row",
            "how_to_read": "digikala_reviews.standard_files.read_split_file(name). Plain pd.read_csv "
            "turns texts such as 'NA' or 'null' into missing values; only an empty label cell is missing",
        },
        "files": {
            "train_supervised.csv": "train rows with at least one label; masked rows removed. "
            "A missing label means that head skips the row",
            "train_pseudo_unlabeled.csv": "masked train rows, id and body only",
            "natural_unlabeled.csv": "train rows without a recommendation label, id and body only. "
            "Rows among them that have a rating band are also in train_supervised for the band head",
            "validation.csv": "every validation row",
            "test.csv": "every test row",
            "audit/masked_labels_reference.csv": "true labels of the masked rows, only for checking "
            "pseudo-labels; never used for training",
        },
        "rows_per_part": {part: int((data["split"] == part).sum()) for part in parts},
        "row_counts": row_counts,
        "class_counts": labels_per_file,
        "text_seen_in_train_pct": {
            part: round(100 * float(data.loc[data["split"] == part, "text_seen_in_train"].mean()), 2)
            for part in parts[1:]
        },
        "sha256": {name: file_sha256(SPLITS_DIR / name) for name in files},
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("Masked per class:")
    for label, counts in masked_per_class.items():
        print(f"  {label:>16}: {counts['masked']:,} of {counts['train_labeled']:,}")
    print(f"Text seen in train: {manifest['text_seen_in_train_pct']}")
    print(f"Saved data/splits/{MANIFEST_PATH.name} ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
