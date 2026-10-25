"""Check the standard data files against the cleaned data and the project rules.

Every file is read back with read_split_file and compared row by row with the source,
so a problem in writing or reading the CSVs shows up here. The expected rows of every
file, the masked rows and the text_seen_in_train flags are worked out again here,
independently of build_standard_files.py where possible.

Rebuild check: the file hashes are compared with reports/split_manifest.json, the copy
kept from the last checked run. Run this script, then build_standard_files.py again,
then this script again: identical hashes show that the files are rebuilt byte for byte.

Writes reports/standard_files_check.json and, when every check passes, refreshes
reports/split_manifest.json. Exits with an error if any check fails.

Usage:
    uv run python scripts/verify_standard_files.py
"""

import hashlib
import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.clean import rating_band
from digikala_reviews.config import (
    COMMENTS_CLEAN_PARQUET,
    MASK_RATE,
    MASK_SEED,
    RECOMMENDATION_CLASSES,
    REPORTS_DIR,
    SPLITS_DIR,
)
from digikala_reviews.ingest import file_sha256
from digikala_reviews.normalize import normalize
from digikala_reviews.standard_files import read_split_file

MANIFEST_PATH = SPLITS_DIR / "split_manifest.json"
KEPT_MANIFEST_PATH = REPORTS_DIR / "split_manifest.json"
REPORT_PATH = REPORTS_DIR / "standard_files_check.json"
CHUNK_ROWS = 250_000

LABELED = ["id", "product_id", "body", "rate", "recommendation_status", "rating_band"]
EXPECTED_COLUMNS = {
    "train_supervised.csv": LABELED,
    "train_pseudo_unlabeled.csv": ["id", "body"],
    "natural_unlabeled.csv": ["id", "body"],
    "validation.csv": [*LABELED, "text_seen_in_train"],
    "test.csv": [*LABELED, "text_seen_in_train"],
    "audit/masked_labels_reference.csv": ["id", "recommendation_status", "rating_band"],
}

checks = []


def check(name, passed, detail=""):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'ok' if passed else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))


def same_values(a: pd.Series, b: pd.Series) -> bool:
    """Same missing positions and the same values everywhere else."""
    a_missing, b_missing = a.isna().to_numpy(), b.isna().to_numpy()
    if len(a) != len(b) or not (a_missing == b_missing).all():
        return False
    a_values = a.to_numpy(dtype=object)[~a_missing]
    b_values = b.to_numpy(dtype=object)[~b_missing]
    return bool((a_values == b_values).all())


def same_ids(a, b) -> bool:
    return np.array_equal(np.sort(np.asarray(a)), np.sort(np.asarray(b)))


def masked_by_protocol(source: pd.DataFrame) -> np.ndarray:
    """The masking rule written out again from the project text, without using mask.py."""
    chosen = []
    labeled_train = source[(source["split"] == "train") & source["recommendation_status"].notna()]
    for label in RECOMMENDATION_CLASSES:
        ids = labeled_train.index[labeled_train["recommendation_status"] == label]
        keyed = []
        for row_id in ids:
            key = f"{MASK_SEED}|{row_id}".encode()
            keyed.append((int(hashlib.sha256(key).hexdigest()[:16], 16), row_id))
        keyed.sort()
        chosen += [row_id for _, row_id in keyed[: math.floor(MASK_RATE * len(keyed))]]
    return np.array(chosen)


def text_seen_in_train(source: pd.DataFrame) -> pd.Series:
    """Worked out again: is the normalized text identical to the text of some train row?"""
    keys = []
    for start in range(0, len(source), CHUNK_ROWS):
        chunk = source["body"].iloc[start : start + CHUNK_ROWS].map(normalize)
        keys.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
    keys = pd.Series(np.concatenate(keys), index=source.index)
    return keys.isin(np.unique(keys[source["split"] == "train"].to_numpy()))


def rebuild_check(hashes, manifest, kept):
    """Compare with the last checked run, but only when that comparison means something."""
    if kept is None:
        return "skipped: no earlier checked run"
    if kept.get("inputs") != manifest.get("inputs"):
        return "skipped: the inputs changed since the last checked run"
    if kept["created_at"] == manifest["created_at"]:
        return "skipped: the files were not rebuilt since the last checked run"
    return "identical" if hashes == kept["sha256"] else "different"


def main():
    start = time.perf_counter()
    REPORT_PATH.unlink(missing_ok=True)  # a failed run never leaves an old "passed" report behind
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    kept = json.loads(KEPT_MANIFEST_PATH.read_text(encoding="utf-8")) if KEPT_MANIFEST_PATH.exists() else None

    columns = ["id", "product_id", "body", "rate", "recommendation", "rating_band"]
    source = pd.read_parquet(COMMENTS_CLEAN_PARQUET, columns=columns)
    source = source.rename(columns={"recommendation": "recommendation_status"})
    source["rate"] = source["rate"].astype("float64")
    assignment = pd.read_parquet(SPLITS_DIR / "split_assignment.parquet", columns=["id", "split"])
    n_assigned = len(assignment)
    source = source.merge(assignment, on="id", how="inner", validate="one_to_one").set_index("id")
    del assignment
    print(f"Read the source data ({time.perf_counter() - start:.0f}s)")

    print("Source and manifest")
    check("every id of the split is in the cleaned data", len(source) == n_assigned, f"{len(source):,}")
    check("no product in more than one part", (source.groupby("product_id")["split"].nunique() == 1).all())
    rows_per_part = source["split"].value_counts().to_dict()
    check("rows per part match the manifest", rows_per_part == manifest["rows_per_part"])
    hashes = {name: file_sha256(SPLITS_DIR / name) for name in EXPECTED_COLUMNS}
    check("file hashes match the manifest", hashes == manifest["sha256"])
    rebuild = rebuild_check(hashes, manifest, kept)
    check("files are rebuilt byte for byte", rebuild != "different", rebuild)

    # Which rows every file should hold, worked out from the source
    is_train = source["split"] == "train"
    has_recommendation = source["recommendation_status"].notna()
    has_label = has_recommendation | source["rating_band"].notna()
    masked = masked_by_protocol(source)
    is_masked = source.index.isin(masked)
    expected_ids = {
        "train_supervised.csv": source.index[is_train & ~is_masked & has_label],
        "train_pseudo_unlabeled.csv": source.index[is_masked],
        "natural_unlabeled.csv": source.index[is_train & ~has_recommendation],
        "validation.csv": source.index[source["split"] == "validation"],
        "test.csv": source.index[source["split"] == "test"],
        "audit/masked_labels_reference.csv": source.index[is_masked],
    }
    print("Masking")
    for label in RECOMMENDATION_CLASSES:
        in_class = is_train & (source["recommendation_status"] == label)
        expected = math.floor(MASK_RATE * int(in_class.sum()))
        actual = int((in_class & is_masked).sum())
        check(f"{label}: floor({MASK_RATE} x class count) rows masked", actual == expected, f"{actual:,}")
    check("validation and test rows are never masked", not (is_masked & ~is_train).any())
    seen = text_seen_in_train(source)
    print(f"Worked out the expected rows ({time.perf_counter() - start:.0f}s)")

    for name, expected_columns in EXPECTED_COLUMNS.items():
        print(name)
        table = read_split_file(name)
        check(f"{name}: columns are exactly as expected", list(table.columns) == expected_columns)
        check(f"{name}: row count matches the manifest", len(table) == manifest["row_counts"][name])
        check(f"{name}: holds exactly the expected rows", same_ids(table["id"], expected_ids[name]))
        if not table["id"].is_unique or not table["id"].isin(source.index).all():
            check(f"{name}: ids are unique and known", False)
            continue
        compared = [c for c in table.columns if c in source.columns]
        rows = source.loc[table["id"], compared]
        for column in compared:
            check(
                f"{name}: {column} is identical to the cleaned data", same_values(table[column], rows[column])
            )
        if "rate" in table:
            check(
                f"{name}: rating_band follows from rate",
                same_values(table["rating_band"], rating_band(table["rate"])),
            )
        if "text_seen_in_train" in table:
            flags = table["text_seen_in_train"]
            check(f"{name}: text_seen_in_train is always True or False", flags.isin([True, False]).all())
            check(
                f"{name}: text_seen_in_train is right for every row",
                same_values(flags, seen.loc[table["id"]]),
            )
        del table, rows

    passed = all(c["passed"] for c in checks)
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "passed": passed,
        "rebuild_check": rebuild,
        "checks": checks,
        "sha256": hashes,
        "seconds": round(time.perf_counter() - start),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{sum(c['passed'] for c in checks)} of {len(checks)} checks passed ({report['seconds']}s)")
    print(f"Rebuild check: {rebuild}")

    if not passed:
        sys.exit("Some checks failed, see reports/standard_files_check.json")
    shutil.copyfile(MANIFEST_PATH, KEPT_MANIFEST_PATH)
    print(f"Copied the manifest to reports/{KEPT_MANIFEST_PATH.name}")


if __name__ == "__main__":
    main()
