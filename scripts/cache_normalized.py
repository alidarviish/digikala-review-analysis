"""Normalize the review texts once and keep them as Parquet, so experiments start fast.

For train_supervised, validation and test, writes data/cache/<name>.parquet with the
same columns as the CSV file, except that `body` is replaced by `text`, the normalized
review. data/cache/cache_info.json records the normalizer version, the hash of each source
file and the hash of each Parquet file, so a stale or damaged copy can be noticed.

Usage:
    uv run python scripts/cache_normalized.py
"""

import json
import time

import pandas as pd

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR, SPLITS_DIR
from digikala_reviews.ingest import file_sha256
from digikala_reviews.normalize import NORMALIZER_VERSION, normalize
from digikala_reviews.resources import peak_memory_gb, require_free_memory
from digikala_reviews.standard_files import read_split_file

FILES = ["train_supervised.csv", "validation.csv", "test.csv"]
CHUNK_ROWS = 250_000


def main():
    start = time.perf_counter()
    manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    info = {"normalizer_version": NORMALIZER_VERSION, "files": {}}

    for name in FILES:
        source_hash = file_sha256(SPLITS_DIR / name)
        assert source_hash == manifest["sha256"][name], f"{name} differs from the checked version"
        require_free_memory(2.5, f"reading {name}")

        table = read_split_file(name)
        texts = [
            table["body"].iloc[i : i + CHUNK_ROWS].map(normalize) for i in range(0, len(table), CHUNK_ROWS)
        ]
        table["text"] = pd.concat(texts)
        assert (table["text"] != "").all(), f"{name} has a review that is empty after normalization"
        table = table.drop(columns="body")

        out = CACHE_DIR / name.replace(".csv", ".parquet")
        table.to_parquet(out, index=False)
        info["files"][out.name] = {
            "source": name,
            "source_sha256": source_hash,
            "rows": len(table),
            "sha256": file_sha256(out),  # checked before every run that reads the cache
        }
        seconds, peak = time.perf_counter() - start, peak_memory_gb()
        print(f"  {out.name}: {len(table):,} rows ({seconds:.0f}s, peak memory {peak:.1f} GB)")
        del table, texts

    (CACHE_DIR / "cache_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
