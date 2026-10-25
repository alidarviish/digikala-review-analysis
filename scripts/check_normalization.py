"""Measure what the normalizer does to the cleaned reviews and write a report.

Reads the cleaned table in chunks, so memory stays low.

Usage:
    uv run python scripts/check_normalization.py
"""

import json
import time
from collections import Counter
from datetime import UTC, datetime

import pandas as pd
import pyarrow.parquet as pq

from digikala_reviews.config import COMMENTS_CLEAN_PARQUET, REPORTS_DIR
from digikala_reviews.normalize import NORMALIZER_VERSION, normalize

REPORT_PATH = REPORTS_DIR / "normalization_report.json"
CHUNK_ROWS = 250_000
SAMPLE_EVERY = 20  # every 20th chunk row goes into the word-count sample (about 300,000 reviews)


def main():
    start = time.perf_counter()
    n_rows = n_changed = n_changed_beyond_trim = n_empty_after = n_not_idempotent = 0
    raw_keys, normalized_keys = set(), set()
    raw_words, normalized_words = Counter(), Counter()
    empty_after_ids, not_idempotent_ids, examples = [], [], []

    parquet = pq.ParquetFile(COMMENTS_CLEAN_PARQUET)
    for batch in parquet.iter_batches(batch_size=CHUNK_ROWS, columns=["id", "body"]):
        chunk = batch.to_pandas()
        normalized = chunk["body"].map(normalize)
        trimmed = chunk["body"].str.split().str.join(" ")  # only spaces and line breaks tidied

        changed = normalized != chunk["body"]
        changed_beyond_trim = normalized != trimmed
        empty = normalized == ""
        not_idempotent = normalized.map(normalize) != normalized
        n_rows += len(chunk)
        n_changed += int(changed.sum())
        n_changed_beyond_trim += int(changed_beyond_trim.sum())
        n_empty_after += int(empty.sum())
        n_not_idempotent += int(not_idempotent.sum())
        empty_after_ids += chunk.loc[empty, "id"].tolist()
        not_idempotent_ids += chunk.loc[not_idempotent, "id"].tolist()

        # Distinct texts before and after (hashes keep memory low)
        raw_keys.update(pd.util.hash_pandas_object(chunk["body"].str.strip(), index=False).tolist())
        normalized_keys.update(pd.util.hash_pandas_object(normalized, index=False).tolist())

        sample = slice(None, None, SAMPLE_EVERY)
        for raw_text, clean_text in zip(chunk["body"].iloc[sample], normalized.iloc[sample], strict=True):
            raw_words.update(raw_text.split())
            normalized_words.update(clean_text.split())

        if len(examples) < 15:
            picked = chunk.loc[changed_beyond_trim].head(3)
            examples += [{"raw": r[:120], "normalized": normalize(r)[:120]} for r in picked["body"]]
        print(f"  {n_rows:,} reviews done")

    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "normalizer_version": NORMALIZER_VERSION,
        "reviews": n_rows,
        "changed": n_changed,
        "changed_pct": round(100 * n_changed / n_rows, 2),
        "changed_beyond_whitespace": n_changed_beyond_trim,
        "changed_beyond_whitespace_pct": round(100 * n_changed_beyond_trim / n_rows, 2),
        "empty_after_normalization": n_empty_after,
        "empty_after_normalization_ids": empty_after_ids[:50],
        "not_idempotent": n_not_idempotent,
        "not_idempotent_ids": not_idempotent_ids[:50],
        "distinct_texts_raw": len(raw_keys),
        "distinct_texts_normalized": len(normalized_keys),
        "sample_reviews_for_words": sum(1 for _ in range(0, n_rows, SAMPLE_EVERY)),
        "distinct_words_raw_sample": len(raw_words),
        "distinct_words_normalized_sample": len(normalized_words),
        "examples": examples[:15],
        "seconds": round(time.perf_counter() - start),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Changed: {n_changed:,} ({report['changed_pct']}%)")
    pct = report["changed_beyond_whitespace_pct"]
    print(f"Changed beyond whitespace: {n_changed_beyond_trim:,} ({pct}%)")
    print(f"Empty after normalization: {n_empty_after:,}")
    print(f"Not idempotent: {n_not_idempotent:,}")
    print(f"Distinct texts: {len(raw_keys):,} -> {len(normalized_keys):,}")
    print(f"Distinct words in sample: {len(raw_words):,} -> {len(normalized_words):,}")
    print(f"Report saved to reports/{REPORT_PATH.name} ({report['seconds']}s)")


if __name__ == "__main__":
    main()
