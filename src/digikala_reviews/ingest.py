"""Convert the raw CSV files to Parquet without changing any value.

Every column is read as text and missing-value detection is switched off, so
values like "" or "nan" stay exactly as they are in the file. Type checks happen
later, in the quality gate, where every problem is counted.
"""

import csv
import hashlib
import time
from pathlib import Path

import pandas as pd

ROW_INDEX_COLUMN = "src_row"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_records(path: Path) -> int:
    """Count records with Python's csv module, as an independent check on pandas."""
    csv.field_size_limit(100_000_000)  # some reviews are very long
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return sum(1 for _ in reader)


def csv_to_parquet(csv_path: Path, out_dir: Path, chunksize: int = 500_000) -> dict:
    """Read the CSV in chunks and write one Parquet file per chunk.

    A `src_row` column keeps the position of each record in the original file.
    """
    start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_part in out_dir.glob("part-*.parquet"):
        old_part.unlink()

    reader = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,  # keep "" and "nan" as text instead of turning them into NaN
        chunksize=chunksize,
        encoding="utf-8-sig",  # some files start with a byte-order mark
    )
    n_rows = 0
    columns = []
    for i, chunk in enumerate(reader):
        columns = list(chunk.columns)
        chunk.insert(0, ROW_INDEX_COLUMN, range(n_rows, n_rows + len(chunk)))
        chunk.to_parquet(out_dir / f"part-{i:03d}.parquet", index=False)
        n_rows += len(chunk)

    size = sum(p.stat().st_size for p in out_dir.glob("part-*.parquet"))
    return {
        "csv_file": csv_path.name,
        "csv_size_bytes": csv_path.stat().st_size,
        "csv_sha256": file_sha256(csv_path),
        "columns": columns,
        "rows_pandas": n_rows,
        "parquet_dir": out_dir.name,
        "parquet_size_bytes": size,
        "seconds": round(time.perf_counter() - start, 1),
    }
