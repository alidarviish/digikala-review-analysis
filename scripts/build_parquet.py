"""Convert the raw Digikala CSV files to Parquet and write an ingest report.

Usage:
    uv run python scripts/build_parquet.py
"""

import json
from datetime import UTC, datetime

from digikala_reviews.config import (
    COMMENTS_CSV,
    COMMENTS_RAW_DIR,
    PRODUCTS_CSV,
    PRODUCTS_RAW_DIR,
    REPORTS_DIR,
)
from digikala_reviews.ingest import count_csv_records, csv_to_parquet

REPORT_PATH = REPORTS_DIR / "data_ingest.json"


def main():
    report = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), "files": []}

    for csv_path, out_dir in [(COMMENTS_CSV, COMMENTS_RAW_DIR), (PRODUCTS_CSV, PRODUCTS_RAW_DIR)]:
        print(f"{csv_path.name}: converting to Parquet...")
        summary = csv_to_parquet(csv_path, out_dir)
        print(f"  pandas: {summary['rows_pandas']:,} rows ({summary['seconds']}s)")

        print(f"{csv_path.name}: counting records again with the csv module...")
        summary["rows_csv_module"] = count_csv_records(csv_path)
        summary["row_counts_match"] = summary["rows_csv_module"] == summary["rows_pandas"]
        print(f"  csv module: {summary['rows_csv_module']:,} rows -> match: {summary['row_counts_match']}")

        report["files"].append(summary)

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report saved to reports/{REPORT_PATH.name}")


if __name__ == "__main__":
    main()
