"""Run the schema and quality checks on the raw data.

Usage:
    uv run python scripts/run_quality_gate.py
"""

import json
import time
from datetime import UTC, datetime

from digikala_reviews.config import COMMENTS_RAW_DIR, PRODUCTS_RAW_DIR, REPORTS_DIR
from digikala_reviews.quality import run_quality_gate

REPORT_PATH = REPORTS_DIR / "quality_gate.json"


def main():
    start = time.perf_counter()
    report = run_quality_gate(COMMENTS_RAW_DIR, PRODUCTS_RAW_DIR)
    report = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **report}

    c = report["comments"]
    print(f"Rows: {c['rows']:,}")
    print(f"Missing body: {c['missing_values']['body']['missing']:,}")
    dup = c["duplicate_ids"]
    print(f"Duplicated ids: {dup['duplicated_ids']:,} (extra rows: {dup['extra_rows']:,})")
    print(f"Rate out of range: {c['rate']['out_of_range']:,} | zero: {c['rate']['zero']:,}")
    print(f"Recommendation labels: {c['recommendation_status']['observed']}")

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    REPORT_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"Report saved to reports/{REPORT_PATH.name} ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
