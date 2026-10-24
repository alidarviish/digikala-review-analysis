"""Apply the cleaning rules to the raw comments and write a cleaning report.

Usage:
    uv run python scripts/build_clean.py
"""

import json
from datetime import UTC, datetime

from digikala_reviews.clean import build_clean_table
from digikala_reviews.config import (
    COMMENTS_CLEAN_PARQUET,
    COMMENTS_RAW_DIR,
    EXCLUDED_ROWS_PARQUET,
    REPORTS_DIR,
)

REPORT_PATH = REPORTS_DIR / "cleaning_report.json"


def main():
    report = build_clean_table(COMMENTS_RAW_DIR, COMMENTS_CLEAN_PARQUET, EXCLUDED_ROWS_PARQUET)
    report = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **report}

    print(f"Raw rows:   {report['rows_raw']:,}")
    for reason, n in report["rows_excluded"].items():
        print(f"  excluded ({reason}): {n:,}")
    print(f"Clean rows: {report['rows_clean']:,} ({report['seconds']}s)")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Rating band ({report['rate_zero_policy']} zero policy): {report['rating_band']}")

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    REPORT_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"Report saved to reports/{REPORT_PATH.name}")


if __name__ == "__main__":
    main()
