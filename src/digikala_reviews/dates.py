"""Helpers for the Persian (Jalali) dates in the reviews, e.g. '23 شهریور 1402'.

For trends over time the Jalali year and month number are enough, so there is no
conversion to the Gregorian calendar.
"""

import pandas as pd

JALALI_MONTHS = [
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
]  # fmt: skip

DATE_PATTERN = r"^(\d{1,2}) (\S+) (\d{4})$"


def parse_jalali(dates: pd.Series) -> pd.DataFrame:
    """Split 'day month-name year' into numeric year and month columns."""
    parts = dates.str.extract(DATE_PATTERN)
    month_number = {name: i for i, name in enumerate(JALALI_MONTHS, start=1)}
    return pd.DataFrame(
        {
            "year": pd.to_numeric(parts[2]),
            "month": parts[1].map(month_number),
        },
        index=dates.index,
    )
