"""Score the hand reading of the rating band error sample (plan: reports/error_sample_plan.md).

Reads data/error_sample/to_label.csv (the reader's codes) and key.csv (groups, labels and
predictions), and writes reports/error_sample.json: per group, the share of each reading,
stars and tag code; how often the stars read from the text equal the true band and FaBERT's
band; the reader's consistency on the repeated reviews; and the decision rule written in the
plan before reading. Report only: nothing in the models, labels or splits changes. No review
text is written to the report.

Codes: reading 1 positive, 2 negative, 3 mixed, 4 neutral; stars 0 cannot tell, 1 low,
2 medium, 3 high; tags any of the letters a to f, or empty.

Intervals: Wilson 95% intervals; the reviews are independent (one per product).

Usage:
    uv run python scripts/score_error_sample.py
"""

import math
from datetime import UTC, datetime

import pandas as pd

from digikala_reviews.config import DATA_DIR, REPORTS_DIR
from digikala_reviews.provenance import git_commit
from digikala_reviews.tuning import save_json

IN_DIR = DATA_DIR / "error_sample"
OUTPUT = REPORTS_DIR / "error_sample.json"
READING = {"1": "positive", "2": "negative", "3": "mixed", "4": "neutral"}
STARS = {"0": "cannot tell", "1": "low", "2": "medium", "3": "high"}
TAGS = {
    "a": "delivery, seller, packaging or price",
    "b": "good but / mixed judgement",
    "c": "negation or sarcasm",
    "d": "too short or no information",
    "e": "against expectations or price",
    "f": "other",
}
GROUPS = ["A", "B", "C", "D", "E"]
PER_GROUP = 20
REPEATS = 10


def read_codes(values: pd.Series, allowed: dict, column: str) -> pd.Series:
    """Codes as text; a whole number written as 1.0 is read as 1, anything else stops the run."""
    out = []
    for v in values.astype(str).str.strip():
        if v.endswith(".0") and v[:-2].isdigit():
            v = v[:-2]
        assert v in allowed, f"{column}: {v!r} is not one of {sorted(allowed)}"
        out.append(v)
    return pd.Series(out, index=values.index)


def read_tags(values: pd.Series) -> pd.Series:
    out = []
    for v in values.fillna("").astype(str):
        letters = "".join(v.lower().split()).replace(",", "")
        assert set(letters) <= set(TAGS), f"tags: {v!r} has letters other than a to f"
        out.append("".join(sorted(set(letters))))
    return pd.Series(out, index=values.index)


def wilson(hits: int, n: int) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    z, p = 1.959964, hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def share(mask: pd.Series) -> dict:
    hits, n = int(mask.sum()), int(len(mask))
    return {"share": hits / n if n else math.nan, "count": hits, "of": n, "interval_95": wilson(hits, n)}


def main():
    answers = pd.read_csv(IN_DIR / "to_label.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    key = pd.read_csv(IN_DIR / "key.csv", encoding="utf-8", dtype=str, keep_default_na=False)
    assert list(answers["row"]) == list(key["row"]), "to_label.csv and key.csv rows differ"
    table = key.copy()
    table["reading"] = read_codes(answers["reading"], READING, "reading").map(READING)
    table["stars"] = read_codes(answers["stars"], STARS, "stars").map(STARS)
    table["tags"] = read_tags(answers["tags"])

    first = table[table["repeat_of"] == "0"]
    assert len(first) == PER_GROUP * len(GROUPS), f"{len(first)} first showings"
    assert first["group"].value_counts().to_dict() == dict.fromkeys(GROUPS, PER_GROUP), "group sizes"
    assert first["product_id"].is_unique and first["id"].is_unique, "a product or review twice"

    groups = {}
    for g in GROUPS:
        part = first[first["group"] == g]
        groups[g] = {
            "rows": int(len(part)),
            "reading": {name: share(part["reading"] == name) for name in READING.values()},
            "stars": {name: share(part["stars"] == name) for name in STARS.values()},
            "tags": {letter: share(part["tags"].str.contains(letter)) for letter in TAGS},
            "no_tag": share(part["tags"] == ""),
            "stars_equal_true_band": share(part["stars"] == part["band"]),
            "stars_equal_fabert_band": share(part["stars"] == part["band_predicted"]),
            "true_band": part["band"].value_counts().to_dict(),
            "fabert_band": part["band_predicted"].value_counts().to_dict(),
        }

    by_row = table.set_index("row")
    repeats = table[table["repeat_of"] != "0"]
    assert len(repeats) == REPEATS, f"{len(repeats)} repeats"
    pairs = [(r, by_row.loc[r["repeat_of"]]) for _, r in repeats.iterrows()]
    consistency = {
        column: share(pd.Series([a[column] == b[column] for a, b in pairs]))
        for column in ["reading", "stars", "tags"]
    }

    a, b, c, d, e = (first[first["group"] == g] for g in GROUPS)
    expected = {
        "A reading positive >= 0.60": share(a["reading"] == "positive"),
        "A stars high >= 0.60": share(a["stars"] == "high"),
        "B reading mixed or neutral >= 0.60": share(b["reading"].isin(["mixed", "neutral"])),
        "C reading mixed, or tag e or d >= 0.50": share(
            (c["reading"] == "mixed") | c["tags"].str.contains("e") | c["tags"].str.contains("d")
        ),
        "D stars not low >= 0.50": share(d["stars"].isin(["medium", "high"])),
        "D tag a": share(d["tags"].str.contains("a")),
        "E stars equal the true band, about 0.75": share(e["stars"] == e["band"]),
        "repeats same stars, about 0.80": consistency["stars"],
    }
    a_high = expected["A stars high >= 0.60"]["share"]
    e_right = expected["E stars equal the true band, about 0.75"]["share"]
    if e_right < 0.60:
        verdict = (
            "E below 0.60: the reader cannot recover the band from text alone, A is not read as evidence"
        )
    elif a_high >= 0.60:
        verdict = "H2' supported by reading: in A the text reads high, the rate disagrees with the text"
    else:
        verdict = "H2' not supported by reading: in A the text does not read high in at least 60%"

    result = {
        "rows": int(len(table)),
        "reviews": int(len(first)),
        "groups": groups,
        "repeat_consistency": consistency,
        "expected_in_plan": expected,
        "decision": {"A_stars_high": a_high, "E_stars_right": e_right, "verdict": verdict},
    }
    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/error_sample_plan.md",
            "codes": {"reading": READING, "stars": STARS, "tags": TAGS},
            **result,
            "code": git_commit(),
        },
    )
    summary(result)
    print(f"saved {OUTPUT}")


def text(x: dict) -> str:
    return f"{x['share']:.2f} ({x['count']}/{x['of']}) [{x['interval_95'][0]:.2f}, {x['interval_95'][1]:.2f}]"


def summary(result):
    print(f"{result['reviews']} reviews, {result['rows']} rows with repeats")
    for g, x in result["groups"].items():
        print(f"group {g}: true band {x['true_band']}, FaBERT {x['fabert_band']}")
        print("   reading  " + ", ".join(f"{k} {v['count']}" for k, v in x["reading"].items()))
        print("   stars    " + ", ".join(f"{k} {v['count']}" for k, v in x["stars"].items()))
        print(
            "   tags     "
            + ", ".join(f"{k} {v['count']}" for k, v in x["tags"].items())
            + f", none {x['no_tag']['count']}"
        )
        right, fabert = text(x["stars_equal_true_band"]), text(x["stars_equal_fabert_band"])
        print(f"   stars = true band {right}; = FaBERT {fabert}")
    print(
        "repeats, same code: " + ", ".join(f"{k} {text(v)}" for k, v in result["repeat_consistency"].items())
    )
    print("expected in the plan:")
    for k, v in result["expected_in_plan"].items():
        print(f"   {k:<44} {text(v)}")
    print(f"decision: {result['decision']['verdict']}")


if __name__ == "__main__":
    main()
