"""Memory of the offline predictor (reports/extensions_plan.md, extension 6).

The process's resident memory (VmRSS) and its peak (VmHWM) before loading, after loading
the models (src/digikala_reviews/predictor.py), and after answering 200 validation reviews one
at a time (the same 2,000-review sample, seed 2026, as scripts/check_predictor.py; its first
200). Runs on the CPU of this machine. The test set is not read.

Writes reports/predictor_memory.json and prints the numbers.

Usage:
    uv run python scripts/measure_predictor_memory.py
"""

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from digikala_reviews.config import REPORTS_DIR, SPLIT_SEED
from digikala_reviews.finetune import fixed_sample
from digikala_reviews.provenance import environment, git_commit, library_version
from digikala_reviews.standard_files import read_split_file
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "predictor_memory.json"
SAMPLE, ANSWERED = 2_000, 200


def memory_gb() -> dict:
    """The current and the peak resident memory of this process, in GB."""
    found = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            found["now" if key == "VmRSS" else "peak"] = round(int(value.split()[0]) / 1024**2, 3)
    return found


def main():
    validation = read_split_file("validation.csv")[["body"]]
    bodies = validation["body"].to_numpy()[fixed_sample(len(validation), SAMPLE, SPLIT_SEED)][:ANSWERED]
    del validation
    before = memory_gb()

    from digikala_reviews.predictor import Predictor

    start = time.perf_counter()
    predictor = Predictor()
    load_seconds = round(time.perf_counter() - start, 1)
    loaded = memory_gb()
    latency = []
    for body in bodies:
        latency.append(sum(predictor.predict(body)["latency_ms"].values()))
    answered = memory_gb()

    result = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan": "reports/extensions_plan.md",
        "memory_gb": {
            "before_loading": before,
            "after_loading": loaded,
            f"after_{ANSWERED}_reviews": answered,
            "models_alone": round(loaded["now"] - before["now"], 3),
        },
        "load_seconds": load_seconds,
        "latency_ms_one_review": {
            "median": float(np.median(latency)),
            "p95": float(np.percentile(latency, 95)),
        },
        "environment": {**environment(), "torch": library_version("torch")},
        "code": git_commit(),
    }
    save_json(OUTPUT, result)
    m = result["memory_gb"]
    done = m[f"after_{ANSWERED}_reviews"]
    print(f"before loading: {m['before_loading']['now']} GB")
    print(f"after loading: {m['after_loading']['now']} GB (models {m['models_alone']} GB, {load_seconds}s)")
    print(f"after {ANSWERED} reviews: {done['now']} GB, peak {done['peak']} GB")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
