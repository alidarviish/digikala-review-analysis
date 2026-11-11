"""The TF-IDF baseline for the exact rate: ridge regression on the final TF-IDF features
(reports/extensions_plan.md, extension 1).

The vectorizer is the final TF-IDF model's (artifacts/tfidf_final/model.joblib, fitted on the
training texts; its hash must match reports/tfidf_final/chosen_model.json). On molab, where the
artifacts are not, it is read from the cache folder as tfidf_model.joblib. A ridge regression
(alpha 1, not tuned) is fitted on the training reviews that have a real rate (a rate of 0 means
"no rating recorded" and is left out), and predicts every validation review; predictions are
clipped to [0, 5]. The test set is not read.

Writes artifacts/tfidf_final/rate_ridge.joblib, artifacts/tfidf_final/rate_validation_predictions.parquet
(id, rate__predicted) and reports/extensions/rate_ridge.json (with the training mean rate, the
constant baseline).

Usage:
    uv run python scripts/fit_rate_ridge.py     # about 3 GB of memory (measured)
"""

import json
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR
from digikala_reviews.ingest import file_sha256
from digikala_reviews.provenance import check_cache, environment, git_commit
from digikala_reviews.resources import peak_memory_gb, require_free_memory
from digikala_reviews.tfidf import load_cache
from digikala_reviews.tuning import save_json

ALPHA = 1.0
OUT_DIR = ARTIFACTS_DIR / "tfidf_final"
OUTPUT = REPORTS_DIR / "extensions" / "rate_ridge.json"


def main():
    if OUTPUT.exists():
        print("Already done", flush=True)
        return
    recorded = json.loads((REPORTS_DIR / "tfidf_final" / "chosen_model.json").read_text(encoding="utf-8"))
    expected = recorded["files"]["artifacts/tfidf_final/model.joblib"]["sha256"]
    model_path = OUT_DIR / "model.joblib"
    if not model_path.exists():
        model_path = CACHE_DIR / "tfidf_model.joblib"
    assert file_sha256(model_path) == expected, f"{model_path} is not the final TF-IDF model"
    data = check_cache(CACHE_DIR, ["train_supervised.parquet", "validation.parquet"])
    require_free_memory(3.5, "the TF-IDF features of the training reviews")
    vectorizer = joblib.load(model_path)["vectorizer"]

    start = time.perf_counter()
    train = load_cache("train_supervised.parquet", extra_columns=["rate"])
    train = train[train["rate"] > 0]
    x = vectorizer.transform(train["text"])
    y = train["rate"].to_numpy(dtype=float)
    model = Ridge(alpha=ALPHA).fit(x, y)
    minutes = round((time.perf_counter() - start) / 60, 1)
    print(f"fitted on {len(y):,} rated training reviews in {minutes} min", flush=True)

    validation = load_cache("validation.parquet")
    predicted = np.clip(model.predict(vectorizer.transform(validation["text"])), 0, 5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "rate_ridge.joblib", compress=3)
    pd.DataFrame(
        {"id": validation["id"].to_numpy(), "rate__predicted": predicted.astype(np.float32)}
    ).to_parquet(OUT_DIR / "rate_validation_predictions.parquet", index=False)
    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/extensions_plan.md",
            "alpha": ALPHA,
            "training_reviews": int(len(y)),
            "training_mean_rate": float(y.mean()),
            "minutes": minutes,
            "tfidf_model_sha256": expected,
            "ridge_sha256": file_sha256(OUT_DIR / "rate_ridge.joblib"),
            "peak_memory_gb": peak_memory_gb(),
            "data": data,
            "environment": environment(),
            "code": git_commit(),
        },
    )
    print(f"saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
