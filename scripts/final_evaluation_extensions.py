"""The second and last reading of the test set, for the optional extensions only
(plan: reports/extensions_plan.md, fixed before they were trained).

The models of the plan's list, each with the offsets fixed on validation; nothing is fitted or
chosen here, and the official system (reports/final_evaluation.json) is not changed:

  rate model        FaBERT with a rate head: its rate, and its two classification heads
  TF-IDF ridge      the TF-IDF baseline for the rate (artifacts/tfidf_final/rate_ridge.joblib)
  frozen            FaBERT with the encoder frozen
  student 0.80, student 0.95, student both heads, student natural   (end of their epoch)

FaBERT, TF-IDF and the official student are scored again from the files of the first reading, as
references. The GPU models' test texts were scored on molab by scripts/score_test_set.py; their
weights must be this machine's files. Scores as in scripts/check_extensions.py: for the rate MAE,
RMSE, MSE, R2, Spearman and the residuals; for every classification head every required score,
with intervals of the differences that draw products.

It runs only once: if reports/final_evaluation_extensions.json exists it stops.

Usage:
    uv run python scripts/final_evaluation_extensions.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_extensions import classification, print_rate, rate_section  # noqa: E402

from digikala_reviews.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CACHE_DIR,
    FINAL_TRANSFORMER_RESULT,
    REPORTS_DIR,
)
from digikala_reviews.experiments import predict_with_offsets  # noqa: E402
from digikala_reviews.ingest import file_sha256  # noqa: E402
from digikala_reviews.predictor import offsets_array  # noqa: E402
from digikala_reviews.provenance import check_cache, git_commit  # noqa: E402
from digikala_reviews.tfidf import HEADS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

OUTPUT = REPORTS_DIR / "final_evaluation_extensions.json"
# name: (scored name on molab, local weights, report with the offsets, key in it)
SCORED = {
    "fabert": ("fabert", "transformer_no_early_stop/model.pt", FINAL_TRANSFORMER_RESULT, "heads"),
    "student 0.90": (
        "student",
        "semi_supervised_student/model_last.pt",
        "semi_supervised/student.json",
        "last_check",
    ),
    "rate model": ("rate", "transformer_rate/model.pt", "transformer_final/rate.json", "heads"),
    "frozen": ("frozen", "transformer_frozen/model.pt", "transformer_final/frozen.json", "heads"),
    "student 0.80": (
        "student_t080",
        "semi_supervised_student_t080/model_last.pt",
        "semi_supervised/student_t080.json",
        "last_check",
    ),
    "student 0.95": (
        "student_t095",
        "semi_supervised_student_t095/model_last.pt",
        "semi_supervised/student_t095.json",
        "last_check",
    ),
    "student both heads": (
        "student_both",
        "semi_supervised_student_both/model_last.pt",
        "semi_supervised/student_both.json",
        "last_check",
    ),
    "student natural": (
        "student_natural",
        "semi_supervised_student_natural/model_last.pt",
        "semi_supervised/student_natural.json",
        "last_check",
    ),
}
PAIRS = [
    ("rate model", "fabert"),
    ("frozen", "fabert"),
    ("frozen", "tfidf"),
    ("student 0.80", "student 0.90"),
    ("student 0.95", "student 0.90"),
    ("student both heads", "student 0.90"),
    ("student natural", "student 0.90"),
    ("student 0.80", "fabert"),
    ("student 0.95", "fabert"),
    ("student both heads", "fabert"),
    ("student natural", "fabert"),
]


def read(relative) -> dict:
    return json.loads((REPORTS_DIR / relative).read_text(encoding="utf-8"))


def main():
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists: the test set has been read for the extensions already")
    offsets, scored_files = {}, {}
    for name, (scored, weights, report, key) in SCORED.items():
        record = read(f"final_evaluation/{scored}_scoring.json")
        assert record["weights_sha256"] == file_sha256(ARTIFACTS_DIR / weights), (
            f"{name}: other weights were scored"
        )
        found = ARTIFACTS_DIR / "final_evaluation" / f"{scored}_test_predictions.parquet"
        assert record["predictions_sha256"] == file_sha256(found), f"{name}: predictions changed"
        heads = read(report)[key]["heads"] if key == "last_check" else read(report)[key]
        offsets[name] = {h: offsets_array(heads[h]["offsets_for_test"], HEADS[h]["labels"]) for h in HEADS}
        scored_files[name] = found
    tfidf = joblib.load(ARTIFACTS_DIR / "tfidf_final" / "model.joblib")
    ridge = joblib.load(ARTIFACTS_DIR / "tfidf_final" / "rate_ridge.joblib")
    ridge_record = read("extensions/rate_ridge.json")
    assert ridge_record["ridge_sha256"] == file_sha256(ARTIFACTS_DIR / "tfidf_final" / "rate_ridge.joblib")

    test = load_cache("test.parquet", extra_columns=["product_id", "rate"])  # the labels: read here only
    data = check_cache(CACHE_DIR, ["test.parquet"])
    x = tfidf["vectorizer"].transform(test["text"])
    tables = {}
    for name, path in scored_files.items():
        found = pd.read_parquet(path)
        assert found["id"].is_unique and set(found["id"]) == set(test["id"]), f"{name}: ids differ"
        found = test[["id"]].merge(found, on="id", how="left", validate="one_to_one")
        table = pd.DataFrame({"id": test["id"].to_numpy()})
        for head, spec in HEADS.items():
            log = found[[f"{head}__log_proba__{c}" for c in spec["labels"]]].to_numpy(dtype=float)
            table[f"{head}__predicted_with_offsets"] = np.array(spec["labels"])[
                predict_with_offsets(log, offsets[name][head])
            ]
        if "rate__predicted" in found:
            table["rate__predicted"] = found["rate__predicted"].to_numpy(dtype=float)
        tables[name] = table
    table = pd.DataFrame({"id": test["id"].to_numpy()})
    for head, spec in HEADS.items():
        model = tfidf["models"][head]
        order = [list(model.classes_.astype(str)).index(c) for c in spec["labels"]]
        log = model.predict_log_proba(x)[:, order]
        table[f"{head}__predicted_with_offsets"] = np.array(spec["labels"])[
            predict_with_offsets(log, offsets_array(tfidf["offsets_for_test"][head], spec["labels"]))
        ]
    tables["tfidf"] = table

    rate = rate_section(
        test,
        {
            "FaBERT rate head": tables["rate model"]["rate__predicted"].to_numpy(dtype=float),
            "TF-IDF ridge": np.clip(ridge.predict(x), 0, 5),
        },
        ridge_record["training_mean_rate"],
    )
    _, product_index = np.unique(test["product_id"].to_numpy(), return_inverse=True)
    heads = classification(test, tables, {}, product_index, PAIRS)
    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/extensions_plan.md",
            "official_result_unchanged": "reports/final_evaluation.json",
            "rate": rate,
            "heads": heads,
            "scoring": {k: read(f"final_evaluation/{v[0]}_scoring.json")["code"] for k, v in SCORED.items()},
            "data": data,
            "code": git_commit(),
        },
    )
    print("=== exact rate (test, reviews with a real rate)")
    print_rate(rate)
    for head, e in heads.items():
        print(f"\n=== {head}, Macro F1 with offsets (test)")
        for name, m in e["models"].items():
            print(f"  {name:<20} {m['macro_f1']:.4f}")
        for pair, d in e["differences"].items():
            low, high = d["interval_95_products"]
            print(f"  {pair}: {d['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    main()
