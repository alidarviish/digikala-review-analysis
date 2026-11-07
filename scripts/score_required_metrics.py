"""Every score the project asks for, for every final model, on the whole validation set.

For TF-IDF, FaBERT (the kept run), the final system (FaBERT + TF-IDF, rebuilt as in step 60),
the further pretrained FaBERT, TookaBERT-Large and the BiLSTM, with labels with offsets as
saved: accuracy, Macro F1, Macro precision, Macro recall, balanced accuracy, precision, recall
and F1 of each class, the confusion matrix and, for the rating band, the ordinal distance error
(how many bands a prediction is off on average, and the share off by two). Macro precision and
recall are the plain means over the classes, as Macro F1.

Each model must first score the Macro F1 recorded for it. Nothing is trained. The test set is
not read: the same scores are computed on it once, at the end, by the final evaluation.

Writes reports/required_metrics.json and prints a table.

Usage:
    uv run python scripts/score_required_metrics.py
"""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR
from digikala_reviews.ensemble import cross_fitted_blend
from digikala_reviews.experiments import product_half
from digikala_reviews.metrics import classification_report, ordinal_distance_error
from digikala_reviews.provenance import check_cache, git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "required_metrics.json"
# name: (folder of its saved predictions, its training report or None)
MODELS = {
    "tfidf": ("tfidf_final", None),
    "fabert": ("transformer_no_early_stop", "transformer_final/fabert_no_early_stop.json"),
    "adapted": ("transformer_adapted", "transformer_final/adapted.json"),
    "large": ("transformer_large", "transformer_final/large.json"),
    "bilstm": ("bilstm", "bilstm/final.json"),
}
FINAL_SYSTEM = "fabert + tfidf"


def read(path: str) -> dict:
    return json.loads((REPORTS_DIR / path).read_text(encoding="utf-8"))


def macro(report: dict) -> dict:
    per_class = report["per_class"].values()
    return {
        "macro_precision": float(np.mean([c["precision"] for c in per_class])),
        "macro_recall": float(np.mean([c["recall"] for c in per_class])),
    }


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    overlap = read("error_overlap.json")["heads"]
    ensemble = read("ensemble.json")["final_fabert_and_tfidf"]
    tables, recorded = {}, {}
    for name, (folder, report) in MODELS.items():
        table = pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet")
        assert table["id"].is_unique and set(table["id"]) == set(validation["id"]), f"{name}: ids differ"
        tables[name] = validation[["id"]].merge(table, on="id", how="left", validate="one_to_one")
        recorded[name] = (
            {h: overlap[h]["macro_f1"]["tfidf"] for h in HEADS}
            if report is None
            else {h: read(report)["heads"][h]["with_offsets"]["macro_f1"] for h in HEADS}
        )
    recorded[FINAL_SYSTEM] = {h: ensemble[h]["blend_macro_f1"] for h in HEADS}

    result = {"data": data, "final_system": FINAL_SYSTEM, "heads": {}}
    for head, spec in HEADS.items():
        labels = spec["labels"]
        rows = validation[spec["column"]].notna().to_numpy()
        y_true = validation[spec["column"]].to_numpy()[rows].astype(str)
        codes = np.array([labels.index(v) for v in y_true])
        predicted = {
            name: t[f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str)
            for name, t in tables.items()
        }
        log = {
            name: tables[name][[f"{head}__log_proba__{c}" for c in labels]].to_numpy(dtype=float)[rows]
            for name in ("fabert", "tfidf")
        }
        halves = product_half(validation["product_id"].to_numpy()[rows])
        blended, _ = cross_fitted_blend(log["fabert"], log["tfidf"], codes, halves)
        predicted[FINAL_SYSTEM] = np.array(labels)[blended]

        result["heads"][head] = {}
        for name, labels_found in predicted.items():
            report = classification_report(y_true, labels_found, labels)
            expected = recorded[name][head]
            assert abs(report["macro_f1"] - expected) < 1e-9, (
                f"{head} {name}: {report['macro_f1']}, {expected}"
            )
            entry = {**report, **macro(report)}
            if head == "rating_band":
                entry["ordinal_distance_error"] = ordinal_distance_error(y_true, labels_found, labels)
            result["heads"][head][name] = entry

    save_json(
        OUTPUT,
        {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **result, "code": git_commit()},
    )
    for head, models in result["heads"].items():
        print(f"\n=== {head} ({next(iter(models.values()))['rows']:,} reviews)")
        extra = "  ordinal: mean, off by two" if head == "rating_band" else ""
        print(f"  {'model':<16} accuracy  macro F1  macro P  macro R  balanced{extra}")
        for name, e in models.items():
            line = (
                f"  {name:<16} {e['accuracy']:.4f}    {e['macro_f1']:.4f}    {e['macro_precision']:.4f}   "
                f"{e['macro_recall']:.4f}   {e['balanced_accuracy']:.4f}"
            )
            if "ordinal_distance_error" in e:
                o = e["ordinal_distance_error"]
                line += f"    {o['mean_distance']:.4f}, {o['share_off_by_two']:.4f}"
            print(line)
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    main()
