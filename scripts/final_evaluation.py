"""The final evaluation: the one reading of the test set's labels (plan: reports/final_evaluation_plan.md).

The four models fixed in the plan, each with the weights, offsets and threshold fixed on
validation; nothing is fitted or chosen on the test set:

  tfidf           TF-IDF (artifacts/tfidf_final/model.joblib), its offsets for the test set
  fabert          FaBERT, the kept run (the supervised model), its offsets for the test set
  final system    FaBERT and TF-IDF mixed, with the weight, offsets and threshold of
                  reports/uncertain.json ("for_test"); the official result
  student         the semi-supervised model: the student at the end of its epoch, its offsets

FaBERT's and the student's test log-probabilities come from scripts/score_test_set.py (molab);
their weights must be this machine's kept files. For every model and head: accuracy, Macro F1,
Macro precision, Macro recall, balanced accuracy, precision, recall and F1 of each class, the
confusion matrix, and for the rating band the ordinal distance error; with offsets (the official
labels) and without; a 95% interval of Macro F1 that draws products (1,000 rounds, seed 2026);
and paired differences: FaBERT minus TF-IDF, the final system minus FaBERT, and the student minus
FaBERT, with the change of every score (the semi-supervised report). For the final system also
the share answered "uncertain" and its scores on the reviews it answers. The validation scores
recorded before are put next to the test ones.

It runs only once: if reports/final_evaluation.json exists it stops.

Usage:
    uv run python scripts/final_evaluation.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recheck_decisions import confusion, macro_f1  # noqa: E402

from digikala_reviews.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CACHE_DIR,
    FINAL_TRANSFORMER_DIR,
    FINAL_TRANSFORMER_RESULT,
    REPORTS_DIR,
)
from digikala_reviews.experiments import predict_with_offsets  # noqa: E402
from digikala_reviews.ingest import file_sha256  # noqa: E402
from digikala_reviews.metrics import classification_report, ordinal_distance_error  # noqa: E402
from digikala_reviews.predictor import final_answer, offsets_array  # noqa: E402
from digikala_reviews.provenance import check_cache, git_commit  # noqa: E402
from digikala_reviews.tfidf import HEADS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

OUTPUT = REPORTS_DIR / "final_evaluation.json"
SCORED = {  # model: (its local weights, which must be the ones scored on molab)
    "fabert": ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "model.pt",
    "student": ARTIFACTS_DIR / "semi_supervised_student" / "model_last.pt",
}
FINAL = "final system"
MODELS = ["tfidf", "fabert", FINAL, "student"]
PAIRS = [("fabert", "tfidf"), (FINAL, "fabert"), ("student", "fabert")]
ROUNDS, SEED = 1000, 2026


def read(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scores(y_true, found, labels, head) -> dict:
    report = classification_report(y_true, found, labels)
    per_class = report["per_class"].values()
    report["macro_precision"] = float(np.mean([c["precision"] for c in per_class]))
    report["macro_recall"] = float(np.mean([c["recall"] for c in per_class]))
    if head == "rating_band":
        report["ordinal_distance_error"] = ordinal_distance_error(y_true, found, labels)
    return report


def change(a: dict, b: dict) -> dict:
    """Every score of a minus the same score of b."""
    keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "balanced_accuracy"]
    found = {k: a[k] - b[k] for k in keys}
    for c in a["per_class"]:
        for k in ("precision", "recall", "f1"):
            found[f"{c} {k}"] = a["per_class"][c][k] - b["per_class"][c][k]
    return found


def main():
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists: the test set has been read already")
    tfidf = joblib.load(ARTIFACTS_DIR / "tfidf_final" / "model.joblib")
    fabert_report = read(REPORTS_DIR / FINAL_TRANSFORMER_RESULT)
    student_report = read(REPORTS_DIR / "semi_supervised" / "student.json")["last_check"]
    final = read(REPORTS_DIR / "uncertain.json")["heads"]
    required = read(REPORTS_DIR / "required_metrics.json")["heads"]
    offsets = {
        "tfidf": tfidf["offsets_for_test"],
        "fabert": {h: fabert_report["heads"][h]["offsets_for_test"] for h in HEADS},
        "student": {h: student_report["heads"][h]["offsets_for_test"] for h in HEADS},
    }
    scoring = {}
    for name, weights in SCORED.items():
        scoring[name] = read(REPORTS_DIR / "final_evaluation" / f"{name}_scoring.json")
        assert scoring[name]["weights_sha256"] == file_sha256(weights), f"{name}: other weights were scored"
        found = ARTIFACTS_DIR / "final_evaluation" / f"{name}_test_predictions.parquet"
        assert scoring[name]["predictions_sha256"] == file_sha256(found), f"{name}: predictions changed"

    test = load_cache("test.parquet", extra_columns=["product_id"])  # the labels: read here only
    data = check_cache(CACHE_DIR, ["test.parquet"])
    tables = {}
    for name in SCORED:
        table = pd.read_parquet(ARTIFACTS_DIR / "final_evaluation" / f"{name}_test_predictions.parquet")
        assert table["id"].is_unique and set(table["id"]) == set(test["id"]), f"{name}: ids differ"
        tables[name] = test[["id"]].merge(table, on="id", how="left", validate="one_to_one")
    x = tfidf["vectorizer"].transform(test["text"])
    _, product_index = np.unique(test["product_id"].to_numpy(), return_inverse=True)

    heads = {}
    for head, spec in HEADS.items():
        labels = spec["labels"]
        n = len(labels)
        rows = test[spec["column"]].notna().to_numpy()
        y_true = test[spec["column"]].to_numpy()[rows].astype(str)
        codes = np.array([labels.index(v) for v in y_true])
        model = tfidf["models"][head]
        order = [list(model.classes_.astype(str)).index(c) for c in labels]
        log = {"tfidf": model.predict_log_proba(x[rows])[:, order]}
        for name, t in tables.items():
            log[name] = t[[f"{head}__log_proba__{c}" for c in labels]].to_numpy(dtype=float)[rows]
        predicted, plain = {}, {}
        for name in ("tfidf", "fabert", "student"):
            predicted[name] = predict_with_offsets(log[name], offsets_array(offsets[name][head], labels))
            plain[name] = log[name].argmax(axis=1)
        f = final[head]["for_test"]
        weight, final_offsets, threshold = (
            f["fabert_weight"],
            offsets_array(f["offsets"], labels),
            f["threshold"],
        )
        predicted[FINAL], _, unsure = final_answer(
            log["fabert"], log["tfidf"], weight, final_offsets, threshold
        )
        plain[FINAL], _, _ = final_answer(log["fabert"], log["tfidf"], weight, np.zeros(n), threshold)

        entry = {"reviews": int(rows.sum()), "models": {}, "differences": {}}
        for name in MODELS:
            entry["models"][name] = {
                "with_offsets": scores(y_true, np.array(labels)[predicted[name]], labels, head),
                "without_offsets": scores(y_true, np.array(labels)[plain[name]], labels, head),
            }
        answered = ~unsure
        entry["models"][FINAL]["uncertain"] = {
            "share": float(unsure.mean()),
            "answered_reviews": scores(
                y_true[answered], np.array(labels)[predicted[FINAL][answered]], labels, head
            ),
        }
        validation_recorded = {
            "tfidf": required[head]["tfidf"],
            "fabert": required[head]["fabert"],
            FINAL: required[head]["fabert + tfidf"],
            "student": student_report["heads"][head]["with_offsets"],
        }
        for name in MODELS:
            entry["models"][name]["validation"] = {
                k: validation_recorded[name][k] for k in ("accuracy", "macro_f1", "balanced_accuracy")
            }

        _, products = np.unique(product_index[rows], return_inverse=True)
        n_products = int(products.max()) + 1
        rng = np.random.default_rng(SEED)
        draws = {name: [] for name in MODELS}
        for _ in range(ROUNDS):
            w = np.bincount(rng.integers(0, n_products, n_products), minlength=n_products)[products].astype(
                float
            )
            for name in MODELS:
                draws[name].append(macro_f1(confusion(codes, predicted[name], w, n)))
        for name in MODELS:
            entry["models"][name]["macro_f1_interval_95_products"] = np.percentile(
                draws[name], [2.5, 97.5]
            ).tolist()
        for a, b in PAIRS:
            found = np.array(draws[a]) - np.array(draws[b])
            entry["differences"][f"{a} minus {b}"] = {
                "macro_f1_interval_95_products": np.percentile(found, [2.5, 97.5]).tolist(),
                "change": change(entry["models"][a]["with_offsets"], entry["models"][b]["with_offsets"]),
            }
        heads[head] = entry

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/final_evaluation_plan.md",
            "official": FINAL,
            "heads": heads,
            "scoring": {
                k: {"weights_sha256": v["weights_sha256"], "code": v["code"]} for k, v in scoring.items()
            },
            "tfidf_sha256": file_sha256(ARTIFACTS_DIR / "tfidf_final" / "model.joblib"),
            "data": data,
            "code": git_commit(),
        },
    )
    for head, e in heads.items():
        print(f"\n=== {head} ({e['reviews']:,} test reviews), with offsets")
        extra = "  ordinal" if head == "rating_band" else ""
        print(
            f"  {'model':<13} accuracy  macro F1 [95%, products]   macro P  macro R  balanced  "
            f"validation F1{extra}"
        )
        for name, m in e["models"].items():
            w = m["with_offsets"]
            low, high = m["macro_f1_interval_95_products"]
            line = (
                f"  {name:<13} {w['accuracy']:.4f}    {w['macro_f1']:.4f} [{low:.4f}, {high:.4f}]   "
                f"{w['macro_precision']:.4f}   {w['macro_recall']:.4f}   {w['balanced_accuracy']:.4f}    "
                f"{m['validation']['macro_f1']:.4f}"
            )
            if head == "rating_band":
                line += f"        {w['ordinal_distance_error']['mean_distance']:.4f}"
            print(line)
        u = e["models"][FINAL]["uncertain"]
        print(
            f"  final system: {u['share']:.1%} 'uncertain'; on the answered reviews accuracy "
            f"{u['answered_reviews']['accuracy']:.4f}, Macro F1 {u['answered_reviews']['macro_f1']:.4f}"
        )
        for pair, d in e["differences"].items():
            low, high = d["macro_f1_interval_95_products"]
            print(f"  {pair}: Macro F1 {d['change']['macro_f1']:+.4f} [{low:+.4f}, {high:+.4f}]")
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    main()
