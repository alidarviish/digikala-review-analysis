"""Look at the final TF-IDF model from several sides, without fitting anything.

Reads the saved model and validation predictions (scripts/fit_final_tfidf.py) and the
local data. Each analysis is written to reports/tfidf_analysis/<name>.json as soon as it
is done, and skipped when its file already exists (use --redo to run them all again).
The test set is not read.

  ceiling       best possible scores on validation texts that appear more than once
  length        scores by review length in words
  sample        how much Macro F1 moves on random validation samples of 20,000 and 50,000
  labels        recommendation against rating band in the training labels
  words         the n-grams with the largest weight for each class
  seen          scores on texts seen word for word in training, and on new texts
  category      scores by product category
  calibration   whether the model's confidence matches how often it is right
  unlabeled     how many unlabeled reviews the model is confident about; on the masked
                training reviews also how often it is right (labels from
                data/splits/audit, used only for this check)

Scores use the labels with offsets, except calibration and the unlabeled pools, which
use the model's own probabilities (offsets move the decision, not the probabilities).

Usage:
    uv run python scripts/analyze_tfidf.py [--redo]
"""

import argparse
import json
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from digikala_reviews.analysis import (
    calibration,
    confident_share,
    label_ceiling,
    length_bucket,
    sample_spread,
    scores_by_group,
    word_count,
)
from digikala_reviews.config import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    PRODUCTS_RAW_DIR,
    PSEUDO_LABEL_THRESHOLD,
    REPORTS_DIR,
)
from digikala_reviews.metrics import classification_report
from digikala_reviews.normalize import normalize
from digikala_reviews.provenance import git_commit
from digikala_reviews.resources import require_free_memory
from digikala_reviews.standard_files import read_split_file
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

MODEL_DIR = ARTIFACTS_DIR / "tfidf_final"
OUT_DIR = REPORTS_DIR / "tfidf_analysis"
THRESHOLDS = [0.8, PSEUDO_LABEL_THRESHOLD, 0.95]
TOP_WORDS = 25
SAMPLE_SIZES = [20_000, 50_000]  # validation samples a Transformer run might score each epoch
CHUNK_ROWS = 100_000


def probabilities(table, head, labels):
    log_proba = table[[f"{head}__log_proba__{label}" for label in labels]].to_numpy()
    proba = np.exp(log_proba)
    return proba.max(axis=1), np.array(labels)[proba.argmax(axis=1)]


def product_categories():
    """Most frequent category of each product; ties broken alphabetically (as in build_splits.py)."""
    products = pd.read_parquet(PRODUCTS_RAW_DIR, columns=["id", "sub_category"])
    pairs = products.value_counts(["id", "sub_category"]).reset_index()
    pairs = pairs.sort_values(["id", "count", "sub_category"], ascending=[True, False, True])
    return pairs.drop_duplicates("id").set_index("id")["sub_category"]


def predict_pool(bundle, bodies):
    """Confidence and predicted label of each head for raw review bodies, normalized as in training."""
    vectorizer, models = bundle["vectorizer"], bundle["models"]
    parts = {head: [] for head in models}
    for i in range(0, len(bodies), CHUNK_ROWS):
        x = vectorizer.transform([normalize(t) for t in bodies[i : i + CHUNK_ROWS]])
        for head, model in models.items():
            parts[head].append(model.predict_proba(x))
    result = {}
    for head, model in models.items():
        proba = np.vstack(parts[head])
        result[head] = (proba.max(axis=1), np.asarray(model.classes_)[proba.argmax(axis=1)])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", action="store_true", help="run every analysis again")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((REPORTS_DIR / "tfidf_final" / "chosen_model.json").read_text(encoding="utf-8"))
    context = {"code": git_commit(), "model_files": summary["files"]}

    def run(name, compute):
        path = OUT_DIR / f"{name}.json"
        if path.exists() and not args.redo:
            print(f"{name}: already done", flush=True)
            return
        start = time.perf_counter()
        result = compute()
        save_json(path, {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **result, **context})
        print(f"{name}: done ({time.perf_counter() - start:.0f}s)", flush=True)

    require_free_memory(2.0, "reading validation")
    validation = load_cache("validation.parquet", extra_columns=["product_id", "text_seen_in_train"])
    predictions = pd.read_parquet(MODEL_DIR / "validation_predictions.parquet")
    assert (predictions["id"].to_numpy() == validation["id"].to_numpy()).all(), (
        "rows are not in the same order"
    )

    def labeled(head):
        column = HEADS[head]["column"]
        rows = validation[column].notna().to_numpy()
        return rows, validation[column].to_numpy()[rows], predictions[f"{head}__predicted_with_offsets"][rows]

    def ceiling():
        result = {}
        for head in HEADS:
            rows, y_true, y_pred = labeled(head)
            found = label_ceiling(validation["text"].to_numpy()[rows], y_true, HEADS[head]["labels"])
            mask = found.pop("rows_mask")
            model = classification_report(y_true[mask], y_pred.to_numpy()[mask], HEADS[head]["labels"])
            found["model_on_same_rows"] = {
                "macro_f1": model["macro_f1"],
                "recall": {k: v["recall"] for k, v in model["per_class"].items()},
            }
            result[head] = found
        return result

    def length():
        buckets = length_bucket(word_count(validation["text"]))
        result = {}
        for head in HEADS:
            rows, y_true, y_pred = labeled(head)
            result[head] = scores_by_group(y_true, y_pred, buckets[rows], HEADS[head]["labels"])
        return result

    def sample():
        result = {}
        for head in HEADS:
            _, y_true, y_pred = labeled(head)
            result[head] = sample_spread(y_true, y_pred.to_numpy(), HEADS[head]["labels"], SAMPLE_SIZES)
        return result

    def labels():
        train = pd.read_parquet(
            CACHE_DIR / "train_supervised.parquet", columns=["recommendation_status", "rating_band"]
        )
        table = pd.crosstab(
            train["recommendation_status"].fillna("missing"), train["rating_band"].fillna("missing")
        )
        shares = table.div(table.sum(axis=1), axis=0)
        return {"counts": table.to_dict(orient="index"), "row_shares": shares.to_dict(orient="index")}

    def words():
        bundle = joblib.load(MODEL_DIR / "model.joblib")
        names = bundle["vectorizer"].get_feature_names_out()
        result = {}
        for head, model in bundle["models"].items():
            result[head] = {}
            for i, label in enumerate(model.classes_):
                order = np.argsort(model.coef_[i])[::-1][:TOP_WORDS]
                result[head][str(label)] = [
                    [str(names[j]), round(float(model.coef_[i][j]), 3)] for j in order
                ]
        return result

    def seen():
        groups = np.where(validation["text_seen_in_train"].to_numpy(), "seen_in_train", "new")
        result = {}
        for head in HEADS:
            rows, y_true, y_pred = labeled(head)
            result[head] = scores_by_group(y_true, y_pred, groups[rows], HEADS[head]["labels"])
        return result

    def category():
        groups = validation["product_id"].map(product_categories()).fillna("unknown").to_numpy()
        result = {}
        for head in HEADS:
            rows, y_true, y_pred = labeled(head)
            result[head] = scores_by_group(y_true, y_pred, groups[rows], HEADS[head]["labels"], min_rows=1000)
        return result

    def calibration_check():
        result = {}
        for head in HEADS:
            rows, y_true, _ = labeled(head)
            confidence, predicted = probabilities(predictions[rows], head, HEADS[head]["labels"])
            correct = predicted == y_true
            result[head] = {
                **calibration(confidence, correct),
                "confident": confident_share(confidence, predicted, THRESHOLDS, correct),
            }
        return result

    def unlabeled():
        bundle = joblib.load(MODEL_DIR / "model.joblib")
        result = {"thresholds": THRESHOLDS}
        require_free_memory(2.5, "predicting the masked training reviews")
        masked = read_split_file("train_pseudo_unlabeled.csv")
        truth = read_split_file("audit/masked_labels_reference.csv").set_index("id").loc[masked["id"]]
        result["masked_training_reviews"] = {}
        pool = predict_pool(bundle, masked["body"].tolist())
        for head in HEADS:
            confidence, predicted = pool[head]
            true = truth[HEADS[head]["column"]].to_numpy()
            has_label = pd.notna(true)
            result["masked_training_reviews"][head] = {
                "rows_with_label": int(has_label.sum()),
                "true_classes": {
                    str(k): float(v)
                    for k, v in pd.Series(true[has_label]).value_counts(normalize=True).items()
                },
                "confident": confident_share(
                    confidence[has_label],
                    predicted[has_label],
                    THRESHOLDS,
                    predicted[has_label] == true[has_label],
                ),
            }
        del masked, truth
        require_free_memory(2.5, "predicting the reviews without a recommendation")
        natural = read_split_file("natural_unlabeled.csv")
        natural_bundle = {**bundle, "models": {"recommendation": bundle["models"]["recommendation"]}}
        confidence, predicted = predict_pool(natural_bundle, natural["body"].tolist())["recommendation"]
        result["no_recommendation_label"] = {
            "rows": int(len(natural)),
            "confident": confident_share(confidence, predicted, THRESHOLDS),
        }
        del natural
        band_pool = pd.read_parquet(MODEL_DIR / "train_unlabeled_predictions.parquet")
        band_pool = band_pool[band_pool["head"] == "rating_band"]
        log_proba = band_pool[[f"log_proba__{c}" for c in HEADS["rating_band"]["labels"]]].to_numpy()
        proba = np.exp(log_proba)
        result["no_rating_band_label"] = {
            "rows": int(len(band_pool)),
            "note": "training reviews with a recommendation but rate 0 (no rating recorded)",
            "confident": confident_share(
                proba.max(axis=1), np.array(HEADS["rating_band"]["labels"])[proba.argmax(axis=1)], THRESHOLDS
            ),
        }
        return result

    for name, compute in [
        ("ceiling", ceiling),
        ("length", length),
        ("sample", sample),
        ("labels", labels),
        ("words", words),
        ("seen", seen),
        ("category", category),
        ("calibration", calibration_check),
        ("unlabeled", unlabeled),
    ]:
        run(name, compute)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
