"""Check the offline predictor against the saved validation predictions
(plan: reports/predictor_plan.md).

On 2,000 fixed validation reviews (seed 2026), from the raw body, the whole path of
src/digikala_reviews/predictor.py is run and compared with what was saved during training:

- normalization: equal to the text in the cache, on every review;
- TF-IDF: largest difference of a probability under 1e-6, and the same labels;
- FaBERT: the label without offsets the same on at least 99.5% of the reviews, and the mean
  absolute difference of the probabilities under 0.005 (the saved ones were made on the GPU
  with bfloat16, these on the CPU with float32, so they are close but not equal);
- the final system: its best guess the same on at least 99.5% (from the saved probabilities
  with the same weight, offsets and threshold).

Also: the share of reviews longer than 128 tokens, and the latency of each model, one review
at a time, on 200 of them. The test set is not read.

Writes reports/predictor_check.json and prints a summary.

Usage:
    uv run python scripts/check_predictor.py
"""

import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, FINAL_TRANSFORMER_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.finetune import fixed_sample
from digikala_reviews.normalize import normalize
from digikala_reviews.predictor import Predictor, final_answer
from digikala_reviews.provenance import check_cache, environment, git_commit
from digikala_reviews.standard_files import read_split_file
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "predictor_check.json"
SAMPLE, TIMED = 2_000, 200
RULES = {
    "tfidf_max_difference": 1e-6,
    "fabert_same_label": 0.995,
    "fabert_mean_difference": 0.005,
    "final_same_label": 0.995,
}


def main():
    predictor = Predictor()
    validation = read_split_file("validation.csv")[["id", "body"]]
    validation = validation.iloc[fixed_sample(len(validation), SAMPLE, SPLIT_SEED)].reset_index(drop=True)
    cache = load_cache("validation.parquet")[["id", "text"]]
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    table = validation.merge(cache, on="id", how="left", validate="one_to_one")
    for name, folder in [("tfidf", "tfidf_final"), ("fabert", FINAL_TRANSFORMER_DIR)]:
        saved = pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet")
        saved = saved.drop(columns=[c for c in saved if c != "id" and "__log_proba__" not in c])
        table = table.merge(saved.add_prefix(f"{name}:").rename(columns={f"{name}:id": "id"}), on="id")
    assert len(table) == SAMPLE, "a sampled review is missing from the saved predictions"

    texts = [normalize(b) for b in table["body"]]
    same_text = float(np.mean([a == b for a, b in zip(texts, table["text"], strict=True)]))
    assert same_text == 1.0, f"normalization differs from the cache on {1 - same_text:.2%} of the reviews"
    log_tfidf, _ = predictor.tfidf_scores(texts)
    start = time.perf_counter()
    log_fabert = predictor.fabert_scores(texts)
    fabert_seconds = time.perf_counter() - start
    lengths = np.array([len(ids) for ids in predictor.tokenizer(texts)["input_ids"]])

    heads = {}
    for head, spec in HEADS.items():
        labels = spec["labels"]
        saved = {
            name: table[[f"{name}:{head}__log_proba__{c}" for c in labels]].to_numpy(dtype=float)
            for name in ("tfidf", "fabert")
        }
        new = {"tfidf": log_tfidf[head], "fabert": log_fabert[head]}
        found = {}
        for name in ("tfidf", "fabert"):
            difference = np.abs(np.exp(new[name]) - np.exp(saved[name]))
            found[name] = {
                "same_label_without_offsets": float(np.mean(new[name].argmax(1) == saved[name].argmax(1))),
                "mean_probability_difference": float(difference.mean()),
                "max_probability_difference": float(difference.max()),
            }
        weight, offsets, threshold = predictor.final[head]
        old_codes, _, old_unsure = final_answer(saved["fabert"], saved["tfidf"], weight, offsets, threshold)
        new_codes, _, new_unsure = final_answer(new["fabert"], new["tfidf"], weight, offsets, threshold)
        found["final_system"] = {
            "same_best_guess": float(np.mean(old_codes == new_codes)),
            "same_uncertain": float(np.mean(old_unsure == new_unsure)),
            "uncertain_share": float(new_unsure.mean()),
        }
        found["passed"] = {
            "tfidf": found["tfidf"]["max_probability_difference"] < RULES["tfidf_max_difference"]
            and found["tfidf"]["same_label_without_offsets"] == 1.0,
            "fabert": found["fabert"]["same_label_without_offsets"] >= RULES["fabert_same_label"]
            and found["fabert"]["mean_probability_difference"] < RULES["fabert_mean_difference"],
            "final_system": found["final_system"]["same_best_guess"] >= RULES["final_same_label"],
        }
        heads[head] = found

    timings = {"normalize": [], "tfidf": [], "fabert": []}
    for body in table["body"].iloc[:TIMED]:
        for name, ms in predictor.predict(body)["latency_ms"].items():
            timings[name].append(ms)
    latency = {
        name: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))}
        for name, v in timings.items()
    }

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/predictor_plan.md",
            "reviews": SAMPLE,
            "rules": RULES,
            "same_normalized_text": same_text,
            "heads": heads,
            "passed": all(all(h["passed"].values()) for h in heads.values()),
            "tokens": {
                "median": float(np.median(lengths)),
                "share_over_128": float(np.mean(lengths > predictor.max_length)),
            },
            "fabert_batched_seconds": round(fabert_seconds, 1),
            "latency_one_review": latency,
            "version": predictor.version,
            "data": data,
            "environment": environment(),
            "code": git_commit(),
        },
    )
    print(f"normalized text equal to the cache: {same_text:.1%}")
    for head, e in heads.items():
        t, f, s = e["tfidf"], e["fabert"], e["final_system"]
        print(f"\n=== {head}")
        print(
            f"  TF-IDF: same label {t['same_label_without_offsets']:.2%}, "
            f"largest probability difference {t['max_probability_difference']:.1e}"
        )
        print(
            f"  FaBERT: same label {f['same_label_without_offsets']:.2%}, probability difference mean "
            f"{f['mean_probability_difference']:.4f}, largest {f['max_probability_difference']:.4f}"
        )
        print(
            f"  final system: same best guess {s['same_best_guess']:.2%}, same 'uncertain' "
            f"{s['same_uncertain']:.2%}, 'uncertain' {s['uncertain_share']:.1%}"
        )
        print("  passed: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in e["passed"].items()))
    print(
        f"\ntokens: median {np.median(lengths):.0f}, over 128: {np.mean(lengths > predictor.max_length):.1%}"
    )
    print(f"FaBERT on {SAMPLE:,} reviews in batches: {fabert_seconds:.0f}s")
    print(
        "latency, one review: "
        + ", ".join(
            f"{k} median {v['median_ms']:.1f} ms (95%: {v['p95_ms']:.1f})" for k, v in latency.items()
        )
    )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
