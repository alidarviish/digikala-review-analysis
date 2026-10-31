"""How much of the perfect-chooser room (step 71) is chance?

Three FaBERT runs that differ only in their seed (step 41: 500,000 training reviews, all of
validation) know the same things; where one of them is right and another wrong, that is
chance. The perfect-chooser bound is measured for them as in step 71 (labels with
cross-fitted offsets), for the three seeds and for each pair, and set next to the bound of
the three different final models (FaBERT, TookaBERT-Large, TF-IDF) and of the pair FaBERT and
TF-IDF, each as a gain over its own single model.

Writes reports/seed_oracle.json. Nothing is trained and the test set is not read. The plan and
the expected results are in reports/error_overlap_plan.md (step 72).

Usage:
    uv run python scripts/check_seed_oracle.py
"""

import json
from datetime import UTC, datetime

import numpy as np

from digikala_reviews.config import ARTIFACTS_DIR, REPORTS_DIR
from digikala_reviews.ensemble import error_overlap, oracle
from digikala_reviews.experiments import cross_fitted_offsets, product_half
from digikala_reviews.metrics import macro_f1_from_codes
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

SEEDS = (2026, 2027, 2028)
PILOT = ARTIFACTS_DIR / "transformer_pilot_whole"


def bound(codes, predictions, fallback, n_classes):
    return macro_f1_from_codes(codes, oracle(codes, predictions, fallback), n_classes)


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"]).set_index("id")
    runs = {s: np.load(PILOT / f"sbunlp__fabert__seed{s}__n500000__checks10.npz") for s in SEEDS}
    ids = runs[SEEDS[0]]["ids"]
    assert all((r["ids"] == ids).all() for r in runs.values()), "the seeds must score the same reviews"
    rows = validation.loc[ids]
    final = json.loads((REPORTS_DIR / "error_overlap.json").read_text(encoding="utf-8"))["heads"]
    result = {}
    for head, h in HEADS.items():
        labeled = rows[h["column"]].notna().to_numpy()
        n = len(h["labels"])
        codes = np.array([h["labels"].index(v) for v in rows[h["column"]].to_numpy()[labeled]])
        halves = product_half(rows["product_id"].to_numpy()[labeled])
        seeds = {str(s): cross_fitted_offsets(r[head][labeled], codes, halves)[0] for s, r in runs.items()}
        single = {s: macro_f1_from_codes(codes, p, n) for s, p in seeds.items()}
        mean_single = float(np.mean(list(single.values())))
        names = list(seeds)
        pair_bounds = {
            f"{a} & {b}": bound(codes, {a: seeds[a], b: seeds[b]}, a, n)
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        }
        three = bound(codes, seeds, names[0], n)
        f = final[head]
        result[head] = {
            "reviews": int(labeled.sum()),
            "seed_macro_f1": single,
            "seed_overlap": error_overlap(codes, seeds),
            "seeds_perfect_chooser": {
                "three": three,
                "three_gain_over_mean_seed": three - mean_single,
                "pairs": pair_bounds,
                "mean_pair_gain_over_mean_seed": float(np.mean(list(pair_bounds.values()))) - mean_single,
            },
            "final_models_perfect_chooser": {
                "three": f["perfect_chooser_macro_f1"],
                "three_gain_over_fabert": f["perfect_chooser_macro_f1"] - f["macro_f1"]["fabert"],
                "some_right": f["overlap"]["some_right"],
            },
        }
    save_json(
        REPORTS_DIR / "seed_oracle.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "heads": result,
            "code": git_commit(),
        },
    )
    for head, r in result.items():
        s, f = r["seeds_perfect_chooser"], r["final_models_perfect_chooser"]
        print(f"{head} ({r['reviews']:,} reviews):")
        print(
            "  seeds alone "
            + ", ".join(f"{v:.4f}" for v in r["seed_macro_f1"].values())
            + f"; some right {r['seed_overlap']['some_right']:.1%} (final models {f['some_right']:.1%})"
        )
        print(
            f"  perfect chooser, three seeds {s['three']:.4f} "
            f"(+{s['three_gain_over_mean_seed']:.4f} over a seed); "
            f"three final models {f['three']:.4f} (+{f['three_gain_over_fabert']:.4f} over FaBERT)"
        )
        print(f"  perfect chooser, two seeds: +{s['mean_pair_gain_over_mean_seed']:.4f} over a seed")


if __name__ == "__main__":
    main()
