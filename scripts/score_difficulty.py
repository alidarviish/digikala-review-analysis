"""How hard each review of the pilot's training sample is, measured out of fold (needs a GPU).

The 500,000 training reviews of the pilot (the same sample as scripts/pilot_transformers.py)
are split in two halves by product. FaBERT is trained on one half with the pilot settings
(1 epoch, seed 2026) and scores the other half, then the other way round. For each review and
head this gives the probability that a model which never saw the review gives to its label;
reviews without a label for a head get NaN there. Only the review body is used; validation
and test are not read.

Writes reports/difficulty/n500000.parquet (id, fold, recommendation_p_true,
rating_band_p_true) and reports/difficulty/scoring.json. The plan is in
reports/difficulty_plan.md; scripts/check_difficulty.py summarizes the result.

Usage:
    python scripts/score_difficulty.py            # needs a GPU
    python scripts/score_difficulty.py --smoke    # a tiny model on a few rows, CPU is fine
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# the notebook may start Python without the script's folder on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_transformers import CACHE_FILES, MODELS, SETTINGS, SMOKE, train_and_score  # noqa: E402

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.experiments import product_half
from digikala_reviews.finetune import IGNORE, fixed_sample, label_codes
from digikala_reviews.provenance import check_cache, environment, git_commit, library_version
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import finished, save_json

MODEL = "sbunlp/fabert"
TRAIN_ROWS = 500_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a tiny model on a few rows, to test the code")
    args = parser.parse_args()
    import torch

    settings = {**SETTINGS, "train_rows": TRAIN_ROWS, "validation_rows": "the other half"}
    model, learning_rate, folder = MODEL, MODELS[MODEL], REPORTS_DIR / "difficulty"
    if args.smoke:
        settings["train_rows"] = SMOKE["train_rows"]
        (model, learning_rate), folder = next(iter(SMOKE["models"].items())), REPORTS_DIR / "difficulty_smoke"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise SystemExit("No GPU found: turn the GPU on for this notebook, or run with --smoke")
    data = check_cache(CACHE_DIR, CACHE_FILES)
    spec = json.loads(
        json.dumps({"model": model, "learning_rate": learning_rate, "settings": settings, "data": data})
    )
    result_path = folder / "scoring.json"
    if finished(result_path, spec):
        print("already done", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)

    train_all = load_cache(CACHE_FILES[0], extra_columns=["product_id"])
    # the same sample as the pilot runs
    sample = train_all.iloc[fixed_sample(len(train_all), settings["train_rows"], SPLIT_SEED)].reset_index(
        drop=True
    )
    del train_all
    half = product_half(sample["product_id"].to_numpy())
    print(
        f"{len(sample):,} training reviews, {half.sum():,} and {(~half).sum():,} in the two halves",
        flush=True,
    )

    p_true = {head: np.full(len(sample), np.nan, dtype=np.float32) for head in HEADS}
    fold = np.empty(len(sample), dtype=object)
    minutes = {}
    for name, trained_on in (("a", half), ("b", ~half)):
        scored = ~trained_on
        print(f"model {name}: trained on {trained_on.sum():,} reviews, scores {scored.sum():,}", flush=True)
        run = train_and_score(
            model,
            learning_rate,
            settings,
            sample[trained_on].reset_index(drop=True),
            sample[scored].reset_index(drop=True),
            device,
            torch,
        )
        minutes[name] = run["train_minutes"]
        rows = np.flatnonzero(scored)
        for head, h in HEADS.items():
            codes = label_codes(sample[h["column"]].to_numpy()[rows], h["labels"])
            labeled = codes != IGNORE
            p_true[head][rows[labeled]] = np.exp(run["log_proba"][head][labeled, codes[labeled]])
        fold[rows] = name  # the review was scored by the model trained on the other half
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame({"id": sample["id"].to_numpy(), "fold": fold})
    for head in HEADS:
        table[f"{head}_p_true"] = p_true[head]
    path = folder / f"n{settings['train_rows']}.parquet"
    table.to_parquet(path, index=False)
    save_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fingerprint": spec,
            "rows": int(len(table)),
            "labeled": {head: int(np.isfinite(p_true[head]).sum()) for head in HEADS},
            "mean_p_true": {head: float(np.nanmean(p_true[head])) for head in HEADS},
            "train_minutes": minutes,
            "file": path.relative_to(REPORTS_DIR).as_posix(),
            "code": git_commit(),
            "environment": {
                **environment(),
                "torch": torch.__version__,
                "transformers": library_version("transformers"),
            },
            "data": data,
        },
    )
    print(f"saved {path}", flush=True)
    for head in HEADS:
        print(f"  {head}: mean p of the true label {np.nanmean(p_true[head]):.3f}", flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
