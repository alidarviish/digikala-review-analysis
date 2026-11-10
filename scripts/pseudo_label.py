"""Guess the hidden recommendation labels with the final FaBERT model (needs a GPU).

The final model's weights (artifacts/transformer_no_early_stop/model.pt on your own machine)
are put into the cache folder as fabert_final.pt, next to train_pseudo_unlabeled.parquet (the
hidden reviews, normalized by scripts/cache_normalized.py --files train_pseudo_unlabeled.csv).
Every hidden review is scored; the recommendation log probabilities are divided by the
temperature of reports/calibration.json, and the guess is the most likely class with its
probability. The true labels are not read here (scripts/check_pseudo_labels.py does that, on
your own machine).

Writes reports/pseudo_labels/masked.parquet (id, label, p) and reports/pseudo_labels/scoring.json
(with the hash of the weights used). The plan is in reports/semi_supervised_plan.md.

With --official (reports/semi_supervised_official_plan.md) the guess and its probability come
from the model's own softmax: no temperature, since the temperature was fitted on validation
labels and the project allows no validation label in making pseudo-labels. Writes
reports/semi_supervised/pseudo_labels.parquet (id, label, p) and reports/semi_supervised/teacher.json.

Two optional extensions (reports/extensions_plan.md), both with the model's own softmax as
--official: --band guesses the rating band of the same hidden reviews and writes
reports/semi_supervised/pseudo_labels_band.parquet and teacher_band.json; --natural guesses the
recommendation of the naturally unlabeled reviews (natural_unlabeled.parquet in the cache folder)
and writes pseudo_labels_natural.parquet and teacher_natural.json.

Usage:
    python scripts/pseudo_label.py            # needs a GPU
    python scripts/pseudo_label.py --smoke    # a tiny untrained model on a few reviews, CPU is fine
    python scripts/pseudo_label.py --official # needs a GPU
    python scripts/pseudo_label.py --band     # needs a GPU
    python scripts/pseudo_label.py --natural  # needs a GPU
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
from train_transformer import SMOKE, predict  # noqa: E402

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR
from digikala_reviews.finetune import build_model
from digikala_reviews.ingest import file_sha256
from digikala_reviews.provenance import check_cache, environment, git_commit, library_version
from digikala_reviews.pseudo import RECOMMENDATION_LABELS, guesses
from digikala_reviews.tfidf import HEADS
from digikala_reviews.tuning import save_json

MASKED = "train_pseudo_unlabeled.parquet"
NATURAL = "natural_unlabeled.parquet"
WEIGHTS = "fabert_final.pt"
SMOKE_ROWS = 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a tiny untrained model on a few reviews")
    parser.add_argument(
        "--official", action="store_true", help="the model's own probabilities, no temperature"
    )
    parser.add_argument("--band", action="store_true", help="the rating band of the hidden reviews")
    parser.add_argument("--natural", action="store_true", help="the naturally unlabeled reviews")
    args = parser.parse_args()
    assert not (args.band and args.natural), "one at a time"
    args.official = args.official or args.band or args.natural  # the model's own softmax
    head = "rating_band" if args.band else "recommendation"
    pool = NATURAL if args.natural else MASKED
    import torch
    from transformers import AutoTokenizer

    assert HEADS["recommendation"]["labels"] == RECOMMENDATION_LABELS, "label order"
    folder = REPORTS_DIR / ("semi_supervised" if args.official else "pseudo_labels")
    folder = folder.with_name(folder.name + "_smoke") if args.smoke else folder
    table_name, record_name = (
        ("pseudo_labels.parquet", "teacher.json") if args.official else ("masked.parquet", "scoring.json")
    )
    if args.band or args.natural:
        extra = "band" if args.band else "natural"
        table_name, record_name = f"pseudo_labels_{extra}.parquet", f"teacher_{extra}.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise SystemExit("No GPU found: turn the GPU on for this notebook, or run with --smoke")
    data = check_cache(CACHE_DIR, [pool])
    masked = pd.read_parquet(CACHE_DIR / pool, columns=["id", "text"])
    if args.official:
        temperature = 1.0  # the model's own softmax
    else:
        temperature = json.loads((REPORTS_DIR / "calibration.json").read_text(encoding="utf-8"))
        temperature = temperature["heads"]["recommendation"]["temperature_all_validation"]

    if args.smoke:
        masked = masked.iloc[:SMOKE_ROWS]
        name, weights_hash = SMOKE["model"], None
        model = build_model(torch, name, 0.1, list(HEADS)).to(device)
    else:
        weights_path = CACHE_DIR / WEIGHTS
        weights_hash = file_sha256(weights_path)
        saved = torch.load(weights_path, map_location=device, weights_only=False)
        name = saved["settings"]["model"]
        model = build_model(torch, name, saved["settings"]["dropout"], list(HEADS)).to(device)
        model.load_state_dict({k: v.float() for k, v in saved["weights"].items()})
    tokenizer = AutoTokenizer.from_pretrained(name)
    print(
        f"{len(masked):,} reviews of {pool}, {head}, model {name}, temperature {temperature:.3f}", flush=True
    )

    encoded = tokenizer(list(masked["text"]), truncation=True, max_length=128)["input_ids"]
    log_proba = predict(torch, model, tokenizer, encoded, device)[head]
    label, p = guesses(log_proba, temperature)

    folder.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "id": masked["id"].to_numpy(),
            "label": np.array(HEADS[head]["labels"])[label],
            "p": p.astype(np.float32),
        }
    )
    table.to_parquet(folder / table_name, index=False)
    save_json(
        folder / record_name,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "reviews": int(len(table)),
            "pool": pool,
            "head": head,
            "model": name,
            "weights_sha256": weights_hash,
            "temperature": temperature,
            "share_at_least_0.9": float((p >= 0.9).mean()),
            "guessed_classes": {k: float(v) for k, v in table["label"].value_counts(normalize=True).items()},
            "code": git_commit(),
            "environment": {
                **environment(),
                "torch": torch.__version__,
                "transformers": library_version("transformers"),
            },
            "data": data,
        },
    )
    print(f"saved {folder / table_name}; {(p >= 0.9).mean():.1%} reach 0.9", flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
