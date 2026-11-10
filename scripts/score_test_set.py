"""Score the test texts with FaBERT or the semi-supervised student, on the GPU, for the final
evaluation (plan: reports/final_evaluation_plan.md).

Only the id and the normalized text of the test set are read, never a label: the labels are
read once, on your own machine, by scripts/final_evaluation.py. The weights come from the cache
folder, so molab does not need the artifacts:

  --model fabert    fabert_final.pt          (the kept run's model.pt)
  --model student   semi_student_last.pt     (the student's model_last.pt, end of its epoch)

The tokenizer and the configuration are those the model was trained with (hub commit
4cad2850ab81). Scoring is the same as for validation in scripts/train_transformer.py: 128
tokens, bfloat16 on the GPU, the model's log-probabilities of each head.

Outputs:
  artifacts/final_evaluation/<model>_test_predictions.parquet   id and log-probabilities
  reports/final_evaluation/<model>_scoring.json                 hash of the weights, data, code

Usage:
    python scripts/score_test_set.py --model fabert     # needs a GPU
    python scripts/score_test_set.py --model student
    python scripts/score_test_set.py --model fabert --smoke    # the first 2,000 texts only
"""

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_transformer import predict  # noqa: E402

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR  # noqa: E402
from digikala_reviews.finetune import build_model  # noqa: E402
from digikala_reviews.ingest import file_sha256  # noqa: E402
from digikala_reviews.provenance import check_cache, environment, git_commit, library_version  # noqa: E402
from digikala_reviews.tfidf import HEADS  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

WEIGHTS = {"fabert": "fabert_final.pt", "student": "semi_student_last.pt"}
HUB_COMMIT = "4cad2850ab81452041924fad29010ec0361322b6"
SMOKE_ROWS = 2_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(WEIGHTS), required=True)
    parser.add_argument("--smoke", action="store_true", help="the first 2,000 texts only")
    args = parser.parse_args()
    import torch
    from transformers import AutoConfig, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise SystemExit("No GPU found: turn the GPU on for this notebook, or run with --smoke")
    name = args.model + ("_smoke" if args.smoke else "")
    out_path = ARTIFACTS_DIR / "final_evaluation" / f"{name}_test_predictions.parquet"
    report_path = REPORTS_DIR / "final_evaluation" / f"{name}_scoring.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    weights_path = CACHE_DIR / WEIGHTS[args.model]
    saved = torch.load(weights_path, map_location=device, weights_only=False)
    settings = saved["settings"]
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=HUB_COMMIT)
    config = AutoConfig.from_pretrained(settings["model"], revision=HUB_COMMIT)
    model = build_model(torch, settings["model"], settings["dropout"], list(HEADS), config=config)
    model.load_state_dict({k: v.float() for k, v in saved["weights"].items()})
    model = model.to(device)
    del saved

    test = pd.read_parquet(CACHE_DIR / "test.parquet", columns=["id", "text"])  # no label is read
    if args.smoke:
        test = test.iloc[:SMOKE_ROWS]
    print(f"Scoring {len(test):,} test texts with {weights_path.name}", flush=True)
    start = time.perf_counter()
    texts = test["text"].to_numpy()
    encoded = []
    for i in range(0, len(texts), 200_000):
        chunk = list(texts[i : i + 200_000])
        encoded += tokenizer(chunk, truncation=True, max_length=settings["max_length"])["input_ids"]
    log_proba = predict(torch, model, tokenizer, encoded, device)
    minutes = round((time.perf_counter() - start) / 60, 1)

    table = pd.DataFrame({"id": test["id"].to_numpy()})
    for head, spec in HEADS.items():
        for i, label in enumerate(spec["labels"]):
            table[f"{head}__log_proba__{label}"] = log_proba[head][:, i].astype(np.float32)
    table.to_parquet(out_path.with_suffix(".tmp"), index=False)
    out_path.with_suffix(".tmp").replace(out_path)
    save_json(
        report_path,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": args.model,
            "weights": WEIGHTS[args.model],
            "weights_sha256": file_sha256(weights_path),
            "hub_commit": HUB_COMMIT,
            "max_length": settings["max_length"],
            "reviews": int(len(table)),
            "minutes": minutes,
            "predictions_sha256": file_sha256(out_path),
            "data": check_cache(CACHE_DIR, ["test.parquet"]),
            "environment": {
                **environment(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "transformers": library_version("transformers"),
            },
            "code": git_commit(),
        },
    )
    print(f"done in {minutes} min; saved {out_path} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
