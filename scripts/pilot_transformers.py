"""Fine-tune each candidate Transformer the same way on a small sample, to choose the main model.

Settings are in reports/transformer_model_choice.md (step 2) and are the same for every
model, except a smaller learning rate for the large one:

- 100,000 training reviews and 50,000 validation reviews, both fixed samples (seed 2026)
- one encoder, two linear heads (recommendation, rating band) on the first token; the loss
  is the sum of the two cross-entropies, and a review without a label for a head is
  skipped by that head
- full fine-tuning, 1 epoch, batch 64, AdamW (weight decay 0.01, not on biases and norms),
  linear warmup over 6% of the steps then linear decay, gradients clipped at 1, bf16 on
  the GPU, 128 tokens
- per-class decision offsets as for TF-IDF: each validation review is scored with offsets
  tuned on the other half of the sample (halves by product)

The final TF-IDF model is scored on the same 50,000 reviews, from its saved predictions,
and every model is compared with it (paired bootstrap interval); when those predictions are
not on this machine, the comparison is left for later. Each model's result is
saved as soon as it is ready in reports/transformer_pilot/, its validation probabilities in
artifacts/transformer_pilot/, and finished models are skipped. The test set is not read.

Usage:
    python scripts/pilot_transformers.py            # all candidates, needs a GPU
    python scripts/pilot_transformers.py --smoke    # a tiny model on a few rows, CPU is fine
"""

import argparse
import json
import math
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.experiments import compare, cross_fitted_offsets, product_half
from digikala_reviews.finetune import IGNORE, fixed_sample, label_codes, linear_warmup
from digikala_reviews.metrics import bootstrap_macro_f1, classification_report
from digikala_reviews.provenance import check_cache, environment, git_commit, library_version
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import finished, save_json

MODELS = {
    "PartAI/TookaBERT-Base": 3e-5,
    "sbunlp/fabert": 3e-5,
    "FacebookAI/xlm-roberta-base": 3e-5,
    "jhu-clsp/mmBERT-base": 3e-5,
    "HooshvareLab/bert-fa-zwnj-base": 3e-5,
    "PartAI/TookaBERT-Large": 2e-5,
}
SETTINGS = {
    "train_rows": 100_000,
    "validation_rows": 50_000,
    "epochs": 1,
    "batch_size": 64,
    "max_length": 128,
    "weight_decay": 0.01,
    "warmup_share": 0.06,
    "dropout": 0.1,
    "seed": SPLIT_SEED,
}
SMOKE = {
    "models": {"hf-internal-testing/tiny-random-BertModel": 1e-3},
    "train_rows": 600,
    "validation_rows": 400,
}
CACHE_FILES = ["train_supervised.parquet", "validation.parquet"]
LOG_EVERY = 100


def safe_name(model_name):
    return model_name.replace("/", "__")


def build_model(torch, name, dropout):
    from transformers import AutoModel

    class TwoHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name)
            size = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(dropout)
            self.heads = torch.nn.ModuleDict({head: torch.nn.Linear(size, 3) for head in HEADS})

        def forward(self, input_ids, attention_mask):
            first = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            first = self.dropout(first)
            return {head: layer(first) for head, layer in self.heads.items()}

    return TwoHeads()


def batches(torch, tokenizer, encoded, order, size, labels=None):
    for start in range(0, len(order), size):
        rows = order[start : start + size]
        padded = tokenizer.pad({"input_ids": [encoded[i] for i in rows]}, return_tensors="pt")
        batch = {"input_ids": padded["input_ids"], "attention_mask": padded["attention_mask"]}
        if labels is not None:
            batch["labels"] = {head: torch.as_tensor(codes[rows]) for head, codes in labels.items()}
        yield batch


def train_and_score(name, learning_rate, settings, train, validation, device, torch):
    from transformers import AutoTokenizer

    torch.manual_seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(name)

    def encode(texts):
        return tokenizer(
            list(texts), add_special_tokens=True, truncation=True, max_length=settings["max_length"]
        )["input_ids"]

    train_ids, validation_ids = encode(train["text"]), encode(validation["text"])
    labels = {
        head: label_codes(train[HEADS[head]["column"]].to_numpy(), HEADS[head]["labels"]) for head in HEADS
    }

    model = build_model(torch, name, settings["dropout"]).to(device)
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]  # biases and norm weights
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    steps = settings["epochs"] * math.ceil(len(train_ids) / settings["batch_size"])
    warmup = max(1, int(settings["warmup_share"] * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: linear_warmup(s, steps, warmup))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    use_bf16 = device.type == "cuda"
    if use_bf16:
        torch.cuda.reset_peak_memory_stats()

    losses, running, step, start = [], [], 0, time.perf_counter()
    model.train()
    for epoch in range(settings["epochs"]):
        order = np.random.default_rng(settings["seed"] + epoch).permutation(len(train_ids))
        for batch in batches(torch, tokenizer, train_ids, order, settings["batch_size"], labels):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = 0.0
            for head, logits in outputs.items():
                target = batch["labels"][head].to(device)
                if (target != IGNORE).any():
                    loss = loss + loss_fn(logits.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running.append(float(loss.detach()))
            step += 1
            if step % LOG_EVERY == 0 or step == steps:
                losses.append({"step": step, "mean_loss": float(np.mean(running))})
                print(f"    step {step}/{steps}, loss {losses[-1]['mean_loss']:.4f}", flush=True)
                running = []
    train_minutes = (time.perf_counter() - start) / 60

    model.eval()
    log_proba = {head: [] for head in HEADS}
    with torch.no_grad():
        order = np.argsort([len(x) for x in validation_ids])  # similar lengths together, less padding
        for batch in batches(torch, tokenizer, validation_ids, order, 256):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            for head, logits in outputs.items():
                log_proba[head].append(torch.log_softmax(logits.float(), dim=-1).cpu().numpy())
    back = np.argsort(order)  # undo the sorting
    log_proba = {head: np.vstack(parts)[back] for head, parts in log_proba.items()}
    return {
        "log_proba": log_proba,
        "train_minutes": round(train_minutes, 2),
        "steps": steps,
        "losses": losses,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if use_bf16 else None,
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }


def scores(head, log_proba, y_true, half_a, tfidf_with_offsets):
    labels = HEADS[head]["labels"]
    code = {label: i for i, label in enumerate(labels)}
    true_codes = np.array([code[v] for v in y_true])
    plain = np.array(labels)[log_proba.argmax(axis=1)]
    crossed, per_half = cross_fitted_offsets(log_proba, true_codes, half_a)
    with_offsets = np.array(labels)[crossed]
    result = {
        "without_offsets": classification_report(y_true, plain, labels),
        "with_offsets": classification_report(y_true, with_offsets, labels),
        "with_offsets_interval_95": bootstrap_macro_f1(y_true, with_offsets, labels),
        "offsets_per_half": {k: dict(zip(labels, v.tolist(), strict=True)) for k, v in per_half.items()},
    }
    if tfidf_with_offsets is not None:
        result["vs_tfidf_with_offsets"] = compare(head, y_true, with_offsets, tfidf_with_offsets)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a tiny model on a few rows, to test the code")
    args = parser.parse_args()
    import torch

    settings = dict(SETTINGS)
    models = MODELS
    results_dir, arrays_dir = REPORTS_DIR / "transformer_pilot", ARTIFACTS_DIR / "transformer_pilot"
    if args.smoke:
        settings.update({k: v for k, v in SMOKE.items() if k != "models"})
        models = SMOKE["models"]
        results_dir, arrays_dir = (
            results_dir.with_name("transformer_pilot_smoke"),
            arrays_dir.with_name("transformer_pilot_smoke"),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise SystemExit("No GPU found: turn the GPU on for this notebook, or run with --smoke")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    if device.type == "cuda":
        (
            torch.ones(8, 8, device=device) @ torch.ones(8, 8, device=device)
        ).sum().item()  # fails early on a bad build
    context = {
        "code": git_commit(),
        "environment": {
            **environment(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu,
            "transformers": library_version("transformers"),
        },
        "data": check_cache(CACHE_DIR, CACHE_FILES),
    }
    print(f"Code {context['code']['commit']}, device {gpu or 'cpu'}, torch {torch.__version__}", flush=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    train_all = load_cache(CACHE_FILES[0])
    train = train_all.iloc[fixed_sample(len(train_all), settings["train_rows"], settings["seed"])]
    train = train.reset_index(drop=True)
    del train_all
    validation_all = load_cache(CACHE_FILES[1], extra_columns=["product_id"])
    validation_all = validation_all[
        validation_all["recommendation_status"].notna() | validation_all["rating_band"].notna()
    ]
    rows = fixed_sample(len(validation_all), settings["validation_rows"], settings["seed"])
    validation = validation_all.iloc[rows].reset_index(drop=True)
    del validation_all

    # the comparison with TF-IDF needs its saved predictions; without them it can be made later
    # from the probabilities saved here
    tfidf_path = ARTIFACTS_DIR / "tfidf_final" / "validation_predictions.parquet"
    tfidf = pd.read_parquet(tfidf_path).set_index("id").loc[validation["id"]] if tfidf_path.exists() else None
    if tfidf is None:
        print(f"{tfidf_path} not found: TF-IDF is not compared in this run", flush=True)
    per_head = {}
    for head in HEADS:
        labeled = validation[HEADS[head]["column"]].notna().to_numpy()
        y_true = validation[HEADS[head]["column"]].to_numpy()[labeled]
        info = {
            "labeled": labeled,
            "y_true": y_true,
            "half_a": product_half(validation["product_id"].to_numpy()[labeled]),
            "tfidf_with_offsets": None,
        }
        if tfidf is not None:
            info["tfidf_with_offsets"] = (
                tfidf[f"{head}__predicted_with_offsets"].to_numpy()[labeled].astype(str)
            )
            tfidf_f1 = classification_report(y_true, info["tfidf_with_offsets"], HEADS[head]["labels"])[
                "macro_f1"
            ]
            info["tfidf_macro_f1"] = tfidf_f1
            print(f"TF-IDF on the same {labeled.sum():,} {head} reviews: {tfidf_f1:.4f}", flush=True)
        per_head[head] = info

    for name, learning_rate in models.items():
        spec = json.loads(
            json.dumps(
                {"model": name, "learning_rate": learning_rate, "settings": settings, "data": context["data"]}
            )
        )
        path = results_dir / f"{safe_name(name)}.json"
        if finished(path, spec):
            print(f"{name}: already done", flush=True)
            continue
        print(f"{name} (lr {learning_rate}):", flush=True)
        try:
            run = train_and_score(name, learning_rate, settings, train, validation, device, torch)
        except Exception as error:  # one model failing (no access, out of memory) should not stop the rest
            print(f"  failed: {type(error).__name__}: {error}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        result = {
            "model": name,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fingerprint": spec,
            **{k: v for k, v in run.items() if k != "log_proba"},
            "heads": {},
            **context,
        }
        for head, info in per_head.items():
            result["heads"][head] = scores(
                head,
                run["log_proba"][head][info["labeled"]],
                info["y_true"],
                info["half_a"],
                info["tfidf_with_offsets"],
            )
            if "tfidf_macro_f1" in info:
                result["heads"][head]["tfidf_with_offsets_macro_f1"] = info["tfidf_macro_f1"]
        np.savez_compressed(
            arrays_dir / f"{safe_name(name)}.npz",
            ids=validation["id"].to_numpy(),
            **{head: run["log_proba"][head].astype(np.float32) for head in HEADS},
        )
        save_json(path, result)
        for head, found in result["heads"].items():
            line = (
                f"  {head}: without offsets {found['without_offsets']['macro_f1']:.4f}, "
                f"with offsets {found['with_offsets']['macro_f1']:.4f}"
            )
            if "vs_tfidf_with_offsets" in found:
                difference = found["vs_tfidf_with_offsets"]
                low, high = difference["difference_interval_95"]
                line += f", vs TF-IDF {difference['macro_f1_difference']:+.4f} [{low:+.4f}, {high:+.4f}]"
            print(line, flush=True)
        print(f"  {run['train_minutes']} min, peak GPU memory {run['peak_gpu_memory_gb']} GB", flush=True)
        del run
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("Done", flush=True)


if __name__ == "__main__":
    main()
