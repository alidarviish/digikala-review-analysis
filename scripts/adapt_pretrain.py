"""Continue the masked-word pretraining of FaBERT on the text of the training reviews (needs a GPU).

Only the text is read: the reviews of train_supervised.parquet and the hidden reviews of
train_pseudo_unlabeled.parquet (normalized by scripts/cache_normalized.py). No label is read,
and the validation and test texts are not used. A fixed 1% of the texts is held out; the
masked-word loss on it is measured before and after, with the same masks both times.

The texts are tokenized as for fine-tuning (at most 128 tokens each, with their special
tokens), joined end to end and cut into blocks of 128 tokens. One pass over the blocks, 15%
of the tokens masked as in BERT, a new random mask for every batch.

Writes artifacts/fabert_adapted/ (the weights and the tokenizer, which from_pretrained reads;
scripts/pilot_transformers.py --adapted starts from them) and reports/domain_adaptation/pretrain.json.
The plan is in reports/domain_adaptation_plan.md.

Usage:
    python scripts/adapt_pretrain.py            # needs a GPU
    python scripts/adapt_pretrain.py --smoke    # a tiny model on a few texts, CPU is fine
"""

import argparse
import math
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.finetune import fixed_sample, linear_warmup
from digikala_reviews.ingest import file_sha256
from digikala_reviews.pretrain import mask_tokens, pack_blocks
from digikala_reviews.provenance import check_cache, environment, git_commit, library_version
from digikala_reviews.tuning import save_json

TEXT_FILES = ["train_supervised.parquet", "train_pseudo_unlabeled.parquet"]
SETTINGS = {
    "model": "sbunlp/fabert",
    "block_size": 128,
    "batch_size": 128,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_share": 0.06,
    "epochs": 1,
    "mask_share": 0.15,
    "held_out_share": 0.01,
    "max_held_out_blocks": 5_000,
    "seed": SPLIT_SEED,
}
SMOKE = {"model": "hf-internal-testing/tiny-random-BertModel", "texts": 3_000, "batch_size": 16}
LOG_EVERY = 200


def held_out_loss(torch, model, blocks, settings, tokenizer, device):
    """Mean masked-word loss on the held-out blocks, with masks fixed by the seed."""
    rng = np.random.default_rng(settings["seed"] + 1)
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(blocks), 256):
            inputs, labels = mask_tokens(
                blocks[start : start + 256],
                rng,
                tokenizer.mask_token_id,
                model.config.vocab_size,
                tokenizer.all_special_ids,
                settings["mask_share"],
            )
            labels = torch.from_numpy(labels).long().to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=torch.from_numpy(inputs).long().to(device)).logits
            chosen = labels != -100
            total += float(
                torch.nn.functional.cross_entropy(logits[chosen].float(), labels[chosen], reduction="sum")
            )
            count += int(chosen.sum())
    model.train()
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a tiny model on a few texts")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    settings = dict(SETTINGS)
    if args.smoke:
        settings.update({k: v for k, v in SMOKE.items() if k != "texts"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise SystemExit("No GPU found: turn the GPU on for this notebook, or run with --smoke")
    out = ARTIFACTS_DIR / ("fabert_adapted_smoke" if args.smoke else "fabert_adapted")
    report_dir = REPORTS_DIR / ("domain_adaptation_smoke" if args.smoke else "domain_adaptation")
    report = report_dir / "pretrain.json"
    data = check_cache(CACHE_DIR, TEXT_FILES)
    texts = pd.concat([pd.read_parquet(CACHE_DIR / f, columns=["text"])["text"] for f in TEXT_FILES])
    texts = texts.to_numpy()
    if args.smoke:
        texts = texts[fixed_sample(len(texts), SMOKE["texts"], settings["seed"])]
    held = np.zeros(len(texts), dtype=bool)
    held[fixed_sample(len(texts), round(settings["held_out_share"] * len(texts)), settings["seed"])] = True
    print(f"{len(texts):,} texts, {held.sum():,} held out, model {settings['model']}", flush=True)

    torch.manual_seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(settings["model"])
    model = AutoModelForMaskedLM.from_pretrained(settings["model"]).to(device)

    def blocks_of(rows):
        encoded = tokenizer(list(rows), truncation=True, max_length=settings["block_size"])["input_ids"]
        return pack_blocks(encoded, settings["block_size"])

    train_blocks, held_blocks = blocks_of(texts[~held]), blocks_of(texts[held])
    held_blocks = held_blocks[: settings["max_held_out_blocks"]]
    assert len(held_blocks) > 0 and len(train_blocks) > 0, "too few texts for one block"
    del texts
    print(f"{len(train_blocks):,} training blocks of {settings['block_size']} tokens", flush=True)
    before = held_out_loss(torch, model, held_blocks, settings, tokenizer, device)
    print(f"held-out masked-word loss before: {before:.4f}", flush=True)

    named = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for _, p in named if p.ndim >= 2], "weight_decay": settings["weight_decay"]},
            {"params": [p for _, p in named if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=settings["learning_rate"],
    )
    steps = settings["epochs"] * math.ceil(len(train_blocks) / settings["batch_size"])
    warmup = max(1, int(settings["warmup_share"] * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: linear_warmup(s, steps, warmup))
    rng = np.random.default_rng(settings["seed"])
    losses, running, step, start = [], [], 0, time.perf_counter()
    model.train()
    for _ in range(settings["epochs"]):
        order = rng.permutation(len(train_blocks))
        for first in range(0, len(order), settings["batch_size"]):
            inputs, labels = mask_tokens(
                train_blocks[order[first : first + settings["batch_size"]]],
                rng,
                tokenizer.mask_token_id,
                model.config.vocab_size,
                tokenizer.all_special_ids,
                settings["mask_share"],
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(
                    input_ids=torch.from_numpy(inputs).long().to(device),
                    labels=torch.from_numpy(labels).long().to(device),
                ).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running.append(float(loss.detach()))
            step += 1
            if step % LOG_EVERY == 0 or step == steps:
                minutes = (time.perf_counter() - start) / 60
                losses.append({"step": step, "mean_loss": float(np.mean(running))})
                print(
                    f"    step {step}/{steps}, loss {losses[-1]['mean_loss']:.4f}, "
                    f"{minutes:.1f} min, about {minutes * (steps - step) / step:.0f} min left",
                    flush=True,
                )
                running = []
    train_minutes = round((time.perf_counter() - start) / 60, 2)
    after = held_out_loss(torch, model, held_blocks, settings, tokenizer, device)
    print(f"held-out masked-word loss after: {after:.4f} (before {before:.4f})", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    save_json(
        report,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": settings,
            "texts": int(len(held)),
            "held_out_texts": int(held.sum()),
            "training_blocks": int(len(train_blocks)),
            "held_out_blocks": int(len(held_blocks)),
            "steps": steps,
            "held_out_loss_before": before,
            "held_out_loss_after": after,
            "losses": losses,
            "train_minutes": train_minutes,
            "weights_sha256": file_sha256(out / "model.safetensors"),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if device.type == "cuda"
            else None,
            "code": git_commit(),
            "environment": {
                **environment(),
                "torch": torch.__version__,
                "transformers": library_version("transformers"),
            },
            "data": data,
        },
    )
    print(f"saved {out}; {train_minutes} min", flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
