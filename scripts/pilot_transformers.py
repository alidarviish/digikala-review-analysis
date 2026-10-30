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
    python scripts/pilot_transformers.py --models PartAI/TookaBERT-Large,sbunlp/fabert --seeds 2027,2028
    python scripts/pilot_transformers.py --models sbunlp/fabert --train-rows 500000

Other seeds or sample sizes are saved under their own names (<model>__seed<s>__n<rows>);
the validation sample is the same in every run.

With --whole-validation the same runs are scored on every labeled validation review instead of
the 50,000-review sample (training is unchanged), and saved apart, in
reports/transformer_pilot_whole/ and artifacts/transformer_pilot_whole/.

With --merge the training sample is merged by text before training (reports/merged_texts_plan.md):
one row per different text, trained against the share of each label among its copies, each
text weighted by its copies (--merge copies) or counted once (--merge once). The run is saved
under its own name (<...>__merge-<mode>).

With --checks N the validation loss is measured N times during training (every 1/N of the
training steps) on a fixed sample of 50,000 validation reviews, so two setups can be compared
while they train, also when their batches are built differently. The time of the checks is not
counted in the training minutes. The run is saved under its own name (<...>__checks<N>).

Two ways to give hard reviews more weight (reports/hard_examples_plan.md), each saved under its
own name: --focal GAMMA multiplies the loss of each review by (1 - p)^GAMMA, p the probability
of its true class (focal loss, <...>__focal<GAMMA>); --class-weights weights each class so that
all classes count as much in total in the training sample (<...>__classweights).

Two uses of the difficulty of each training review, measured out of fold by
scripts/score_difficulty.py (reports/difficulty_plan.md), each saved under its own name:
--leave-out-hardest F drops, for each head, the label of the fraction F of reviews with the
lowest probability of their label (<...>__leaveout<F>); --curriculum orders the epoch from easy
to hard reviews in 10 groups, random inside a group (<...>__curriculum); --gradual-curriculum
uses every review once too, but lets the harder reviews join step by step and mixes all of them
at the end (<...>__gradual).

Two settings of the optimizer (reports/learning_rate_plan.md), each saved under its own name:
--learning-rate LR replaces the model's usual learning rate (<...>__lr<LR>); --layer-decay D
gives the heads the usual learning rate and multiplies it by D for every layer further down,
to the embeddings (<...>__llrd<D>).
"""

import argparse
import json
import math
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.difficulty import curriculum_order, gradual_order, hardest, review_difficulty
from digikala_reviews.experiments import compare, cross_fitted_offsets, product_half
from digikala_reviews.finetune import (
    IGNORE,
    balanced_class_weights,
    batches,
    build_model,
    fixed_sample,
    focal_weights,
    label_codes,
    layer_learning_rates,
    linear_warmup,
    mean_log_loss,
)
from digikala_reviews.merge_texts import merge, soft_targets, weighted_soft_loss
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
CHECK_ROWS = 50_000  # validation reviews measured at every check (--checks)
CURRICULUM_GROUPS = 10
LOG_EVERY = 100


def safe_name(model_name):
    return model_name.replace("/", "__")


def train_and_score(name, learning_rate, settings, train, validation, device, torch):
    from transformers import AutoTokenizer

    torch.manual_seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(name)

    def encode(texts):
        return tokenizer(
            list(texts), add_special_tokens=True, truncation=True, max_length=settings["max_length"]
        )["input_ids"]

    merged = settings.get("merge")
    if merged:
        texts, _, counts = merge(
            train["text"], {head: (train[h["column"]], h["labels"]) for head, h in HEADS.items()}
        )
        train_texts, labels = list(texts), {}
        for head in HEADS:
            labels[head], labels[f"{head}__weight"] = soft_targets(counts[head], merged)
        print(f"  {len(train):,} training reviews merged into {len(train_texts):,} texts", flush=True)
    else:
        train_texts = train["text"]
        labels = {
            head: label_codes(train[HEADS[head]["column"]].to_numpy(), HEADS[head]["labels"])
            for head in HEADS
        }
    train_ids, validation_ids = encode(train_texts), encode(validation["text"])
    difficulty = None
    if settings.get("leave_out_hardest") or settings.get("curriculum") or settings.get("gradual_curriculum"):
        assert not merged, "the difficulty is per review, not per merged text"
        found = pd.read_parquet(REPORTS_DIR / settings["difficulty_file"]).set_index("id")
        p_true = {head: found[f"{head}_p_true"].reindex(train["id"]).to_numpy() for head in HEADS}
        for head in HEADS:  # every labeled review of the sample must have its difficulty
            assert not np.isnan(p_true[head][labels[head] != IGNORE]).any(), head
        if settings.get("leave_out_hardest"):
            for head in HEADS:
                dropped = hardest(p_true[head], settings["leave_out_hardest"])
                labels[head] = np.where(dropped, IGNORE, labels[head])
                print(f"  {head}: label left out for {dropped.sum():,} hardest reviews", flush=True)
        if settings.get("curriculum") or settings.get("gradual_curriculum"):
            difficulty = review_difficulty(p_true)

    model = build_model(torch, name, settings["dropout"], list(HEADS)).to(device)
    named = list(model.named_parameters())
    if settings.get("layer_decay"):
        # every parameter gets its own group: the learning rate falls layer by layer downwards
        layers = model.encoder.config.num_hidden_layers
        rates = layer_learning_rates([n for n, _ in named], layers, learning_rate, settings["layer_decay"])
        groups = [
            {"params": [p], "lr": rate, "weight_decay": settings["weight_decay"] if p.ndim >= 2 else 0.0}
            for (_, p), rate in zip(named, rates, strict=True)
        ]
    else:
        groups = [
            {"params": [p for _, p in named if p.ndim >= 2], "weight_decay": settings["weight_decay"]},
            {"params": [p for _, p in named if p.ndim < 2], "weight_decay": 0.0},  # biases and norms
        ]
    optimizer = torch.optim.AdamW(groups, lr=learning_rate)
    steps = settings["epochs"] * math.ceil(len(train_ids) / settings["batch_size"])
    warmup = max(1, int(settings["warmup_share"] * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: linear_warmup(s, steps, warmup))
    loss_fns = {head: torch.nn.CrossEntropyLoss(ignore_index=IGNORE) for head in HEADS}
    if settings.get("class_weights"):
        for head in HEADS:
            weight = balanced_class_weights(labels[head], len(HEADS[head]["labels"]))
            loss_fns[head] = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(weight, dtype=torch.float32, device=device), ignore_index=IGNORE
            )
            print(f"  {head} class weights: " + ", ".join(f"{w:.2f}" for w in weight), flush=True)
    gamma = settings.get("focal")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    use_bf16 = device.type == "cuda"

    def predict(rows):
        """Log-probabilities of the given validation rows, in their order."""
        model.eval()
        found = {head: [] for head in HEADS}
        with torch.no_grad():
            order = rows[np.argsort([len(validation_ids[i]) for i in rows])]  # less padding
            for batch in batches(torch, tokenizer, validation_ids, order, 256):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                for head, logits in outputs.items():
                    found[head].append(torch.log_softmax(logits.float(), dim=-1).cpu().numpy())
        position = {row: i for i, row in enumerate(order)}
        back = np.array([position[row] for row in rows])  # undo the sorting
        model.train()
        return {head: np.vstack(parts)[back] for head, parts in found.items()}

    check_at, checks, check_seconds = set(), [], 0.0
    if settings.get("checks"):
        check_rows = fixed_sample(len(validation), min(CHECK_ROWS, len(validation)), SPLIT_SEED)
        check_codes = {
            head: label_codes(validation[HEADS[head]["column"]].to_numpy()[check_rows], HEADS[head]["labels"])
            for head in HEADS
        }
        check_at = {round(k * steps / settings["checks"]) for k in range(1, settings["checks"] + 1)}

    losses, running, step, start = [], [], 0, time.perf_counter()
    model.train()
    for epoch in range(settings["epochs"]):
        order = np.random.default_rng(settings["seed"] + epoch).permutation(len(train_ids))
        if difficulty is not None and settings.get("gradual_curriculum"):
            order = gradual_order(difficulty, settings["seed"] + epoch)
        elif difficulty is not None:  # curriculum: easy to hard instead of random
            order = curriculum_order(difficulty, CURRICULUM_GROUPS, settings["seed"] + epoch)
        for batch in batches(torch, tokenizer, train_ids, order, settings["batch_size"], labels):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = 0.0
            for head, logits in outputs.items():
                target = batch["labels"][head].to(device)
                if merged:
                    weight = batch["labels"][f"{head}__weight"].to(device)
                    if (weight > 0).any():
                        log_proba = torch.log_softmax(logits.float(), dim=-1)
                        loss = loss + weighted_soft_loss(log_proba, target, weight)
                elif (target != IGNORE).any():
                    if gamma:
                        # focal loss: -(1 - p)^gamma log p of the true class, mean over labeled reviews
                        labeled = target != IGNORE
                        log_p = torch.log_softmax(logits.float()[labeled], dim=-1)
                        log_p = log_p.gather(1, target[labeled].unsqueeze(1)).squeeze(1)
                        loss = loss - (focal_weights(log_p.exp(), gamma) * log_p).mean()
                    else:
                        loss = loss + loss_fns[head](logits.float(), target)
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
            if step in check_at:
                check_start = time.perf_counter()
                found = predict(check_rows)
                check = {
                    "step": step,
                    "epoch": round(step / math.ceil(len(train_ids) / settings["batch_size"]), 4),
                }
                for head, codes in check_codes.items():
                    labeled = codes != IGNORE
                    check[f"{head}_validation_loss"] = mean_log_loss(found[head][labeled], codes[labeled])
                checks.append(check)
                check_seconds += time.perf_counter() - check_start
                print(
                    f"    check at step {step}: "
                    + ", ".join(f"{head} loss {check[f'{head}_validation_loss']:.4f}" for head in HEADS),
                    flush=True,
                )
    train_minutes = (time.perf_counter() - start - check_seconds) / 60

    log_proba = predict(np.arange(len(validation_ids)))
    model.eval()
    return {
        "log_proba": log_proba,
        "train_minutes": round(train_minutes, 2),
        "steps": steps,
        "train_texts": len(train_ids),
        "losses": losses,
        "checks": checks,
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
    parser.add_argument("--models", help="comma-separated model names (default: all candidates)")
    parser.add_argument("--seeds", help="comma-separated training seeds (default: 2026)")
    parser.add_argument("--train-rows", type=int, help="training sample size (default: 100,000)")
    parser.add_argument(
        "--whole-validation", action="store_true", help="score on every labeled validation review"
    )
    parser.add_argument(
        "--merge",
        choices=["copies", "once"],
        help="merge the training sample by text; weight each text by its copies or count it once",
    )
    parser.add_argument(
        "--checks", type=int, help="measure the validation loss this many times during training"
    )
    parser.add_argument("--focal", type=float, help="focal loss with this gamma")
    parser.add_argument("--class-weights", action="store_true", help="weight the classes to count the same")
    parser.add_argument(
        "--leave-out-hardest", type=float, help="drop the label of this fraction of the hardest reviews"
    )
    parser.add_argument("--curriculum", action="store_true", help="order the epoch from easy to hard")
    parser.add_argument(
        "--gradual-curriculum", action="store_true", help="let the harder reviews join step by step"
    )
    parser.add_argument("--learning-rate", type=float, help="instead of the model's usual learning rate")
    parser.add_argument(
        "--layer-decay", type=float, help="multiply the learning rate by this for every layer further down"
    )
    args = parser.parse_args()
    import torch

    settings = dict(SETTINGS)
    models = MODELS
    if args.models:
        models = {name: MODELS[name] for name in args.models.split(",")}
    if args.train_rows:
        settings["train_rows"] = args.train_rows
    if args.merge:
        settings["merge"] = args.merge  # only set when used, so the other runs keep their fingerprint
    if args.checks:
        settings["checks"] = args.checks  # the same
    if args.focal:
        settings["focal"] = args.focal
    if args.class_weights:
        settings["class_weights"] = True
    assert not (args.merge and (args.focal or args.class_weights)), "merging is not combined with these"
    assert not (args.focal and args.class_weights), "focal loss and class weights are tried apart"
    if args.leave_out_hardest:
        settings["leave_out_hardest"] = args.leave_out_hardest
    if args.curriculum:
        settings["curriculum"] = True
    if args.gradual_curriculum:
        settings["gradual_curriculum"] = True
    assert not (args.curriculum and args.gradual_curriculum), "one curriculum at a time"
    uses_difficulty = bool(args.leave_out_hardest or args.curriculum or args.gradual_curriculum)
    assert not (uses_difficulty and (args.merge or args.focal or args.class_weights)), "tried apart"
    if args.layer_decay:
        settings["layer_decay"] = args.layer_decay
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [SETTINGS["seed"]]
    results_dir, arrays_dir = REPORTS_DIR / "transformer_pilot", ARTIFACTS_DIR / "transformer_pilot"
    if args.whole_validation:
        settings["validation_rows"] = "all"
        results_dir, arrays_dir = (
            results_dir.with_name("transformer_pilot_whole"),
            arrays_dir.with_name("transformer_pilot_whole"),
        )
    if args.smoke:
        keep = {"models"} | ({"validation_rows"} if args.whole_validation else set())
        settings.update({k: v for k, v in SMOKE.items() if k not in keep})
        models = SMOKE["models"]
        results_dir, arrays_dir = (
            results_dir.with_name("transformer_pilot_smoke"),
            arrays_dir.with_name("transformer_pilot_smoke"),
        )
    if args.learning_rate:  # after the test-run models are chosen, so the name always fits the run
        models = {name: args.learning_rate for name in models}
    if uses_difficulty:  # made by scripts/score_difficulty.py for the same training sample
        folder = "difficulty_smoke" if args.smoke else "difficulty"
        settings["difficulty_file"] = f"{folder}/n{settings['train_rows']}.parquet"
        if not (REPORTS_DIR / settings["difficulty_file"]).exists():
            raise SystemExit(
                f"{settings['difficulty_file']} not found: run scripts/score_difficulty.py first"
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
    # the samples always use the split seed; a run's own seed only changes the training itself
    train = train_all.iloc[fixed_sample(len(train_all), settings["train_rows"], SPLIT_SEED)]
    train = train.reset_index(drop=True)
    del train_all
    validation_all = load_cache(CACHE_FILES[1], extra_columns=["product_id"])
    validation_all = validation_all[
        validation_all["recommendation_status"].notna() | validation_all["rating_band"].notna()
    ]
    if settings["validation_rows"] == "all":
        rows = np.arange(len(validation_all))
    else:
        rows = fixed_sample(len(validation_all), settings["validation_rows"], SPLIT_SEED)
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

    runs = [(seed, name, lr) for seed in seeds for name, lr in models.items()]
    for seed, name, learning_rate in runs:
        settings = {**settings, "seed": seed}
        tag = safe_name(name)
        if seed != SETTINGS["seed"] or settings["train_rows"] != SETTINGS["train_rows"]:
            tag += f"__seed{seed}__n{settings['train_rows']}"
        if settings.get("merge"):
            tag += f"__merge-{settings['merge']}"
        if settings.get("focal"):
            tag += f"__focal{settings['focal']:g}"
        if settings.get("class_weights"):
            tag += "__classweights"
        if settings.get("leave_out_hardest"):
            tag += f"__leaveout{settings['leave_out_hardest']:g}"
        if settings.get("curriculum"):
            tag += "__curriculum"
        if settings.get("gradual_curriculum"):
            tag += "__gradual"
        if args.learning_rate:
            tag += f"__lr{learning_rate:g}"
        if settings.get("layer_decay"):
            tag += f"__llrd{settings['layer_decay']:g}"
        if settings.get("checks"):
            tag += f"__checks{settings['checks']}"
        spec = json.loads(
            json.dumps(
                {"model": name, "learning_rate": learning_rate, "settings": settings, "data": context["data"]}
            )
        )
        path = results_dir / f"{tag}.json"
        if finished(path, spec):
            print(f"{tag}: already done", flush=True)
            continue
        print(f"{name} (lr {learning_rate}, seed {seed}, {settings['train_rows']:,} rows):", flush=True)
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
            arrays_dir / f"{tag}.npz",
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
