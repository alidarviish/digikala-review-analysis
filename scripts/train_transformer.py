"""Fine-tune the chosen Transformer (FaBERT, decision in reports/transformer_model_choice.md) on all
labeled training reviews, and score it on the whole validation set.

Same model and settings as the comparison runs (scripts/pilot_transformers.py), except:

- every labeled review of train_supervised, up to 4 epochs
- four checks per epoch on the fixed 50,000-review validation sample of the comparison runs,
  scored as the mean Macro F1 of the two heads with offsets (tuned on the other half of the
  sample, by product); the best check is kept, and training stops after two checks in a
  row without improvement
- the kept weights are saved in bfloat16, and everything after training is scored with
  exactly those saved weights
- a resume point is saved at every check, so an interrupted run continues where it was
  (the random state of dropout is not restored, so a resumed run is not bit-identical)

At the end the kept model scores every validation review. Labels with offsets are scored
(each review with offsets tuned on the other half of validation, by product) and compared
with the final TF-IDF model on the same reviews; the offsets for the test set are tuned on
all of validation. The test set is not read.

Outputs:
  artifacts/transformer_final/model.pt                   kept weights (bfloat16) and settings
  artifacts/transformer_final/validation_predictions.parquet
  reports/transformer_final/fabert.json                  checks, scores, comparison, offsets

With --no-early-stop every epoch is trained (the learning rate reaches 0 at the end) and
the best check is still the one kept. Its outputs get their own names, so the first run's
files stay as they are:
  artifacts/transformer_no_early_stop/   and   reports/transformer_final/fabert_no_early_stop.json

Every check records the validation loss of each head next to its Macro F1.

Usage:
    python scripts/train_transformer.py            # needs a GPU
    python scripts/train_transformer.py --no-early-stop
    python scripts/train_transformer.py --smoke    # a tiny model on a few rows, CPU is fine
    python scripts/train_transformer.py --smoke --stop-after 20   # then run again: tests resuming
"""

import argparse
import json
import math
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.experiments import compare, cross_fitted_offsets, product_half, search_offsets
from digikala_reviews.finetune import (
    IGNORE,
    batches,
    build_model,
    fixed_sample,
    label_codes,
    linear_warmup,
    mean_log_loss,
)
from digikala_reviews.metrics import bootstrap_macro_f1, classification_report, macro_f1_from_codes
from digikala_reviews.provenance import check_cache, environment, git_commit, library_version
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import finished, save_json

SETTINGS = {
    "model": "sbunlp/fabert",
    "learning_rate": 3e-5,
    "max_epochs": 4,
    "checks_per_epoch": 4,
    "patience": 2,
    "batch_size": 64,
    "max_length": 128,
    "weight_decay": 0.01,
    "warmup_share": 0.06,
    "dropout": 0.1,
    "selection_rows": 50_000,
    "train_rows": None,  # None = every labeled training review
    "seed": SPLIT_SEED,
}
SMOKE = {
    "model": "hf-internal-testing/tiny-random-BertModel",
    "learning_rate": 1e-3,
    "train_rows": 2_000,
    "selection_rows": 400,
    "max_epochs": 2,
}
SMOKE_VALIDATION_ROWS = 1_500  # the final scoring in a smoke run uses only this many reviews
CACHE_FILES = ["train_supervised.parquet", "validation.parquet"]
LOG_EVERY = 500


def predict(torch, model, tokenizer, encoded, device):
    """Log-probabilities of each head for already tokenized reviews, in their order."""
    model.eval()
    order = np.argsort([len(x) for x in encoded])  # similar lengths together, less padding
    parts = {head: [] for head in HEADS}
    with torch.no_grad():
        for batch in batches(torch, tokenizer, encoded, order, 512):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            for head, logits in outputs.items():
                parts[head].append(torch.log_softmax(logits.float(), dim=-1).cpu().numpy())
    back = np.argsort(order)
    return {head: np.vstack(p)[back] for head, p in parts.items()}


def head_targets(table, head):
    labels = HEADS[head]["labels"]
    labeled = table[HEADS[head]["column"]].notna().to_numpy()
    y_true = table[HEADS[head]["column"]].to_numpy()[labeled]
    codes = np.array([labels.index(v) for v in y_true])
    halves = product_half(table["product_id"].to_numpy()[labeled])
    return labeled, y_true, codes, halves


def selection_score(log_proba, selection):
    """Mean over the heads of Macro F1 with cross-fitted offsets, on the selection sample,
    and the validation loss of each head on the same reviews."""
    found, losses = {}, {}
    for head, (labeled, _, codes, halves) in selection.items():
        predicted, _ = cross_fitted_offsets(log_proba[head][labeled], codes, halves)
        found[head] = macro_f1_from_codes(codes, predicted, len(HEADS[head]["labels"]))
        losses[head] = mean_log_loss(log_proba[head][labeled], codes)
    return float(np.mean(list(found.values()))), found, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a tiny model on a few rows, to test the code")
    parser.add_argument(
        "--stop-after", type=int, help="end the session after this step's check (tests resuming)"
    )
    parser.add_argument(
        "--no-early-stop", action="store_true", help="train every epoch; keep the best check all the same"
    )
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer

    settings = {**SETTINGS, **(SMOKE if args.smoke else {})}
    if args.no_early_stop:
        settings["patience"] = None
    name = ("smoke" if args.smoke else "fabert") + ("_no_early_stop" if args.no_early_stop else "")
    out_dir = ARTIFACTS_DIR / ("transformer_final_smoke" if args.smoke else "transformer_final")
    if args.no_early_stop:
        out_dir = out_dir.with_name(out_dir.name.replace("final", "no_early_stop"))
    result_path = (
        REPORTS_DIR / ("transformer_final_smoke" if args.smoke else "transformer_final") / f"{name}.json"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise SystemExit("No GPU found: turn the GPU on for this notebook, or run with --smoke")
    context = {
        "code": git_commit(),
        "environment": {
            **environment(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "transformers": library_version("transformers"),
        },
        "data": check_cache(CACHE_DIR, CACHE_FILES),
    }
    spec = json.loads(json.dumps({"settings": settings, "data": context["data"]}))
    if finished(result_path, spec):
        print("Already done", flush=True)
        return
    print(f"Code {context['code']['commit']}, device {context['environment']['gpu'] or 'cpu'}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(settings["seed"])

    train = load_cache(CACHE_FILES[0])
    if settings["train_rows"]:
        train = train.iloc[fixed_sample(len(train), settings["train_rows"], SPLIT_SEED)].reset_index(
            drop=True
        )
    validation = load_cache(CACHE_FILES[1], extra_columns=["product_id"])
    labeled_any = validation["recommendation_status"].notna() | validation["rating_band"].notna()
    selection_table = validation[labeled_any]
    # the same 50,000 reviews as the comparison runs
    selection_table = selection_table.iloc[
        fixed_sample(len(selection_table), settings["selection_rows"], SPLIT_SEED)
    ].reset_index(drop=True)
    if args.smoke:
        validation = validation.iloc[:SMOKE_VALIDATION_ROWS].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(settings["model"])

    def encode(texts):
        start = time.perf_counter()
        ids = []
        for i in range(0, len(texts), 200_000):
            chunk = list(texts[i : i + 200_000])
            ids += tokenizer(chunk, truncation=True, max_length=settings["max_length"])["input_ids"]
        print(f"  tokenized {len(ids):,} reviews ({time.perf_counter() - start:.0f}s)", flush=True)
        return ids

    train_ids = encode(train["text"].to_numpy())
    selection_ids = encode(selection_table["text"].to_numpy())
    labels = {
        head: label_codes(train[HEADS[head]["column"]].to_numpy(), HEADS[head]["labels"]) for head in HEADS
    }
    selection = {head: head_targets(selection_table, head) for head in HEADS}

    model = build_model(torch, settings["model"], settings["dropout"], list(HEADS)).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]  # biases and norm weights
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=settings["learning_rate"],
    )
    steps_per_epoch = math.ceil(len(train_ids) / settings["batch_size"])
    total_steps = settings["max_epochs"] * steps_per_epoch
    warmup = max(1, int(settings["warmup_share"] * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: linear_warmup(s, total_steps, warmup))
    check_every = max(1, steps_per_epoch // settings["checks_per_epoch"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    use_bf16 = device.type == "cuda"

    resume_path, best_path = out_dir / "resume.pt", out_dir / "model.pt"
    state = {
        "step": 0,
        "best": -1.0,
        "best_step": None,
        "checks_without_gain": 0,
        "history": [],
        "spec": spec,
    }
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["state"]["spec"] == spec:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            state = saved["state"]
            print(f"Resumed at step {state['step']:,} (best {state['best']:.4f})", flush=True)
        del saved

    def save_atomic(obj, path):
        temporary = path.with_suffix(".tmp")
        torch.save(obj, temporary)
        temporary.replace(path)

    print(
        f"{len(train_ids):,} training reviews, {steps_per_epoch:,} steps per epoch, "
        f"check every {check_every:,}",
        flush=True,
    )

    # a run resumed after it already stopped early goes straight to the final scoring
    def patience_used_up():
        return settings["patience"] is not None and state["checks_without_gain"] >= settings["patience"]

    start, running, stop = time.perf_counter(), [], patience_used_up()
    while state["step"] < total_steps and not stop:
        epoch, done_in_epoch = divmod(state["step"], steps_per_epoch)
        order = np.random.default_rng(settings["seed"] + epoch).permutation(len(train_ids))
        order = order[done_in_epoch * settings["batch_size"] :]  # continue inside the epoch after a resume
        model.train()
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
            state["step"] += 1
            if state["step"] % LOG_EVERY == 0:
                minutes = (time.perf_counter() - start) / 60
                print(
                    f"    step {state['step']:,}/{total_steps:,}, loss {np.mean(running):.4f}, "
                    f"{minutes:.1f} min",
                    flush=True,
                )
                running = []
            if state["step"] % check_every == 0 or state["step"] == total_steps:
                score, per_head, losses = selection_score(
                    predict(torch, model, tokenizer, selection_ids, device), selection
                )
                model.train()
                improved = score > state["best"]
                state["history"].append(
                    {
                        "step": state["step"],
                        "epoch": round(state["step"] / steps_per_epoch, 2),
                        "score": score,
                        **{f"{h}_macro_f1": v for h, v in per_head.items()},
                        **{f"{h}_validation_loss": v for h, v in losses.items()},
                        "validation_loss_both_heads": sum(losses.values()),
                        "kept": improved,
                    }
                )
                if improved:
                    state.update(best=score, best_step=state["step"], checks_without_gain=0)
                    weights = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
                    save_atomic({"weights": weights, "settings": settings, "step": state["step"]}, best_path)
                else:
                    state["checks_without_gain"] += 1
                print(
                    f"  check at epoch {state['history'][-1]['epoch']}: {score:.4f}, "
                    f"validation loss {sum(losses.values()):.4f} ({'kept' if improved else 'not better'})",
                    flush=True,
                )
                stop = patience_used_up()
                save_atomic(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "state": state,
                    },
                    resume_path,
                )
                if stop:
                    print("  stopped early: no gain in the last checks", flush=True)
                    break
                if args.stop_after and state["step"] >= args.stop_after:
                    print(
                        f"Session ended at step {state['step']:,} as asked; run again to resume", flush=True
                    )
                    return
            if state["step"] >= total_steps:
                break
    train_minutes = round((time.perf_counter() - start) / 60, 1)

    # everything below uses exactly the saved weights
    saved = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict({k: v.float() for k, v in saved["weights"].items()})
    print(f"Scoring the whole validation set with the weights from step {saved['step']:,}", flush=True)
    log_proba = predict(torch, model, tokenizer, encode(validation["text"].to_numpy()), device)

    tfidf_path = ARTIFACTS_DIR / "tfidf_final" / "validation_predictions.parquet"
    tfidf = pd.read_parquet(tfidf_path).set_index("id").loc[validation["id"]] if tfidf_path.exists() else None
    table = pd.DataFrame({"id": validation["id"].to_numpy()})
    heads, offsets_for_test = {}, {}
    for head in HEADS:
        labels_ = HEADS[head]["labels"]
        labeled, y_true, codes, halves = head_targets(validation, head)
        crossed, per_half = cross_fitted_offsets(log_proba[head][labeled], codes, halves)
        with_offsets = np.array(labels_)[crossed]
        plain = np.array(labels_)[log_proba[head].argmax(axis=1)]
        offsets_for_test[head] = dict(
            zip(labels_, search_offsets(log_proba[head][labeled], codes).tolist(), strict=True)
        )
        heads[head] = {
            "without_offsets": classification_report(y_true, plain[labeled], labels_),
            "with_offsets": classification_report(y_true, with_offsets, labels_),
            "with_offsets_interval_95": bootstrap_macro_f1(y_true, with_offsets, labels_),
            "offsets_per_half": {k: dict(zip(labels_, v.tolist(), strict=True)) for k, v in per_half.items()},
            "offsets_for_test": offsets_for_test[head],
        }
        if tfidf is not None:
            tfidf_labels = tfidf[f"{head}__predicted_with_offsets"].to_numpy()[labeled].astype(str)
            heads[head]["vs_tfidf_with_offsets"] = compare(head, y_true, with_offsets, tfidf_labels)
        for i, label in enumerate(labels_):
            table[f"{head}__log_proba__{label}"] = log_proba[head][:, i].astype(np.float32)
        table[f"{head}__predicted"] = plain
        column = np.full(len(table), None, dtype=object)
        column[labeled] = with_offsets
        table[f"{head}__predicted_with_offsets"] = column
    predictions_path = out_dir / "validation_predictions.parquet"
    table.to_parquet(predictions_path.with_suffix(".tmp"), index=False)
    predictions_path.with_suffix(".tmp").replace(predictions_path)
    torch.save({**saved, "offsets_for_test": offsets_for_test}, best_path.with_suffix(".tmp"))
    best_path.with_suffix(".tmp").replace(best_path)

    save_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fingerprint": spec,
            "training_reviews": len(train_ids),
            "steps_per_epoch": steps_per_epoch,
            "kept_step": saved["step"],
            "kept_epoch": round(saved["step"] / steps_per_epoch, 2),
            "stopped_at_step": state["step"],
            "train_minutes_this_session": train_minutes,
            "checks": state["history"],
            "heads": heads,
            "validation_reviews": int(len(validation)),
            **context,
        },
    )
    resume_path.unlink(missing_ok=True)  # finished: the resume point is no longer needed
    for head, found in heads.items():
        line = (
            f"  {head}: without offsets {found['without_offsets']['macro_f1']:.4f}, "
            f"with offsets {found['with_offsets']['macro_f1']:.4f}"
        )
        if "vs_tfidf_with_offsets" in found:
            difference = found["vs_tfidf_with_offsets"]
            low, high = difference["difference_interval_95"]
            line += f", vs TF-IDF {difference['macro_f1_difference']:+.4f} [{low:+.4f}, {high:+.4f}]"
        print(line, flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
