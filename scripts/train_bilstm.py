"""A BiLSTM trained from scratch on the review text, with the same two heads as FaBERT
(plan: reports/bilstm_plan.md).

Words come from the project's `persian` token pattern (src/digikala_reviews/tfidf.py), so
emoji and punctuation are tokens; Latin letters are lowercased. The vocabulary is every token
seen at least 5 times in train_supervised, at most 200,000 (id 0 padding, id 1 any other
token); a review keeps its first 128 tokens. Network: embeddings of size 300 trained from
scratch, a bidirectional LSTM, the mean and the maximum of its outputs over the real tokens
side by side, dropout, and one linear layer per head. Loss: the sum of the two cross-entropies
(a review without a label for a head is skipped for that head), AdamW with weight decay 0.01,
6% warmup then linear decay, batch 256, gradient norm clipped at 1.0. Scores are Macro F1 with
per-class offsets tuned on the other half of the reviews, by product, as for FaBERT.

Two parts, run in this order:

--tune   the 8 settings of the plan (hidden size 256 or 512, 1 or 2 layers, learning rate
         1e-3 or 2e-3; dropout 0.3), each on the same fixed 500,000 training reviews for 2
         epochs, scored at the end of each epoch on the fixed 50,000-review validation sample
         of every FaBERT check; a setting's score is its better epoch. Each finished setting
         is saved at once, so a stopped run goes on with the next one.
         Writes reports/bilstm/tuning/<setting>.json and reports/bilstm/tuning.json.
(none)   the best setting of tuning.json on all training reviews, up to 4 epochs, four checks
         per epoch on the same 50,000 reviews, the best check kept, stopping after two checks
         in a row without improvement; a resume point is saved at every check. The kept
         weights then score every validation review.
         Writes artifacts/bilstm/ (vocabulary, kept weights, validation_predictions.parquet
         with the same columns as the Transformer runs) and reports/bilstm/final.json.

The four-model comparison later leaves out the 50,000 reviews used here for choosing; this
script reports the whole validation set, like the other runs. The test set is not read.

Usage:
    python scripts/train_bilstm.py --tune        # needs a GPU
    python scripts/train_bilstm.py               # needs a GPU, after --tune
    python scripts/train_bilstm.py --tune --smoke
    python scripts/train_bilstm.py --smoke       # tiny model on a few rows, CPU is fine
    python scripts/train_bilstm.py --smoke --stop-after 6   # then run again: tests resuming
"""

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from train_transformer import head_targets, selection_score

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.experiments import cross_fitted_offsets, search_offsets
from digikala_reviews.finetune import IGNORE, fixed_sample, label_codes, linear_warmup
from digikala_reviews.metrics import bootstrap_macro_f1, classification_report
from digikala_reviews.provenance import check_cache, environment, git_commit, library_version
from digikala_reviews.tfidf import HEADS, TOKEN_PATTERNS, load_cache
from digikala_reviews.tuning import finished, save_json

CACHE_FILES = ["train_supervised.parquet", "validation.parquet"]
TOKEN = re.compile(TOKEN_PATTERNS["persian"])
PAD, OTHER = 0, 1
SETTINGS = {
    "min_count": 5,
    "max_vocabulary": 200_000,
    "max_tokens": 128,
    "embedding": 300,
    "dropout": 0.3,
    "batch_size": 256,
    "weight_decay": 0.01,
    "warmup_share": 0.06,
    "selection_rows": 50_000,
    "seed": SPLIT_SEED,
}
GRID = [
    {"hidden": h, "layers": n, "learning_rate": lr} for h in (256, 512) for n in (1, 2) for lr in (1e-3, 2e-3)
]
TUNE = {"train_rows": 500_000, "max_epochs": 2, "checks_per_epoch": 1, "patience": None}
FINAL = {"train_rows": None, "max_epochs": 4, "checks_per_epoch": 4, "patience": 2}
SMOKE = {
    "min_count": 1,
    "embedding": 16,
    "train_rows": 2_000,
    "selection_rows": 400,
    "batch_size": 64,
}
SMOKE_GRID = [{"hidden": h, "layers": n, "learning_rate": 2e-3} for h in (8, 16) for n in (1, 2)]
SMOKE_VALIDATION_ROWS = 1_500
LOG_EVERY = 500


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def build_vocabulary(texts, min_count: int, max_size: int) -> dict[str, int]:
    """Token -> id for tokens seen at least min_count times, most frequent first, ties by text."""
    counts = Counter()
    for text in texts:
        counts.update(tokens(text))
    kept = sorted((t for t, c in counts.items() if c >= min_count), key=lambda t: (-counts[t], t))
    return {t: i + 2 for i, t in enumerate(kept[: max_size - 2])}


def encode(texts, vocabulary: dict[str, int], max_tokens: int) -> list[np.ndarray]:
    """Token ids of each review, cut at max_tokens; an empty review gets one OTHER token."""
    out = []
    for text in texts:
        ids = [vocabulary.get(t, OTHER) for t in tokens(text)[:max_tokens]] or [OTHER]
        out.append(np.asarray(ids, dtype=np.int64))
    return out


def build_network(torch, vocabulary_size: int, settings: dict, heads: list[str]):
    nn = torch.nn

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, settings["embedding"], padding_idx=PAD)
            self.lstm = nn.LSTM(
                settings["embedding"],
                settings["hidden"],
                num_layers=settings["layers"],
                batch_first=True,
                bidirectional=True,
                dropout=settings["dropout"] if settings["layers"] > 1 else 0.0,
            )
            self.dropout = nn.Dropout(settings["dropout"])
            self.heads = nn.ModuleDict({h: nn.Linear(4 * settings["hidden"], 3) for h in heads})

        def forward(self, ids, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(
                self.dropout(self.embedding(ids)), lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=ids.shape[1])
            mask = (ids != PAD).unsqueeze(-1)
            mean = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
            largest = output.masked_fill(~mask, float("-inf")).max(1).values
            pooled = self.dropout(torch.cat([mean, largest], dim=-1))
            return {h: layer(pooled) for h, layer in self.heads.items()}

    return BiLSTM()


def batches(torch, encoded, order, size, labels=None):
    for start in range(0, len(order), size):
        rows = order[start : start + size]
        lengths = np.array([len(encoded[i]) for i in rows])
        ids = np.full((len(rows), lengths.max()), PAD, dtype=np.int64)
        for j, i in enumerate(rows):
            ids[j, : lengths[j]] = encoded[i]
        batch = {"ids": torch.as_tensor(ids), "lengths": torch.as_tensor(lengths)}
        if labels is not None:
            batch["labels"] = {h: torch.as_tensor(codes[rows]) for h, codes in labels.items()}
        yield batch


def predict(torch, model, encoded, device) -> dict:
    """Log-probabilities of each head, in the order of the reviews."""
    model.eval()
    order = np.argsort([len(x) for x in encoded], kind="stable")
    parts = {h: [] for h in HEADS}
    with torch.no_grad():
        for batch in batches(torch, encoded, order, 2048):
            outputs = model(batch["ids"].to(device), batch["lengths"])
            for h, logits in outputs.items():
                parts[h].append(torch.log_softmax(logits.float(), dim=-1).cpu().numpy())
    back = np.argsort(order)
    return {h: np.vstack(p)[back] for h, p in parts.items()}


def setting_name(s: dict) -> str:
    return f"h{s['hidden']}_l{s['layers']}_lr{s['learning_rate']:g}"


def train(
    torch, settings, train_ids, labels, selection_ids, selection, device, out_dir=None, stop_after=None
):
    """Train one setting; with out_dir, keep the best check there and resume from it."""
    torch.manual_seed(settings["seed"])
    model = build_network(torch, settings["vocabulary_size"], settings, list(HEADS)).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=settings["learning_rate"],
    )
    steps_per_epoch = math.ceil(len(train_ids) / settings["batch_size"])
    total = settings["max_epochs"] * steps_per_epoch
    warmup = max(1, int(settings["warmup_share"] * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: linear_warmup(s, total, warmup))
    check_every = max(1, steps_per_epoch // settings["checks_per_epoch"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    state = {"step": 0, "best": -1.0, "best_step": None, "checks_without_gain": 0, "history": []}
    best_weights = None
    resume_path = best_path = None
    if out_dir is not None:
        resume_path, best_path = out_dir / "resume.pt", out_dir / "model.pt"
        if resume_path.exists():
            saved = torch.load(resume_path, map_location=device, weights_only=False)
            if saved["settings"] == settings:
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                state = saved["state"]
                print(f"Resumed at step {state['step']:,} (best {state['best']:.4f})", flush=True)

    def save_atomic(obj, path):
        temporary = path.with_suffix(".tmp")
        torch.save(obj, temporary)
        temporary.replace(path)

    def patience_used_up():
        return settings["patience"] is not None and state["checks_without_gain"] >= settings["patience"]

    start, running, stop = time.perf_counter(), [], patience_used_up()
    while state["step"] < total and not stop:
        epoch, done = divmod(state["step"], steps_per_epoch)
        order = np.random.default_rng(settings["seed"] + epoch).permutation(len(train_ids))
        order = order[done * settings["batch_size"] :]
        model.train()
        for batch in batches(torch, train_ids, order, settings["batch_size"], labels):
            outputs = model(batch["ids"].to(device), batch["lengths"])
            loss = 0.0
            for h, logits in outputs.items():
                target = batch["labels"][h].to(device)
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
                    f"    step {state['step']:,}/{total:,}, loss {np.mean(running):.4f}, {minutes:.1f} min",
                    flush=True,
                )
                running = []
            if state["step"] % check_every == 0 or state["step"] == total:
                score, per_head, losses = selection_score(
                    predict(torch, model, selection_ids, device), selection
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
                        "kept": improved,
                    }
                )
                if improved:
                    state.update(best=score, best_step=state["step"], checks_without_gain=0)
                    weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if best_path is not None:
                        save_atomic(
                            {"weights": weights, "settings": settings, "step": state["step"]}, best_path
                        )
                    else:
                        best_weights = weights
                else:
                    state["checks_without_gain"] += 1
                print(
                    f"  check at epoch {state['history'][-1]['epoch']}: {score:.4f} "
                    f"({'kept' if improved else 'not better'})",
                    flush=True,
                )
                stop = patience_used_up()
                if resume_path is not None:
                    save_atomic(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "state": state,
                            "settings": settings,
                        },
                        resume_path,
                    )
                if stop:
                    print("  stopped early: no gain in the last checks", flush=True)
                    break
                if stop_after and state["step"] >= stop_after:
                    print(
                        f"Session ended at step {state['step']:,} as asked; run again to resume", flush=True
                    )
                    return None
            if state["step"] >= total:
                break
    minutes = round((time.perf_counter() - start) / 60, 1)
    return {
        "model": model,
        "state": state,
        "steps_per_epoch": steps_per_epoch,
        "minutes": minutes,
        "best_weights": best_weights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="the 8 settings of the plan on 500,000 reviews")
    parser.add_argument("--smoke", action="store_true", help="a tiny model on a few rows, to test the code")
    parser.add_argument(
        "--stop-after", type=int, help="end the session after this step's check (tests resuming)"
    )
    args = parser.parse_args()
    import torch

    base = {**SETTINGS, **(TUNE if args.tune else FINAL), **(SMOKE if args.smoke else {})}
    suffix = "_smoke" if args.smoke else ""
    report_dir = REPORTS_DIR / f"bilstm{suffix}"
    out_dir = ARTIFACTS_DIR / f"bilstm{suffix}"
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
        },
        "data": check_cache(CACHE_DIR, CACHE_FILES),
    }
    print(f"Code {context['code']['commit']}, device {context['environment']['gpu'] or 'cpu'}", flush=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_table = load_cache(CACHE_FILES[0])
    validation = load_cache(CACHE_FILES[1], extra_columns=["product_id"])
    labeled_any = validation["recommendation_status"].notna() | validation["rating_band"].notna()
    selection_table = validation[labeled_any]
    selection_table = selection_table.iloc[
        fixed_sample(len(selection_table), base["selection_rows"], SPLIT_SEED)
    ].reset_index(drop=True)

    # the vocabulary comes from every training review (in a smoke run, from its few rows)
    vocab_rows = fixed_sample(len(train_table), base["train_rows"], SPLIT_SEED) if args.smoke else None
    vocab_texts = (
        train_table["text"].to_numpy() if vocab_rows is None else train_table["text"].to_numpy()[vocab_rows]
    )
    started = time.perf_counter()
    vocabulary = build_vocabulary(vocab_texts, base["min_count"], base["max_vocabulary"])
    vocab_hash = hashlib.sha256(
        json.dumps(vocabulary, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    print(f"vocabulary {len(vocabulary) + 2:,} ids ({time.perf_counter() - started:.0f}s)", flush=True)
    base["vocabulary_size"] = len(vocabulary) + 2
    base["vocabulary_sha256"] = vocab_hash

    if base["train_rows"]:
        train_table = train_table.iloc[
            fixed_sample(len(train_table), base["train_rows"], SPLIT_SEED)
        ].reset_index(drop=True)
    train_ids = encode(train_table["text"].to_numpy(), vocabulary, base["max_tokens"])
    labels = {h: label_codes(train_table[HEADS[h]["column"]].to_numpy(), HEADS[h]["labels"]) for h in HEADS}
    selection_ids = encode(selection_table["text"].to_numpy(), vocabulary, base["max_tokens"])
    selection = {h: head_targets(selection_table, h) for h in HEADS}
    print(f"{len(train_ids):,} training reviews encoded", flush=True)

    if args.tune:
        results = []
        for grid in SMOKE_GRID if args.smoke else GRID:
            settings = {**base, **grid}
            path = report_dir / "tuning" / f"{setting_name(grid)}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            spec = json.loads(json.dumps({"settings": settings, "data": context["data"]}))
            if not finished(path, spec):
                print(f"Setting {setting_name(grid)}", flush=True)
                run = train(torch, settings, train_ids, labels, selection_ids, selection, device)
                best = max(run["state"]["history"], key=lambda c: c["score"])
                save_json(
                    path,
                    {
                        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "fingerprint": spec,
                        "setting": grid,
                        "score": best["score"],
                        "best_epoch": best["epoch"],
                        "checks": run["state"]["history"],
                        "train_minutes": run["minutes"],
                        **context,
                    },
                )
            results.append(json.loads(path.read_text(encoding="utf-8")))
        ranked = sorted(results, key=lambda r: -r["score"])
        save_json(
            report_dir / "tuning.json",
            {
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "base": base,
                "settings": [
                    {
                        "setting": r["setting"],
                        "score": r["score"],
                        "best_epoch": r["best_epoch"],
                        "minutes": r["train_minutes"],
                    }
                    for r in ranked
                ],
                "best": ranked[0]["setting"],
                **context,
            },
        )
        for r in ranked:
            print(f"  {setting_name(r['setting'])}: {r['score']:.4f} (epoch {r['best_epoch']})", flush=True)
        print(f"Best: {setting_name(ranked[0]['setting'])}", flush=True)
        return

    tuning_path = report_dir / "tuning.json"
    if not tuning_path.exists():
        raise SystemExit(f"{tuning_path} not found: run with --tune first")
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    settings = {**base, **tuning["best"]}
    result_path = report_dir / "final.json"
    spec = json.loads(json.dumps({"settings": settings, "data": context["data"]}))
    if finished(result_path, spec):
        print("Already done", flush=True)
        return
    (out_dir / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    run = train(
        torch, settings, train_ids, labels, selection_ids, selection, device, out_dir, args.stop_after
    )
    if run is None:
        return

    if args.smoke:
        validation = validation.iloc[:SMOKE_VALIDATION_ROWS].reset_index(drop=True)
    model = run["model"]
    saved = torch.load(out_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["weights"])
    print(f"Scoring the whole validation set with the weights from step {saved['step']:,}", flush=True)
    log_proba = predict(
        torch, model, encode(validation["text"].to_numpy(), vocabulary, settings["max_tokens"]), device
    )

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
        for i, label in enumerate(labels_):
            table[f"{head}__log_proba__{label}"] = log_proba[head][:, i].astype(np.float32)
        table[f"{head}__predicted"] = plain
        column = np.full(len(table), None, dtype=object)
        column[labeled] = with_offsets
        table[f"{head}__predicted_with_offsets"] = column
    predictions_path = out_dir / "validation_predictions.parquet"
    table.to_parquet(predictions_path.with_suffix(".tmp"), index=False)
    predictions_path.with_suffix(".tmp").replace(predictions_path)
    torch.save({**saved, "offsets_for_test": offsets_for_test}, (out_dir / "model.pt").with_suffix(".tmp"))
    (out_dir / "model.pt").with_suffix(".tmp").replace(out_dir / "model.pt")

    save_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fingerprint": spec,
            "training_reviews": len(train_ids),
            "steps_per_epoch": run["steps_per_epoch"],
            "kept_step": saved["step"],
            "kept_epoch": round(saved["step"] / run["steps_per_epoch"], 2),
            "stopped_at_step": run["state"]["step"],
            "train_minutes_this_session": run["minutes"],
            "checks": run["state"]["history"],
            "heads": heads,
            "validation_reviews": int(len(validation)),
            "torch_library": library_version("torch"),
            **context,
        },
    )
    (out_dir / "resume.pt").unlink(missing_ok=True)
    for head, found in heads.items():
        print(
            f"  {head}: without offsets {found['without_offsets']['macro_f1']:.4f}, "
            f"with offsets {found['with_offsets']['macro_f1']:.4f}",
            flush=True,
        )
    print("Done", flush=True)


if __name__ == "__main__":
    main()
