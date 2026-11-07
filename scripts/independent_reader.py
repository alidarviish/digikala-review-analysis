"""An independent reader that never saw our labels (plan: reports/independent_reader_plan.md).

Three parts, run in this order:

--make-sample  (your machine: needs the three models' predictions) From the 872,795 compared
               validation reviews, for each head and true class, up to 300 reviews all three
               models got right and up to 300 all three got wrong with the same wrong label
               (seed 2026). Writes reports/independent_reader/sample.json: ids, groups and
               labels only, no text.
(none)         (molab GPU) Qwen/Qwen3-8B, thinking mode off, is asked one question per
               sampled review, zero-shot; the answer is the most probable of the tokens
               "1", "2" and "3" after the question. Texts come from the validation cache.
               Writes reports/independent_reader/answers.json (ids and answers, no text).
               With --smoke: Qwen/Qwen3-0.6B on 20 reviews per head, to test the code.
--score        (your machine) Agreement of the reader with the true labels and with the three
               models, per head, group and true class, with Wilson 95% intervals, and the
               plan's rule. Writes reports/independent_reader.json.

The test set is not read.

Usage:
    uv run python scripts/independent_reader.py --make-sample
    python scripts/independent_reader.py            # needs a GPU
    python scripts/independent_reader.py --smoke
    uv run python scripts/independent_reader.py --score
"""

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# the notebook may start Python without the script's folder on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR, SPLIT_SEED  # noqa: E402
from digikala_reviews.provenance import check_cache, environment, git_commit, library_version  # noqa: E402
from digikala_reviews.tfidf import HEADS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

FOLDER = REPORTS_DIR / "independent_reader"
PER_GROUP = 300
MODEL, SMOKE_MODEL, SMOKE_ROWS = "Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B", 20
MAX_CHARS = 1500  # a review longer than this is cut (over 99.9% of reviews are shorter)
BATCH = 32
ALL_RIGHT, ALL_WRONG_SAME = "all right", "all wrong, same label"
INTRO = (
    "Here is a customer review of a product, written in Persian on an Iranian online shop.\n\n"
    "Review:\n{text}\n\n"
)
QUESTIONS = {
    "recommendation": (
        "Would the writer recommend buying this product? Answer with one digit only:\n"
        "1 = yes, recommends it\n"
        "2 = no, does not recommend it\n"
        "3 = no clear opinion either way"
    ),
    "rating_band": (
        "From the text alone, what star rating out of 5 did the writer most likely give? "
        "Answer with one digit only:\n"
        "1 = low (1 or 2 stars)\n"
        "2 = medium (about 3 stars)\n"
        "3 = high (4 or 5 stars)"
    ),
}
ANSWERS = {
    "recommendation": {"1": "recommended", "2": "not_recommended", "3": "no_idea"},
    "rating_band": {"1": "low", "2": "medium", "3": "high"},
}


def make_sample():
    import compare_reading_models as crm

    names = list(crm.MODELS)
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    keep = crm.comparison_mask(validation)
    assert int((~keep).sum()) == crm.SELECTION_ROWS
    predictions, checks = crm.load_predictions(validation, names)
    rng = np.random.default_rng(SPLIT_SEED)
    sample, available = {}, {}
    for head, spec in HEADS.items():
        truth = validation[spec["column"]].to_numpy(dtype=object)
        rows = keep & pd.notna(truth)
        labels = np.column_stack([predictions[m][head] for m in names])
        right = (labels == truth[:, None]) & rows[:, None]
        all_right = rows & right.all(axis=1)
        all_wrong_same = rows & ~right.any(axis=1) & (labels == labels[:, :1]).all(axis=1)
        sample[head], available[head] = [], {}
        for c in spec["labels"]:
            for group, mask in [(ALL_RIGHT, all_right), (ALL_WRONG_SAME, all_wrong_same)]:
                positions = np.flatnonzero(mask & (truth == c))
                chosen = np.sort(rng.choice(positions, min(PER_GROUP, len(positions)), replace=False))
                available[head][f"{c}, {group}"] = int(len(positions))
                for p in chosen:
                    sample[head].append(
                        {
                            "id": int(validation["id"].iat[p]),
                            "group": group,
                            "true": c,
                            "models_said": None if group == ALL_RIGHT else str(labels[p, 0]),
                        }
                    )
    FOLDER.mkdir(parents=True, exist_ok=True)
    save_json(
        FOLDER / "sample.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/independent_reader_plan.md",
            "per_group": PER_GROUP,
            "seed": SPLIT_SEED,
            "reviews_available": available,
            "model_checks": checks,
            "sample": sample,
            "code": git_commit(),
        },
    )
    for head, rows in sample.items():
        print(f"{head}: {len(rows):,} reviews; available per group {available[head]}")


def read(smoke: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not smoke and device.type != "cuda":
        raise SystemExit("No GPU found: turn the GPU on for this notebook, or run with --smoke")
    model_name = SMOKE_MODEL if smoke else MODEL
    sample = json.loads((FOLDER / "sample.json").read_text(encoding="utf-8"))
    validation = load_cache("validation.parquet").set_index("id")
    context = {
        "code": git_commit(),
        "environment": {
            **environment(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "transformers": library_version("transformers"),
        },
        "data": check_cache(CACHE_DIR, ["validation.parquet"]),
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = []
    for digit in ("1", "2", "3"):
        ids = tokenizer.encode(digit, add_special_tokens=False)
        assert len(ids) == 1, f"{digit!r} is not one token: {ids}"
        choice_ids.append(ids[0])
    dtype = torch.bfloat16 if device.type == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model = model.to(device).eval()
    start = time.perf_counter()
    answers = {}
    for head, rows in sample["sample"].items():
        rows = rows[:SMOKE_ROWS] if smoke else rows
        missing = [r["id"] for r in rows if r["id"] not in validation.index]
        assert not missing, f"{head}: ids not in the validation cache: {missing[:5]}"
        prompts = []
        for r in rows:
            text = str(validation.at[r["id"], "text"])[:MAX_CHARS]
            messages = [{"role": "user", "content": INTRO.format(text=text) + QUESTIONS[head]}]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            )
        found = []
        with torch.no_grad():
            for i in range(0, len(prompts), BATCH):
                batch = tokenizer(prompts[i : i + BATCH], return_tensors="pt", padding=True).to(device)
                logits = model(**batch).logits[:, -1, :].float()
                log_p = torch.log_softmax(logits, dim=-1)[:, choice_ids].cpu().numpy()
                found.append(log_p)
        log_p = np.vstack(found)
        mass = np.exp(log_p).sum(axis=1)  # share of the next-token probability on the three digits
        p = np.exp(log_p) / mass[:, None]
        answers[head] = [
            {
                "id": r["id"],
                "answer": ANSWERS[head][str(int(np.argmax(p[j])) + 1)],
                "p": [round(float(x), 6) for x in p[j]],
                "mass_on_digits": round(float(mass[j]), 6),
            }
            for j, r in enumerate(rows)
        ]
        print(f"{head}: {len(rows):,} reviews read; mean mass on the digits {mass.mean():.3f}", flush=True)
    save_json(
        FOLDER / ("answers_smoke.json" if smoke else "answers.json"),
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "reader": model_name,
            "reader_revision": getattr(model.config, "_commit_hash", None),
            "prompt_intro": INTRO,
            "questions": QUESTIONS,
            "answer_codes": ANSWERS,
            "max_chars": MAX_CHARS,
            "minutes": round((time.perf_counter() - start) / 60, 1),
            "answers": answers,
            **context,
        },
    )
    print("Done", flush=True)


def wilson(hits: int, n: int) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    z, p = 1.959964, hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def share(hits: int, n: int) -> dict:
    return {
        "share": hits / n if n else None,
        "count": int(hits),
        "of": int(n),
        "interval_95": wilson(hits, n),
    }


def shown(s: dict) -> str:
    low, high = s["interval_95"]
    return f"{s['share']:.3f} ({s['count']}/{s['of']}) [{low:.3f}, {high:.3f}]"


def score():
    sample = json.loads((FOLDER / "sample.json").read_text(encoding="utf-8"))
    answers = json.loads((FOLDER / "answers.json").read_text(encoding="utf-8"))
    result = {"reader": answers["reader"], "reader_revision": answers["reader_revision"], "heads": {}}
    verdicts = {}
    for head, rows in sample["sample"].items():
        found = pd.DataFrame(answers["answers"][head])
        table = pd.DataFrame(rows).merge(found, on="id", validate="one_to_one")
        assert len(table) == len(rows), f"{head}: answers missing"
        entry = {"mean_mass_on_digits": float(table["mass_on_digits"].mean()), "groups": {}}
        for group in (ALL_RIGHT, ALL_WRONG_SAME):
            part = table[table["group"] == group]
            agree_label = part["answer"] == part["true"]
            out = {
                "overall": share(int(agree_label.sum()), len(part)),
                "by_true_class": {},
                "reader_said": part["answer"].value_counts().to_dict(),
            }
            if group == ALL_WRONG_SAME:
                agree_models = part["answer"] == part["models_said"]
                out["agrees_with_models"] = share(int(agree_models.sum()), len(part))
                out["agrees_with_neither"] = share(int((~agree_label & ~agree_models).sum()), len(part))
            for c in HEADS[head]["labels"]:
                sub = part[part["true"] == c]
                row = {"agrees_with_label": share(int((sub["answer"] == sub["true"]).sum()), len(sub))}
                if group == ALL_WRONG_SAME:
                    hits = int((sub["answer"] == sub["models_said"]).sum())
                    row["agrees_with_models"] = share(hits, len(sub))
                    row["models_said"] = sub["models_said"].value_counts().to_dict()
                out["by_true_class"][c] = row
            entry["groups"][group] = out
        control = entry["groups"][ALL_RIGHT]["overall"]["share"]
        wrong = entry["groups"][ALL_WRONG_SAME]
        with_models, with_label = wrong["agrees_with_models"]["share"], wrong["overall"]["share"]
        if control < 0.70:
            verdict = "reader not usable: under 70% agreement with the label on all-right reviews"
        elif with_models >= 2 * with_label:
            verdict = (
                "shared-bias objection rejected: the reader sides with the models at least twice as often"
            )
        elif with_label > with_models:
            verdict = (
                "shared-bias objection stands: the reader sides with the label more often than the models"
            )
        else:
            verdict = "inconclusive: the reader sides with the models more often, but not twice as often"
        entry["decision"] = {
            "control": control,
            "with_models": with_models,
            "with_label": with_label,
            "verdict": verdict,
        }
        verdicts[head] = verdict
        result["heads"][head] = entry
    save_json(
        REPORTS_DIR / "independent_reader.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/independent_reader_plan.md",
            **result,
            "code": git_commit(),
        },
    )
    for head, e in result["heads"].items():
        print(f"\n=== {head} (mean probability on the three digits {e['mean_mass_on_digits']:.3f})")
        for group, g in e["groups"].items():
            o = g["overall"]
            line = f"  {group:<22} agrees with label {shown(o)}"
            if "agrees_with_models" in g:
                m = g["agrees_with_models"]
                line += f"; with the models {shown(m)}"
            print(line)
            for c, row in g["by_true_class"].items():
                a = row["agrees_with_label"]
                text = f"    true {c:<16} label {a['share']:.3f} ({a['count']}/{a['of']})"
                if "agrees_with_models" in row:
                    m = row["agrees_with_models"]
                    text += f", models {m['share']:.3f} ({m['count']}/{m['of']})"
                    text += f", models said {row['models_said']}"
                print(text)
            print(f"    reader said {g['reader_said']}")
        print(f"  decision: {e['decision']['verdict']}")
    print(f"saved {REPORTS_DIR / 'independent_reader.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-sample", action="store_true", help="choose the reviews (your machine)")
    parser.add_argument("--score", action="store_true", help="score the answers (your machine)")
    parser.add_argument("--smoke", action="store_true", help="a small reader on 20 reviews per head")
    args = parser.parse_args()
    assert not (args.make_sample and args.score)
    if args.make_sample:
        make_sample()
    elif args.score:
        score()
    else:
        read(args.smoke)


if __name__ == "__main__":
    main()
