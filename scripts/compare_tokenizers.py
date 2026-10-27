"""Compare the tokenizers of the candidate Transformer models on our own reviews.

Nothing is trained; only the tokenizer files of each model are downloaded. On a fixed
sample of labeled validation reviews (normalized as in training) it measures, for each
tokenizer:

- tokens per review: mean, median, 95th and 99th percentile, and the share of reviews
  longer than 128 and 256 tokens (two places kept for the start and end tokens)
- tokens per word (how finely words are cut) and the share of words cut into 3+ pieces
- the share of unknown tokens
- whether the half-space (ZWNJ) survives tokenization
- how a fixed list of informal, misspelled or mixed words is cut

Results go to reports/tokenizers/compare.json, with the commit of each model on the
Hugging Face hub. The test set is not read.

Usage:
    python scripts/compare_tokenizers.py
"""

import time
from datetime import UTC, datetime

import numpy as np

from digikala_reviews.config import CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.provenance import check_cache, environment, git_commit
from digikala_reviews.tfidf import load_cache
from digikala_reviews.tuning import save_json

MODELS = [
    "PartAI/TookaBERT-Base",
    "sbunlp/fabert",
    "FacebookAI/xlm-roberta-base",
    "jhu-clsp/mmBERT-base",
    "HooshvareLab/bert-fa-zwnj-base",
]
SAMPLE_ROWS = 50_000
LIMITS = [128, 256]
SPECIAL = 2  # start and end tokens
ZWNJ = "‌"
PROBES = [
    "می‌خواهم",  # formal, with half-space
    "میخوام",  # informal, no half-space
    "نمی‌کنم",
    "نمیکنم",
    "پیشنهاد",
    "پشنهاد",  # common misspelling
    "خوبه",
    "خوبع",  # misspelled ending
    "عالییییی",  # stretched
    "اصلن",  # informal spelling of اصلا
    "مرسی",
    "گوشیه",
    "کیفیتش",
    "بد نبود",
    "😍",
    "کیفیت good",
    "۱۰۰٪",
]


def hub_commit(name):
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(name).sha
    except Exception as error:  # the number is still useful without it
        return f"unknown ({type(error).__name__})"


def measure(tokenizer, texts, words_per_text):
    start = time.perf_counter()
    pieces = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    seconds = time.perf_counter() - start
    lengths = np.array([len(p) for p in pieces])
    unknown_id = tokenizer.unk_token_id
    unknown = sum(sum(1 for t in p if t == unknown_id) for p in pieces) if unknown_id is not None else 0

    word_pieces = []
    rng = np.random.default_rng(SPLIT_SEED)
    for text in rng.choice(texts, min(5_000, len(texts)), replace=False):
        for word in text.split():
            word_pieces.append(len(tokenizer(word, add_special_tokens=False)["input_ids"]))
    word_pieces = np.array(word_pieces)

    with_zwnj = [t for t in texts if ZWNJ in t][:2_000]
    kept = 0
    for text in with_zwnj:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        kept += ZWNJ in tokenizer.decode(ids)
    return {
        "tokens_per_review": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
        },
        "longer_than": {str(n): float((lengths > n - SPECIAL).mean()) for n in LIMITS},
        "tokens_per_word": float(lengths.sum() / words_per_text.sum()),
        "words_in_3_or_more_pieces": float((word_pieces >= 3).mean()),
        "unknown_token_share": float(unknown / max(lengths.sum(), 1)),
        "half_space_kept_after_decoding": float(kept / len(with_zwnj)) if with_zwnj else None,
        "vocabulary_size": int(len(tokenizer)),
        "seconds_for_sample": round(seconds, 1),
        "probes": {
            probe: tokenizer.convert_ids_to_tokens(tokenizer(probe, add_special_tokens=False)["input_ids"])
            for probe in PROBES
        },
    }


def main():
    from transformers import AutoTokenizer

    context = {
        "code": git_commit(),
        "environment": environment(),
        "data": check_cache(CACHE_DIR, ["validation.parquet"]),
    }
    validation = load_cache("validation.parquet")
    labeled = validation[validation["recommendation_status"].notna() | validation["rating_band"].notna()]
    rows = np.random.default_rng(SPLIT_SEED).choice(len(labeled), SAMPLE_ROWS, replace=False)
    texts = labeled["text"].to_numpy()[np.sort(rows)]
    words_per_text = np.array([len(t.split()) for t in texts])
    print(f"{len(texts):,} reviews, {words_per_text.sum():,} words", flush=True)

    results = {}
    for name in MODELS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
        except Exception as error:
            print(f"{name}: could not load ({type(error).__name__}: {error})", flush=True)
            results[name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        result = measure(tokenizer, texts, words_per_text)
        result["tokenizer_class"] = type(tokenizer).__name__
        result["hub_commit"] = hub_commit(name)
        results[name] = result
        print(
            f"{name}: {result['tokens_per_review']['mean']:.1f} tokens per review, "
            f"{result['tokens_per_word']:.2f} per word, "
            f">128: {result['longer_than']['128']:.2%}, >256: {result['longer_than']['256']:.2%}, "
            f"unknown {result['unknown_token_share']:.3%}",
            flush=True,
        )

    path = REPORTS_DIR / "tokenizers" / "compare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        path,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sample": {"rows": SAMPLE_ROWS, "seed": SPLIT_SEED, "words": int(words_per_text.sum())},
            "models": results,
            **context,
        },
    )
    print("Done", flush=True)


if __name__ == "__main__":
    main()
