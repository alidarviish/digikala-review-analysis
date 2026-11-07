"""How many reviews does FaBERT's tokenizer see with unknown tokens, what are they, and would
fixing them be worth a new training run?

Hypothesis from friends, checked here: after tokenizing, many reviews are seen as unknown
because of spacing or similar, so the text should have been changed first.

Earlier, reports/tokenizers/compare.json measured on 50,000 reviews that 0.068% of FaBERT's
tokens are unknown, that the tokenizer removes the half-space (the words around it are joined,
not made unknown), and that an emoji becomes [UNK]. This script measures the same on the whole
validation set, per review, and what it costs:

1. Tokens: share of reviews with at least one [UNK], share of unknown tokens, share of reviews
   with a half-space (which the tokenizer drops).
2. Sources: the text pieces that became [UNK] (from the tokenizer's offsets), the 40 most
   frequent, and their kinds (emoji, Latin letters, digits, Persian or Arabic letters, other).
3. Cost, for each head of the kept FaBERT run: share right on reviews with and without [UNK],
   with product-level bootstrap intervals, and an upper bound of what any fix could gain:
   (error with [UNK] - error without) x share of reviews with [UNK], in points of accuracy.

The text is the normalized text the model was trained on (the validation cache), and the
tokenizer is the one saved with the further pretrained weights, the same as sbunlp/fabert
(vocabulary 50,000, as sbunlp/fabert). The project environment has neither transformers nor
tokenizers, so the tokenizer is rebuilt in plain Python from its tokenizer.json
(scripts/bert_wordpiece.py). Before anything is measured, the rebuilt tokenizer must give
exactly what the real one gave in reports/tokenizers/compare.json on the same 50,000 reviews
of the same cache: tokens per review (mean, median, 95th and 99th percentile, maximum), share
longer than 128 and 256, tokens per word, share of words in 3 or more pieces, share of
unknown tokens, and the 17 probe words. Any difference stops the run. Nothing is trained; the
test set is not read.

Writes reports/unknown_tokens.json and prints a summary.

Usage:
    uv run python scripts/check_unknown_tokens.py
"""

import json
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from analyze_rate_recommendation import BAND, REC, load_fabert, load_recommendation
from analyze_rating_boundaries import CACHE_FILE, SPLIT_FILE, check_rates, cluster_bootstrap, load_validation
from bert_wordpiece import BertWordPiece

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.provenance import check_cache, git_commit
from digikala_reviews.tfidf import load_cache
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "unknown_tokens.json"
TOKENIZER_DIR = ARTIFACTS_DIR / "fabert_adapted"
ZWNJ = "‌"
BATCH = 50_000


def kind(piece: str) -> str:
    """A rough kind of the text that became [UNK]."""
    kinds = set()
    for ch in piece:
        name = unicodedata.name(ch, "")
        category = unicodedata.category(ch)
        if (
            category == "So"
            or "EMOJI" in name
            or 0x1F000 <= ord(ch) <= 0x1FAFF
            or 0x2600 <= ord(ch) <= 0x27BF
        ):
            kinds.add("emoji or symbol")
        elif ch.isdigit():
            kinds.add("digits")
        elif "LATIN" in name:
            kinds.add("latin letters")
        elif "ARABIC" in name:
            kinds.add("persian or arabic letters")
        elif category.startswith("M") or category in {"Cf", "Mn"}:
            kinds.add("marks or format characters")
        else:
            kinds.add("other")
    return " + ".join(sorted(kinds)) if kinds else "empty"


def check_against_real_tokenizer(tokenizer) -> dict:
    """Recompute FaBERT's numbers of reports/tokenizers/compare.json exactly as that script did,
    and stop at the first difference."""
    recorded = json.loads((REPORTS_DIR / "tokenizers" / "compare.json").read_text(encoding="utf-8"))
    real = recorded["models"]["sbunlp/fabert"]
    cache = check_cache(CACHE_DIR, ["validation.parquet"])["sha256"]["validation.parquet"]
    assert cache == recorded["data"]["sha256"]["validation.parquet"], "compare.json used another cache"

    validation = load_cache("validation.parquet")
    labeled = validation[validation["recommendation_status"].notna() | validation["rating_band"].notna()]
    rows = np.random.default_rng(SPLIT_SEED).choice(len(labeled), recorded["sample"]["rows"], replace=False)
    texts = labeled["text"].to_numpy()[np.sort(rows)]
    words = np.array([len(t.split()) for t in texts])
    assert int(words.sum()) == recorded["sample"]["words"], "not the same sample of reviews"
    pieces = [tokenizer.encode(t)[0] for t in texts]
    lengths = np.array([len(p) for p in pieces])
    unknown = sum(sum(1 for i in p if i == tokenizer.unk_id) for p in pieces)
    word_pieces = []
    rng = np.random.default_rng(SPLIT_SEED)
    for text in rng.choice(texts, min(5_000, len(texts)), replace=False):
        for word in text.split():
            word_pieces.append(len(tokenizer.encode(word)[0]))
    word_pieces = np.array(word_pieces)
    found = {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "128": float((lengths > 128 - 2).mean()),
        "256": float((lengths > 256 - 2).mean()),
        "tokens_per_word": float(lengths.sum() / words.sum()),
        "words_in_3_or_more_pieces": float((word_pieces >= 3).mean()),
        "unknown_token_share": float(unknown / max(lengths.sum(), 1)),
    }
    expected = {
        **real["tokens_per_review"],
        **real["longer_than"],
        "tokens_per_word": real["tokens_per_word"],
        "words_in_3_or_more_pieces": real["words_in_3_or_more_pieces"],
        "unknown_token_share": real["unknown_token_share"],
    }
    for key, value in expected.items():
        assert abs(found[key] - value) < 1e-12, f"rebuilt tokenizer: {key} {found[key]}, real {value}"
    for probe, tokens in real["probes"].items():
        assert tokenizer.tokens(probe) == tokens, (
            f"rebuilt tokenizer: {probe!r} {tokenizer.tokens(probe)}, real {tokens}"
        )
    return {"numbers": len(expected), "probes": len(real["probes"]), "reviews": int(len(texts))}


def share_right(right: np.ndarray, products) -> dict:
    r = right.astype(float)

    def statistic(w):
        return {"share": float((w * r).sum() / w.sum())}

    return {
        "rows": int(len(r)),
        "share_right": float(r.mean()),
        "interval_95": cluster_bootstrap(products, statistic)["share"],
    }


def difference(right: np.ndarray, has_unk: np.ndarray, products) -> dict:
    """Share right without [UNK] minus with [UNK], on the same draws of products."""
    r, u = right.astype(float), has_unk.astype(float)

    def statistic(w):
        wu, wn = w * u, w * (1 - u)
        return {"d": float((wn * r).sum() / wn.sum() - (wu * r).sum() / wu.sum())}

    return {
        "value": statistic(np.ones(len(r)))["d"],
        "interval_95": cluster_bootstrap(products, statistic)["d"],
    }


def main():
    start = time.perf_counter()
    manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["class_counts"][SPLIT_FILE]
    validation, source = load_validation(verify_csv=False)
    check_rates(validation, counts[BAND])
    buyer, _ = load_recommendation(validation, False, counts[REC])
    predicted, join = load_fabert(validation)
    text = pd.read_parquet(CACHE_DIR / CACHE_FILE, columns=["id", "text"])
    assert (text["id"].to_numpy() == validation["id"].to_numpy()).all(), "row order differs"
    texts = text["text"].astype(str).tolist()

    tokenizer = BertWordPiece(TOKENIZER_DIR / "tokenizer.json")
    vocabulary = tokenizer.vocab_size()
    assert vocabulary == 50_000, f"vocabulary size {vocabulary}, expected 50,000 as sbunlp/fabert"
    unk_id = tokenizer.unk_id
    replication = check_against_real_tokenizer(tokenizer)
    print(f"0. the rebuilt tokenizer gives exactly the recorded numbers: {replication}", flush=True)

    n_tokens = np.zeros(len(texts), dtype=np.int64)
    n_unk = np.zeros(len(texts), dtype=np.int64)
    sources, kinds = Counter(), Counter()
    for begin in range(0, len(texts), BATCH):
        part = texts[begin : begin + BATCH]
        for i, (ids, offsets) in enumerate(tokenizer.encode(t) for t in part):
            n_tokens[begin + i] = len(ids)
            for token, (a, b) in zip(ids, offsets, strict=True):
                if token == unk_id:
                    n_unk[begin + i] += 1
                    piece = part[i][a:b]
                    sources[piece] += 1
                    kinds[kind(piece)] += 1
        print(f"  tokenized {min(begin + BATCH, len(texts)):,} of {len(texts):,}", flush=True)

    has_unk = n_unk > 0
    has_zwnj = np.array([ZWNJ in t for t in texts])
    products_all = validation["product_id"].to_numpy()
    result = {
        "reviews": len(texts),
        "data_checks": {"source": source, "prediction_join": join},
        "tokenizer": {
            "folder": str(TOKENIZER_DIR.name),
            "vocabulary_size": vocabulary,
            "unk_id": unk_id,
            "rebuilt_in_plain_python": "scripts/bert_wordpiece.py",
            "checked_against_compare_json": replication,
        },
        "1_tokens": {
            "tokens": int(n_tokens.sum()),
            "unknown_tokens": int(n_unk.sum()),
            "unknown_token_share": float(n_unk.sum() / max(n_tokens.sum(), 1)),
            "reviews_with_unknown": int(has_unk.sum()),
            "reviews_with_unknown_share": float(has_unk.mean()),
            "reviews_only_unknown": int(((n_unk == n_tokens) & (n_tokens > 0)).sum()),
            "reviews_with_half_space_share": float(has_zwnj.mean()),
            "unknown_per_review_with_unknown": float(n_unk[has_unk].mean()) if has_unk.any() else 0.0,
        },
        "2_sources": {
            "kinds": dict(kinds.most_common()),
            "most_frequent": [{"text": s, "count": c, "kind": kind(s)} for s, c in sources.most_common(40)],
            "distinct": len(sources),
        },
        "3_cost": {},
    }
    for head, truth_all in [("recommendation", buyer), ("rating_band", validation[BAND].to_numpy())]:
        name = REC if head == "recommendation" else BAND
        rows = pd.notna(truth_all)
        right = predicted[name][rows].astype(str) == truth_all[rows].astype(str)
        u, p = has_unk[rows], products_all[rows]
        with_unk, without = share_right(right[u], p[u]), share_right(right[~u], p[~u])
        gap = difference(right, u, p)
        result["3_cost"][head] = {
            "with_unknown": with_unk,
            "without_unknown": without,
            "right_without_minus_with": gap,
            "upper_bound_gain_accuracy_points": 100 * max(gap["value"], 0.0) * float(u.mean()),
        }

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {
                "text": "normalized validation text, as in training",
                "intervals": "product-level cluster bootstrap, 1000 rounds, seed 2026",
            },
            **result,
            "minutes": round((time.perf_counter() - start) / 60, 1),
            "code": git_commit(),
        },
    )
    summary(result)
    print(f"saved {OUTPUT} ({(time.perf_counter() - start) / 60:.1f} min)")


def summary(result):
    t = result["1_tokens"]
    print(
        f"1. {result['reviews']:,} reviews, {t['tokens']:,} tokens; "
        f"unknown tokens {t['unknown_token_share']:.4%}; "
        f"reviews with at least one [UNK] {t['reviews_with_unknown']:,} "
        f"({t['reviews_with_unknown_share']:.2%}); "
        f"made only of [UNK] {t['reviews_only_unknown']:,}; reviews with a half-space "
        f"{t['reviews_with_half_space_share']:.2%}"
    )
    print(f"2. kinds of [UNK]: {result['2_sources']['kinds']}")
    print(
        "   most frequent: "
        + ", ".join(f"{x['text']!r} {x['count']:,}" for x in result["2_sources"]["most_frequent"][:25])
    )
    print("3. share right, with [UNK] / without [UNK] [95%]; upper bound of a fix:")
    for head, c in result["3_cost"].items():
        w, n, g = c["with_unknown"], c["without_unknown"], c["right_without_minus_with"]
        print(
            f"   {head:<15} {w['share_right']:.3f} "
            f"[{w['interval_95'][0]:.3f}, {w['interval_95'][1]:.3f}] ({w['rows']:,}) / "
            f"{n['share_right']:.3f} [{n['interval_95'][0]:.3f}, {n['interval_95'][1]:.3f}] ({n['rows']:,}); "
            f"gap {g['value']:+.3f} [{g['interval_95'][0]:+.3f}, {g['interval_95'][1]:+.3f}]; "
            f"at most {c['upper_bound_gain_accuracy_points']:.2f} accuracy points"
        )


if __name__ == "__main__":
    main()
