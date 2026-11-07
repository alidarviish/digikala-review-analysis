"""The main model decisions again, with paired product-level bootstrap intervals (plan:
reports/decision_recheck_plan.md).

Uses the saved validation predictions of FaBERT (kept run), the further pretrained FaBERT,
TF-IDF and the BiLSTM; rebuilds the FaBERT + TF-IDF combinations as in steps 60 and 75; checks
that each scores its recorded Macro F1; then, for each comparison and head, the Macro F1
difference with a 95% interval drawing products with replacement (1,000 rounds, seed 2026).
Nothing is trained. The test set is not read.

Writes reports/decision_recheck.json and prints a summary.

Usage:
    uv run python scripts/recheck_decisions.py
"""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR
from digikala_reviews.ensemble import cross_fitted_blend
from digikala_reviews.experiments import product_half
from digikala_reviews.metrics import classification_report
from digikala_reviews.provenance import check_cache, git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "decision_recheck.json"
ROUNDS, SEED = 1000, 2026
FOLDERS = {
    "fabert": "transformer_no_early_stop",
    "adapted": "transformer_adapted",
    "tfidf": "tfidf_final",
    "bilstm": "bilstm",
}
COMPARISONS = [
    ("D58 final system", "fabert + tfidf", "fabert"),
    ("D65 best single model", "adapted", "fabert"),
    ("D65 final system unchanged", "adapted + tfidf", "fabert + tfidf"),
    ("model order", "fabert", "tfidf"),
    ("model order", "bilstm", "tfidf"),
    ("model order", "fabert", "bilstm"),
]


def recorded() -> dict:
    """Macro F1 recorded for each model and combination, per head."""
    read = lambda path: json.loads((REPORTS_DIR / path).read_text(encoding="utf-8"))  # noqa: E731
    overlap = read("error_overlap.json")["heads"]
    ensemble = read("ensemble.json")["final_fabert_and_tfidf"]
    adapted = read("transformer_final/adapted_vs_fabert.json")["heads"]
    bilstm = read("bilstm/final.json")["heads"]
    return {
        head: {
            "fabert": overlap[head]["macro_f1"]["fabert"],
            "tfidf": overlap[head]["macro_f1"]["tfidf"],
            "adapted": adapted[head]["adapted"]["macro_f1"],
            "bilstm": bilstm[head]["with_offsets"]["macro_f1"],
            "fabert + tfidf": ensemble[head]["blend_macro_f1"],
            "adapted + tfidf": adapted[head]["with_tfidf"]["adapted"]["macro_f1"],
        }
        for head in HEADS
    }


def confusion(codes_true, codes_pred, weights, n) -> np.ndarray:
    return np.bincount(codes_true * n + codes_pred, weights=weights, minlength=n * n).reshape(n, n)


def macro_f1(matrix) -> float:
    tp = np.diag(matrix)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    return float(f1.mean())


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    tables = {}
    for name, folder in FOLDERS.items():
        table = pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet")
        assert table["id"].is_unique and set(table["id"]) == set(validation["id"]), f"{name}: ids differ"
        tables[name] = validation[["id"]].merge(table, on="id", how="left", validate="one_to_one")
    expected = recorded()
    _, product_index = np.unique(validation["product_id"].to_numpy(), return_inverse=True)

    result = {"rounds": ROUNDS, "seed": SEED, "data": data, "heads": {}}
    for head, spec in HEADS.items():
        labels = spec["labels"]
        n = len(labels)
        rows = validation[spec["column"]].notna().to_numpy()
        y_true = validation[spec["column"]].to_numpy()[rows].astype(str)
        codes = np.array([labels.index(v) for v in y_true])
        halves = product_half(validation["product_id"].to_numpy()[rows])
        log = {
            name: t[[f"{head}__log_proba__{c}" for c in labels]].to_numpy(dtype=float)[rows]
            for name, t in tables.items()
        }
        predicted = {
            name: np.array([labels.index(v) for v in t[f"{head}__predicted_with_offsets"].to_numpy()[rows]])
            for name, t in tables.items()
        }
        for base in ("fabert", "adapted"):
            predicted[f"{base} + tfidf"], _ = cross_fitted_blend(log[base], log["tfidf"], codes, halves)

        checks = {}
        for name, codes_pred in predicted.items():
            found = classification_report(y_true, np.array(labels)[codes_pred], labels)["macro_f1"]
            assert abs(found - expected[head][name]) < 1e-9, (
                f"{head} {name}: {found}, recorded {expected[head][name]}"
            )
            checks[name] = found

        _, products = np.unique(product_index[rows], return_inverse=True)
        n_products = int(products.max()) + 1
        rng = np.random.default_rng(SEED)
        differences = [[] for _ in COMPARISONS]
        for _ in range(ROUNDS):  # one draw of products serves every comparison (paired)
            w = np.bincount(rng.integers(0, n_products, n_products), minlength=n_products)[products].astype(
                float
            )
            score = {name: macro_f1(confusion(codes, p, w, n)) for name, p in predicted.items()}
            for i, (_, variant, reference) in enumerate(COMPARISONS):
                differences[i].append(score[variant] - score[reference])
        comparisons = []
        for (decision, variant, reference), found in zip(COMPARISONS, differences, strict=True):
            low, high = np.percentile(found, [2.5, 97.5])
            comparisons.append(
                {
                    "decision": decision,
                    "variant": variant,
                    "reference": reference,
                    "difference": checks[variant] - checks[reference],
                    "interval_95_products": [float(low), float(high)],
                    "above_zero": bool(low > 0),
                    "below_zero": bool(high < 0),
                }
            )
        result["heads"][head] = {
            "reviews": int(rows.sum()),
            "products": n_products,
            "macro_f1": checks,
            "comparisons": comparisons,
        }

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/decision_recheck_plan.md",
            **result,
            "code": git_commit(),
        },
    )
    for head, e in result["heads"].items():
        print(f"\n=== {head}: {e['reviews']:,} reviews, {e['products']:,} products")
        print(
            "  Macro F1 (match the records): " + ", ".join(f"{k} {v:.4f}" for k, v in e["macro_f1"].items())
        )
        for c in e["comparisons"]:
            low, high = c["interval_95_products"]
            verdict = "better" if c["above_zero"] else "worse" if c["below_zero"] else "tie"
            print(
                f"  {c['decision']:<28} {c['variant']:<16} vs {c['reference']:<15} "
                f"{c['difference']:+.4f} [{low:+.4f}, {high:+.4f}] {verdict}"
            )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
