"""How much the full runs depend on chance (plan: reports/full_seeds_plan.md).

For the kept FaBERT run trained with seeds 2026 (the kept run itself), 2027 and 2028, and the
BiLSTM trained for every epoch (no early stopping) with the same three seeds:

- Macro F1 with offsets of each run on the whole validation set, the spread and the mean;
- the model order with each seed: FaBERT minus BiLSTM, BiLSTM minus TF-IDF;
- D58 with each FaBERT seed: FaBERT + TF-IDF (rebuilt as in step 60) minus FaBERT alone, with
  a paired 95% interval drawing products with replacement (1,000 rounds, seed 2026);
- where the full further-pretrained run (D65) stands against the three FaBERT seeds (report
  only: D65 was decided on the pilot with three seeds);
- the three-model comparison of scripts/compare_reading_models.py (all right, all wrong, the
  same wrong label, no known cause) with each pair of seeds, on the same 872,795 reviews.

Every run's saved labels with offsets must score its recorded Macro F1, and the FaBERT +
TF-IDF combination with seed 2026 must score the recorded D58 value. Nothing is trained. The
test set is not read.

Writes reports/full_seeds.json and prints a summary.

Usage:
    uv run python scripts/check_full_seeds.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_reading_models as crm  # noqa: E402
from recheck_decisions import confusion, macro_f1  # noqa: E402

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR  # noqa: E402
from digikala_reviews.ensemble import cross_fitted_blend  # noqa: E402
from digikala_reviews.experiments import product_half  # noqa: E402
from digikala_reviews.metrics import classification_report  # noqa: E402
from digikala_reviews.provenance import check_cache, git_commit  # noqa: E402
from digikala_reviews.tfidf import HEADS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

OUTPUT = REPORTS_DIR / "full_seeds.json"
SEEDS = [2026, 2027, 2028]
ROUNDS, SEED = 1000, 2026


def seed_part(seed: int) -> str:
    return "" if seed == 2026 else f"_seed{seed}"


# name: (folder of its predictions, its training report)
RUNS = {
    **{
        f"fabert_{s}": (
            f"transformer_no_early_stop{seed_part(s)}",
            f"transformer_final/fabert_no_early_stop{seed_part(s)}.json",
        )
        for s in SEEDS
    },
    **{
        f"bilstm_{s}": (
            f"bilstm_no_early_stop{seed_part(s)}",
            f"bilstm/final_no_early_stop{seed_part(s)}.json",
        )
        for s in SEEDS
    },
    "adapted": ("transformer_adapted", "transformer_final/adapted.json"),
}


def read(path: str) -> dict:
    return json.loads((REPORTS_DIR / path).read_text(encoding="utf-8"))


def main():
    validation = load_cache("validation.parquet", extra_columns=["product_id", "rate", "text_seen_in_train"])
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    tables, reports = {}, {}
    for name, (folder, report) in {**RUNS, "tfidf": ("tfidf_final", None)}.items():
        table = pd.read_parquet(ARTIFACTS_DIR / folder / "validation_predictions.parquet")
        assert table["id"].is_unique and set(table["id"]) == set(validation["id"]), f"{name}: ids differ"
        tables[name] = validation[["id"]].merge(table, on="id", how="left", validate="one_to_one")
        reports[name] = read(report) if report else None
    overlap = read("error_overlap.json")["heads"]
    ensemble = read("ensemble.json")["final_fabert_and_tfidf"]
    keep = crm.comparison_mask(validation)
    assert int((~keep).sum()) == crm.SELECTION_ROWS
    causes = crm.known_causes(validation)
    any_cause = np.column_stack(list(causes.values())).any(axis=1)
    _, product_index = np.unique(validation["product_id"].to_numpy(), return_inverse=True)

    result = {
        "seeds": SEEDS,
        "kept_epoch": {name: reports[name]["kept_epoch"] for name in RUNS},
        "data": data,
        "heads": {},
    }
    for head, spec in HEADS.items():
        labels = spec["labels"]
        n = len(labels)
        rows = validation[spec["column"]].notna().to_numpy()
        y_true = validation[spec["column"]].to_numpy()[rows].astype(str)
        codes = np.array([labels.index(v) for v in y_true])
        halves = product_half(validation["product_id"].to_numpy()[rows])
        predicted, scores = {}, {}
        for name, t in tables.items():
            found = t[f"{head}__predicted_with_offsets"].to_numpy()[rows]
            predicted[name] = np.array([labels.index(v) for v in found])
            scores[name] = classification_report(y_true, found.astype(str), labels)["macro_f1"]
            expected = (
                overlap[head]["macro_f1"]["tfidf"]
                if name == "tfidf"
                else reports[name]["heads"][head]["with_offsets"]["macro_f1"]
            )
            assert abs(scores[name] - expected) < 1e-9, f"{head} {name}: {scores[name]}, recorded {expected}"
        log = {
            name: t[[f"{head}__log_proba__{c}" for c in labels]].to_numpy(dtype=float)[rows]
            for name, t in tables.items()
        }
        for s in SEEDS:
            predicted[f"fabert_{s} + tfidf"], _ = cross_fitted_blend(
                log[f"fabert_{s}"], log["tfidf"], codes, halves
            )
            scores[f"fabert_{s} + tfidf"] = classification_report(
                y_true, np.array(labels)[predicted[f"fabert_{s} + tfidf"]], labels
            )["macro_f1"]
        assert abs(scores["fabert_2026 + tfidf"] - ensemble[head]["blend_macro_f1"]) < 1e-9, "D58 differs"

        # D58 with each FaBERT seed: paired product-level bootstrap
        pairs = [(f"fabert_{s} + tfidf", f"fabert_{s}") for s in SEEDS]
        _, products = np.unique(product_index[rows], return_inverse=True)
        n_products = int(products.max()) + 1
        rng = np.random.default_rng(SEED)
        differences = [[] for _ in pairs]
        for _ in range(ROUNDS):
            w = np.bincount(rng.integers(0, n_products, n_products), minlength=n_products)[products]
            w = w.astype(float)
            for i, (a, b) in enumerate(pairs):
                differences[i].append(
                    macro_f1(confusion(codes, predicted[a], w, n))
                    - macro_f1(confusion(codes, predicted[b], w, n))
                )
        d58 = {}
        for (a, b), found, s in zip(pairs, differences, SEEDS, strict=True):
            low, high = np.percentile(found, [2.5, 97.5])
            d58[str(s)] = {
                "difference": scores[a] - scores[b],
                "interval_95_products": [float(low), float(high)],
                "above_zero": bool(low > 0),
            }

        fabert = np.array([scores[f"fabert_{s}"] for s in SEEDS])
        bilstm = np.array([scores[f"bilstm_{s}"] for s in SEEDS])
        old_bilstm = read("bilstm_seeds.json")["heads"][head]["bilstm_macro_f1_whole_validation"]

        # the three-model comparison with each pair of seeds, on the reviews outside the sample
        truth_all = validation[spec["column"]].to_numpy(dtype=object)
        keep_rows = keep[rows]
        comparison = {}
        for s in SEEDS:
            trio = [f"fabert_{s}", "tfidf", f"bilstm_{s}"]
            stacked = np.column_stack([predicted[m][keep_rows] for m in trio])
            right = stacked == codes[keep_rows][:, None]
            all_wrong = ~right.any(axis=1)
            comparison[str(s)] = {
                "all_right": int(right.all(axis=1).sum()),
                "all_wrong": int(all_wrong.sum()),
                "all_wrong_same_label": int((all_wrong & (stacked == stacked[:, :1]).all(axis=1)).sum()),
                "all_wrong_no_known_cause": int((all_wrong & ~any_cause[rows][keep_rows]).sum()),
            }
        counts = {k: [comparison[str(s)][k] for s in SEEDS] for k in comparison[str(SEEDS[0])]}
        assert int(keep_rows.sum()) == int((keep & pd.notna(truth_all)).sum())
        result["heads"][head] = {
            "macro_f1": {name: float(v) for name, v in scores.items()},
            "fabert": {"spread": float(fabert.max() - fabert.min()), "mean": float(fabert.mean())},
            "bilstm_no_early_stop": {
                "spread": float(bilstm.max() - bilstm.min()),
                "mean": float(bilstm.mean()),
            },
            "bilstm_with_early_stop_for_reference": old_bilstm,
            "fabert_minus_bilstm": {
                str(s): float(scores[f"fabert_{s}"] - scores[f"bilstm_{s}"]) for s in SEEDS
            },
            "bilstm_minus_tfidf": {str(s): float(scores[f"bilstm_{s}"] - scores["tfidf"]) for s in SEEDS},
            "d58_with_each_fabert_seed": d58,
            "adapted_against_fabert_seeds": {
                "adapted": float(scores["adapted"]),
                "fabert_lowest": float(fabert.min()),
                "fabert_highest": float(fabert.max()),
                "above_every_fabert_seed": bool(scores["adapted"] > fabert.max()),
            },
            "three_model_comparison": comparison,
            "relative_spread_of_counts": {
                k: float((max(v) - min(v)) / max(np.mean(v), 1)) for k, v in counts.items()
            },
            "reviews_compared": int(keep_rows.sum()),
        }

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/full_seeds_plan.md",
            **result,
            "code": git_commit(),
        },
    )
    print("kept epoch: " + ", ".join(f"{k} {v}" for k, v in result["kept_epoch"].items()))
    for head, e in result["heads"].items():
        f = e["macro_f1"]
        print(f"\n=== {head}")
        print(
            "  FaBERT: "
            + ", ".join(f"{s} {f[f'fabert_{s}']:.4f}" for s in SEEDS)
            + f"; spread {e['fabert']['spread']:.4f}"
        )
        print(
            "  BiLSTM, every epoch: "
            + ", ".join(f"{s} {f[f'bilstm_{s}']:.4f}" for s in SEEDS)
            + f"; spread {e['bilstm_no_early_stop']['spread']:.4f}"
            + " (with early stopping: "
            + ", ".join(f"{s} {v:.4f}" for s, v in e["bilstm_with_early_stop_for_reference"].items())
            + ")"
        )
        print(f"  TF-IDF {f['tfidf']:.4f}; further pretrained FaBERT {f['adapted']:.4f}")
        print(
            "  FaBERT minus BiLSTM: "
            + ", ".join(f"{s} {v:+.4f}" for s, v in e["fabert_minus_bilstm"].items())
        )
        print(
            "  BiLSTM minus TF-IDF: " + ", ".join(f"{s} {v:+.4f}" for s, v in e["bilstm_minus_tfidf"].items())
        )
        for s, d in e["d58_with_each_fabert_seed"].items():
            low, high = d["interval_95_products"]
            print(f"  D58 with FaBERT seed {s}: {d['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
        a = e["adapted_against_fabert_seeds"]
        print(
            f"  further pretrained {a['adapted']:.4f} against FaBERT seeds "
            f"{a['fabert_lowest']:.4f} to {a['fabert_highest']:.4f}: "
            + ("above every seed" if a["above_every_fabert_seed"] else "not above every seed")
        )
        print(f"  three models on {e['reviews_compared']:,} reviews, per pair of seeds:")
        for s, x in e["three_model_comparison"].items():
            print(
                f"    seed {s}: all right {x['all_right']:,}, all wrong {x['all_wrong']:,} "
                f"(same label {x['all_wrong_same_label']:,}, "
                f"no known cause {x['all_wrong_no_known_cause']:,})"
            )
        spread = ", ".join(f"{k} {100 * v:.2f}%" for k, v in e["relative_spread_of_counts"].items())
        print(f"  relative spread of the counts: {spread}")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
