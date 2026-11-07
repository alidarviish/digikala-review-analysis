"""Compare models that read the text differently, review by review (plan:
reports/reading_models_plan.md).

For each head, overall and for each true class, with exact counts: Macro F1 of each model;
every right/wrong pattern of the models; all right, all wrong, exactly one right (which
model) and the other groups; whether the models agree on the same wrong label when all are
wrong; and how often known causes appear in each group.

Validation only, without the 50,000-review sample used for FaBERT's checks and the BiLSTM's
tuning. Each model's saved labels with offsets are used, after checking that they score the
recorded Macro F1 on the whole validation set. Report only. The test set is not read.

Writes reports/reading_models.json and prints a summary.

Usage:
    uv run python scripts/compare_reading_models.py
"""

import itertools
import json
import unicodedata
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, CACHE_DIR, REPORTS_DIR, SPLIT_SEED
from digikala_reviews.finetune import fixed_sample
from digikala_reviews.metrics import classification_report
from digikala_reviews.provenance import check_cache, git_commit
from digikala_reviews.tfidf import HEADS, load_cache
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "reading_models.json"
SELECTION_ROWS = 50_000
# name: (folder of its predictions, how to read its recorded Macro F1 for a head)
MODELS = {
    "fabert": ("transformer_no_early_stop", ("error_overlap.json", "fabert")),
    "tfidf": ("tfidf_final", ("error_overlap.json", "tfidf")),
    "bilstm": ("bilstm", ("bilstm/final.json", None)),
}
MAPPING = {"high": "recommended", "medium": "no_idea", "low": "not_recommended"}  # label_sample_plan.md
CAUSES = [
    "buyer_labels_disagree",
    "buyer_labels_strongly_disagree",
    "rate_exactly_3",
    "rate_exactly_4",
    "rate_exactly_2",
    "text_also_in_train",
    "emoji",
    "three_words_or_fewer",
]


def recorded_macro_f1(name: str, head: str) -> float:
    path, key = MODELS[name][1]
    report = json.loads((REPORTS_DIR / path).read_text(encoding="utf-8"))
    if key is None:  # a training report
        return report["heads"][head]["with_offsets"]["macro_f1"]
    return report["heads"][head]["macro_f1"][key]


def has_emoji(text: str) -> bool:
    return any(
        unicodedata.category(c) == "So" or 0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF
        for c in text
    )


def comparison_mask(validation: pd.DataFrame) -> np.ndarray:
    """True for the reviews outside the 50,000-review choosing sample (as train_transformer.py)."""
    labeled_any = validation["recommendation_status"].notna() | validation["rating_band"].notna()
    positions = np.flatnonzero(labeled_any.to_numpy())
    chosen = positions[fixed_sample(len(positions), SELECTION_ROWS, SPLIT_SEED)]
    mask = np.ones(len(validation), dtype=bool)
    mask[chosen] = False
    return mask


def load_predictions(validation: pd.DataFrame, names: list[str]) -> dict:
    """Each model's labels with offsets for both heads, in validation's order, checked."""
    predictions, checks = {}, {}
    for name in names:
        path = ARTIFACTS_DIR / MODELS[name][0] / "validation_predictions.parquet"
        columns = ["id"] + [f"{h}__predicted_with_offsets" for h in HEADS]
        table = pd.read_parquet(path, columns=columns)
        assert table["id"].is_unique, f"{name}: duplicate ids"
        assert set(table["id"]) == set(validation["id"]), f"{name}: ids differ from validation"
        table = validation[["id"]].merge(table, on="id", how="left", validate="one_to_one")
        predictions[name], checks[name] = {}, {}
        for head, spec in HEADS.items():
            labels = table[f"{head}__predicted_with_offsets"].to_numpy(dtype=object)
            truth = validation[spec["column"]].to_numpy(dtype=object)
            has = pd.notna(truth)
            assert (pd.isna(labels) == ~has).all(), f"{name} {head}: predictions missing or extra"
            found = classification_report(truth[has].astype(str), labels[has].astype(str), spec["labels"])[
                "macro_f1"
            ]
            recorded = recorded_macro_f1(name, head)
            assert abs(found - recorded) < 1e-9, f"{name} {head}: Macro F1 {found}, recorded {recorded}"
            checks[name][head] = found
            predictions[name][head] = labels
    return predictions, checks


def share(count: int, total: int) -> dict:
    return {"count": int(count), "share": round(count / total, 6) if total else None}


def groups(right: np.ndarray, names: list[str]) -> dict:
    """Pattern and group counts for a boolean matrix (reviews x models)."""
    n, k = right.shape
    out = {"reviews": int(n), "patterns": {}, "groups": {}}
    for combo in itertools.product([True, False], repeat=k):
        key = ", ".join(f"{m} {'right' if r else 'wrong'}" for m, r in zip(names, combo, strict=True))
        out["patterns"][key] = share(int((right == np.array(combo)).all(axis=1).sum()), n)
    number_right = right.sum(axis=1)
    out["groups"]["all right"] = share(int((number_right == k).sum()), n)
    out["groups"]["all wrong"] = share(int((number_right == 0).sum()), n)
    for j in range(1, k):
        out["groups"][f"exactly {j} right"] = share(int((number_right == j).sum()), n)
    one = number_right == 1
    out["only_this_model_right"] = {m: share(int((one & right[:, i]).sum()), n) for i, m in enumerate(names)}
    wrong_one = number_right == k - 1
    if k > 2:
        out["only_this_model_wrong"] = {
            m: share(int((wrong_one & ~right[:, i]).sum()), n) for i, m in enumerate(names)
        }
    return out


def known_causes(validation: pd.DataFrame) -> dict:
    """For every validation review, whether each known cause applies (see the plan)."""
    texts = validation["text"].astype(str).to_numpy()
    rate = validation["rate"].to_numpy(dtype=float)
    rec = validation["recommendation_status"].to_numpy(dtype=object)
    band = validation["rating_band"].to_numpy(dtype=object)
    both = pd.notna(rec) & pd.notna(band)
    mapped = np.array([MAPPING.get(b) for b in band], dtype=object)
    strong = both & (
        ((rec == "recommended") & (band == "low")) | ((rec == "not_recommended") & (band == "high"))
    )
    return {
        "buyer_labels_disagree": both & (mapped != rec),
        "buyer_labels_strongly_disagree": strong,
        "rate_exactly_3": np.isclose(rate, 3.0),
        # added after the first run, before running again (reports/reading_models_plan.md)
        "rate_exactly_4": np.isclose(rate, 4.0),
        # added after the second run, before running again (reports/reading_models_plan.md)
        "rate_exactly_2": np.isclose(rate, 2.0),
        "text_also_in_train": validation["text_seen_in_train"].to_numpy(dtype=bool),
        "emoji": np.array([has_emoji(t) for t in texts]),
        "three_words_or_fewer": np.array([len(t.split()) <= 3 for t in texts]),
    }


def main():
    names = list(MODELS)
    validation = load_cache("validation.parquet", extra_columns=["product_id", "rate", "text_seen_in_train"])
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    keep = comparison_mask(validation)
    assert int((~keep).sum()) == SELECTION_ROWS, "the choosing sample is not 50,000 reviews"
    predictions, checks = load_predictions(validation, names)

    causes_all = known_causes(validation)

    result = {
        "models": {m: MODELS[m][0] for m in names},
        "reviews_compared": int(keep.sum()),
        "left_out_choosing_sample": SELECTION_ROWS,
        "checks": {"macro_f1_whole_validation_matches_record": checks, "data": data},
        "heads": {},
    }
    for head, spec in HEADS.items():
        truth_all = validation[spec["column"]].to_numpy(dtype=object)
        rows = keep & pd.notna(truth_all)
        truth = truth_all[rows].astype(str)
        labels = {m: predictions[m][head][rows].astype(str) for m in names}
        right = np.column_stack([labels[m] == truth for m in names])
        entry = {
            "reviews": int(rows.sum()),
            "macro_f1": {
                m: classification_report(truth, labels[m], spec["labels"])["macro_f1"] for m in names
            },
            "overall": groups(right, names),
            "by_true_class": {c: groups(right[truth == c], names) for c in spec["labels"]},
        }
        all_wrong = ~right.any(axis=1)
        stacked = np.column_stack([labels[m] for m in names])
        same = all_wrong & (stacked == stacked[:, :1]).all(axis=1)
        agreed = {}
        for c in spec["labels"]:
            for d in spec["labels"]:
                if c != d:
                    agreed[f"true {c}, all said {d}"] = int(
                        (same & (truth == c) & (stacked[:, 0] == d)).sum()
                    )
        entry["all_wrong_same_label"] = {**share(int(same.sum()), int(all_wrong.sum())), "by_label": agreed}

        number_right = right.sum(axis=1)
        parts = {
            "all right": number_right == len(names),
            "some right": (number_right > 0) & (number_right < len(names)),
            "all wrong": number_right == 0,
        }
        causes = {k: v[rows] for k, v in causes_all.items()}
        none = ~np.column_stack(list(causes.values())).any(axis=1)
        entry["causes"] = {}
        for part, mask in parts.items():
            total = int(mask.sum())
            entry["causes"][part] = {
                **{k: share(int((v & mask).sum()), total) for k, v in causes.items()},
                "none_of_these": share(int((none & mask).sum()), total),
                "reviews": total,
            }
        entry["causes_all_wrong_by_true_class"] = {}
        for c in spec["labels"]:
            mask = parts["all wrong"] & (truth == c)
            total = int(mask.sum())
            entry["causes_all_wrong_by_true_class"][c] = {
                **{k: share(int((v & mask).sum()), total) for k, v in causes.items()},
                "none_of_these": share(int((none & mask).sum()), total),
                "reviews": total,
            }
        result["heads"][head] = entry

    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "plan": "reports/reading_models_plan.md",
            **result,
            "code": git_commit(),
        },
    )
    summary(result, names)
    print(f"saved {OUTPUT}")


def line(x: dict) -> str:
    return f"{x['count']:>9,} ({100 * x['share']:6.2f}%)" if x["share"] is not None else f"{x['count']:>9,}"


def summary(result, names):
    print(f"{result['reviews_compared']:,} validation reviews compared (the 50,000 choosing sample left out)")
    for head, e in result["heads"].items():
        print(
            f"\n=== {head}: {e['reviews']:,} reviews; Macro F1 "
            + ", ".join(f"{m} {v:.4f}" for m, v in e["macro_f1"].items())
        )
        o = e["overall"]
        for g, x in o["groups"].items():
            print(f"  {g:<18} {line(x)}")
        print(
            "  only this model right: "
            + "; ".join(f"{m} {line(x)}" for m, x in o["only_this_model_right"].items())
        )
        if "only_this_model_wrong" in o:
            print(
                "  only this model wrong: "
                + "; ".join(f"{m} {line(x)}" for m, x in o["only_this_model_wrong"].items())
            )
        print("  every pattern:")
        for p, x in o["patterns"].items():
            print(f"    {p:<50} {line(x)}")
        s = e["all_wrong_same_label"]
        print(f"  all wrong with the same wrong label: {line(s)} of all wrong")
        for k, v in s["by_label"].items():
            if v:
                print(f"    {k:<36} {v:>9,}")
        print("  by true class (all right / all wrong / exactly one right):")
        for c, x in e["by_true_class"].items():
            print(
                f"    {c:<16} {x['reviews']:>8,}: {line(x['groups']['all right'])} / "
                f"{line(x['groups']['all wrong'])} / {line(x['groups']['exactly 1 right'])}"
            )
        print("  causes (share of the group):")
        print(f"    {'':<32}" + "".join(f"{p:>14}" for p in e["causes"]))
        for k in [*CAUSES, "none_of_these"]:
            print(f"    {k:<32}" + "".join(f"{100 * e['causes'][p][k]['share']:13.2f}%" for p in e["causes"]))
        print("  causes in all wrong, by true class (share):")
        for c, x in e["causes_all_wrong_by_true_class"].items():
            parts = ", ".join(
                f"{k} {100 * x[k]['share']:.1f}%" for k in [*CAUSES, "none_of_these"] if x["reviews"]
            )
            print(f"    {c:<16} {x['reviews']:>8,}: {parts}")


if __name__ == "__main__":
    main()
