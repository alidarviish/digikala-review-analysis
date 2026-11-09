"""The semi-supervised experiment as the project defines it, scored on your own machine
(plan: reports/semi_supervised_official_plan.md).

1. Pseudo-labels: the teacher's guesses (reports/semi_supervised/pseudo_labels.parquet) against
   the reference labels of the hidden reviews (data/splits/audit/masked_labels_reference.csv,
   read here only, after training, to measure): share kept, class shares, accuracy, and for each
   true class the share kept with the right label, at 0.80, 0.90 and 0.95. Only 0.90 was used.
2. The supervised model (the kept FaBERT run), the control (one more epoch on train_supervised)
   and the student (one more epoch with the pseudo-labels), each at its best check (the rule) and
   at the end of its epoch (added before the second run, so that the semi-supervised model
   the project compares is the one the extra training made), on the whole validation set: every
   required score with and without offsets, and the paired differences in Macro F1 with
   offsets, with intervals that draw products (1,000 rounds, seed 2026).
3. The rule of the plan: the pseudo-labels help if the student beats the control on
   recommendation by more than 0.0012 with the whole interval above zero.

Checks first that the teacher, the student and the control all started from this machine's
kept FaBERT weights, and that each run's saved labels score its recorded Macro F1. The test set
is not read: the same comparison on it is part of the final evaluation.

Writes reports/semi_supervised.json and prints a summary.

Usage:
    uv run python scripts/check_semi_supervised.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recheck_decisions import confusion, macro_f1  # noqa: E402

from digikala_reviews.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CACHE_DIR,
    FINAL_TRANSFORMER_DIR,
    PSEUDO_LABEL_THRESHOLD,
    REPORTS_DIR,
    SPLITS_DIR,
)
from digikala_reviews.ingest import file_sha256  # noqa: E402
from digikala_reviews.metrics import classification_report, ordinal_distance_error  # noqa: E402
from digikala_reviews.provenance import check_cache, git_commit  # noqa: E402
from digikala_reviews.pseudo import quality  # noqa: E402
from digikala_reviews.standard_files import read_split_file  # noqa: E402
from digikala_reviews.tfidf import HEADS, load_cache  # noqa: E402
from digikala_reviews.tuning import save_json  # noqa: E402

FOLDER = REPORTS_DIR / "semi_supervised"
OUTPUT = REPORTS_DIR / "semi_supervised.json"
THRESHOLDS = [0.8, PSEUDO_LABEL_THRESHOLD, 0.95]
MIN_LEAD = 0.0012  # the spread of the kept run over three seeds (reports/full_seeds.json)
ROUNDS, SEED = 1000, 2026
MODELS = {  # name: (folder of its predictions, its report relative to reports/, which weights)
    "supervised": (FINAL_TRANSFORMER_DIR, "transformer_final/fabert_no_early_stop.json", "kept"),
    "control": ("semi_supervised_control", "semi_supervised/control.json", "kept"),
    "student": ("semi_supervised_student", "semi_supervised/student.json", "kept"),
    # the weights at the end of the extra epoch, which that training actually made
    "control at the end": ("semi_supervised_control", "semi_supervised/control.json", "last"),
    "student at the end": ("semi_supervised_student", "semi_supervised/student.json", "last"),
}
PAIRS = [
    ("student", "control"),
    ("student", "supervised"),
    ("control", "supervised"),
    ("student at the end", "control at the end"),
    ("student at the end", "supervised"),
    ("control at the end", "supervised"),
]


def read(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scores(y_true, found, labels, head) -> dict:
    report = classification_report(y_true, found, labels)
    per_class = report["per_class"].values()
    report["macro_precision"] = float(np.mean([c["precision"] for c in per_class]))
    report["macro_recall"] = float(np.mean([c["recall"] for c in per_class]))
    if head == "rating_band":
        report["ordinal_distance_error"] = ordinal_distance_error(y_true, found, labels)
    return report


def main():
    kept_hash = file_sha256(ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "model.pt")
    teacher = read(FOLDER / "teacher.json")
    records = {name: read(REPORTS_DIR / report) for name, (_, report, _) in MODELS.items()}
    assert teacher["weights_sha256"] == kept_hash, "the guesses were not made with the kept weights"
    assert teacher["temperature"] == 1.0, "the guesses must come from the model's own softmax"
    for name in ("control", "student"):
        settings = records[name]["fingerprint"]["settings"]
        assert settings["start_sha256"] == kept_hash, f"the {name} did not start from the kept weights"
    guessed = pd.read_parquet(FOLDER / "pseudo_labels.parquet")
    assert records["student"]["fingerprint"]["settings"]["pseudo_labels_sha256"] == file_sha256(
        FOLDER / "pseudo_labels.parquet"
    ), "the student used other pseudo-labels"
    accepted = int((guessed["p"] >= PSEUDO_LABEL_THRESHOLD).sum())
    assert records["student"]["pseudo_labeled_reviews"] == accepted, "accepted count differs"

    reference = read_split_file(SPLITS_DIR / "audit" / "masked_labels_reference.csv")
    joined = guessed.merge(
        reference[["id", "recommendation_status"]], on="id", how="left", validate="one_to_one"
    )
    assert joined["recommendation_status"].notna().all(), "a hidden review has no reference label"
    pseudo = quality(joined["label"], joined["p"], joined["recommendation_status"], THRESHOLDS)
    pseudo["reference_class_shares"] = joined["recommendation_status"].value_counts(normalize=True).to_dict()

    validation = load_cache("validation.parquet", extra_columns=["product_id"])
    data = check_cache(CACHE_DIR, ["validation.parquet"])
    tables = {}
    for name, (folder, _, weights) in MODELS.items():
        file = (
            "validation_predictions.parquet" if weights == "kept" else "validation_predictions_last.parquet"
        )
        table = pd.read_parquet(ARTIFACTS_DIR / folder / file)
        assert table["id"].is_unique and set(table["id"]) == set(validation["id"]), f"{name}: ids differ"
        tables[name] = validation[["id"]].merge(table, on="id", how="left", validate="one_to_one")
    _, product_index = np.unique(validation["product_id"].to_numpy(), return_inverse=True)

    heads = {}
    for head, spec in HEADS.items():
        labels = spec["labels"]
        rows = validation[spec["column"]].notna().to_numpy()
        y_true = validation[spec["column"]].to_numpy()[rows].astype(str)
        codes = np.array([labels.index(v) for v in y_true])
        entry, predicted = {"models": {}}, {}
        for name, t in tables.items():
            with_offsets = t[f"{head}__predicted_with_offsets"].to_numpy()[rows].astype(str)
            log_proba = t[[f"{head}__log_proba__{c}" for c in labels]].to_numpy(dtype=float)[rows]
            plain = np.array(labels)[log_proba.argmax(axis=1)]  # the model's own top class
            found = {"with_offsets": scores(y_true, with_offsets, labels, head)}
            found["without_offsets"] = scores(y_true, plain, labels, head)
            record = records[name] if MODELS[name][2] == "kept" else records[name]["last_check"]
            recorded = record["heads"][head]["with_offsets"]["macro_f1"]
            assert abs(found["with_offsets"]["macro_f1"] - recorded) < 1e-9, (
                f"{head} {name}: Macro F1 differs"
            )
            entry["models"][name] = found
            predicted[name] = np.array([labels.index(v) for v in with_offsets])
        _, products = np.unique(product_index[rows], return_inverse=True)
        n_products = int(products.max()) + 1
        rng = np.random.default_rng(SEED)
        draws = {pair: [] for pair in PAIRS}
        for _ in range(ROUNDS):
            w = np.bincount(rng.integers(0, n_products, n_products), minlength=n_products)[products].astype(
                float
            )
            f1 = {name: macro_f1(confusion(codes, p, w, len(labels))) for name, p in predicted.items()}
            for a, b in PAIRS:
                draws[(a, b)].append(f1[a] - f1[b])
        entry["differences_with_offsets"] = {}
        for (a, b), found in draws.items():
            low, high = np.percentile(found, [2.5, 97.5])
            entry["differences_with_offsets"][f"{a} minus {b}"] = {
                "difference": entry["models"][a]["with_offsets"]["macro_f1"]
                - entry["models"][b]["with_offsets"]["macro_f1"],
                "interval_95_products": [float(low), float(high)],
            }
        heads[head] = entry

    lead = heads["recommendation"]["differences_with_offsets"]["student minus control"]
    helps = lead["interval_95_products"][0] > 0 and lead["difference"] > MIN_LEAD
    result = {
        "plan": "reports/semi_supervised_official_plan.md",
        "threshold": PSEUDO_LABEL_THRESHOLD,
        "hidden_reviews": int(len(guessed)),
        "accepted": accepted,
        "pseudo_labels": pseudo,
        "kept_check": {name: records[name]["kept_epoch"] for name in ("supervised", "control", "student")},
        "heads": heads,
        "rule": {"min_lead": MIN_LEAD, "lead": lead, "pseudo_labels_help": bool(helps)},
        "data": data,
    }
    save_json(
        OUTPUT,
        {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **result, "code": git_commit()},
    )

    print(f"hidden reviews {len(guessed):,}; accepted at {PSEUDO_LABEL_THRESHOLD}: {accepted:,}")
    for t, q in pseudo.items():
        if t == "reference_class_shares":
            continue
        classes = ", ".join(f"{k} {v:.1%}" for k, v in q["kept_classes"].items())
        per_true = ", ".join(f"{k} {v:.1%}" for k, v in q["kept_right_per_true_class"].items())
        print(f"  {t}: kept {q['share_kept']:.1%}, right {q['accuracy']:.1%}; kept classes {classes}")
        print(f"        kept and right, share of each true class: {per_true}")
    print("kept check (epoch): " + ", ".join(f"{k} {v}" for k, v in result["kept_check"].items()))
    for head, e in heads.items():
        print(f"\n=== {head}")
        print(
            "  model                accuracy  macro F1  macro P  macro R  (with offsets)"
            "   macro F1 without offsets"
        )
        for name, m in e["models"].items():
            w, p = m["with_offsets"], m["without_offsets"]
            print(
                f"  {name:<20} {w['accuracy']:.4f}    {w['macro_f1']:.4f}    {w['macro_precision']:.4f}   "
                f"{w['macro_recall']:.4f}                   {p['macro_f1']:.4f}"
            )
        for pair, d in e["differences_with_offsets"].items():
            low, high = d["interval_95_products"]
            print(f"  {pair}: {d['difference']:+.4f} [{low:+.4f}, {high:+.4f}]")
    print(f"\npseudo-labels help by the rule: {'yes' if helps else 'no'}")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
