"""How good the guessed labels of the hidden reviews are, against their true labels.

Reads reports/pseudo_labels/masked.parquet (scripts/pseudo_label.py) and the true labels of
the hidden reviews (data/splits/audit/masked_labels_reference.csv, on your own machine; used
here only to measure, never for training). Checks first that the guesses were made with this
machine's final FaBERT weights. For the thresholds 0.8, 0.9 and 0.95: the share kept, how
often the kept guesses are right, the share of each class among them, and for each true
class the share of its reviews kept with the right label.

Writes reports/pseudo_labels/check.json. The plan is in reports/semi_supervised_plan.md.

Usage:
    uv run python scripts/check_pseudo_labels.py
"""

import json
from datetime import UTC, datetime

import pandas as pd

from digikala_reviews.config import ARTIFACTS_DIR, FINAL_TRANSFORMER_DIR, PSEUDO_LABEL_THRESHOLD, REPORTS_DIR
from digikala_reviews.ingest import file_sha256
from digikala_reviews.provenance import git_commit
from digikala_reviews.pseudo import RECOMMENDATION_LABELS, quality
from digikala_reviews.standard_files import read_split_file
from digikala_reviews.tuning import save_json

THRESHOLDS = [0.8, PSEUDO_LABEL_THRESHOLD, 0.95]
FOLDER = REPORTS_DIR / "pseudo_labels"


def main():
    scoring = json.loads((FOLDER / "scoring.json").read_text(encoding="utf-8"))
    local = file_sha256(ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "model.pt")
    assert scoring["weights_sha256"] == local, "the guesses were not made with the final FaBERT weights"
    guessed = pd.read_parquet(FOLDER / "masked.parquet")
    truth = read_split_file("audit/masked_labels_reference.csv").set_index("id")
    true = truth.loc[guessed["id"], "recommendation_status"].to_numpy()
    assert pd.notna(true).all(), "every hidden review had a recommendation label"

    result = {
        "reviews": int(len(guessed)),
        "true_classes": {k: float(v) for k, v in pd.Series(true).value_counts(normalize=True).items()},
        "thresholds": quality(guessed["label"].to_numpy(), guessed["p"].to_numpy(), true, THRESHOLDS),
    }
    save_json(
        FOLDER / "check.json",
        {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **result, "code": git_commit()},
    )
    print(
        f"{result['reviews']:,} hidden reviews; true classes: "
        + ", ".join(f"{k} {v:.1%}" for k, v in result["true_classes"].items())
    )
    for threshold, r in result["thresholds"].items():
        print(
            f"p >= {threshold}: kept {r['kept']:,} ({r['share_kept']:.1%}), right {r['accuracy']:.1%}; "
            "kept classes "
            + ", ".join(f"{k} {r['kept_classes'][k]:.1%}" for k in RECOMMENDATION_LABELS)
            + "; kept and right per true class "
            + ", ".join(f"{k} {r['kept_right_per_true_class'][k]:.1%}" for k in RECOMMENDATION_LABELS)
        )


if __name__ == "__main__":
    main()
