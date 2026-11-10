"""Save FaBERT's tokenizer and configuration once, so prediction needs no internet
(plan: reports/predictor_plan.md).

The model was trained with sbunlp/fabert as it was on the Hugging Face hub; the hub commit
then was 4cad285 (reports/transformer_model_choice.md). This reads the hub's current commit
and stops if it is another one, since the tokenizer might differ. The weights are not
downloaded: the trained ones are in artifacts/transformer_no_early_stop/model.pt.

Writes artifacts/fabert_files/ (not in git: the model's license is not declared) and
reports/fabert_files.json with the hub commit and the hash of every file.

Usage:
    uv run python scripts/save_fabert_files.py
"""

from datetime import UTC, datetime

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer

from digikala_reviews.config import REPORTS_DIR
from digikala_reviews.ingest import file_sha256
from digikala_reviews.predictor import FABERT_FILES
from digikala_reviews.provenance import git_commit
from digikala_reviews.tuning import save_json

MODEL = "sbunlp/fabert"
TRAINED_WITH = "4cad285"
OUTPUT = REPORTS_DIR / "fabert_files.json"


def main():
    commit = HfApi().model_info(MODEL).sha
    assert commit.startswith(TRAINED_WITH), (
        f"the hub's {MODEL} is now at {commit}, not {TRAINED_WITH}: check before using it"
    )
    FABERT_FILES.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(MODEL, revision=commit).save_pretrained(FABERT_FILES)
    AutoConfig.from_pretrained(MODEL, revision=commit).save_pretrained(FABERT_FILES)
    saved = {p.name: file_sha256(p) for p in sorted(FABERT_FILES.iterdir())}
    save_json(
        OUTPUT,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": MODEL,
            "hub_commit": commit,
            "files": saved,
            "code": git_commit(),
        },
    )
    for name, digest in saved.items():
        print(f"  {name}  {digest[:12]}")
    print(f"hub commit {commit}; saved {FABERT_FILES} and {OUTPUT}")


if __name__ == "__main__":
    main()
