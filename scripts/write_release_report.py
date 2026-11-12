"""Write reports/release_report.json: what is delivered, how to check it, and its main results.

Everything is read from the saved reports, and every delivered file is hashed again here, so
the report states what is actually on this machine. Nothing is trained or scored again, and
the test set is not read: its scores come from reports/final_evaluation.json.

Where the files are (decided before writing this report):
- public, as assets of the GitHub release RELEASE_TAG: the split files and their manifest
  (the dataset is MIT-licensed) and the TF-IDF model;
- private, given to the reviewer with the submission: the fine-tuned FaBERT weights, the
  semi-supervised student, the rate model (for predicted_rate) and FaBERT's tokenizer files
  (FaBERT declares no licence).

Usage:
    uv run python scripts/write_release_report.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from digikala_reviews.config import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    FINAL_TRANSFORMER_DIR,
    FINAL_TRANSFORMER_RESULT,
    PROJECT_ROOT,
    REPORTS_DIR,
    SPLITS_DIR,
)
from digikala_reviews.ingest import file_sha256
from digikala_reviews.provenance import environment, git_commit, library_version
from digikala_reviews.tfidf import HEADS
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "release_report.json"
RELEASE_TAG = "v1.0"
DATASET = "https://huggingface.co/datasets/RadeAI/Digikala_comments_products"
PUBLIC = f"public: asset of the GitHub release {RELEASE_TAG}"
PRIVATE = "private: given to the reviewer with the submission (FaBERT declares no licence)"


def read(relative) -> dict:
    return json.loads((REPORTS_DIR / relative).read_text(encoding="utf-8"))


def entry(path: Path, availability: str, expected: str | None = None) -> dict:
    digest = file_sha256(path)
    assert expected is None or digest == expected, f"{path} differs from its recorded hash"
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "availability": availability,
    }


def headline(scores: dict, head: str) -> dict:
    found = {
        k: scores[k] for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "balanced_accuracy")
    }
    found["f1_per_class"] = {c: v["f1"] for c, v in scores["per_class"].items()}
    if head == "rating_band":
        found["ordinal_distance_error"] = scores["ordinal_distance_error"]
    return found


def main():
    code = git_commit()
    assert not code["uncommitted_changes"], "commit the code first"
    ingest = read("data_ingest.json")
    cleaning = read("cleaning_report.json")
    manifest = read("split_manifest.json")
    chosen = read("tfidf_final/chosen_model.json")
    fabert = read(FINAL_TRANSFORMER_RESULT)
    student = read("semi_supervised/student.json")
    uncertain = read("uncertain.json")["heads"]
    final = read("final_evaluation.json")
    semi = read("semi_supervised.json")
    predictor = read("predictor_check.json")
    tokenizer = read("fabert_files.json")
    extensions_validation = read("extensions_validation.json")
    extensions_test = read("final_evaluation_extensions.json")
    memory = read("predictor_memory.json")
    docker = read("docker_check.json")
    cache = json.loads((CACHE_DIR / "cache_info.json").read_text(encoding="utf-8"))

    files = {"split_manifest": entry(REPORTS_DIR / "split_manifest.json", PUBLIC)}
    for name, digest in manifest["sha256"].items():
        files[f"split/{name}"] = entry(SPLITS_DIR / name, PUBLIC, digest)
    files["tfidf_model"] = entry(ARTIFACTS_DIR / "tfidf_final" / "model.joblib", PUBLIC)
    files["fabert_weights"] = entry(ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "model.pt", PRIVATE)
    files["semi_supervised_student_weights"] = entry(
        ARTIFACTS_DIR / "semi_supervised_student" / "model_last.pt", PRIVATE
    )
    files["rate_model_weights"] = entry(ARTIFACTS_DIR / "transformer_rate" / "model.pt", PRIVATE)
    for name, digest in tokenizer["files"].items():
        files[f"fabert_tokenizer/{name}"] = entry(ARTIFACTS_DIR / "fabert_files" / name, PRIVATE, digest)
    assert files["fabert_weights"]["sha256"] == final["scoring"]["fabert"]["weights_sha256"]
    assert files["semi_supervised_student_weights"]["sha256"] == final["scoring"]["student"]["weights_sha256"]
    assert files["tfidf_model"]["sha256"] == final["tfidf_sha256"]
    assert files["rate_model_weights"]["sha256"] == predictor["version"]["rate_model_sha256"]

    models = {
        "tfidf": {
            "what": "TF-IDF word 1-3-grams (Persian tokens, min_df 5) and one logistic regression per head",
            "features": chosen["features"],
            "settings": {h: v["settings"] for h, v in chosen["heads"].items()},
            "offsets": {h: v["offsets_for_test"] for h, v in chosen["heads"].items()},
            "file": "tfidf_model",
        },
        "fabert": {
            "what": "sbunlp/fabert fine-tuned on train_supervised, a linear head per task on the first token",
            "hub_commit": tokenizer["hub_commit"],
            "licence": "not declared by the model's authors (checked 2026-11-08)",
            "settings": fabert["fingerprint"]["settings"],
            "kept_epoch": fabert["kept_epoch"],
            "offsets": {h: fabert["heads"][h]["offsets_for_test"] for h in HEADS},
            "file": "fabert_weights",
        },
        "final_system": {
            "what": "weighted mean of FaBERT's and TF-IDF's probabilities, per-class offsets, and "
            "'uncertain' when the mixed probability of the answer is under the threshold (official)",
            "per_head": {h: uncertain[h]["for_test"] for h in HEADS},
            "uses": ["fabert_weights", "tfidf_model"],
        },
        "semi_supervised_student": {
            "what": "FaBERT fine-tuned one more epoch with the accepted pseudo-labels; for comparison only",
            "pseudo_labeled_reviews": student["pseudo_labeled_reviews"],
            "offsets": {h: student["last_check"]["heads"][h]["offsets_for_test"] for h in HEADS},
            "file": "semi_supervised_student_weights",
        },
    }

    results = {}
    for head in HEADS:
        e = final["heads"][head]
        results[head] = {"test_reviews": e["reviews"], "models": {}, "differences_macro_f1": {}}
        for name, m in e["models"].items():
            results[head]["models"][name] = {
                "test": headline(m["with_offsets"], head),
                "test_macro_f1_interval_95_products": m["macro_f1_interval_95_products"],
                "validation": m["validation"],
            }
        u = e["models"]["final system"]["uncertain"]
        results[head]["final_system_uncertain"] = {
            "share": u["share"],
            "answered_accuracy": u["answered_reviews"]["accuracy"],
            "answered_macro_f1": u["answered_reviews"]["macro_f1"],
        }
        for pair, d in e["differences"].items():
            results[head]["differences_macro_f1"][pair] = {
                "difference": d["change"]["macro_f1"],
                "interval_95_products": d["macro_f1_interval_95_products"],
            }

    pseudo = semi["pseudo_labels"][str(semi["threshold"])]
    extensions = {
        "plan": "reports/extensions_plan.md",
        "official_result_unchanged": True,
        "exact_rate": {
            "valid_rate": "0 < rate <= 5 (D75)",
            "validation": {name: r["scores"] for name, r in extensions_validation["rate"].items()},
            "test": {name: r["scores"] for name, r in extensions_test["rate"].items()},
        },
        "test_macro_f1": {
            head: {name: m["macro_f1"] for name, m in extensions_test["heads"][head]["models"].items()}
            for head in HEADS
        },
        "test_differences_macro_f1": {head: extensions_test["heads"][head]["differences"] for head in HEADS},
        "pseudo_labels_help_by_the_rule": extensions_validation["rule_pseudo_labels_help"],
        "weights": "the rate model's weights are delivered (privately) for predicted_rate; the other "
        "extensions' weights are not",
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "project": "Digikala review analysis: recommendation and rating band from the review text alone",
        "release_tag": RELEASE_TAG,
        "official_system": "final_system",
        "input": "the review body only, normalized (normalizer version in data)",
        "data": {
            "source": DATASET,
            "licence": "MIT (dataset README)",
            "raw_files": {
                f["csv_file"]: {"sha256": f["csv_sha256"], "rows": f["rows_pandas"]} for f in ingest["files"]
            },
            "cleaning": {k: cleaning[k] for k in ("rows_raw", "rows_excluded", "rows_clean")},
            "split": manifest["split"],
            "seeds": manifest["seeds"],
            "masking": {k: manifest["masking"][k] for k in ("rate", "rule", "rounding", "masked_total")},
            "rating_band": manifest["rating_band"],
            "row_counts": manifest["row_counts"],
            "normalizer_version": cache["normalizer_version"],
            "test_set_read": "once for the four official models (scripts/final_evaluation.py), and once more "
            "for the extensions' models, reported apart (scripts/final_evaluation_extensions.py, D74); "
            "nothing was chosen on it",
        },
        "files": files,
        "models": models,
        "results": results,
        "semi_supervised": {
            "threshold": semi["threshold"],
            "hidden_reviews": semi["hidden_reviews"],
            "accepted": semi["accepted"],
            "accepted_share": pseudo["share_kept"],
            "accepted_accuracy": pseudo["accuracy"],
            "accepted_classes": pseudo["kept_classes"],
            "answer": "no measurable gain: the semi-supervised model is below the supervised one in Macro F1 "
            "on validation and test",
        },
        "offline_prediction": {
            "script": "scripts/predict.py",
            "check": {
                "passed": predictor["passed"],
                "reviews": predictor["reviews"],
                "rate": predictor["rate"],
            },
            "predicted_rate": "from the rate model (an extension); it changes no label",
            "latency_one_review_ms": {k: v["median_ms"] for k, v in predictor["latency_one_review"].items()},
            "demo": "reports/predict_demo.md",
        },
        "extensions": extensions,
        "service": {
            "code": ["app/main.py", "Dockerfile", "compose.yaml", "requirements-service.txt"],
            "endpoints": [
                "/health",
                "/ready",
                "/normalize",
                "/tokenize",
                "/predict",
                "/predict/tfidf",
                "/predict/transformer",
                "/predict/compare",
            ],
            "tests": "tests/test_api.py",
            "memory_gb": memory["memory_gb"],
            "latency_ms_one_review": memory["latency_ms_one_review"],
            "docker": {
                k: docker[k]
                for k in (
                    "image_size_gb",
                    "seconds_until_ready",
                    "models_loaded_seconds",
                    "health_check",
                    "from_inside",
                    "torch_inside",
                    "gpu_inside",
                    "all_as_expected",
                    "review_text_in_log",
                )
            },
        },
        "limitations": [
            "FaBERT declares no licence: its fine-tuned weights are not published",
            "on the test set the final system matches FaBERT on recommendation (+0.03 point, interval across "
            "zero) and leads it by 0.09 point on the rating band",
            "the middle classes (no_idea, medium) stay near F1 0.5: identical texts often carry other labels",
            "reviews longer than 128 tokens (about 0.2%) are cut for FaBERT",
            "GPU training is not bit-for-bit repeatable; decisions rest on intervals and seed spread",
            "one validation review has a rate of 2,500; the exact-rate scores leave it out (D75)",
            "the service is built and checked for the CPU only",
        ],
        "reproduce": [
            "uv sync --group transformer",
            "put the dataset's two CSV files in data/raw (see README)",
            "uvx marimo edit notebooks/run_all.py  # the steps of src/digikala_reviews/pipeline.py",
            "uv run python scripts/verify_standard_files.py  # the split files must match the hashes above",
            'uv run python scripts/predict.py "<review text>"  # after scripts/save_fabert_files.py, once',
            "uv sync --all-groups && uv run uvicorn app.main:app  # or: docker compose up --build",
        ],
        "environment": {
            **environment(),
            "torch": library_version("torch"),
        },
        "code": code,
    }
    save_json(OUTPUT, report)
    print(f"{len(files)} files hashed; official system: final_system")
    for head, r in results.items():
        m = r["models"]["final system"]
        print(
            f"  {head}: test Macro F1 {m['test']['macro_f1']:.4f}, "
            f"validation {m['validation']['macro_f1']:.4f}"
        )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
