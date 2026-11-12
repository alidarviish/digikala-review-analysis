"""The saved models, loaded once, answering for review texts (reports/predictor_plan.md).

Only the review text goes in. It is normalized as in training, then scored by:
- TF-IDF (artifacts/tfidf_final/model.joblib) with its offsets for the test set;
- FaBERT (artifacts/transformer_no_early_stop/model.pt) on the CPU, cut at 128 tokens as in
  training, with its offsets for the test set; the tokenizer and the model's configuration
  come from artifacts/fabert_files (scripts/save_fabert_files.py), so nothing is downloaded;
- the final system: the weighted mean of both models' probabilities and its offsets, with
  "uncertain" when the mixed probability of its answer is under the threshold
  (reports/uncertain.json, "for_test");
- the exact rate (an extension, reports/extensions_plan.md): FaBERT with a rate head
  (artifacts/transformer_rate/model.pt, pipeline step 113), its prediction clipped to [0, 5].
  Only its rate is used; it changes no label. Without that file the predicted rate is None.

The explanation is the words of the review that pushed the TF-IDF model most towards its
answer (weight of the word for that class, minus its mean weight over the classes, times
its TF-IDF value).
"""

import json
import time

import numpy as np

from digikala_reviews.config import (
    ARTIFACTS_DIR,
    FINAL_TRANSFORMER_DIR,
    FINAL_TRANSFORMER_RESULT,
    REPORTS_DIR,
)
from digikala_reviews.ensemble import blend
from digikala_reviews.experiments import predict_with_offsets
from digikala_reviews.ingest import file_sha256
from digikala_reviews.normalize import NORMALIZER_VERSION, normalize
from digikala_reviews.provenance import git_commit
from digikala_reviews.tfidf import HEADS

TFIDF_MODEL = ARTIFACTS_DIR / "tfidf_final" / "model.joblib"
FABERT_MODEL = ARTIFACTS_DIR / FINAL_TRANSFORMER_DIR / "model.pt"
FABERT_FILES = ARTIFACTS_DIR / "fabert_files"
RATE_MODEL = ARTIFACTS_DIR / "transformer_rate" / "model.pt"
RATE_RESULT = REPORTS_DIR / "transformer_final" / "rate.json"
UNCERTAIN = REPORTS_DIR / "uncertain.json"
UNCERTAIN_LABEL = "uncertain"
TOP_WORDS = 5


def offsets_array(offsets: dict, labels: list[str]) -> np.ndarray:
    return np.array([offsets[c] for c in labels], dtype=float)


def final_answer(log_fabert, log_tfidf, weight, offsets, threshold):
    """The final system's class codes, the mixed probability of each answer, and whether that
    probability is under the threshold ("uncertain")."""
    mixed = blend(log_fabert, log_tfidf, weight)
    codes = predict_with_offsets(mixed, offsets)
    confidence = np.exp(mixed[np.arange(len(codes)), codes])
    return codes, confidence, confidence < threshold


def top_words(row, coef, names, code, n=TOP_WORDS) -> list[dict]:
    """The terms of one review (a 1-row sparse matrix) that pushed most towards class `code`."""
    row = row.tocsr()
    columns, values = row.indices, row.data
    push = values * (coef[code, columns] - coef[:, columns].mean(axis=0))
    order = np.argsort(-push)[:n]
    return [{"term": str(names[columns[i]]), "push": round(float(push[i]), 4)} for i in order if push[i] > 0]


class Predictor:
    def __init__(self):
        import joblib
        import torch
        from transformers import AutoConfig, AutoTokenizer

        from digikala_reviews.finetune import build_model

        assert FABERT_FILES.is_dir(), "run scripts/save_fabert_files.py once first"
        self.torch = torch
        tfidf = joblib.load(TFIDF_MODEL)
        self.vectorizer, self.tfidf_models = tfidf["vectorizer"], tfidf["models"]
        self.names = self.vectorizer.get_feature_names_out()
        # the models keep their classes in alphabetical order; here they follow HEADS' labels
        self.tfidf_order = {
            head: [list(model.classes_.astype(str)).index(c) for c in HEADS[head]["labels"]]
            for head, model in self.tfidf_models.items()
        }
        fabert_report = json.loads((REPORTS_DIR / FINAL_TRANSFORMER_RESULT).read_text(encoding="utf-8"))
        final = json.loads(UNCERTAIN.read_text(encoding="utf-8"))["heads"]
        self.offsets = {"tfidf": {}, "fabert": {}}
        self.final = {}
        for head, spec in HEADS.items():
            labels = spec["labels"]
            self.offsets["tfidf"][head] = offsets_array(tfidf["offsets_for_test"][head], labels)
            self.offsets["fabert"][head] = offsets_array(
                fabert_report["heads"][head]["offsets_for_test"], labels
            )
            f = final[head]["for_test"]
            self.final[head] = (f["fabert_weight"], offsets_array(f["offsets"], labels), f["threshold"])

        self.tokenizer = AutoTokenizer.from_pretrained(FABERT_FILES)
        config = AutoConfig.from_pretrained(FABERT_FILES)

        def load(path, rate_head):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            settings = saved["settings"]
            model = build_model(
                torch, settings["model"], settings["dropout"], list(HEADS), config=config, rate_head=rate_head
            )
            model.load_state_dict({k: v.float() for k, v in saved["weights"].items()})
            model.eval()
            return model, settings

        self.model, settings = load(FABERT_MODEL, rate_head=False)
        self.max_length = settings["max_length"]
        self.rate_model = None
        if RATE_MODEL.exists():
            self.rate_model, rate = load(RATE_MODEL, rate_head=True)
            assert rate["max_length"] == self.max_length and rate["model"] == settings["model"]
            self.rate_scale = (rate["rate_mean"], rate["rate_std"])
        self.version = {
            "code": git_commit(),
            "normalizer_version": NORMALIZER_VERSION,
            "tfidf_sha256": file_sha256(TFIDF_MODEL),
            "fabert_sha256": file_sha256(FABERT_MODEL),
            "fabert_kept_epoch": fabert_report["kept_epoch"],
        }
        if self.rate_model is not None:
            self.version["rate_model_sha256"] = file_sha256(RATE_MODEL)
            self.version["rate_model_kept_epoch"] = json.loads(RATE_RESULT.read_text(encoding="utf-8"))[
                "kept_epoch"
            ]

    def tfidf_scores(self, texts):
        """Log-probabilities of each head, in the order of HEADS' labels, and the features."""
        x = self.vectorizer.transform(texts)
        found = {}
        for head in HEADS:
            found[head] = (
                self.tfidf_models[head].predict_log_proba(x)[:, self.tfidf_order[head]].astype(float)
            )
        return found, x

    def batches(self, texts, batch_size):
        for i in range(0, len(texts), batch_size):
            yield self.tokenizer(
                list(texts[i : i + batch_size]),
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )

    def fabert_scores(self, texts, batch_size=64):
        """Log-probabilities of each head, in the order of HEADS' labels."""
        torch = self.torch
        parts = {head: [] for head in HEADS}
        with torch.no_grad():
            for batch in self.batches(texts, batch_size):
                outputs = self.model(batch["input_ids"], batch["attention_mask"])
                for head in HEADS:
                    parts[head].append(torch.log_softmax(outputs[head].float(), dim=-1).numpy())
        return {head: np.vstack(p).astype(float) for head, p in parts.items()}

    def rate_scores(self, texts, batch_size=64):
        """The predicted rate of each text, on the 0 to 5 scale."""
        torch = self.torch
        mean, std = self.rate_scale
        parts = []
        with torch.no_grad():
            for batch in self.batches(texts, batch_size):
                parts.append(
                    self.rate_model(batch["input_ids"], batch["attention_mask"])["rate"].float().numpy()
                )
        return np.clip(np.concatenate(parts).astype(float) * std + mean, 0, 5)

    def tokenize(self, body: str) -> dict:
        """The review as FaBERT reads it: normalized, cut at max_length tokens."""
        text = normalize(body)
        count = len(self.tokenizer(text)["input_ids"])  # before the cut
        cut = self.tokenizer(text, truncation=True, max_length=self.max_length)
        return {
            "normalized_text": text,
            "tokens": self.tokenizer.convert_ids_to_tokens(cut["input_ids"]),
            "input_ids": cut["input_ids"],
            "attention_mask": cut["attention_mask"],
            "token_count": count,
            "truncated": count > self.max_length,
        }

    def predict(self, body: str) -> dict:
        """Everything the system says about one review."""
        start = time.perf_counter()
        text = normalize(body)
        normalize_ms = (time.perf_counter() - start) * 1000
        ids = self.tokenizer(text)["input_ids"]  # not cut, to count the tokens

        start = time.perf_counter()
        log_tfidf, x = self.tfidf_scores([text])
        tfidf_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        log_fabert = self.fabert_scores([text])
        fabert_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        predicted_rate = None if self.rate_model is None else round(float(self.rate_scores([text])[0]), 2)
        rate_ms = (time.perf_counter() - start) * 1000

        models = {"tfidf": {}, "fabert": {}, "final_system": {}}
        explanation = {}
        for head, spec in HEADS.items():
            labels = spec["labels"]
            for name, log in [("tfidf", log_tfidf[head]), ("fabert", log_fabert[head])]:
                code = int(predict_with_offsets(log, self.offsets[name][head])[0])
                models[name][head] = {
                    "label": labels[code],
                    "probabilities": dict(zip(labels, np.exp(log[0]).round(4).tolist(), strict=True)),
                }
            weight, offsets, threshold = self.final[head]
            codes, confidence, unsure = final_answer(
                log_fabert[head], log_tfidf[head], weight, offsets, threshold
            )
            mixed = np.exp(blend(log_fabert[head], log_tfidf[head], weight)[0])
            models["final_system"][head] = {
                "label": UNCERTAIN_LABEL if unsure[0] else labels[int(codes[0])],
                "best_guess": labels[int(codes[0])],
                "confidence": round(float(confidence[0]), 4),
                "threshold": round(float(threshold), 4),
                "probabilities": dict(zip(labels, mixed.round(4).tolist(), strict=True)),
            }
            tfidf_code = labels.index(models["tfidf"][head]["label"])
            coef = self.tfidf_models[head].coef_[self.tfidf_order[head]]
            explanation[head] = {
                "models_agree": models["tfidf"][head]["label"] == models["fabert"][head]["label"],
                "tfidf_words_for_its_answer": top_words(x[0], coef, self.names, tfidf_code),
            }
        return {
            "text": body,
            "normalized_text": text,
            "tokens": self.tokenizer.convert_ids_to_tokens(ids),
            "token_count": len(ids),
            "truncated": len(ids) > self.max_length,
            "models": models,
            "predicted_rate": predicted_rate,
            "explanation": explanation,
            "latency_ms": {
                "normalize": round(normalize_ms, 2),
                "tfidf": round(tfidf_ms, 2),
                "fabert": round(fabert_ms, 2),
                "rate": round(rate_ms, 2),
            },
            "version": self.version,
        }
