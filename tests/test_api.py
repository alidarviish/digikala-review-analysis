import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import MAX_CHARACTERS, create_app  # noqa: E402

LABELS = {
    "recommendation": ["recommended", "not_recommended", "no_idea"],
    "rating_band": ["low", "medium", "high"],
}


class FakePredictor:
    """Answers like src/digikala_reviews/predictor.py without any model."""

    version = {"code": {"commit": "test"}}

    def predict(self, body):
        if body == "boom":
            raise RuntimeError("an unexpected failure")
        models = {}
        for name in ("tfidf", "fabert", "final_system"):
            models[name] = {
                head: {
                    "label": labels[0],
                    "best_guess": labels[0],
                    "confidence": 0.9,
                    "probabilities": {c: (0.8 if i == 0 else 0.1) for i, c in enumerate(labels)},
                }
                for head, labels in LABELS.items()
            }
        return {
            "token_count": 4,
            "truncated": False,
            "models": models,
            "explanation": {h: {"models_agree": True, "tfidf_words_for_its_answer": []} for h in LABELS},
            "latency_ms": {"normalize": 0.1, "tfidf": 1.0, "fabert": 40.0},
            "version": self.version,
        }

    def tokenize(self, body):
        return {"tokens": ["[CLS]", "[SEP]"], "input_ids": [2, 4], "attention_mask": [1, 1], "token_count": 2}


def client(loader=FakePredictor):
    return TestClient(create_app(loader), raise_server_exceptions=False)


def not_loaded():
    raise FileNotFoundError("artifacts are missing")


def test_health_answers_even_without_models():
    with client(not_loaded) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_ready_is_503_until_the_models_are_loaded():
    with client(not_loaded) as c:
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json={"text": "عالی"}).status_code == 503
    with client() as c:
        assert c.get("/ready").status_code == 200


def test_empty_or_too_long_text_is_422():
    with client() as c:
        assert c.post("/predict", json={"text": ""}).status_code == 422
        assert c.post("/predict", json={"text": "   "}).status_code == 422
        assert c.post("/predict", json={"text": "خ" * (MAX_CHARACTERS + 1)}).status_code == 422
        assert c.post("/predict", json={}).status_code == 422


def test_unknown_model_is_400():
    with client() as c:
        assert c.post("/predict", json={"text": "عالی", "model": "bert"}).status_code == 400


def test_predict_gives_both_heads_with_probabilities():
    with client() as c:
        found = c.post("/predict", json={"text": "عالی"}).json()
    assert found["model"] == "final"
    for head, labels in LABELS.items():
        assert set(found[head]["probabilities"]) == set(labels)
    assert {"model_version", "token_count", "latency_ms"} <= set(found)


def test_each_model_and_the_comparison():
    with client() as c:
        assert c.post("/predict/tfidf", json={"text": "عالی"}).json()["model"] == "tfidf"
        assert c.post("/predict/transformer", json={"text": "عالی"}).json()["model"] == "transformer"
        both = c.post("/predict/compare", json={"text": "عالی"}).json()
    assert set(both["models"]) == {"final", "tfidf", "transformer"}


def test_normalize_and_tokenize():
    with client() as c:
        found = c.post("/normalize", json={"text": "كيفيت"}).json()
        assert found["text"] == "كيفيت" and found["normalized_text"] == "کیفیت"
        assert c.post("/tokenize", json={"text": "عالی"}).json()["token_count"] == 2


def test_unexpected_error_is_500_without_details():
    with client() as c:
        response = c.post("/predict", json={"text": "boom"})
    assert response.status_code == 500 and response.json() == {"detail": "internal error"}
