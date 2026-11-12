"""The prediction service: the saved models behind a small HTTP API (FastAPI).

The models are loaded once, when the service starts (src/digikala_reviews/predictor.py, the
same code as scripts/predict.py). Requests carry the review text only, never a label or a rate.

  GET  /health                 the service is alive
  GET  /ready                  the models are loaded (503 until they are)
  POST /normalize              the text and its normalized form
  POST /tokenize               FaBERT's tokens, input_ids, attention_mask, token count, truncation
  POST /predict                one model, "final" (the default), "tfidf" or "transformer"
  POST /predict/tfidf          the TF-IDF model
  POST /predict/transformer    the fine-tuned FaBERT model
  POST /predict/compare        the three side by side, with each one's latency

Every prediction also has predicted_rate: the exact rate (0 to 5) from FaBERT with a rate head,
an extension (reports/extensions_plan.md) that changes no label.

Failures: 422 for an empty text or one longer than MAX_CHARACTERS, 400 for an unknown model,
503 when the models are not ready, 500 for anything unexpected. The review text is never logged:
the log has the method, the path, the status and the time only.

Run:
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from digikala_reviews.normalize import normalize
from digikala_reviews.tfidf import HEADS

MAX_CHARACTERS = 5_000
MODELS = {
    "final": "final_system",
    "tfidf": "tfidf",
    "transformer": "fabert",
}  # name in the API: in the predictor
log = logging.getLogger("digikala.api")


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARACTERS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("the text is empty")
        return text


class PredictIn(TextIn):
    model: str = "final"


def load_predictor():
    from digikala_reviews.predictor import Predictor

    return Predictor()


def one_model(result: dict, model: str) -> dict:
    """The answer of one model, in the shape the project asks for."""
    key = MODELS[model]
    answer = {
        "model": model,
        "model_version": result["version"],
        "token_count": result["token_count"],
        "truncated": result["truncated"],
        "predicted_rate": result["predicted_rate"],
    }
    for head in HEADS:
        found = result["models"][key][head]
        answer[head] = {"label": found["label"], "probabilities": found["probabilities"]}
        if model == "final":  # its best guess when it answers "uncertain", and how sure it is
            answer[head].update(best_guess=found["best_guess"], confidence=found["confidence"])
    latency = result["latency_ms"]
    answer["latency_ms"] = {
        "tfidf": latency["normalize"] + latency["tfidf"],
        "transformer": latency["normalize"] + latency["fabert"],
        "final": latency["normalize"] + latency["tfidf"] + latency["fabert"],
    }[model]
    return answer


def create_app(loader=load_predictor) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        start = time.perf_counter()
        try:
            app.state.predictor = loader()
            log.info("models loaded in %.1f s", time.perf_counter() - start)
        except Exception:  # the service stays up and answers 503 until it is restarted
            log.exception("the models could not be loaded")
        yield

    app = FastAPI(title="Digikala review analysis", lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        log.info("%s %s %d %.1f ms", request.method, request.url.path, response.status_code, ms)
        return response

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        log.error("unexpected error on %s: %s", request.url.path, type(error).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    def predictor():
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="the models are not loaded")
        return app.state.predictor

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "model_version": predictor().version}

    @app.post("/normalize")
    def normalize_text(request: TextIn):
        return {"text": request.text, "normalized_text": normalize(request.text)}

    @app.post("/tokenize")
    def tokenize(request: TextIn):
        return predictor().tokenize(request.text)

    @app.post("/predict")
    def predict(request: PredictIn):
        if request.model not in MODELS:
            raise HTTPException(status_code=400, detail=f"unknown model; use one of {sorted(MODELS)}")
        return one_model(predictor().predict(request.text), request.model)

    @app.post("/predict/tfidf")
    def predict_tfidf(request: TextIn):
        return one_model(predictor().predict(request.text), "tfidf")

    @app.post("/predict/transformer")
    def predict_transformer(request: TextIn):
        return one_model(predictor().predict(request.text), "transformer")

    @app.post("/predict/compare")
    def compare(request: TextIn):
        result = predictor().predict(request.text)
        return {
            "models": {model: one_model(result, model) for model in MODELS},
            "predicted_rate": {"value": result["predicted_rate"], "latency_ms": result["latency_ms"]["rate"]},
            "models_agree": {h: e["models_agree"] for h, e in result["explanation"].items()},
            "tfidf_words": {h: e["tfidf_words_for_its_answer"] for h, e in result["explanation"].items()},
        }

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
