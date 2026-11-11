"""The prediction service inside its Docker container (reports/extensions_plan.md, extension 8).

Starts the container of compose.yaml, waits until /ready answers, sends the requests below from
outside the container, reads the image size, the health check's state, the loading time from the
service's log and whether torch sees a GPU inside, checks that no review text reached the log, and
stops the container. The image must be built first. The review texts are made up; no data file is
read.

Writes reports/docker_check.json and prints the numbers.

Usage:
    docker compose build
    uv run python scripts/check_docker.py
"""

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

from digikala_reviews.config import PROJECT_ROOT, REPORTS_DIR
from digikala_reviews.provenance import git_commit
from digikala_reviews.tuning import save_json

OUTPUT = REPORTS_DIR / "docker_check.json"
IMAGE = "digikala-review-api"  # compose.yaml
URL = "http://127.0.0.1:8000"
REVIEW = "کیفیتش خوبه ولی قیمتش بالاست"
REQUESTS = {  # name: path, body, expected status
    "health": ("/health", None, 200),
    "ready": ("/ready", None, 200),
    "normalize": ("/normalize", {"text": REVIEW}, 200),
    "tokenize": ("/tokenize", {"text": REVIEW}, 200),
    "predict (final system)": ("/predict", {"text": REVIEW}, 200),
    "predict tfidf": ("/predict/tfidf", {"text": REVIEW}, 200),
    "predict transformer": ("/predict/transformer", {"text": REVIEW}, 200),
    "compare": ("/predict/compare", {"text": REVIEW}, 200),
    "empty text": ("/predict", {"text": "   "}, 422),
    "text too long": ("/predict", {"text": "خوب " * 2_000}, 422),
    "unknown model": ("/predict", {"text": REVIEW, "model": "bert"}, 400),
}


def run(*command) -> str:
    return subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def request(path, body=None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    sent = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(sent, timeout=60) as answer:
            return answer.status, json.load(answer)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def wait_until_ready(limit_seconds=300) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < limit_seconds:
        try:
            status, _ = request("/ready")
        except OSError:  # the service is still loading the models and does not answer yet
            time.sleep(1)
            continue
        if status != 200:
            raise SystemExit("the service runs but the models were not loaded: see docker compose logs")
        return round(time.perf_counter() - start, 1)
    raise SystemExit(f"the service was not ready after {limit_seconds} seconds")


def health_state(container, limit_seconds=120) -> str:
    """The Docker health check's state, once it is no longer "starting" (it runs every 30 seconds)."""
    start = time.perf_counter()
    state = "starting"
    while state == "starting" and time.perf_counter() - start < limit_seconds:
        time.sleep(5)
        state = run("docker", "inspect", "--format", "{{.State.Health.Status}}", container)
    return state


def main():
    image_bytes = int(run("docker", "image", "inspect", "--format", "{{.Size}}", IMAGE))
    run("docker", "compose", "up", "-d")
    try:
        seconds_until_ready = wait_until_ready()
        checks, labels = {}, None
        for name, (path, body, expected) in REQUESTS.items():
            start = time.perf_counter()
            status, answer = request(path, body)
            checks[name] = {
                "status": status,
                "expected": expected,
                "ms": round((time.perf_counter() - start) * 1000, 1),
            }
            if name == "predict (final system)" and status == 200:
                labels = {head: answer[head]["label"] for head in ("recommendation", "rating_band")}
        container = run("docker", "compose", "ps", "-q", "api")
        state = health_state(container)
        inside = run(
            "docker",
            "compose",
            "exec",
            "-T",
            "api",
            "python",
            "-c",
            "import torch; print(torch.__version__, torch.cuda.is_available())",
        ).split()
        log = run("docker", "compose", "logs", "api")
    finally:
        run("docker", "compose", "down")

    loaded = re.search(r"models loaded in ([0-9.]+) s", log)
    result = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan": "reports/extensions_plan.md",
        "image": IMAGE,
        "image_size_gb": round(image_bytes / 1e9, 3),
        "image_list": json.loads(run("docker", "image", "ls", IMAGE, "--format", "{{json .}}")),
        "docker_version": run("docker", "version", "--format", "{{.Server.Version}}"),
        "seconds_until_ready": seconds_until_ready,
        "models_loaded_seconds": float(loaded.group(1)) if loaded else None,
        "health_check": state,
        "torch_inside": inside[0],
        "gpu_inside": inside[1] == "True",
        "requests": checks,
        "all_as_expected": all(c["status"] == c["expected"] for c in checks.values()),
        "example_answer": labels,
        "review_text_in_log": REVIEW in log or "خوب خوب" in log,
        "code": git_commit(),
    }
    save_json(OUTPUT, result)
    print(f"image {result['image_size_gb']} GB, ready after {seconds_until_ready} s", end="")
    print(f" (models loaded in {result['models_loaded_seconds']} s), health check: {state}")
    print(f"torch {inside[0]}, GPU inside: {result['gpu_inside']}")
    for name, c in checks.items():
        print(f"  {name:24} {c['status']} (expected {c['expected']})  {c['ms']} ms")
    print(f"all as expected: {result['all_as_expected']}; review text in the log: ", end="")
    print(result["review_text_in_log"])
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
