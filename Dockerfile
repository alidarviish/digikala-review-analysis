# The prediction service (app/) on the CPU. Build and run: docker compose up --build
FROM python:3.13-slim

WORKDIR /app

# The locked versions of the service's libraries (uv export, see the README); torch from its CPU index
COPY requirements-service.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-service.txt

# The code, and the two result files the final system reads (its offsets, weight and threshold)
COPY src ./src
COPY app ./app
COPY reports/uncertain.json reports/
COPY reports/transformer_final/fabert_no_early_stop.json reports/transformer_final/
ENV PYTHONPATH=/app/src

# The model files are not in the image: compose.yaml mounts artifacts/ at run time
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
