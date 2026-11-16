# ClinVerify runtime image (API + UI share this image; see ADR-0007).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models/hf \
    FASTEMBED_CACHE_PATH=/models/fastembed

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["uvicorn", "clinverify.api:app", "--host", "0.0.0.0", "--port", "8000"]
