# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgomp1 \
    && python -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md alembic.ini /app/
COPY core /app/core
COPY api /app/api
COPY adapters /app/adapters
COPY db /app/db
COPY workers /app/workers

RUN pip install -e .


FROM python:${PYTHON_VERSION}-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=deps /opt/venv /opt/venv
COPY pyproject.toml README.md alembic.ini /app/
COPY core /app/core
COPY api /app/api
COPY adapters /app/adapters
COPY db /app/db
COPY workers /app/workers

RUN mkdir -p /app/chroma_store /app/data/exhibits \
    /home/appuser/.cache/huggingface/hub \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]


FROM runtime-base AS api

EXPOSE 8081
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8081", "--timeout-graceful-shutdown", "30"]


FROM runtime-base AS adapter-telegram

CMD ["python", "-m", "adapters.telegram.app"]


FROM runtime-base AS vlm-worker

CMD ["python", "-m", "workers.vlm_worker"]


FROM runtime-base AS migrator

CMD ["alembic", "upgrade", "head"]
