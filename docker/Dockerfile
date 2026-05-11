# syntax=docker/dockerfile:1
# Targets: deps (venv), macro (CLI pipeline exemplo), streamlit (painel diário).

FROM python:3.11-slim-bookworm AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml /app/
COPY README.md /app/
COPY src /app/src

RUN uv sync --no-dev --extra dashboard

FROM python:3.11-slim-bookworm AS macro

ARG GIT_COMMIT=docker-build
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    MACRO_GIT_COMMIT=$GIT_COMMIT

COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml /app/
COPY src /app/src

CMD ["macro-var-example"]

FROM python:3.11-slim-bookworm AS streamlit

ARG GIT_COMMIT=docker-dashboard
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MACRO_GIT_COMMIT=$GIT_COMMIT

COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml /app/
COPY src /app/src
COPY dashboard /app/dashboard
COPY data/demo/macro_ipca_selic.parquet /app/data/demo/macro_ipca_selic.parquet

EXPOSE 8501

ENTRYPOINT []

CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
