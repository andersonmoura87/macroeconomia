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
COPY README.md /app/
COPY src /app/src
COPY dashboard /app/dashboard
COPY scripts/gen_demo_panel.py /app/scripts/gen_demo_panel.py
# Parquet demo não precisa de estar no Git: gera-se no build (clone-and-run).
RUN mkdir -p /app/data/demo \
    && cd /app \
    && PYTHONPATH=/app/src /app/.venv/bin/python /app/scripts/gen_demo_panel.py

EXPOSE 8501

ENTRYPOINT []

CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
