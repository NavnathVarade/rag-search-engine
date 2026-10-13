# Multi-stage build (spec #57): build deps in one stage, ship a minimal
# runtime image with no compilers/build tools, running as non-root.
# Stage 1: build the React web UI (served by the API at /).
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS builder

# uv builds the virtualenv from uv.lock (exact, reproducible versions). It lives at /app/.venv in both stages so
# its scripts keep working when copied into the runtime image.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Dependencies first (cached until pyproject.toml or uv.lock change), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-editable
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 raguser

COPY --from=builder /app/.venv /app/.venv
COPY configs /app/configs
COPY migrations /app/migrations
COPY alembic.ini /app/alembic.ini
COPY --from=frontend /frontend/dist /app/frontend/dist
COPY benchmarks/results /app/benchmarks/results

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/data && chown -R raguser:raguser /app
USER raguser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "rag_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
