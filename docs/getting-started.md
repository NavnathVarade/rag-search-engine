# Getting Started

## Prerequisites
- Docker + Docker Compose (for the containerized path), OR
- Python 3.12+ and a local PostgreSQL 16 instance (for the local dev path)

## Fastest path (Docker)
```bash
git clone <repository>
cd rag-search-engine
cp .env.example .env
docker compose up --build
# in another terminal:
curl http://localhost:8000/health
```

## Local development path (no Docker)
```bash
uv sync                # creates .venv from uv.lock (Python 3.12 + exact dependency versions); commands below run through `uv run`
cp .env.example .env   # adjust DATABASE_URL if not using Docker's postgres

# apply the schema
make migrate

# ingest the bundled sample docs and build the index
uv run rag ingest ./data/sample
uv run rag index build

# search
uv run rag search "How does Kubernetes restart a failed container?" --mode hybrid

# ask a question (uses the offline NullLLMProvider by default -- see
# docs/generation.md to point LLM_PROVIDER at a real OpenAI-compatible
# endpoint)
uv run rag chat "What does CREATE INDEX CONCURRENTLY do?"

# run the sample evaluation
uv run rag evaluate --dataset data/evaluation/sample.jsonl

# run the API
uv run uvicorn rag_engine.api.app:app --reload
```

## Python environment (uv)

This project uses [uv](https://docs.astral.sh/uv/) for the Python environment. Install uv once: `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux), `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows), or `pipx install uv`.

| Task | Command |
|---|---|
| Create or update `.venv` from the lockfile (dev tools included) | `uv sync` |
| Production-style install without dev tools | `uv sync --no-dev` |
| Add the optional real-embedding dependencies | `uv sync --extra embeddings` |
| Run any command in the environment | `uv run <command>` (no activation needed) |
| Add or upgrade a dependency | `uv add <package>`, `uv lock --upgrade-package <package>` |

`uv.lock` pins every dependency; commit it when dependencies change. `pyproject.toml` stays the single source of truth for what the project needs.

## Running tests
```bash
make test              # everything, including Postgres integration tests
make test-unit          # pure-algorithm tests only, no DB required
```

## What "hashing-embedder" means in the default output
By default `EMBEDDING_PROVIDER=hashing`, a fully offline, deterministic
embedder (see `src/rag_engine/embeddings/local.py`) used so the whole
pipeline runs with zero network access and zero model downloads. It has
no real semantic understanding of language. For production-quality
semantic search, set `EMBEDDING_PROVIDER=sentence_transformers` (requires
`uv sync --extra embeddings` and network access to download model
weights on first use) -- see docs/embeddings.md.
