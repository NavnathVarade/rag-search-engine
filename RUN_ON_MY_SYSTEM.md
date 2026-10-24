# Run this project on your own computer

Needs [uv](https://docs.astral.sh/uv/), which also fetches the right Python (3.12) for you. Docker and Postgres are optional.
Install uv once: `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux), `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows), or `pipx install uv`.

## Fastest way (no Docker, no database)
```bash
uv sync                              # creates .venv from uv.lock: Python 3.12 plus the exact locked dependencies

uv run rag ingest ./data/sample             # index the 3 sample documents
uv run rag search "how does kubernetes restart a failed container" --mode hybrid
uv run rag chat "What does CREATE INDEX CONCURRENTLY do?"   # stub answer unless you set an LLM key
uv run rag debug "CREATE INDEX CONCURRENTLY"                # shows how BM25 and dense search voted

uv run uvicorn rag_engine.api.app:app --port 8000           # API; docs at http://localhost:8000/docs
```
Every command is run through `uv run`, so you never need to activate `.venv`. If `rag` is not found, use `uv run python -m rag_engine.cli.main`.

## Run the tests
```bash
uv run pytest -q -m "not integration"       # no database needed
```
The Postgres tests need a database named `rag_test` (user `rag`, password `rag`) on localhost:5432.

## With Docker (Postgres storage, migrations run automatically)
```bash
docker compose up --build            # API on http://localhost:8000
```
This has not been built in the development sandbox. If it fails, send the error output.

## Production-style compose (auth on)
Set `POSTGRES_PASSWORD`, `API_KEYS_JSON`, `METRICS_TOKEN` first; see docs/deployment.md.
Create an API key with: `rag auth create-key --user-id alice --tenant-id t1`

## Benchmarks on real datasets
Download BEIR SciFact or NFCorpus, then:
```bash
uv run python scripts/run_beir.py --dataset /path/to/scifact --stem porter --tune-split train

uv run python scripts/run_beir.py `
  --dataset .\datasets\scifact `
  --stem porter `
  --tune-split train `
  --embedder sentence-transformers/all-MiniLM-L6-v2
# with a real embedding model (needs internet): run it as `uv run --extra embeddings python scripts/run_beir.py ...` and add
#   --embedder sentence-transformers/all-MiniLM-L6-v2
```

## Real LLM answers
Set `LLM_API_KEY` (and `LLM_BASE_URL`/model settings, see docs/configuration.md) in a `.env` file.

Start with README.md, docs/getting-started.md and docs/final-report.md (honest limitations).

## Web UI

The zip includes a prebuilt UI in `frontend/dist`, so once the API runs, open <http://localhost:8000/>:

```bash
uv run uvicorn rag_engine.api.app:app --port 8000      # run from the project root so it finds frontend/dist
```

To change the UI you need Node 20+ (tested on 22):

```bash
cd frontend
npm ci
npm run dev        # http://localhost:5173 with hot reload, proxying to the API on :8000
npm run build      # rebuilds frontend/dist
```

For generated answers (instead of the offline stub) set `LLM_API_KEY` in `.env` (any OpenAI-compatible endpoint via `LLM_BASE_URL`), or `LLM_PROVIDER=local` for Ollama, then restart the API. With `AUTH_ENABLED=true` create a key with `rag auth create-key` and paste it into the sign-in screen. Details: `frontend/README.md` and `docs/frontend.md`.
