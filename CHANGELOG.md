# Changelog

## Unreleased
### Changed
- Python environment management moved from pip to [uv](https://docs.astral.sh/uv/): `uv.lock` (reproducible installs), `.python-version` (3.12), dev tools in a `[dependency-groups] dev` group (installed by `uv sync`), `uv run` in the Makefile, CI workflows and docs, and uv-built virtualenvs in `Dockerfile` and `Dockerfile.worker`. The optional `embeddings` extra is installed with `uv sync --extra embeddings`. Application code and behaviour are unchanged; only the setup commands and two install hints in error messages changed.
### Added
- Web UI (`frontend/`, React + TypeScript + Vite + Tailwind + TanStack Query): dashboard, search with per-ranker score breakdown, cited answers, document library, file/URL/text ingest, index rebuild, analytics (live Prometheus counters and recorded benchmarks), health and settings; light and dark themes; responsive; 21 unit tests and 15 browser tests against real servers. Served by the API at `/` when `frontend/dist` exists; built into the Docker image.
- API: `GET /api/v1/documents`, `POST /api/v1/documents/upload` (multipart), `POST /api/v1/documents/ingest-url` (SSRF-checked, size-capped), `GET /api/v1/info`, `GET /api/v1/benchmarks`; `GET /api/v1/documents/{id}` now returns chunk text. Settings `URL_INGEST_ALLOW_PRIVATE`, `BENCHMARKS_DIR`, `FRONTEND_DIR`.
- Makefile targets `frontend-*` and CI jobs `frontend` and `frontend-e2e` (the CI jobs have not been run on a runner).
### Fixed
- The engine always used the offline stub answerer: `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` were never read. They now select the OpenAI-compatible provider (with a key) or Ollama (`LLM_PROVIDER=local`); without credentials the stub is still used and says so.
- `scripts/run_beir.py`: evaluation on BEIR-format datasets (nDCG@10, Recall, MRR, HNSW recall vs exact); SciFact results in `benchmarks/results/scifact.json`.
- HNSW regression tests (recall on clustered data, directed-edge removal).
- From-scratch Porter stemmer (`BM25_STEMMER=porter`, now the default): BM25 nDCG@10 0.648 -> 0.674 (SciFact), 0.299 -> 0.315 (NFCorpus).
- NFCorpus benchmark results.
### Added (storage)
- `STORE_BACKEND=postgres`: the API and CLI now store documents/versions/chunks in Postgres (database written first, atomically; indexes rebuilt from it at startup and on `/index/rebuild`; `/ready` checks the database). Compose runs migrations and enables it. Tested against a real PostgreSQL (store, API restart, tenant ACLs, failure handling, CLI).
### Fixed
- Re-ingesting identical content created a duplicate in the CLI (and after an API restart); content hashes are now stored with chunks and the duplicate trackers are restored on load.
- `retrieval_mode=tfidf` silently returned no results (retriever never wired into the engine); now wired, with a regression test.
- Chat reported "sufficient evidence" for off-topic questions because noise-level dense matches counted as evidence; added an evidence gate (`MIN_DENSE_SIMILARITY`) and the answerer now skips the LLM call when nothing qualifies.
- Alembic migrated a hardcoded test database (`rag_test`) and ignored `DATABASE_URL`; `migrations/env.py` now uses `ALEMBIC_DATABASE_URL`/`DATABASE_URL` (asyncpg URLs converted) and `alembic check` reports no drift.
- API documents were lost on restart: added `STATE_FILE` snapshot persistence for non-Postgres runs, superseded for deployments by `STORE_BACKEND=postgres` above.
- docker-compose: production overlay no longer leaves Postgres/Redis ports published (empty `ports: []` does not clear inherited lists), no longer pairs `replicas: 2` with a fixed host port, and now requires `POSTGRES_PASSWORD`/`API_KEYS_JSON`/`METRICS_TOKEN` so auth-enabled startup cannot crash on missing credentials.
- Test suite collection: renamed `tests/evaluation/test_metrics.py` to `test_retrieval_metrics.py` (basename clashed with `tests/api/test_metrics.py`).
- BM25 scoring is term-at-a-time (avgdl/IDF computed once; running length total): about 40x faster per query on SciFact (111 ms to 2.7 ms p50), identical results.
- HNSW: directed edges + heuristic neighbor selection (recall@10 vs exact 0.75 -> 0.95 on SciFact); `remove()` now scans in-edges.
- `mypy src` is clean (23 errors fixed).

## [0.1.0] - Unreleased

### Added
- From-scratch retrieval core: tokenizer, TF-IDF, BM25, cosine similarity,
  brute-force + HNSW ANN, RRF fusion, hybrid retrieval orchestrator
- Seven chunking strategies: fixed, recursive, sentence, markdown-aware,
  code-aware, parent-child, semantic
- Heuristic (offline) and cross-encoder (network-gated) reranking
- Context selection: deduplication, token budgeting, source diversity,
  extractive compression
- Citation-validated, prompt-injection-resistant RAG generation with
  OpenAI-compatible, local/Ollama, and offline null LLM providers
- Retrieval evaluation metrics (Recall@K, Precision@K, MRR, NDCG, Hit
  Rate) and generation-quality metrics (deterministic + LLM-judge,
  clearly separated)
- PostgreSQL schema (14 tables) + Alembic migrations, repository pattern
- FastAPI application (health/ready/search/chat/documents/evaluation) and
  Typer CLI sharing one in-process retrieval engine
- Security: SSRF/path-traversal/prompt-injection defense, authorization
  filtering with cross-tenant isolation guarantees
- Docker/docker-compose (dev + prod overlay), background worker skeleton
- 164 passing tests including real PostgreSQL integration tests and
  full end-to-end FastAPI TestClient tests

### Known limitations
See docs/final-report.md "Limitations" — most notably: full 200-500
question labeled benchmark not yet built (schema + 3-question real sample
provided), production auth middleware not wired, git-repo cloning ingestor
not implemented, metrics/tracing endpoints not wired (timings are
captured, not yet exported to Prometheus/OTel).
