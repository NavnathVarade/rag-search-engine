# Configuration

All settings are defined in `src/rag_engine/config.py` (Pydantic
Settings) and can be set via environment variables or a `.env` file (see
`.env.example` for the full list with defaults and comments). Config is
validated at startup -- e.g. `BM25_B` outside `[0,1]` or `BM25_K1 <= 0`
raises immediately rather than producing silently wrong scores later.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://rag:rag@localhost:5432/rag` | Postgres connection |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` (offline) or `sentence_transformers` |
| `EMBEDDING_DIMENSION` | `384` | must match the chosen model |
| `BM25_K1` / `BM25_B` | `1.5` / `0.75` | see docs/bm25.md |
| `ANN_BACKEND` | `brute_force` | `brute_force` or `hnsw` |
| `HNSW_M` / `HNSW_EF_CONSTRUCTION` / `HNSW_EF_SEARCH` | `16` / `200` / `64` | see docs/hnsw.md |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | `50` / `50` | candidates fetched per ranker before fusion |
| `RRF_K` | `60` | fusion dampening constant, see docs/hybrid-search.md |
| `RERANKER_ENABLED` / `RERANK_TOP_K` | `true` / `5` | see docs/reranking.md |
| `MAX_CONTEXT_TOKENS` | `6000` | context budget passed to the LLM |
| `CHUNKING_STRATEGY` | `recursive` | one of `fixed/recursive/sentence/markdown/code/parent_child/semantic` |
| `LLM_PROVIDER` | `openai_compatible` | `openai_compatible` / `local` / `none` |
| `AUTH_ENABLED` | `false` | `false` in dev = every request is treated as an authenticated admin |
| `RATE_LIMIT_ANONYMOUS_PER_MIN` / `RATE_LIMIT_AUTHENTICATED_PER_MIN` | `30` / `120` | |

Production profile (`configs/production.yaml`) sets safer defaults:
`AUTH_ENABLED=true`, `LOG_JSON=true`, stricter rate limits.
