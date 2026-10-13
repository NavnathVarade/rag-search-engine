# Troubleshooting

**`IndexNotBuiltError` from search/chat** — run `rag index build` (or
`POST /api/v1/index/rebuild`) first; the in-process index starts empty.

**Postgres connection refused** — confirm `DATABASE_URL` matches a
running instance; `docker compose up` waits on Postgres's healthcheck
before starting the API, but a bare `uvicorn` run against a stopped local
Postgres will fail fast with a clear connection error rather than hang
(see `pool_pre_ping=True` in `storage/db.py`, which also handles silent
reconnects after an idle-timeout disconnect).

**Search returns nothing for an obviously-relevant query** — check
`response.degraded`; if it lists `dense_unavailable...` or
`bm25_unavailable...`, one retriever wasn't wired up. Otherwise, use
`rag debug "<query>"` to see each ranker's individual rank/score before
fusion — a query with zero BM25 term overlap and a poor dense match (e.g.
using the offline `HashingEmbedder`, which has no real semantic
understanding — see docs/embeddings.md) can legitimately score everything
near zero.

**Citations look wrong / missing** — check `ChatAnswer.injection_flags`
and the raw LLM output; `extract_and_validate_citations` strips any `[n]`
marker that doesn't map to a supplied context chunk, so a "missing"
citation is more likely the model citing a number that was never in
range than a bug in validation.

**Embedding-related `EmbeddingError`** on `SentenceTransformerEmbedder` —
this provider requires `uv sync --extra embeddings` and network
access to huggingface.co to download weights on first use; neither is
available in this project's own development sandbox, which is why the
default provider is `HashingEmbedder`.

**mypy errors** — `mypy src` is clean (0 errors across 111 files) under the pragmatic
configuration in `pyproject.toml` (not full `--strict`). If you add code and mypy fails,
fix the annotation rather than adding an ignore.
