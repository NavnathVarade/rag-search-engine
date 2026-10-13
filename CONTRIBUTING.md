# Contributing

## Workflow
1. `uv sync` (installs the locked dependencies and dev tools into `.venv`; commit `uv.lock` whenever you change dependencies)
2. `make check` before opening a PR (lint + typecheck + test)
3. Follow Conventional Commits: `feat: add bm25 retrieval`,
   `fix: prevent duplicate document ingestion`, `test: add rrf regression
   tests`, `docs: explain hybrid retrieval`, `refactor: isolate embedding
   provider`

## Principles for this codebase specifically
- **No black-box retrieval.** Core algorithms (tokenizer, BM25, TF-IDF,
  cosine, ANN, RRF) must remain implemented from scratch in plain
  Python/NumPy — see docs/adr/ADR-003-bm25-implementation.md for why.
  Don't add a dependency that silently replaces one of these.
- **No mocking the algorithm under test.** Mocks are fine at true external
  boundaries (HTTP APIs, model downloads) — never mock BM25 in a BM25
  test, RRF in an RRF test, etc.
- **No fabricated numbers.** Benchmark/evaluation numbers in docs must be
  reproducible by running the referenced script; if you haven't run it,
  say so and give the reproduction command instead of inventing a figure.
- **Every new retrieval code path must call `filter_authorized()`** before
  results reach context selection — see docs/adr/ADR-010-security-model.md.

## Tests
- `tests/unit/`, `tests/retrieval/`, `tests/ingestion/`, `tests/evaluation/`
  — no external dependencies required
- `tests/integration/`, `tests/api/` — some require a running Postgres
  (`TEST_DATABASE_URL` env var, defaults to `postgresql+asyncpg://rag:rag@localhost:5432/rag_test`)
- `tests/security/` — SSRF, path traversal, prompt injection, authorization
