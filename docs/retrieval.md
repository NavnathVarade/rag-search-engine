# Retrieval

## Modes (`config.py::RetrievalMode`, spec #20)
`dense` | `bm25` | `tfidf` | `hybrid` (RRF, equal weight) | `hybrid_weighted`
(RRF with configurable `dense_weight`/`sparse_weight`). Selected per-query
via `retrieval_mode` in `POST /api/v1/search` or `--mode` on the CLI.

## Dense retrieval
`retrieval/dense.py::DenseRetriever`: query → `embedder.embed_query()` →
`ann_index.search()` (brute-force exact or HNSW approximate, spec #14) →
ranked `RetrievalResult`s with `dense_rank`/`dense_score` populated.

## ANN
See docs/hnsw.md. `BruteForceIndex` is the exact reference
implementation (O(N) per query, always correct by construction) used both
directly (small/medium corpora) and as the ground truth HNSW's recall is
measured against.

## Graceful degradation (spec #104)
`HybridRetriever` accepts `dense_retriever=None` or `bm25_retriever=None`
and falls back to whichever retriever *is* available, reporting the
degradation explicitly in `HybridSearchResponse.degraded` rather than
silently returning empty/wrong results --
see `tests/retrieval/test_hybrid.py::test_dense_retriever_unavailable_degrades_to_bm25_only`.

## Latency
Each stage's timing is measured and returned (`HybridSearchTimings`):
`embedding_ms`, `dense_ms`, `bm25_ms`, `fusion_ms`. In this reference
implementation dense and BM25 run sequentially for simplicity/clarity;
since they're independent, a production deployment can run them
concurrently (`asyncio.gather` or a thread pool) to cut wall-clock latency
roughly to `max(dense_ms, bm25_ms)` instead of their sum -- this is called
out explicitly rather than silently claimed as already-parallel.
