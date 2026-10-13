# Reranking

## Retriever vs reranker
The retriever (dense + BM25 + RRF) must be **broad and fast**: it scans
the entire corpus (or an ANN approximation of it) and returns a
recall-oriented top-50. The reranker must be **narrow and precise**: it
re-scores only those 50 candidates with a much more expensive, much more
accurate model, and returns the top-5 that actually go to the LLM.

## Bi-encoder vs cross-encoder
A **bi-encoder** (what `DenseRetriever` uses) embeds the query and each
document *independently* -- document embeddings can be precomputed once
and stored, so search is "compare one new query vector against N stored
vectors", which scales to millions of documents. The cost is that the
model never sees the query and document together, so it can miss
fine-grained interactions.

A **cross-encoder** feeds the *concatenated* (query, document) pair
through the model together, so every query token can attend to every
document token (true cross-attention). This is substantially more
accurate but cannot be precomputed -- it requires one full model forward
pass per (query, candidate) pair at query time, which is why it is only
ever run on a small candidate set (spec #22: never the whole corpus).

## Why not send the top-50 straight to the LLM?
Cost and quality both suffer: 50 chunks at even 200 tokens each is
10,000 tokens of context before the question is even asked, most of it
irrelevant noise that dilutes the model's attention and increases the
chance of citing the wrong thing. Reranking down to a precise top-5 (or
whatever `RERANK_TOP_K` is set to) keeps the prompt focused and cheap.

## This project's two reranker implementations
- `reranking/scoring.py::HeuristicReranker` -- offline, deterministic:
  recomputes a BM25-like score using inverse-document-frequency *within
  just the candidate set* (not the full corpus) plus an exact-phrase-match
  bonus. This is explicitly **not** a neural cross-encoder; it exists so
  the reranking *stage* of the pipeline is real, testable, and runs with
  zero network access (this project's development sandbox cannot reach
  huggingface.co to download a real cross-encoder model).
- `reranking/cross_encoder.py::CrossEncoderReranker` -- a real
  `sentence-transformers.CrossEncoder` wrapper (e.g.
  `cross-encoder/ms-marco-MiniLM-L-6-v2`). Genuine code, gated behind
  `uv sync --extra embeddings` and network access to download
  model weights; use this in any production deployment.
