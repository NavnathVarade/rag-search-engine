# ADR-007: Cross-encoder reranking, with an offline heuristic fallback

**Context:** Need a precision-oriented second-stage reranker (spec #22),
but this project's development sandbox has no network route to
huggingface.co to download a real cross-encoder model.

**Decision:** Define `Reranker` Protocol with two implementations:
`HeuristicReranker` (offline, deterministic, BM25-like recomputation over
just the candidate set — used everywhere in this repo's own tests/CLI/API
demos) and `CrossEncoderReranker` (real `sentence-transformers.CrossEncoder`
wrapper, network-gated, intended for production use).

**Consequences:** Every reranking-quality behavior demonstrated in this
repo's tests is from the heuristic reranker, honestly labeled as *not* a
neural cross-encoder. Swapping to `CrossEncoderReranker` in a deployment
with network access is a one-line config change (`RERANKER_MODEL` +
constructing `CrossEncoderReranker` instead of `HeuristicReranker` in
`api/engine.py::build_engine_state`) since both implement the same
`Reranker` Protocol.
