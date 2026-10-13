# ADR-004: Dense retrieval via a swappable EmbeddingProvider Protocol

**Context:** Need semantic retrieval, but must not hard-couple the core
retrieval logic to one embedding vendor/model (spec #115).

**Decision:** `embeddings/base.py::EmbeddingProvider` Protocol with two
implementations: `HashingEmbedder` (offline, deterministic, no real
semantics — default for dev/CI/this sandbox) and
`SentenceTransformerEmbedder` (real semantic embeddings, network-gated).

**Consequences:** Every test and CLI/API smoke-run in this repository
uses `HashingEmbedder` by default, which means retrieval-quality claims
in this repo's own test suite are about the *pipeline mechanics*
(does hybrid retrieval fuse correctly, does context selection dedup
correctly), not about real-world semantic retrieval quality — that
depends entirely on which real embedding model you configure in
production. This is stated explicitly rather than left implicit.
