# ADR-005: From-scratch HNSW, with BruteForceIndex as ground truth

**Context:** Need an ANN option for corpora too large for brute-force
O(N) search to stay fast, while keeping the "transparent algorithms"
principle.

**Decision:** Implement HNSW from scratch (`retrieval/ann/hnsw.py`),
following Malkov & Yashunin (2016/2018), with `BruteForceIndex` kept as
the exact reference implementation both for direct use (small/medium
corpora) and as ground truth for measuring HNSW's actual recall.

**Alternatives considered:** FAISS (mature, fast, but a C++ black box —
defeats the transparency goal) or Postgres's pgvector HNSW (couples the
retrieval algorithm to the database, see ADR-002).

**Consequences:** This implementation is not competitive with FAISS on
raw throughput at millions of vectors (pure Python object graph, no SIMD).
It is fully inspectable, steppable in a debugger, and unit-testable
(including two real bugs found and fixed: dangling edges after pruning, and
low recall from symmetric edges with naive neighbor selection, caught by the
SciFact benchmark; see docs/hnsw.md).
