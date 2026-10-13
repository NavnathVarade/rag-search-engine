# ADR-006: Reciprocal Rank Fusion for hybrid combination

**Context:** Need to combine dense and BM25 rankings, which live on
incomparable score scales.

**Decision:** RRF (Cormack, Clarke, Buettcher, SIGIR 2009), implemented
from scratch with full per-ranker contribution tracking for explainability.

**Alternatives considered:** Weighted score averaging after min-max
normalization (sensitive to outlier scores skewing the normalization
range per-query, less robust than a rank-based approach); a learned
fusion model (requires labeled training data this project doesn't have,
and adds an opaque ML component where a simple, auditable formula
suffices).

**Consequences:** RRF requires no training and is easy to explain (every
result's fused score decomposes into simple, inspectable per-ranker
terms — see `FusedResult.reason()`), at the cost of not being provably
optimal for any specific corpus/query distribution the way a learned
fusion model tuned on labeled data could be.
