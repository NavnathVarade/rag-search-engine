# ADR-003: From-scratch BM25 (not rank_bm25 / Elasticsearch / Postgres FTS)

**Context:** Multiple mature libraries could provide BM25-style scoring
(`rank_bm25` package, Elasticsearch, Postgres's built-in full-text search
ranking).

**Decision:** Implement BM25 from scratch (`retrieval/sparse/bm25.py`),
including our own tokenizer and inverted index.

**Rationale:** The stated goal of this project (spec item #162) is to
demonstrate understanding of information retrieval, not framework
proficiency. A from-scratch implementation makes the exact formula,
smoothing choices, and length-normalization behavior fully inspectable
and testable against hand-calculated examples — which a black-box library
call cannot provide.

**Consequences:** We give up the operational maturity of Elasticsearch
(no built-in distributed sharding, no decades of query-performance tuning)
in exchange for transparency. For a real production deployment at large
scale, swapping this module for Elasticsearch/OpenSearch behind the same
`Bm25Retriever` interface is a contained change — the `RetrievalResult`
contract (`retrieval/models.py`) does not depend on the from-scratch
implementation internals.
