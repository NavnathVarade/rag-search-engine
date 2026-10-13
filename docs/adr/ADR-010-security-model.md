# ADR-010: Authorization filters before context selection, never after

**Context:** Retrieved chunks flow: retrieve -> rerank -> select context
-> generate. A permission check could conceptually happen at any of these
points, or even just before the final answer is returned.

**Decision:** `security/authorization.py::filter_authorized()` is the
single mandatory choke point applied to raw retrieval results, before
reranking or context selection ever see them.

**Rationale:** Filtering late (e.g. redacting unauthorized citations from
the final answer) still means an unauthorized chunk's *content* passed
through the reranker and the LLM prompt — a determined adversary could
extract it via prompt injection or by observing subtle behavior changes,
even if the final displayed citation is hidden. Filtering first means
unauthorized content genuinely never enters the rest of the pipeline.

**Consequences:** Every new retrieval code path (a new retrieval mode, a
new API route) must remember to call `filter_authorized()` — this is a
discipline that isn't statically enforced by the type system in this
reference implementation. A stricter version of this project could wrap
`HybridRetriever.search()` itself to make the filter structurally
impossible to skip; documented here as a known design trade-off rather
than done, given the added complexity of threading `AuthContext` through
every retriever constructor.
