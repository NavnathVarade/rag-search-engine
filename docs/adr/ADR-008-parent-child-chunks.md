# ADR-008: Parent-child chunking as an explicit strategy, not the default

**Context:** Retrieval wants small, focused chunks; generation wants
enough surrounding context to be coherent. A single chunk size can't
satisfy both.

**Decision:** Implement `ParentChildChunker` as one of seven selectable
strategies (default remains `recursive`, a simpler general-purpose
choice), rather than making parent-child the universal default.

**Rationale:** Parent-child roughly doubles the indexed content (parents
+ children) and adds bookkeeping (`parent_chunk_id` resolution at context-
selection time) that isn't justified for every corpus — markdown-aware or
plain recursive chunking is simpler and sufficient when documents are
already reasonably-sized and well-structured. Parent-child earns its
complexity specifically when source documents are long-form and
retrieval precision on sub-sections matters (spec #9.G).

**Consequences:** Context selection (`context/selector.py`) does not
currently auto-expand a selected child chunk to its parent — that
expansion is a deliberate manual step left for the caller (or a follow-up
enhancement, see docs/final-report.md) rather than silently changing
what content reaches the LLM.
