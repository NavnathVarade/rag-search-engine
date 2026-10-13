# ADR-009: Deterministic metrics and LLM-judge metrics kept strictly separate

**Context:** Retrieval metrics (Recall@K, MRR, NDCG) are objective given
relevance labels. Generation-quality metrics (faithfulness, relevance)
are inherently more subjective and commonly evaluated via "LLM-as-judge".

**Decision:** Every deterministic metric and every LLM-judge metric lives
in its own clearly-named function (`lexical_faithfulness_proxy` vs
`llm_judge_faithfulness`), and no report in this codebase presents an
LLM-judge score as ground truth without that label.

**Rationale:** Conflating the two invites false confidence — a
faithfulness score of "0.92" sounds objective regardless of which method
produced it, but an LLM-judge-produced 0.92 carries judge-model bias and
non-determinism that a lexical-overlap-produced 0.92 does not (and vice
versa, the lexical proxy under-counts legitimate paraphrase). Naming and
documentation make the distinction impossible to miss at the call site.

**Consequences:** Anyone building a dashboard on top of this evaluation
framework must consciously choose which family of metric to trust for
which purpose, rather than the framework quietly picking for them.
