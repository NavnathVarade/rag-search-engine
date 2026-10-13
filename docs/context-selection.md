# Context Selection

`context/selector.py::ContextSelector.select()` runs a single greedy pass
over reranked candidates (already sorted by relevance), in order:

1. **Near-duplicate check** (`context/deduplicator.py`): cosine similarity
   on embeddings when available, else Jaccard similarity over 4-word
   shingles (punctuation-stripped). A candidate that's a near-duplicate of
   something already selected is skipped -- prevents showing the LLM five
   near-identical paraphrases of the same fact.
2. **Source diversity cap** (`max_chunks_per_document`, default 3): once a
   document has contributed its cap, further chunks from it are skipped
   even if individually high-scoring, to avoid one document crowding out
   the rest of the corpus.
3. **Token budget** (`context/budget.py::TokenBudget`, default 6000): a
   candidate that would exceed the remaining budget is skipped.

Every selected chunk records *why* it was selected (relevance score,
document-contribution count, remaining budget at time of selection) --
`ContextSelectionResult.selected[i].reasons`. Every skipped chunk records
why it was skipped. Nothing about the final context is opaque.

`context/compressor.py::ExtractiveCompressor` optionally shrinks an
over-budget chunk to its most query-relevant sentences (deterministic
term-overlap scoring, no LLM call) rather than dropping it outright.

## Evidence gate (why off-topic questions get "insufficient evidence")
Dense search always returns its top-k neighbours, however unrelated, so "something was retrieved"
is not evidence. `ContextSelector` therefore keeps a chunk only if it has a **lexical match**
(BM25/TF-IDF) or a **dense cosine similarity >= `MIN_DENSE_SIMILARITY`** (default 0.2). Dropped chunks
appear in `ContextSelectionResult.skipped` with the reason "below evidence threshold". When nothing
survives, `RagAnswerer` returns a fixed "I don't have enough evidence..." answer, makes **no LLM call**,
and reports `had_sufficient_evidence=false`.

Measured on the 5-document sample corpus with the offline hashing embedder: a real match scores about
0.29 dense and has a BM25 hit; "airspeed velocity of an unladen swallow" scores about 0.09 dense with no
BM25 hit and is rejected. The threshold is embedder-specific: 0.2 suits the hashing embedder, real
sentence embedders usually want 0.3 or more, and 0 disables the gate. It has only been tuned on this tiny
corpus, so treat the default as a starting point and calibrate it on your own queries.
