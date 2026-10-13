# Interview Guide

Twenty questions, each with a short answer, detailed explanation, a
practical example, a common mistake, and where this project demonstrates
it. Code references point at the actual implementation in this repo.

---

### 1. Why does semantic (dense) search fail?
**Short:** It can't reliably match exact identifiers, versions, or rare
technical tokens, because embeddings are trained for topical similarity,
not token-level precision.
**Detail:** A bi-encoder compresses a chunk into a fixed vector optimized
so that *topically similar* text lands nearby. "CREATE INDEX" and "CREATE
INDEX CONCURRENTLY" embed close together because they're about the same
topic — but a user asking specifically about `CONCURRENTLY` needs the
distinction preserved, and nothing in the embedding objective rewards
that.
**Example:** Query "PostgreSQL 16 CREATE INDEX CONCURRENTLY" can retrieve
a chunk about Postgres 12 index basics ahead of the exact chunk, because
both embed similarly.
**Common mistake:** Assuming a bigger/better embedding model fixes this.
It helps some, but the fundamental limitation (embeddings optimize
topical similarity, not exact-token precision) doesn't go away.
**In this project:** `docs/bm25.md` "Why BM25 can beat dense/semantic
search", demonstrated in `tests/retrieval/test_hybrid.py::test_bm25_only_finds_exact_technical_term`.

### 2. Why does BM25 fail?
**Short:** No notion of synonymy or paraphrase — it needs literal token
overlap.
**Detail:** BM25's entire signal is term frequency weighted by inverse
document frequency. A query phrased with different words than the source
("bring a crashed container back up" vs "kubelet restarts a failed
container") scores poorly despite being exactly what's needed.
**Example:** Query "how does k8s heal itself" won't match a chunk about
"restartPolicy" via BM25 alone if none of those words overlap.
**Common mistake:** Tuning `k1`/`b` to try to fix a fundamentally lexical
mismatch — no amount of BM25 parameter tuning adds semantic
understanding.
**In this project:** `src/rag_engine/retrieval/sparse/bm25.py` docstring;
`docs/hybrid-search.md`.

### 3. Why hybrid search?
**Short:** Dense and BM25 fail in complementary ways — fusing them covers
more query types than either alone.
**Detail:** RRF combines both rankers' *ranks* (not raw scores, which
live on incomparable scales) into one fused ranking.
**Example:** "PostgreSQL 16 CREATE INDEX CONCURRENTLY" — BM25 nails the
exact terms, dense catches paraphrased "build an index without locking".
**Common mistake:** Averaging raw scores instead of fusing ranks — BM25's
unbounded scale can dominate a bounded [-1,1] cosine score for reasons
unrelated to relevance.
**In this project:** `src/rag_engine/retrieval/rrf.py`.

### 4. How does BM25 work?
**Short:** TF-IDF variant with term-frequency saturation and document-length
normalization. See docs/bm25.md for the full formula and worked example.
**Common mistake:** Forgetting the `+0.5` smoothing terms in IDF, which
matters when a term appears in most/all documents (classic IDF can go
negative there; BM25's `+1` inside the log keeps it non-negative).
**In this project:** `src/rag_engine/retrieval/sparse/bm25.py`,
hand-calculated in `tests/retrieval/test_bm25.py`.

### 5. How does HNSW work?
**Short:** A multi-layer proximity graph, sparser at higher layers,
searched via greedy descent + best-first search at the base layer.
**Detail:** See docs/hnsw.md.
**Common mistake:** Assuming HNSW always beats brute force — at small
corpus sizes brute force can be faster overall once you include HNSW's
build cost (measured, not assumed — `scripts/benchmark.py`).
**In this project:** `src/rag_engine/retrieval/ann/hnsw.py`, including a
real bug (asymmetric graph edges after pruning) found and fixed during
development — see docs/hnsw.md "A real bug this implementation had".

### 6. What is ANN (approximate nearest neighbor)?
**Short:** Trading exactness for speed at scale — returns *likely* nearest
neighbors, not guaranteed exact ones.
**Detail:** Exact search is O(N) per query; ANN structures (HNSW, IVF,
LSH) achieve sub-linear query time by not examining every vector, at the
cost of sometimes missing the true nearest neighbor.
**Common mistake:** Not measuring recall — deploying ANN without knowing
what fraction of true top-K results it's actually finding on your data.
**In this project:** `BruteForceIndex` (exact) vs `HNSWIndex` (approximate),
with recall measured against the exact index as ground truth.

### 7. Why cosine similarity (over Euclidean distance, say)?
**Short:** It measures *direction*, not magnitude — appropriate when
embedding magnitude isn't meaningful (which is true for most sentence
embedding models).
**Detail:** `cos(A,B) = A·B / (||A|| ||B||)`. Two embeddings pointing the
same direction score 1.0 regardless of their length, which matters
because many embedding models don't reliably encode meaning in vector
*magnitude* — L2-normalizing and using cosine (equivalently, dot product
on normalized vectors) avoids being misled by it.
**Common mistake:** Mixing normalized and unnormalized vectors — a
similarity computed between one normalized and one unnormalized embedding
is not meaningful.
**In this project:** `src/rag_engine/retrieval/cosine.py`, property-tested
for symmetry and boundedness in `tests/retrieval/test_cosine.py`.

### 8. What is embedding drift?
**Short:** Vectors from different model versions aren't comparable —
"drift" between what an old model considered similar and what a new one
does.
**Detail:** If you swap `all-MiniLM-L6-v2` for a different model (or even
a new revision of the same model), previously-indexed vectors are stale
relative to new query vectors from the new model. Comparing them produces
meaningless similarity scores.
**Common mistake:** Silently swapping the embedding model without
rebuilding the index — results degrade gradually and confusingly rather
than failing loudly.
**In this project:** `DocumentMetadata.embedding_model`/`embedding_dimension`
+ `IndexVersion` tracks which model produced which vectors specifically
so this can be detected — docs/embeddings.md.

### 9. How do you choose chunk size?
**Short:** Balance retrieval precision (smaller chunks = more focused
matches) against generation coherence (larger chunks = more complete
thoughts) — there's no universally correct number, it's corpus- and
query-dependent.
**Common mistake:** Picking one chunk size for an entire heterogeneous
corpus (long-form docs vs terse API references) instead of using a
structure-aware strategy (markdown/code-aware) that adapts to content.
**In this project:** `docs/chunking.md`, 7 strategies implemented
side-by-side in `src/rag_engine/chunking/`.

### 10. What is chunk overlap, and when is it harmful?
**Short:** Duplicating a bit of text between adjacent chunks so a
boundary-straddling fact still appears whole somewhere. Harmful when
excessive — it bloats the index with near-duplicate content that then has
to be filtered at context-selection time.
**In this project:** `docs/chunking.md`, `context/deduplicator.py`.

### 11. Why can large chunks hurt retrieval?
**Short:** Averaging multiple topics into one embedding/BM25 profile
dilutes the signal for any one of them.
**In this project:** `docs/chunking.md`.

### 12. Why can small chunks hurt generation?
**Short:** Fragments force the LLM to reconstruct meaning across many
retrieved pieces, increasing context-budget pressure and the chance of
losing a qualifying clause that changes a fact's meaning.
**In this project:** `docs/chunking.md`; `parent_child` chunking exists
specifically to decouple retrieval granularity from generation context
size (`chunking/parent_child.py`).

### 13. What is reranking?
**Short:** A second, more expensive/accurate scoring pass over a small
candidate set from the (fast, broad) first-stage retriever.
**In this project:** `docs/reranking.md`, `reranking/scoring.py` +
`reranking/cross_encoder.py`.

### 14. Bi-encoder vs cross-encoder?
**Short:** Bi-encoder embeds query and document independently
(precomputable, scalable); cross-encoder scores them jointly (accurate,
not precomputable). See docs/reranking.md for the full explanation.
**Common mistake:** Using a cross-encoder as the *first-stage* retriever
over a large corpus — it doesn't scale, because there's no way to
precompute anything.

### 15. Recall@K vs Precision@K?
**Short:** Recall asks "of everything relevant, how much did we find in
the top K"; Precision asks "of what we returned in the top K, how much
was actually relevant". They trade off against each other as K grows.
**In this project:** `evaluation/metrics/recall.py`, hand-calculated in
`tests/evaluation/test_metrics.py`.

### 16. MRR vs NDCG?
**Short:** MRR only cares about the rank of the *first* relevant result;
NDCG accounts for *all* relevant results in the top-K, weighted by
position, and supports graded (not just binary) relevance.
**Common mistake:** Using MRR to evaluate a system where multiple
relevant documents genuinely matter (e.g. multi-hop questions) — MRR is
blind to everything after the first hit.
**In this project:** `evaluation/metrics/mrr.py`, `ndcg.py`.

### 17. How would you debug bad retrieval?
**Short:** Look at each ranker's individual rank/score before fusion, not
just the final result.
**In this project:** `rag debug "<query>"` — prints normalized query,
per-ranker rank/score, RRF contribution, and reranker score for the top
results in one report (`cli/main.py::debug`).

### 18. How would you detect hallucinations?
**Short:** Validate every citation marker against the chunks actually
supplied (catches fabricated citations deterministically); use a lexical
overlap proxy or an LLM judge for unsupported *content* (softer signal,
see limitations below).
**Common mistake:** Trusting an LLM-as-judge faithfulness score as ground
truth without accounting for judge-model bias and non-determinism.
**In this project:** `generation/citations.py` (deterministic),
`evaluation/metrics/faithfulness.py` (both a deterministic lexical proxy
and an explicitly-labeled LLM-judge option).

### 19. How would you reduce RAG latency?
**Short:** Parallelize independent stages (dense + BM25 concurrently),
cache embeddings/queries, use ANN over brute force at scale, batch
reranker calls, keep the reranked/selected context small, stream
generation.
**In this project:** every stage is individually timed
(`HybridSearchTimings`) so you can see exactly where latency goes before
optimizing; `embeddings/cache.py` caches by (model, revision, text).

### 20. How would you reduce RAG cost?
**Short:** Smaller reranker candidate sets, tighter context budgets,
response/query caching, context compression, cheaper models for
easier query classes.
**In this project:** `config.py::max_context_tokens`/`rerank_candidate_k`,
`context/compressor.py`, `embeddings/cache.py`.
