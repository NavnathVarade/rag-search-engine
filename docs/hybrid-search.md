# Hybrid Search & RRF

## Why dense alone is insufficient
Dense retrieval finds semantically related content even without exact
word overlap ("how do pods come back after crashing" can match a chunk
about "container restartPolicy" despite sharing almost no words). But it
is comparatively weak on exact identifiers, versions, and rare technical
tokens (see docs/bm25.md) and is sensitive to embedding-model quality and
training-data coverage of the domain.

## Why BM25 alone is insufficient
BM25 requires literal token overlap. A query phrased differently from the
source text ("how do I make Kubernetes bring a container back" vs "the
kubelet restarts a failed container") can score poorly even though the
content is exactly what's needed, because BM25 has no notion of synonymy
or paraphrase.

## Reciprocal Rank Fusion

```
RRF(d) = Σ_i  weight_i / (k + rank_i(d))
```

`rank_i(d)` is document `d`'s 1-based rank in ranker `i`'s list; a
document absent from a ranker's list contributes nothing for that ranker
(it is not penalized with an infinite rank). `k` (default 60, from the
original paper -- Cormack, Clarke, Buettcher, SIGIR 2009) dampens the
influence of any single very-high rank.

**Why fuse ranks instead of averaging raw scores?** BM25 scores and
cosine similarities live on incomparable scales (BM25 is unbounded and
corpus-dependent; cosine is bounded [-1, 1]). Averaging them directly
would let whichever score happens to have larger magnitude dominate for
reasons unrelated to actual relevance. Rank is scale-free, so RRF sidesteps
this entirely. See `src/rag_engine/retrieval/rrf.py` for the from-scratch
implementation and `tests/retrieval/test_rrf.py` for a hand-calculated
fusion example.

## Explanation object
Every hybrid result carries, per spec #19/#21/#105:
```json
{
  "chunk_id": "...",
  "final_rank": 1,
  "rrf_score": 0.0321,
  "dense": {"rank": 2, "score": 0.87},
  "bm25": {"rank": 1, "score": 14.82},
  "explanation": [
    "Ranked by 2 retrievers: dense, bm25",
    "dense: rank 2, score 0.870, RRF contribution 0.01613",
    "bm25: rank 1, score 14.820, RRF contribution 0.01639",
    "Strong agreement across retrievers boosts confidence"
  ]
}
```
This is generated programmatically by `FusedResult.reason()` -- it is not
free-text from an LLM, so it is 100% reproducible and auditable.
