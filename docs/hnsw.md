# HNSW (Hierarchical Navigable Small World)

Implementation: `src/rag_engine/retrieval/ann/hnsw.py`, following Malkov &
Yashunin (2016/2018), https://arxiv.org/abs/1603.09320.

## Concept
A multi-layer graph where layer 0 contains every vector and each higher
layer contains an exponentially-shrinking random subset (probability
`~1/ln(M)` of promotion per layer) -- functioning like a skip-list over a
proximity graph. Each node keeps up to `M` neighbor edges per layer
(`M0 = 2M` at layer 0). Search starts at the top layer's entry point,
greedily descends to the closest node at each layer, then runs a
best-first search of width `ef_search` at layer 0.

## Parameters
- `M`: max neighbors per node. Higher -> better recall, more memory.
- `ef_construction`: candidate list width during insertion. Higher ->
  better graph quality, slower builds.
- `ef_search`: candidate list width during search. Higher -> better
  recall, slower queries. This is the parameter you tune per-query for a
  recall/latency trade-off without rebuilding the index.

## Recall/latency trade-off (measured, not assumed)
`tests/retrieval/test_ann.py::test_hnsw_recall_at_10_against_brute_force_ground_truth`
measures actual recall@10 of this implementation against `BruteForceIndex`
(the exact ground truth) on a random 500-vector, 32-dim synthetic corpus,
and asserts recall > 0.7 as a sanity floor -- the measured value printed by
that test is typically ~1.0 on such a small, easy corpus. `scripts/benchmark.py`
measures build time, query time, and recall at several corpus sizes on
whatever machine you run it on; run it yourself to get numbers for your
hardware and corpus rather than trusting a number written in a doc, since
recall/latency depend heavily on both.

Brute force is O(N) per query with zero index-memory overhead beyond the
raw vectors; HNSW trades index-build time and graph memory (each vector
gets up to `M0` edges at layer 0 plus fewer at higher layers) for
sub-linear query time at large N. For corpora under a few thousand chunks,
brute force is often simply faster in wall-clock terms once you include
HNSW's build cost -- see the benchmark output for exactly where the
crossover point is on your machine.

## Two real problems this implementation had, and how they were found
**1. Dangling edges after pruning (found in development).** Pruning dropped an
edge on one side only, so a deleted node could stay referenced and cause a
`KeyError` on search. The first fix deleted the reverse edge too, forcing a
symmetric graph.

**2. Low recall on real text vectors (found by the SciFact benchmark).** On
5,183 SciFact abstracts embedded with the hashing embedder, recall@10 against
brute force was only **0.75** (M=16, efC=100, ef=64), far below what a correct
HNSW should reach. Two causes:
- The symmetric-edge fix from (1) was itself wrong: HNSW graphs are *directed*.
  Deleting reverse edges starved other nodes of connectivity.
- Neighbor selection was "keep the M closest", which lets tight clusters link
  only to each other.

Fix: edges are directed again, `remove()` scans all nodes for in-edges (so no
dangling references), and selection uses the paper's heuristic (Algorithm 4:
keep a candidate only if it is closer to the base point than to any neighbor
already kept, back-filling with the closest discarded ones). Similarity is now a
single dot product on unit vectors (about 3x faster index builds).

Measured after the fix (SciFact, 300 test queries, recall@10 vs brute force):
0.948 at M=16, efConstruction=200, ef_search=200. At the default ef_search=64 it
is lower (about 0.88 on a 2,500-doc subset). On synthetic clustered Gaussian
vectors recall is 1.0, so the implementation is correct; the hashing embedder
produces tightly bunched similarities, which is a hard case for any ANN index.
Tune `ef_search` upward (accepting higher latency) if you keep that embedder.
Regression tests: `test_hnsw_recall_on_clustered_data_is_high` and
`test_hnsw_remove_with_directed_edges_leaves_no_dangling_references`.
