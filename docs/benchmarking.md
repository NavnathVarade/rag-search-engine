# Benchmarking

Run `python scripts/benchmark.py --sizes 100 1000 5000` (or your own
sizes). Every number it prints is measured live on the machine it runs
on -- this doc does not hardcode results, because they are meaningless
without knowing the hardware they were measured on. Example output shape
from a run in this project's development sandbox (included to show the
*format*, not as a performance claim you should expect on your machine):

```
=== ANN benchmark (dim=128, 20 queries) ===
       N  BF build ms  BF query ms  HNSW build ms  HNSW query ms  recall@10
     100          0.1        0.057          189.9          0.641      1.000
    1000          0.9        0.319         4799.3          4.013      1.000
```

Reproduce it yourself: `PYTHONPATH=src python3 scripts/benchmark.py`.

## What's benchmarked
- **ANN**: brute-force vs HNSW build time, query time, and measured
  recall@10 (HNSW against brute-force ground truth), at each corpus size.
- **BM25**: inverted-index build time and query latency at each corpus
  size.
- **RRF**: fusion latency as a function of candidate-list size.

## Interpreting the numbers
- HNSW's build cost is real and grows faster than brute force's; at small
  N (hundreds to low thousands of chunks), brute force can be faster
  end-to-end once you include HNSW's build time -- there is a genuine
  crossover point, and it depends on your corpus size, dimension, and
  hardware. Run the benchmark on your target corpus size before choosing
  `ANN_BACKEND=hnsw` over the default `brute_force`.
- Recall@10 measured here is on a *random synthetic* corpus (Gaussian
  vectors), which is an easy case for ANN (no clustering structure to get
  lost in). Real embedding spaces have cluster structure that can be
  either easier or harder for HNSW depending on how well-separated the
  clusters are -- measure recall on your actual corpus/queries before
  trusting a number from a synthetic benchmark.
- Not benchmarked in this repository (would require real infrastructure
  this development sandbox doesn't have): sustained throughput under
  concurrent load, multi-instance/distributed indexing, and performance
  at the 100K-1M chunk scale the original spec mentions -- see
  docs/final-report.md "Limitations".


## Retrieval quality on a real dataset: BEIR SciFact

Reproduce: `python scripts/run_beir.py --dataset <path-to-scifact> --stem porter --tune-split train --hnsw-ef-construction 200 --hnsw-ef-search 200 --out benchmarks/results/scifact.json`
(raw output committed in `benchmarks/results/scifact.json`).

Setup (BM25 uses the from-scratch Porter stemmer, the engine default `BM25_STEMMER=porter`): 5,183 documents (title + abstract, one chunk each), the 300 test queries
with human relevance labels, CPU only, pure-Python implementations, hashing
embedder (384 dim). Measured on the development sandbox:

| Mode | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 latency |
|---|---|---|---|---|---|
| BM25 (Porter stemming) | 0.674 | 0.816 | 0.911 | 0.633 | 2.7 ms |
| Dense, brute force (hashing embedder) | 0.247 | 0.317 | 0.536 | 0.233 | 6.3 ms |
| Dense, HNSW ef=200 | 0.226 | 0.292 | 0.475 | 0.212 | 6.7 ms |
| Hybrid RRF, equal weights | 0.482 | 0.700 | 0.882 | 0.427 | 11.0 ms |
| Hybrid RRF, weight tuned on train split | 0.674 | 0.816 | 0.911 | 0.633 | 10.6 ms |

HNSW recall@10 vs brute force: 0.948 (M=16, efConstruction=200, ef_search=200).

What this does and does not show:
- BM25 at 0.674 nDCG@10 is in the range usually reported for BM25 on SciFact (about 0.67 in the BEIR paper), which supports the from-scratch BM25 and Porter stemmer being correct. Without stemming it scored 0.648; light stemming 0.654.
- The default `HashingEmbedder` has no semantics, so dense retrieval is weak and
  **fusing it with BM25 hurts**. The dense weight chosen on the train split
  (809 queries) was 0.0, i.e. the tuned hybrid equals BM25. Hybrid search needs a
  real embedding model to pay off, and that has **not been tested** here: the
  sandbox cannot download model weights. Run with `--embedder sentence-transformers/all-MiniLM-L6-v2`
  on a machine with network access to find out.
- One embedder, one machine. A second dataset (NFCorpus) is below.
- Latency note: an earlier version scored every candidate with an O(corpus) average-length computation (BM25 p50 about 57 ms without stemming, 111 ms with). Switching to term-at-a-time scoring with IDF/avgdl computed once fixed that without changing any result (identical nDCG@10). Latencies above are from the final code, single-threaded pure Python on a shared sandbox CPU; treat as indicative only.

## Second dataset: BEIR NFCorpus

Reproduce: `python scripts/run_beir.py --dataset <path-to-nfcorpus> --stem porter --tune-split dev --hnsw-ef-construction 200 --hnsw-ef-search 200 --out benchmarks/results/nfcorpus.json`

3,633 documents, 323 test queries, same setup as SciFact (hashing embedder, CPU).

| Mode | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 latency |
|---|---|---|---|---|---|
| BM25 (Porter stemming) | 0.315 | 0.146 | 0.238 | 0.520 | 0.7 ms |
| Dense, brute force (hashing embedder) | 0.102 | 0.056 | 0.126 | 0.192 | 3.0 ms |
| Dense, HNSW ef=200 | 0.098 | 0.051 | 0.117 | 0.185 | 5.3 ms |
| Hybrid RRF, equal weights | 0.237 | 0.122 | 0.233 | 0.401 | 5.8 ms |
| Hybrid RRF, weight tuned on dev split | 0.316 | 0.147 | 0.246 | 0.520 | 6.2 ms |

HNSW recall@10 vs brute force: 0.920 (M=16, efC=200, ef=200).

Findings: same pattern as SciFact. The dense weight chosen on the dev split was 0.0,
so tuned hybrid is essentially BM25 and the hashing embedder does not help.
BM25 reaches 0.315 nDCG@10 with Porter stemming (0.299 without, 0.309 with the light stemmer),
still a little under the roughly 0.325 commonly reported for BM25 on NFCorpus (BEIR paper).
Remaining difference not investigated (tokenizer details, stopword list, BM25 parameters).

### Stemming ablation (BM25 only, nDCG@10)

| Stemmer | SciFact | NFCorpus |
|---|---|---|
| none | 0.648 | 0.299 |
| light suffix stripper | 0.654 | 0.309 |
| Porter (from scratch) | 0.674 | 0.315 |

