# rag-search-engine

A hybrid RAG (Retrieval-Augmented Generation) search engine for technical
documentation, built to demonstrate **transparent, from-scratch
information retrieval** — not a wrapper around LangChain/LlamaIndex. The
core algorithms (tokenizer, TF-IDF, BM25, cosine similarity, brute-force
and HNSW approximate nearest-neighbor search, and Reciprocal Rank Fusion)
are implemented in plain Python + NumPy and are fully unit-tested,
including hand-calculated worked examples.

## Measured results (BEIR, hashing embedder, CPU)

| Dataset | BM25 + Porter nDCG@10 | BM25 Recall@100 | HNSW recall@10 vs exact |
|---|---|---|---|
| SciFact (5,183 docs, 300 queries) | 0.674 | 0.911 | 0.948 |
| NFCorpus (3,633 docs, 323 queries) | 0.315 | 0.238 | 0.919 |

With the offline hashing embedder, hybrid does **not** beat BM25 on either dataset
(the tuned dense weight was 0); a real embedding model is needed and has not been
tested here. NFCorpus BM25 is slightly below the usual published number. Full tables,
the stemming ablation, method and caveats: [docs/benchmarking.md](docs/benchmarking.md).

## Features

- **Hybrid retrieval**: dense (embeddings) + BM25 (lexical), fused via
  RRF, selectable per-query (`dense` / `bm25` / `tfidf` / `hybrid` /
  `hybrid_weighted`)
- **Full explainability**: every result shows its per-ranker rank/score
  and exactly how it was fused — never a black box
- **Reranking + context selection**: precision-oriented reranking,
  near-duplicate detection, token budgeting, source diversity
- **Citation-grounded generation**: answers cite specific retrieved
  chunks; hallucinated citations are detected and stripped; the model
  explicitly says "I don't have enough evidence" rather than guessing
- **Evaluation framework**: Recall@K, Precision@K, MRR, NDCG, Hit Rate,
  plus deterministic and (separately-labeled) LLM-judge generation
  metrics
- **Security**: SSRF/path-traversal/prompt-injection defense,
  authorization filtering with cross-tenant isolation tests
- **Postgres storage** (`STORE_BACKEND=postgres`, the docker-compose default; memory + JSON snapshot otherwise),
  Alembic migrations, FastAPI + CLI sharing one retrieval engine

## Quickstart

```bash
git clone <repository>
cd rag-search-engine
cp .env.example .env
docker compose up --build
```

Or locally (see docs/getting-started.md for the full walkthrough):

```bash
uv sync                # creates .venv from uv.lock (install uv first: https://docs.astral.sh/uv/)
make migrate
uv run rag ingest ./data/sample
uv run rag index build
uv run rag search "How does Kubernetes restart a failed container?" --mode hybrid
uv run rag chat "What does CREATE INDEX CONCURRENTLY do?"
uv run rag evaluate --dataset data/evaluation/sample.jsonl
```

## Web UI

A React console (search with per-ranker score breakdown, cited answers, document library, uploads and URL ingest, index
rebuild, analytics and health) is in [`frontend/`](frontend/README.md). Build it once and the API serves it at <http://localhost:8000/>:

```bash
cd frontend && npm ci && npm run build && cd ..
uv run uvicorn rag_engine.api.app:app --port 8000
```

See [docs/frontend.md](docs/frontend.md) for what was verified and what was not.

## Documentation

Start at [docs/architecture.md](docs/architecture.md). Algorithm-specific
deep dives: [BM25](docs/bm25.md), [HNSW](docs/hnsw.md),
[hybrid search / RRF](docs/hybrid-search.md), [chunking](docs/chunking.md),
[reranking](docs/reranking.md). Preparing for an interview about this
project? See [docs/interview-guide.md](docs/interview-guide.md). Curious
what's real vs. still scaffolded? See
[docs/final-report.md](docs/final-report.md) "Limitations".

## Development

```bash
make test          # full suite (needs Postgres for integration tests)
make test-unit      # pure-algorithm tests, no DB required
make lint
make typecheck
make check          # lint + typecheck + test
```

## License

MIT — see [LICENSE](LICENSE).
