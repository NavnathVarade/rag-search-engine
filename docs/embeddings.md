# Embeddings

## Interface (`embeddings/base.py::EmbeddingProvider`)
`embed_documents(texts) -> (N,D)`, `embed_query(text) -> (D,)`,
`dimension()`, `model_name()`, `model_revision()`. The retrieval core
depends only on this Protocol.

## Two implementations
- **`HashingEmbedder`** (default, `EMBEDDING_PROVIDER=hashing`): offline,
  dependency-free, deterministic signed feature hashing of word n-grams
  (Weinberger et al. 2009) into a fixed-size vector, L2-normalized. Zero
  semantic understanding -- exists purely so the full pipeline runs
  end-to-end without a model download, in this project's sandbox, in CI,
  and for anyone who wants to try the project before setting up a real
  model. **Not for production semantic search.**
- **`SentenceTransformerEmbedder`** (`EMBEDDING_PROVIDER=sentence_transformers`):
  real `sentence-transformers` model (default
  `all-MiniLM-L6-v2`, 384-dim). Requires `uv sync --extra embeddings`
  and network access to download weights on first use. This is the one to
  use in any real deployment.

## Checking an embedding model on your machine
```bash
uv sync --extra embeddings                 # installs sentence-transformers + torch (large download)
# .env:  EMBEDDING_PROVIDER=sentence_transformers
#        EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2   (downloaded on first use)
uv run python scripts/check_embeddings.py
```
The script loads the model the same way the API does and prints its dimension, speed, whether vectors are
unit length, and a small semantic test (a related text must outrank an unrelated one). The hashing
embedder fails that test by design; a real model should pass all four. It also prints the similarity range
of related and unrelated pairs to help you choose `MIN_DENSE_SIMILARITY` (default `0.2`, chosen for the
hashing embedder; confirm it on your own documents). The API and the CLI both honour `EMBEDDING_PROVIDER`;
the vector size comes from the model, so `EMBEDDING_DIMENSION` is only for hashing. After switching models,
restart: vectors are recomputed from the stored text at startup (the first start embeds every chunk).

## Cosine similarity, dot product, normalization
This project L2-normalizes every embedding on write, which makes cosine
similarity and dot product equivalent (`cos(a,b) = a·b` when `||a||=||b||=1`)
-- `retrieval/cosine.py::batch_cosine_similarity` exploits this for a
single vectorized matrix-vector product per query rather than N separate
normalizations at query time.

## Embedding drift & model upgrades
If you change `EMBEDDING_MODEL` (or its revision), previously-indexed
vectors are **not comparable** to new query vectors from the new model --
cosine similarity between vectors from two different models is
meaningless. `DocumentMetadata.embedding_model` / `embedding_dimension`
and `IndexVersion` (storage/models.py) track which model produced which
vectors specifically so a model change can be detected and the affected
index rebuilt (`rag index rebuild`), rather than silently mixing
incompatible embeddings (spec #49).
