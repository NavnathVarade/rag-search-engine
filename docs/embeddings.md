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
