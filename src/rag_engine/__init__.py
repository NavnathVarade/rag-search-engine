"""rag_engine: transparent hybrid RAG search engine.

Core retrieval algorithms (tokenization, TF-IDF, BM25, cosine similarity,
RRF fusion, chunking, context selection) are implemented from scratch in
this package and depend on nothing but the Python standard library and
NumPy. Framework/infra concerns (FastAPI, SQLAlchemy, HTTP clients) live
only at the edges (api/, storage/, ingestion/loaders) and are never
imported by the retrieval core, so the algorithms can be tested and
understood in isolation.
"""

__version__ = "0.1.0"
