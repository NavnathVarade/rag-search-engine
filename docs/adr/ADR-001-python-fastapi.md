# ADR-001: Python 3.12 + FastAPI

**Context:** Need a backend stack with strong typing support, async I/O for
concurrent retrieval calls and LLM streaming, and a mature ecosystem for
NumPy-based numerical code (the retrieval algorithms).

**Decision:** Python 3.12+ with FastAPI/Pydantic v2 for the API layer.

**Alternatives considered:** Go (better raw performance, but NumPy-quality
numerical/ML ecosystem doesn't exist; would mean reimplementing vectorized
math by hand). Node/TypeScript (good async story, but the ML/embedding
ecosystem is thin compared to Python's).

**Consequences:** Python's GIL means CPU-bound retrieval work (BM25
scoring, HNSW search) doesn't parallelize across threads within one
process without extra care (multiprocessing or releasing the GIL in
NumPy's C code, which NumPy's vectorized operations already do for the
hot paths in `cosine.py`). Async I/O is used for the network-bound
operations (DB, HTTP to the LLM) where it actually helps.
