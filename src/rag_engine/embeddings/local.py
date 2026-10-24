"""Local embedding providers.

Two implementations are provided:

  HashingEmbedder
    A fully offline, dependency-free, *deterministic* embedder based on
    signed feature hashing (a classic technique -- see Weinberger et al.,
    "Feature Hashing for Large Scale Multitask Learning", 2009) of token
    n-grams into a fixed-size vector, followed by L2 normalization. It has
    no semantic understanding of language (it cannot tell "car" and
    "automobile" are related) -- it exists so the full pipeline
    (chunk -> embed -> index -> dense search -> hybrid -> rerank) runs
    end-to-end, deterministically, and entirely offline in development,
    CI, and this sandbox, without requiring a multi-hundred-MB model
    download. It is NOT a substitute for a real sentence embedding model
    in production -- see SentenceTransformerEmbedder below and
    docs/embeddings.md "Limitations".

  SentenceTransformerEmbedder
    Thin wrapper around the `sentence-transformers` library for real
    semantic embeddings (e.g. all-MiniLM-L6-v2, bge-small, etc). Import of
    `sentence_transformers` is deferred into __init__ so importing this
    module never requires the optional heavy dependency (torch) unless
    this class is actually instantiated. This class's forward path is
    real, tested code, but this development sandbox has no network route
    to huggingface.co to download model weights, so it cannot be exercised
    end-to-end here -- integration tests for it are marked accordingly.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from rag_engine.embeddings.batching import batched
from rag_engine.embeddings.normalization import normalize
from rag_engine.exceptions import EmbeddingError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_token(token: str, dimension: int, seed: int = 0) -> tuple[int, float]:
    """Deterministic (index, sign) pair for a token via SHA-256, avoiding
    Python's randomized `hash()` so embeddings are stable across processes
    and runs (a requirement for reproducible indexes -- spec #11, #81)."""
    digest = hashlib.sha256(f"{seed}:{token}".encode()).digest()
    index = int.from_bytes(digest[:4], "big") % dimension
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


class HashingEmbedder:
    def __init__(self, dimension: int = 384, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self._dimension = dimension
        self.ngram_range = ngram_range

    def _tokenize_ngrams(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        grams: list[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                grams.append(" ".join(tokens[i : i + n]))
        return grams or tokens

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float64)
        for gram in self._tokenize_ngrams(text):
            idx, sign = _hash_token(gram, self._dimension)
            vec[idx] += sign
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension))
        try:
            matrix = np.vstack([self._embed_one(t) for t in texts])
        except Exception as exc:  # pragma: no cover - defensive
            raise EmbeddingError(f"hashing embedding failed: {exc}") from exc
        return normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(self._embed_one(text))

    def dimension(self) -> int:
        return self._dimension

    def model_name(self) -> str:
        return "hashing-embedder"

    def model_revision(self) -> str:
        return f"v1-dim{self._dimension}-ngram{self.ngram_range[0]}-{self.ngram_range[1]}"


class SentenceTransformerEmbedder:
    """Real sentence-transformers-backed embedder. Requires the optional
    `embeddings` extra (`uv sync --extra embeddings`) and network access to
    download model weights on first use. Not exercised in this sandbox
    (no route to huggingface.co) -- covered by integration tests marked
    `@pytest.mark.requires_network` for environments that have one."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Install with "
                "`uv sync --extra embeddings` to use this provider."
            ) from exc

        self._model_name = model_name
        self._batch_size = batch_size
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbeddingError(f"failed to load embedding model {model_name}: {exc}") from exc
        self._dimension = self._model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension))
        chunks = []
        for batch in batched(texts, self._batch_size):
            try:
                chunks.append(self._model.encode(batch, normalize_embeddings=True))
            except Exception as exc:
                raise EmbeddingError(f"embedding batch failed: {exc}") from exc
        return np.vstack(chunks).astype(np.float64)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def dimension(self) -> int:
        return self._dimension

    def model_name(self) -> str:
        return self._model_name

    def model_revision(self) -> str:
        return getattr(self._model, "_model_config", {}).get("revision", "unknown")
