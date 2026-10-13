"""Embedding provider abstraction (spec #11).

Any embedder used by the engine implements this Protocol. This keeps the
retrieval core decoupled from any specific embedding backend/vendor
(spec #115, dependency inversion) -- dense retrieval only depends on
`EmbeddingProvider`, never on `sentence_transformers` or a specific model.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of document/chunk texts. Returns (N, D) float64 array."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns (D,) float64 array."""
        ...

    def dimension(self) -> int: ...

    def model_name(self) -> str: ...

    def model_revision(self) -> str:
        """Version/revision identifier -- used to detect stale/incompatible
        indexes when the embedding model changes (spec #49, #80)."""
        ...
