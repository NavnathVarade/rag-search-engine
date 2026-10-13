"""ANN backend abstraction. Both BruteForceIndex and HNSWIndex implement
this Protocol so dense retrieval (retrieval/dense.py) can swap backends
via config without changing calling code (spec #13, #14, #92)."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class AnnIndex(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    def remove(self, ids: list[str]) -> None: ...

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return up to top_k (id, cosine_similarity) pairs, descending score."""
        ...

    @property
    def size(self) -> int: ...
