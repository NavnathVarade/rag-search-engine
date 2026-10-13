"""LRU cache wrapper around an EmbeddingProvider (spec #45 caching).

Cache keys include the model name + revision so that changing the
embedding model (or its version) automatically invalidates stale cached
vectors instead of silently mixing incompatible embeddings (spec #49).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

from rag_engine.embeddings.base import EmbeddingProvider


class CachedEmbeddingProvider:
    def __init__(self, provider: EmbeddingProvider, max_size: int = 10_000) -> None:
        self._provider = provider
        self._max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        raw = f"{self._provider.model_name()}:{self._provider.model_revision()}:{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, text: str) -> np.ndarray | None:
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def _put(self, text: str, vector: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = vector
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        results: list[np.ndarray | None] = [self._get_cached(t) for t in texts]
        missing_idx = [i for i, r in enumerate(results) if r is None]
        if missing_idx:
            fresh = self._provider.embed_documents([texts[i] for i in missing_idx])
            for offset, i in enumerate(missing_idx):
                results[i] = fresh[offset]
                self._put(texts[i], fresh[offset])
        filled = [r for r in results if r is not None]  # every slot is filled by now
        return np.vstack(filled) if filled else np.zeros((0, self._provider.dimension()))

    def embed_query(self, text: str) -> np.ndarray:
        cached = self._get_cached(text)
        if cached is not None:
            return cached
        vector = self._provider.embed_query(text)
        self._put(text, vector)
        return vector

    def dimension(self) -> int:
        return self._provider.dimension()

    def model_name(self) -> str:
        return self._provider.model_name()

    def model_revision(self) -> str:
        return self._provider.model_revision()

    @property
    def cache_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
