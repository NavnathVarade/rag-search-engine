"""Near-duplicate detection for context selection (spec #24).

Uses cosine similarity between chunk embeddings when available, falling
back to a Jaccard similarity over token shingles (no embeddings required)
so dedup still works when only lexical retrieval was used. Words are
punctuation-stripped before shingling so trivial differences (trailing
'.' vs '!', a stray comma) don't defeat the match -- shingle identity is
about word sequence, not exact punctuation.
"""
from __future__ import annotations

import re

import numpy as np

from rag_engine.retrieval.cosine import cosine_similarity

_WORD_RE = re.compile(r"[a-z0-9]+")


def _shingles(text: str, n: int = 4) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_near_duplicate(
    content_a: str,
    content_b: str,
    embedding_a: np.ndarray | None = None,
    embedding_b: np.ndarray | None = None,
    threshold: float = 0.92,
) -> bool:
    if embedding_a is not None and embedding_b is not None:
        return cosine_similarity(embedding_a, embedding_b) >= threshold
    return jaccard_similarity(content_a, content_b) >= threshold
