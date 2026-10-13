"""From-scratch cosine similarity, vectorized with NumPy.

    cos(A, B) = (A . B) / (||A|| * ||B||)

Handles:
  * zero vectors (defined as similarity 0.0 rather than raising/NaN)
  * batch (matrix x matrix) similarity for dense retrieval scans
  * numerical stability via a small epsilon in the denominator
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0.0 if either is
    the zero vector (cosine is undefined there; 0.0 is the conventional
    "no similarity" fallback used throughout this codebase)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < _EPS or norm_b < _EPS:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of a single query vector (D,) against a matrix of
    document vectors (N, D). Returns an (N,) array of similarities.

    Vectorized: computes ||matrix_i|| for all rows at once and a single
    matrix-vector dot product, avoiding a Python-level loop over N docs.
    """
    query = np.asarray(query, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D (N, D)")
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(f"dimension mismatch: query {query.shape} vs matrix {matrix.shape}")

    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(matrix, axis=1)

    dots = matrix @ query
    denom = doc_norms * query_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(denom > _EPS, dots / denom, 0.0)
    return sims


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a (N, D) matrix (or a single (D,) vector).
    Zero vectors are left as zero (not divided)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / norm if norm > _EPS else vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms > _EPS, norms, 1.0)
    normalized = vectors / safe_norms
    normalized[norms.flatten() <= _EPS] = 0.0
    return normalized
