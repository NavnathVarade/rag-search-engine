"""Unit + property-based tests for cosine similarity (spec #12, #54)."""
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from rag_engine.retrieval.cosine import batch_cosine_similarity, cosine_similarity, normalize


def test_identical_vectors_have_similarity_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_minus_one():
    a = np.array([1.0, 2.0, 3.0])
    b = -a
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_known_worked_example():
    # A = [1, 2, 3], B = [4, 5, 6]
    # A.B = 4+10+18 = 32; ||A|| = sqrt(14); ||B|| = sqrt(77)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    expected = 32 / (np.sqrt(14) * np.sqrt(77))
    assert cosine_similarity(a, b) == pytest.approx(expected, rel=1e-9)


def test_zero_vector_returns_zero_not_nan():
    a = np.zeros(5)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(a, a) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_batch_cosine_matches_loop_of_scalar_cosine():
    rng = np.random.default_rng(42)
    query = rng.normal(size=16)
    matrix = rng.normal(size=(50, 16))

    batch = batch_cosine_similarity(query, matrix)
    loop = np.array([cosine_similarity(query, matrix[i]) for i in range(matrix.shape[0])])
    np.testing.assert_allclose(batch, loop, rtol=1e-9, atol=1e-12)


def test_batch_cosine_handles_zero_rows():
    query = np.array([1.0, 0.0])
    matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sims = batch_cosine_similarity(query, matrix)
    assert sims[0] == 0.0
    assert sims[1] == pytest.approx(1.0)
    assert sims[2] == pytest.approx(0.0)


def test_normalize_unit_length():
    v = np.array([3.0, 4.0])
    n = normalize(v)
    assert np.linalg.norm(n) == pytest.approx(1.0)


def test_normalize_zero_vector_stays_zero():
    v = np.zeros(4)
    n = normalize(v)
    np.testing.assert_array_equal(n, v)


@given(
    arrays(dtype=np.float64, shape=8, elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)),
    arrays(dtype=np.float64, shape=8, elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)),
)
def test_property_cosine_is_symmetric(a, b):
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a), abs=1e-9)


@given(arrays(dtype=np.float64, shape=8, elements=st.floats(0.1, 100, allow_nan=False)))
def test_property_positive_scalar_multiple_has_similarity_one(v):
    scaled = v * 3.7
    sim = cosine_similarity(v, scaled)
    assert sim == pytest.approx(1.0, abs=1e-6)


@given(
    arrays(dtype=np.float64, shape=8, elements=st.floats(-50, 50, allow_nan=False, allow_infinity=False))
)
def test_property_cosine_bounded(v):
    rng = np.random.default_rng(abs(hash(v.tobytes())) % (2**32))
    other = rng.normal(size=8)
    sim = cosine_similarity(v, other)
    assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6
