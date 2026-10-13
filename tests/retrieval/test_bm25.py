"""Unit tests for the from-scratch BM25 implementation, including a
hand-calculated worked example (per spec item #16 / #54)."""
import math

import pytest

from rag_engine.retrieval.sparse.bm25 import Bm25Scorer
from rag_engine.retrieval.sparse.inverted_index import InvertedIndex
from rag_engine.retrieval.sparse.tokenizer import Tokenizer, TokenizerConfig


def build_index(docs: dict[str, str], **tok_kwargs) -> InvertedIndex:
    tok = Tokenizer(TokenizerConfig(**tok_kwargs)) if tok_kwargs else Tokenizer()
    idx = InvertedIndex(tokenizer=tok)
    for doc_id, text in docs.items():
        idx.add_document(doc_id, text)
    return idx


def test_hand_calculated_bm25_score():
    """Three tiny documents, manually computed against the BM25 formula.

    Corpus (stopwords removed by default tokenizer):
      d1: "cat sat mat"        -> 3 tokens
      d2: "cat sat on the mat" -> "on"/"the" are stopwords -> "cat sat mat" -> 3 tokens
      d3: "dog ran fast"       -> 3 tokens
    avgdl = 3.0, N = 3

    Query: "cat"
      df(cat) = 2 (d1, d2)
      idf(cat) = ln((3 - 2 + 0.5)/(2 + 0.5) + 1) = ln(1.6/2.5 + 1) = ln(1.64)
      For d1: f(cat,d1)=1, |d1|=3 => denom = 1 + 1.5*(1-0.75+0.75*3/3) = 1+1.5*1 = 2.5
              numerator = 1*(1.5+1) = 2.5
              score = idf * (2.5/2.5) = idf
    """
    idx = build_index({"d1": "cat sat mat", "d2": "cat sat on the mat", "d3": "dog ran fast"})
    scorer = Bm25Scorer(idx, k1=1.5, b=0.75)

    expected_idf_cat = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1.0)
    hits = scorer.score("cat", top_k=10)

    scores = {h.doc_id: h.score for h in hits}
    assert set(scores.keys()) == {"d1", "d2"}
    assert scores["d1"] == pytest.approx(expected_idf_cat, rel=1e-9)
    assert scores["d2"] == pytest.approx(expected_idf_cat, rel=1e-9)
    assert "d3" not in scores  # no term overlap at all


def test_bm25_rewards_exact_rare_term_over_topical_similarity():
    """Simulates the CREATE INDEX CONCURRENTLY scenario: a document that
    contains the exact rare phrase should outrank a document merely 'about'
    the same topic without the exact tokens."""
    idx = build_index(
        {
            "generic_indexing_doc": "database indexes speed up query performance significantly",
            "exact_doc": "postgresql 16 create index concurrently avoids locking the table",
        }
    )
    scorer = Bm25Scorer(idx)
    hits = scorer.score("create index concurrently postgresql 16", top_k=10)
    assert hits[0].doc_id == "exact_doc"
    assert hits[0].score > 0


def test_bm25_length_normalization_penalizes_padding():
    """A doc that repeats irrelevant content to inflate length should not
    outrank a short, tightly on-topic doc for the same term frequency."""
    idx = build_index(
        {
            "short": "kubernetes restarts a failed container automatically",
            "long_padded": "kubernetes restarts a failed container automatically. "
            + ("filler filler filler filler filler filler filler filler. " * 10),
        }
    )
    scorer = Bm25Scorer(idx)
    hits = scorer.score("kubernetes restart failed container", top_k=10)
    scores = {h.doc_id: h.score for h in hits}
    assert scores["short"] > scores["long_padded"]


def test_bm25_empty_query_returns_no_results():
    idx = build_index({"d1": "some content here"})
    scorer = Bm25Scorer(idx)
    assert scorer.score("", top_k=10) == []
    assert scorer.score("the a of", top_k=10) == []  # all stopwords


def test_bm25_empty_corpus_returns_no_results():
    idx = InvertedIndex()
    scorer = Bm25Scorer(idx)
    assert scorer.score("anything", top_k=10) == []


def test_bm25_invalid_parameters_rejected():
    idx = build_index({"d1": "content"})
    with pytest.raises(ValueError):
        Bm25Scorer(idx, k1=-1.0)
    with pytest.raises(ValueError):
        Bm25Scorer(idx, b=1.5)


def test_bm25_results_sorted_descending():
    idx = build_index(
        {
            "a": "kubernetes kubernetes kubernetes pod",
            "b": "kubernetes pod",
            "c": "pod",
        }
    )
    scorer = Bm25Scorer(idx)
    hits = scorer.score("kubernetes pod", top_k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_bm25_top_k_limits_results():
    idx = build_index({f"d{i}": "kubernetes pod restart" for i in range(20)})
    scorer = Bm25Scorer(idx)
    hits = scorer.score("kubernetes pod", top_k=5)
    assert len(hits) == 5


def test_porter_stemmer_matches_published_examples():
    from rag_engine.retrieval.sparse.stemmer import porter_stem

    cases = {"caresses": "caress", "ponies": "poni", "agreed": "agre", "motoring": "motor", "hopping": "hop",
             "relational": "relat", "conditional": "condit", "generalizations": "gener", "oscillators": "oscil",
             "happy": "happi", "controll": "control", "k8s": "k8s", "asyncio.gather": "asyncio.gather"}
    for word, expected in cases.items():
        assert porter_stem(word) == expected, word


def test_porter_tokenizer_conflates_inflections_so_bm25_matches_them():
    from rag_engine.retrieval.sparse.tokenizer import Tokenizer, TokenizerConfig

    tok = Tokenizer(TokenizerConfig(stem=True, stemmer="porter"))
    assert tok.tokenize("indexing indexes indexed") == ["index", "index", "index"]
    assert tok.tokenize("restarting containers") == tok.tokenize("restart container")
