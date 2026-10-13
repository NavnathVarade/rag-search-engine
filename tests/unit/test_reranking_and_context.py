"""Tests for the reranking and context-selection stages (spec #22-25)."""
from rag_engine.context.compressor import ExtractiveCompressor
from rag_engine.context.deduplicator import is_near_duplicate, jaccard_similarity
from rag_engine.context.selector import ContextSelector
from rag_engine.reranking.scoring import HeuristicReranker
from rag_engine.retrieval.models import RetrievalMethod, RetrievalResult


def make_result(chunk_id, doc_id, content, score=1.0, rank=1):
    return RetrievalResult(
        chunk_id=chunk_id, document_id=doc_id, content=content, score=score, rank=rank,
        method=RetrievalMethod.hybrid_rrf,
    )


# --- Reranker ---------------------------------------------------------
def test_reranker_promotes_exact_phrase_match():
    candidates = [
        make_result("a", "d1", "This document discusses containers and orchestration in general."),
        make_result("b", "d1", "Kubernetes restarts a failed container automatically via restartPolicy."),
    ]
    reranker = HeuristicReranker()
    reranked = reranker.rerank("kubernetes restarts a failed container", candidates, top_k=2)
    assert reranked[0].chunk_id == "b"
    assert reranked[0].reranker_score is not None
    assert reranked[0].reranker_score > reranked[1].reranker_score


def test_reranker_assigns_sequential_new_ranks():
    candidates = [make_result(f"c{i}", "d1", f"content about topic {i}") for i in range(5)]
    reranked = HeuristicReranker().rerank("topic", candidates, top_k=3)
    assert [r.rank for r in reranked] == [1, 2, 3]
    assert len(reranked) == 3


def test_reranker_handles_empty_candidates():
    assert HeuristicReranker().rerank("query", [], top_k=5) == []


# --- Deduplication --------------------------------------------------------
def test_jaccard_similarity_identical_text():
    assert jaccard_similarity("the quick brown fox jumps", "the quick brown fox jumps") == 1.0


def test_jaccard_similarity_unrelated_text():
    assert jaccard_similarity("kubernetes pod restart policy", "postgresql index concurrently locking") < 0.3


def test_is_near_duplicate_detects_lexical_overlap():
    a = "Kubernetes restarts a failed container based on restartPolicy settings for the pod"
    b = "Kubernetes restarts a failed container based on restartPolicy settings for the pod."
    assert is_near_duplicate(a, b, threshold=0.8)


# --- Context selector -------------------------------------------------
def test_selector_deduplicates_near_identical_chunks():
    a = make_result("a", "d1", "Kubernetes restarts a failed container based on the restart policy configured.", score=0.9)
    b = make_result("b", "d1", "Kubernetes restarts a failed container based on the restart policy configured!", score=0.8)
    c = make_result("c", "d2", "PostgreSQL CREATE INDEX CONCURRENTLY avoids locking the table during build.", score=0.7)

    selector = ContextSelector(max_context_tokens=1000, dedup_threshold=0.8)
    result = selector.select([a, b, c])

    selected_ids = {s.result.chunk_id for s in result.selected}
    assert "a" in selected_ids
    assert "b" not in selected_ids  # deduped against a
    assert "c" in selected_ids


def test_selector_respects_max_chunks_per_document():
    chunks = [make_result(f"c{i}", "same-doc", f"unique sentence number {i} about a totally different subject entirely") for i in range(5)]
    selector = ContextSelector(max_context_tokens=10_000, max_chunks_per_document=2)
    result = selector.select(chunks)
    assert len(result.selected) == 2
    assert len(result.skipped) == 3


def test_selector_respects_token_budget():
    big_content = " ".join(f"word{i}" for i in range(500))
    chunks = [make_result(f"c{i}", f"d{i}", big_content) for i in range(5)]
    selector = ContextSelector(max_context_tokens=600, max_chunks_per_document=10)
    result = selector.select(chunks)
    assert result.total_tokens <= 600
    assert len(result.selected) < 5


def test_selector_explanations_are_populated():
    chunks = [make_result("a", "d1", "some relevant technical content about indexing")]
    selector = ContextSelector(max_context_tokens=1000)
    result = selector.select(chunks)
    assert result.selected[0].reasons
    assert any("relevance score" in r for r in result.selected[0].reasons)


# --- Compressor ----------------------------------------------------
def test_compressor_leaves_short_content_untouched():
    compressor = ExtractiveCompressor()
    content = "Short sentence about kubernetes."
    assert compressor.compress("kubernetes", content, target_tokens=100) == content


def test_compressor_shrinks_long_content_and_keeps_relevant_sentences():
    compressor = ExtractiveCompressor()
    content = (
        "This is an introductory sentence with no keywords at all whatsoever whatsoever. "
        "Kubernetes restarts a failed container using the restart policy. "
        "This is another filler sentence about nothing important today either. "
        "The kubelet monitors container health continuously in the background process."
    )
    compressed = compressor.compress("kubernetes container restart", content, target_tokens=15)
    assert "restart" in compressed.lower()
    assert len(compressed.split()) <= 20


def _result(chunk_id, *, dense=None, bm25=None, method="hybrid_rrf"):
    from rag_engine.retrieval.models import RetrievalMethod, RetrievalResult

    return RetrievalResult(chunk_id=chunk_id, document_id=chunk_id, content=f"text of {chunk_id}", score=0.03, rank=1,
                           method=RetrievalMethod(method), dense_score=dense, bm25_score=bm25)


def test_evidence_gate_drops_noise_level_dense_only_matches():
    from rag_engine.context.selector import ContextSelector

    sel = ContextSelector(min_dense_similarity=0.2).select([_result("weak", dense=0.09), _result("strong", dense=0.31), _result("lexical", dense=0.01, bm25=4.2)])
    assert {s.result.chunk_id for s in sel.selected} == {"strong", "lexical"}
    assert any("below evidence threshold" in reason for _, reason in sel.skipped)


def test_answerer_skips_llm_when_no_evidence():
    import asyncio

    from rag_engine.context.selector import ContextSelector
    from rag_engine.generation.answerer import RagAnswerer

    class ExplodingLLM:
        async def generate(self, *a, **k):
            raise AssertionError("LLM must not be called without evidence")

    selection = ContextSelector(min_dense_similarity=0.2).select([_result("weak", dense=0.05)])
    answer = asyncio.run(RagAnswerer(ExplodingLLM()).answer("unrelated question", selection))
    assert answer.had_sufficient_evidence is False and answer.citations == [] and answer.prompt_tokens == 0
