"""Answer relevance and context relevance (spec #34).

  lexical_relevance_proxy()   -- deterministic query/answer term-overlap
                                   proxy (same caveats as faithfulness proxy)
  llm_judge_relevance()       -- LLM-as-judge, same non-determinism caveats
                                   as llm_judge_faithfulness above.
"""
from __future__ import annotations

from rag_engine.evaluation.metrics.faithfulness import _ngrams
from rag_engine.generation.base import LLMProvider


def lexical_relevance_proxy(query: str, text: str, n: int = 2) -> float:
    query_ngrams = _ngrams(query, n)
    if not query_ngrams:
        return 0.0
    text_ngrams = _ngrams(text, n)
    return len(query_ngrams & text_ngrams) / len(query_ngrams)


_JUDGE_PROMPT = """Rate how directly and completely the ANSWER addresses the QUESTION, \
ignoring whether the answer is factually correct. Respond with ONLY a number 0 to 1.

Question: {question}
Answer: {answer}

Score (0 to 1 only):"""


async def llm_judge_relevance(llm: LLMProvider, question: str, answer: str) -> float:
    result = await llm.generate(
        "You are a strict, careful evaluation judge.",
        _JUDGE_PROMPT.format(question=question, answer=answer),
        temperature=0.0,
        max_tokens=10,
    )
    try:
        score = float(result.text.strip().split()[0])
        return max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        return 0.0
