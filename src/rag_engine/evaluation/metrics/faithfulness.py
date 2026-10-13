"""Faithfulness / hallucination-rate estimation (spec #34).

Two tiers, clearly separated per spec's requirement to not present
LLM-judge scores as objective truth:

  lexical_faithfulness_proxy()
    Deterministic, reproducible: what fraction of the answer's content
    n-grams also appear in the cited context chunks. Cheap, has no
    understanding of paraphrase or entailment, so it under-counts
    faithful answers that legitimately paraphrase -- treat it as a lower
    bound / fast smoke-test signal, not ground truth.

  llm_judge_faithfulness()
    Delegates to an LLMProvider to judge whether the answer is entailed
    by the context, following the "LLM-as-judge" pattern common in RAG
    evaluation (see e.g. RAGAS). This is inherently non-deterministic
    (depends on judge model, prompt, sampling) and judge-model-biased;
    docs/evaluation.md documents this limitation explicitly and this
    function's docstring repeats it so it can't be missed at the call site.
"""
from __future__ import annotations

import re

from rag_engine.generation.base import LLMProvider

_WORD_RE = re.compile(r"[a-z0-9]+")


def _ngrams(text: str, n: int = 3) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def lexical_faithfulness_proxy(answer: str, context_texts: list[str], n: int = 3) -> float:
    answer_ngrams = _ngrams(answer, n)
    if not answer_ngrams:
        return 0.0
    context_ngrams: set[str] = set()
    for text in context_texts:
        context_ngrams |= _ngrams(text, n)
    supported = answer_ngrams & context_ngrams
    return len(supported) / len(answer_ngrams)


_JUDGE_PROMPT = """You are evaluating whether an AI-generated answer is faithful to (i.e. \
fully supported by, not contradicting or inventing beyond) the provided context. \
Respond with ONLY a number from 0 to 1, where 1 means every claim in the answer is \
directly supported by the context, and 0 means the answer contains claims not \
supported by or contradicting the context.

Context:
{context}

Answer:
{answer}

Score (0 to 1 only):"""


async def llm_judge_faithfulness(llm: LLMProvider, answer: str, context_texts: list[str]) -> float:
    """NOTE: non-deterministic, judge-model-dependent. See module docstring
    and docs/evaluation.md 'LLM-as-judge limitations' before treating this
    as ground truth."""
    context_block = "\n\n".join(context_texts)
    result = await llm.generate(
        "You are a strict, careful evaluation judge.",
        _JUDGE_PROMPT.format(context=context_block, answer=answer),
        temperature=0.0,
        max_tokens=10,
    )
    try:
        score = float(result.text.strip().split()[0])
        return max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        return 0.0
