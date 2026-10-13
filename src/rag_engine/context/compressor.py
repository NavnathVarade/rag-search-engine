"""Context compression (spec #24): optional extractive compression that
trims a selected chunk down to the sentences most relevant to the query
when a chunk is large relative to the remaining budget, instead of
dropping the whole chunk. Uses simple query-term sentence scoring --
no LLM call, so it is fast, free, and deterministic.
"""
from __future__ import annotations

from rag_engine.chunking.base import estimate_tokens
from rag_engine.chunking.recursive import _split_sentences
from rag_engine.retrieval.sparse.tokenizer import Tokenizer


class ExtractiveCompressor:
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or Tokenizer()

    def compress(self, query: str, content: str, target_tokens: int) -> str:
        if estimate_tokens(content) <= target_tokens:
            return content

        sentences = _split_sentences(content)
        if not sentences:
            return content

        query_terms = set(self.tokenizer.tokenize(query))
        scored = []
        for i, sentence in enumerate(sentences):
            terms = set(self.tokenizer.tokenize(sentence))
            overlap = len(terms & query_terms)
            scored.append((overlap, -i, sentence, i))  # -i breaks ties toward earlier sentences

        scored.sort(reverse=True)
        selected: list[tuple[int, str]] = []
        tokens = 0
        for _, _, sentence, i in scored:
            sentence_tokens = estimate_tokens(sentence)
            if tokens + sentence_tokens > target_tokens and selected:
                continue
            selected.append((i, sentence))
            tokens += sentence_tokens
            if tokens >= target_tokens:
                break

        selected.sort(key=lambda x: x[0])  # restore original order
        return " ".join(s for _, s in selected)
