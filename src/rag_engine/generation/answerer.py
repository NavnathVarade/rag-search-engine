"""End-to-end RAG answering: selected context -> prompt -> LLM ->
citation validation -> structured ChatAnswer (spec #26-28, #41, #140-141).
This is the "LLM" + "Citations" boxes of the architecture diagram.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag_engine.context.selector import ContextSelectionResult
from rag_engine.generation.base import LLMProvider
from rag_engine.generation.citations import (
    Citation,
    extract_and_validate_citations,
    strip_invalid_citations,
)
from rag_engine.generation.prompts import SYSTEM_PROMPT, PromptContext, build_user_prompt
from rag_engine.security.prompt_injection import scan_for_prompt_injection


@dataclass
class ChatAnswer:
    answer: str
    citations: list[Citation]
    context_chunk_ids: list[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
    had_sufficient_evidence: bool
    injection_flags: list[str] = field(default_factory=list)


INSUFFICIENT_EVIDENCE_ANSWER = "I don't have enough evidence in the indexed documentation to answer that reliably."


class RagAnswerer:
    def __init__(self, llm: LLMProvider, temperature: float = 0.1, max_tokens: int = 1000) -> None:
        self.llm = llm
        self.temperature = temperature  # LLM_TEMPERATURE
        self.max_tokens = max_tokens  # LLM_MAX_TOKENS

    async def answer(
        self, question: str, selection: ContextSelectionResult, temperature: float | None = None, max_tokens: int | None = None
    ) -> ChatAnswer:
        if not selection.selected:
            # Nothing relevant was retrieved: answer deterministically and skip the LLM call entirely
            # (no cost, and no chance of an answer made up from the model's own knowledge).
            return ChatAnswer(
                answer=INSUFFICIENT_EVIDENCE_ANSWER, citations=[], context_chunk_ids=[], model="none",
                prompt_tokens=0, completion_tokens=0, had_sufficient_evidence=False,
            )
        contexts = [
            PromptContext(
                chunk_id=s.result.chunk_id,
                content=s.result.content,
                source_label=s.result.metadata.get("source", s.result.document_id),
            )
            for s in selection.selected
        ]

        injection_flags: list[str] = []
        for ctx in contexts:
            scan = scan_for_prompt_injection(ctx.content)
            if scan.suspicious:
                injection_flags.append(f"chunk {ctx.chunk_id}: {scan.matched_patterns}")

        user_prompt = build_user_prompt(question, contexts)
        result = await self.llm.generate(
            SYSTEM_PROMPT, user_prompt,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )

        document_ids = {s.result.chunk_id: s.result.document_id for s in selection.selected}
        validation = extract_and_validate_citations(result.text, contexts, document_ids)

        answer_text = result.text
        if not validation.all_valid:
            answer_text = strip_invalid_citations(answer_text, validation.invalid_markers)

        had_sufficient_evidence = bool(contexts) and (
            validation.has_any_citation or "don't have enough evidence" not in answer_text.lower()
        )

        return ChatAnswer(
            answer=answer_text,
            citations=[c for c in validation.citations if c.valid],
            context_chunk_ids=[c.chunk_id for c in contexts],
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            had_sufficient_evidence=had_sufficient_evidence,
            injection_flags=injection_flags,
        )
