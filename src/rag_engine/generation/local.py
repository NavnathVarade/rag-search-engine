"""Local/offline LLM providers.

  NullLLMProvider
    Deterministic, no-network stand-in used by tests and by
    `llm_provider: "none"` config -- lets the retrieval/citation pipeline
    be exercised end-to-end without any model. It synthesizes an answer by
    concatenating the supplied context with citation markers, honestly
    labeled as "not a real generation" in its output, and is exactly why
    the evaluation harness treats faithfulness/relevance metrics from this
    provider as trivially perfect (see docs/evaluation.md).

  OllamaProvider
    Real code targeting a local Ollama server's OpenAI-compatible
    `/v1/chat/completions` endpoint (reuses OpenAICompatibleProvider's
    HTTP contract). Not exercised in this sandbox (no local Ollama
    process/model available here), analogous to the network-gated
    embedding/reranking providers.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from rag_engine.generation.base import GenerationResult
from rag_engine.generation.openai import OpenAICompatibleProvider


class NullLLMProvider:
    def model_name(self) -> str:
        return "null-provider"

    async def generate(
        self, system_prompt: str, user_prompt: str, *, temperature: float = 0.1, max_tokens: int = 1000
    ) -> GenerationResult:
        import re

        context_ids = sorted({int(m) for m in re.findall(r'id="(\d+)"', user_prompt)})
        if not context_ids:
            text = "I don't have enough evidence in the indexed documentation to answer that reliably."
        else:
            cites = "".join(f"[{i}]" for i in context_ids)
            text = (
                f"[offline stub answer -- no LLM configured] Based on the retrieved context {cites}, "
                "see the cited source chunks for the relevant documentation."
            )
        return GenerationResult(
            text=text, prompt_tokens=len(user_prompt.split()), completion_tokens=len(text.split()), model="null"
        )

    async def generate_stream(
        self, system_prompt: str, user_prompt: str, *, temperature: float = 0.1, max_tokens: int = 1000
    ) -> AsyncIterator[str]:
        result = await self.generate(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
        for word in result.text.split(" "):
            yield word + " "


class OllamaProvider(OpenAICompatibleProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", timeout_s: float = 60.0) -> None:
        super().__init__(base_url=base_url, api_key="ollama", model=model, timeout_s=timeout_s)
