"""Provider selection and strict-model handling.

* `build_embedder` / `build_llm` honour the settings (a regression: the engine used to always build the
  hashing embedder, and `LLM_PROVIDER=openai` silently meant "stub").
* The `sentence_transformers` library is not installed in CI, so a tiny stand-in module is injected. These
  tests prove the *wiring* (right class, right arguments, right dimension reaches the index). They say
  nothing about the quality of a real model; that needs the real library (see docs/embeddings.md).
* The OpenAI provider is exercised against a mocked HTTP transport (the network boundary only) that replays
  the error wording OpenAI uses when a model refuses `max_tokens` or a custom temperature.
"""
import json
import sys
import types

import httpx
import numpy as np
import pytest

from rag_engine.api.engine import build_embedder, build_engine_state, build_llm
from rag_engine.config import Settings
from rag_engine.embeddings.local import HashingEmbedder
from rag_engine.exceptions import ConfigurationError, GenerationError
from rag_engine.generation.answerer import RagAnswerer
from rag_engine.generation.local import NullLLMProvider
from rag_engine.generation.openai import OpenAICompatibleProvider
from rag_engine.retrieval.dense import ChunkRecord


class _FakeSentenceTransformer:
    last_args: tuple = ()

    def __init__(self, name: str, device: str = "cpu") -> None:
        type(self).last_args = (name, device)

    def get_sentence_embedding_dimension(self) -> int:
        return 8

    def encode(self, batch, normalize_embeddings: bool = True):  # noqa: ANN001
        out = np.array([[float(len(t) % 5 + 1)] * 8 for t in batch])
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def fake_st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


# --- embedder ---------------------------------------------------------------------------------
def test_default_embedder_is_hashing():
    assert isinstance(build_embedder(Settings()), HashingEmbedder)


def test_sentence_transformers_provider_is_used_with_configured_model(fake_st):
    emb = build_embedder(Settings(embedding_provider="sentence_transformers", embedding_model="BAAI/bge-small-en-v1.5", embedding_device="cpu"))
    assert emb.model_name() == "BAAI/bge-small-en-v1.5"
    assert emb.dimension() == 8  # taken from the model, not from EMBEDDING_DIMENSION
    assert _FakeSentenceTransformer.last_args == ("BAAI/bge-small-en-v1.5", "cpu")


def test_engine_index_uses_the_model_dimension(fake_st):
    state = build_engine_state(Settings(embedding_provider="sentence_transformers", embedding_dimension=384))
    state.rebuild_from_chunks([ChunkRecord(chunk_id="c1", document_id="d1", content="hello world", metadata={})])
    assert state.embedder.dimension() == 8
    assert state.ann_index is not None and state.ann_index.dimension == 8


def test_missing_sentence_transformers_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes `import` raise ImportError
    with pytest.raises(Exception, match="uv sync --extra embeddings"):
        build_embedder(Settings(embedding_provider="sentence_transformers"))


def test_unknown_embedding_provider_fails_loudly():
    with pytest.raises(ConfigurationError, match="EMBEDDING_PROVIDER"):
        build_embedder(Settings(embedding_provider="sentence-transformer"))


# --- llm selection ----------------------------------------------------------------------------
def test_openai_is_an_alias_for_openai_compatible():
    assert isinstance(build_llm(Settings(llm_provider="openai", llm_api_key="k", llm_model="gpt-6-luna")), OpenAICompatibleProvider)
    assert build_llm(Settings(llm_provider="openai", llm_api_key="k", llm_model="gpt-6-luna")).model_name() == "gpt-6-luna"


def test_openai_without_key_uses_stub():
    assert isinstance(build_llm(Settings(llm_provider="openai", llm_api_key="")), NullLLMProvider)


def test_unknown_llm_provider_fails_loudly():
    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        build_llm(Settings(llm_provider="opnai", llm_api_key="k"))


def test_answerer_uses_configured_temperature_and_token_budget():
    state = build_engine_state(Settings(llm_temperature=0.7, llm_max_tokens=4096))
    assert state.answerer is not None
    assert (state.answerer.temperature, state.answerer.max_tokens) == (0.7, 4096)
    assert isinstance(state.answerer, RagAnswerer)


# --- strict models ----------------------------------------------------------------------------
def _provider(handler) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider(base_url="https://x/v1", api_key="k", model="gpt-6-luna", client=client, max_retries=1)


def _ok(payload_seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        payload_seen.append(json.loads(request.content))
        body = {
            "choices": [{"message": {"content": "hi [1]"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}, "model": "gpt-6-luna",
        }
        return httpx.Response(200, json=body)
    return handler


async def test_default_request_shape_is_unchanged():
    seen: list = []
    result = await _provider(_ok(seen)).generate("s", "u", temperature=0.1, max_tokens=500)
    assert result.text == "hi [1]"
    assert seen[0]["max_tokens"] == 500 and seen[0]["temperature"] == 0.1 and "max_completion_tokens" not in seen[0]


async def test_model_refusing_max_tokens_and_temperature_is_handled_once_then_remembered():
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "max_tokens" in body:
            msg = "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."
            return httpx.Response(400, json={"error": {"message": msg}})
        if "temperature" in body:
            msg = "Unsupported value: 'temperature' does not support 0.1 with this model. Only the default (1) value is supported."
            return httpx.Response(400, json={"error": {"message": msg}})
        return _ok([])(request)

    provider = _provider(handler)
    assert (await provider.generate("s", "u", temperature=0.1, max_tokens=500)).text == "hi [1]"
    assert len(seen) == 3 and seen[-1]["max_completion_tokens"] == 500
    assert "max_tokens" not in seen[-1] and "temperature" not in seen[-1]
    seen.clear()
    await provider.generate("s", "u", temperature=0.1, max_tokens=500)
    assert len(seen) == 1  # learned: no further rejected attempts


async def test_unrelated_400_is_not_retried_or_swallowed():
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "model not found"}})

    with pytest.raises(GenerationError, match="400"):
        await _provider(handler).generate("s", "u")
    assert len(calls) == 1


async def test_empty_answer_from_exhausted_budget_is_a_clear_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}, "finish_reason": "length"}]})

    with pytest.raises(GenerationError, match="LLM_MAX_TOKENS"):
        await _provider(handler).generate("s", "u")
