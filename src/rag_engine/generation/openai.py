"""OpenAI-compatible LLM provider (works with OpenAI, and any
OpenAI-compatible endpoint -- vLLM, Ollama's OpenAI shim, Azure OpenAI,
etc -- since the wire format is the same `/chat/completions` API).

Real, network-dependent code: not exercised in this sandbox (api.openai.com
is not in the allowed egress list here), but the HTTP contract, retry
policy, and error handling below are genuine and unit-tested against a
mocked transport (mocking the network boundary is explicitly allowed by
spec #154 -- only the algorithms under test must not be mocked).
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_engine.exceptions import GenerationError, GenerationTimeoutError
from rag_engine.generation.base import GenerationResult

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class _RetryableError(Exception):
    pass


class _BadRequest(GenerationError):
    """HTTP 400 from the API; keeps the response body so the caller can see which parameter was refused."""

    def __init__(self, message: str, body: str) -> None:
        super().__init__(message)
        self.body = body


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        # Newer OpenAI models reject `max_tokens` (they want `max_completion_tokens`) and reasoning models only
        # accept the default temperature. Rather than hard-coding model names, the provider starts with the
        # widely supported form and, if the API says a parameter is refused, adapts once and remembers.
        self._token_param = "max_tokens"
        self._send_temperature = True

    def model_name(self) -> str:
        return self.model

    def _payload(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, stream: bool) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            self._token_param: max_tokens,
            "stream": stream,
        }
        if self._send_temperature:
            payload["temperature"] = temperature
        return payload

    def _adapt_to_rejection(self, payload: dict, body: str) -> bool:
        """If the 400 says a parameter we sent is unsupported, drop/rename it in `payload` and return True."""
        text = body.lower()
        if not any(w in text for w in ("unsupported", "not supported", "does not support", "unrecognized", "use 'max_completion_tokens'")):
            return False
        if "max_tokens" in payload and "max_tokens" in text:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            self._token_param = "max_completion_tokens"
            return True
        if "temperature" in payload and "temperature" in text:
            payload.pop("temperature")
            self._send_temperature = False
            return True
        return False

    async def _post_adapting(self, payload: dict) -> dict:
        for _ in range(3):  # at most: rename max_tokens, then drop temperature, then the real attempt
            try:
                return await self._post_chat_completion(payload)
            except _BadRequest as exc:
                if not self._adapt_to_rejection(payload, exc.body):
                    raise
        return await self._post_chat_completion(payload)

    async def _post_chat_completion(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(_RetryableError),
            reraise=True,
        )
        async def _call() -> dict:
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout_s
                )
            except httpx.TimeoutException as exc:
                raise GenerationTimeoutError(f"LLM request timed out after {self.timeout_s}s") from exc
            except httpx.HTTPError as exc:
                raise _RetryableError(str(exc)) from exc

            if resp.status_code in _RETRYABLE_STATUS:
                raise _RetryableError(f"transient upstream error: {resp.status_code}")
            if resp.status_code >= 400:
                # permanent errors (401, 400 malformed request, etc) must NOT retry
                message = f"LLM request failed: {resp.status_code} {resp.text[:200]}"
                if resp.status_code == 400:
                    raise _BadRequest(message, resp.text)
                raise GenerationError(message)
            return resp.json()

        try:
            return await _call()
        except _RetryableError as exc:
            raise GenerationError(f"LLM request failed after {self.max_retries} retries: {exc}") from exc

    async def generate(
        self, system_prompt: str, user_prompt: str, *, temperature: float = 0.1, max_tokens: int = 1000
    ) -> GenerationResult:
        payload = self._payload(system_prompt, user_prompt, temperature, max_tokens, stream=False)
        data = await self._post_adapting(payload)
        choice = data["choices"][0]
        usage = data.get("usage", {})
        text = choice["message"].get("content") or ""
        if not text.strip():
            reason = choice.get("finish_reason", "unknown")
            raise GenerationError(
                f"the model returned an empty answer (finish_reason={reason}). Reasoning models spend part of the "
                "token budget on thinking: if finish_reason is 'length', raise LLM_MAX_TOKENS."
            )
        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def generate_stream(
        self, system_prompt: str, user_prompt: str, *, temperature: float = 0.1, max_tokens: int = 1000
    ) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._payload(system_prompt, user_prompt, temperature, max_tokens, stream=True)
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout_s
            ) as resp:
                if resp.status_code >= 400:
                    raise GenerationError(f"LLM stream failed: {resp.status_code}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    import json

                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
                    except (KeyError, IndexError, ValueError):
                        continue
        except httpx.TimeoutException as exc:
            raise GenerationTimeoutError(f"LLM stream timed out after {self.timeout_s}s") from exc
