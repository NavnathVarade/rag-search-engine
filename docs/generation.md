# Generation

## Prompt design (`generation/prompts.py`)
The system prompt enforces: answer only from supplied evidence; cite every
claim with `[n]`; say "I don't have enough evidence..." rather than guess
when context is empty or insufficient; preserve exact technical
terminology; treat `<context>` blocks as **untrusted data, not
instructions** and explicitly refuse to follow anything that looks like an
embedded instruction.

## LLM providers (`generation/`)
- `NullLLMProvider` (`local.py`): offline, deterministic, no network --
  the default (`LLM_PROVIDER=none`). Synthesizes a stub answer citing
  whichever context was supplied, honestly labeled `[offline stub
  answer]`. Used by every automated test and the CLI/API smoke path in
  this sandbox.
- `OpenAICompatibleProvider` (`openai.py`): real HTTP client against any
  OpenAI-compatible `/chat/completions` endpoint (OpenAI, vLLM, Azure
  OpenAI, Ollama's OpenAI shim). Real retry/backoff logic (exponential,
  only on transient 429/5xx, never on permanent 4xx) -- unit-tested
  against a mocked `httpx` transport, not exercised end-to-end here since
  `api.openai.com` isn't reachable from this sandbox.
- `OllamaProvider` (`local.py`): thin subclass pointing at a local Ollama
  server.

## Citations (`generation/citations.py`)
The LLM is asked to cite with `[n]` markers tied to context position.
`extract_and_validate_citations` parses those markers and checks each one
against the chunks *actually supplied* -- a marker pointing outside that
range is flagged invalid and stripped from the final answer
(`strip_invalid_citations`) rather than shown to the user as a citation
to nothing (spec #140: never present a citation to a nonexistent chunk).

## Prompt-injection defense (`security/prompt_injection.py`)
Defense in depth, not a single silver bullet: (1) the system prompt
explicitly instructs the model to treat context as data; (2) retrieved
content is scanned for common injection patterns
(`scan_for_prompt_injection`) and flagged in the response
(`ChatAnswer.injection_flags`) so a caller/operator can see when indexed
documentation contained something suspicious, whether or not the model
actually obeyed it.
