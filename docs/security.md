# Security

## Threat model
Retrieved documents are **untrusted data** -- they may come from public
websites, third-party repos, or user uploads, and could contain
adversarial content aimed at the LLM (prompt injection) or the ingestion
pipeline itself (SSRF via a malicious URL, path traversal via a malicious
archive/filename, resource-exhaustion via oversized files or zip bombs).

## Mitigations implemented
- **SSRF** (`security/sanitization.py::validate_public_url`): blocks
  loopback/private/link-local/cloud-metadata IP ranges and disallowed
  schemes *before* any network call; re-validates on redirect rather than
  blindly following it. Tested in `tests/security/test_sanitization.py`
  and `tests/ingestion/test_ingestion.py::test_url_loader_blocks_ssrf_before_any_network_call`.
- **Path traversal** (`validate_safe_relative_path`): rejects `..`
  components and absolute paths, resolves against a real base directory.
- **Prompt injection** (`security/prompt_injection.py` +
  `generation/prompts.py`): pattern-based detection + system-prompt-level
  instruction to never treat context as commands. See docs/generation.md.
- **Resource limits** (`config.py`): `max_document_size_mb`,
  `max_pdf_pages`, `max_files_per_repo`, enforced in the loaders/parsers
  (`ingestion/loaders/local_file.py`, `parsers/pdf.py`) before/during
  parsing, not after.
- **Authorization** (`security/authorization.py`): `filter_authorized()`
  is the single choke point every retrieval path calls *before* results
  reach context selection -- never retrieve-then-filter. Includes a
  cross-tenant isolation guarantee (a document tagged `tenant_id=X` is
  never visible to a caller in tenant `Y`, even if `visibility=public` or
  the caller is explicitly listed in `allowed_users` -- see
  `tests/security/test_authorization.py::test_cross_tenant_document_never_accessible`).
- **Rate limiting** (`api/middleware.py::RateLimitMiddleware`):
  differentiated anonymous/authenticated limits on ingestion/chat/
  evaluation/index endpoints.
- **Structured error responses** (`exceptions.py` + `api/app.py`
  exception handler): every engine error carries a stable `code` and
  never leaks a raw traceback to the client.

## Known gaps (stated plainly, not hidden)
- No SQL injection surface exists in the current codebase because every
  query goes through SQLAlchemy's parameterized query builder -- there is
  no raw string-interpolated SQL anywhere in `storage/`.
- **Authentication** (`security/authentication.py`): JWT bearer tokens (explicit
  algorithm allow-list, `exp`/`sub` required, optional `iss`/`aud`) and SHA-256-hashed API
  keys mapped to a user, tenant, groups and an admin flag. With `AUTH_ENABLED=true` and no
  credential mechanism configured, startup fails closed. Failures return one generic 401.
  Covered by `tests/security/test_authentication.py` and `tests/api/test_api_auth.py`; the
  API-key path, tenant isolation and fail-closed startup were also exercised against a live
  `uvicorn`. Not provided: user management, key rotation/expiry or revocation lists, OAuth/OIDC
  discovery, and JWKS key fetching (the JWT key is a static secret or PEM public key).
- **SSRF / DNS rebinding**: the URL loader resolves the hostname, requires every address to be
  public, then connects to that exact validated IP (Host header and TLS SNI preserved), and
  re-validates every redirect hop. Tests run against real local HTTP and HTTPS servers.
  Residual risk: the allow/deny list is static and `--allow-private-network` is an operator
  override that disables the check.
- **/metrics** is open unless `METRICS_TOKEN` is set; series names reveal usage patterns, so set
  the token or restrict the route at the network layer.
