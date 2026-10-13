"""Authentication: JWT bearer tokens and hashed API keys -> AuthContext.

Design rules (each covered by tests/security/test_authentication.py):
  * Fail closed: when auth is enabled but no credential mechanism is configured,
    startup raises (validate_auth_config) instead of silently allowing or denying all.
  * JWTs: the allowed algorithms are an explicit allow-list (never taken from the token
    header, `none` is rejected at config time), `exp` and `sub` are required, and
    `iss`/`aud` are verified when configured.
  * API keys are stored only as SHA-256 hashes (keys are 256-bit random values, so a
    fast hash is appropriate -- unlike passwords); lookup uses constant-time comparison.
  * Failures always surface as the same generic 401 message; the precise reason is only
    logged, so clients cannot probe which part of a credential was wrong.
  * `admin` comes from the token's `roles` claim (or the API-key entry), never from a
    client-controlled boolean claim.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jwt
import structlog

from rag_engine.config import Settings
from rag_engine.exceptions import AuthenticationError, ConfigurationError
from rag_engine.security.authorization import AuthContext

if TYPE_CHECKING:
    from jwt.types import Options

logger = structlog.get_logger("security.auth")

_GENERIC_FAILURE = "invalid or missing credentials"
_MIN_HS_KEY_BYTES = 32


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Return (plaintext_key, sha256_hash). Show the key once; store only the hash."""
    key = "rag_" + secrets.token_urlsafe(32)
    return key, hash_api_key(key)


@dataclass(frozen=True)
class ApiKeyRecord:
    user_id: str
    tenant_id: str | None
    groups: frozenset[str]
    admin: bool


def load_api_keys(api_keys_json: str) -> dict[str, ApiKeyRecord]:
    if not api_keys_json.strip():
        return {}
    try:
        raw = json.loads(api_keys_json)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"API_KEYS_JSON is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("API_KEYS_JSON must be a JSON object keyed by sha256 hash")
    records: dict[str, ApiKeyRecord] = {}
    for key_hash, entry in raw.items():
        if not (isinstance(key_hash, str) and len(key_hash) == 64 and all(c in "0123456789abcdef" for c in key_hash)):
            raise ConfigurationError("API_KEYS_JSON keys must be lowercase 64-char sha256 hex digests (never plaintext keys)")
        if not isinstance(entry, dict) or not entry.get("user_id"):
            raise ConfigurationError("each API key entry needs at least a user_id")
        records[key_hash] = ApiKeyRecord(
            user_id=str(entry["user_id"]), tenant_id=entry.get("tenant_id"),
            groups=frozenset(entry.get("groups", [])), admin=bool(entry.get("admin", False)),
        )
    return records


def validate_auth_config(settings: Settings) -> None:
    """Raise ConfigurationError if auth is enabled with an unusable configuration."""
    if not settings.auth_enabled:
        return
    has_keys = bool(load_api_keys(settings.api_keys_json))
    has_jwt = bool(settings.jwt_key)
    if not (has_keys or has_jwt):
        raise ConfigurationError("AUTH_ENABLED=true but neither JWT_KEY nor API_KEYS_JSON is configured")
    if has_jwt and any(a.upper().startswith("HS") for a in settings.jwt_algorithms) and len(settings.jwt_key.encode()) < _MIN_HS_KEY_BYTES:
        raise ConfigurationError(f"JWT_KEY must be at least {_MIN_HS_KEY_BYTES} bytes for HMAC algorithms")


class Authenticator:
    def __init__(self, settings: Settings) -> None:
        validate_auth_config(settings)
        self.enabled = settings.auth_enabled
        self._jwt_key = settings.jwt_key
        self._algorithms = list(settings.jwt_algorithms)
        self._issuer = settings.jwt_issuer or None
        self._audience = settings.jwt_audience or None
        self._api_keys = load_api_keys(settings.api_keys_json)

    def authenticate(self, headers: Mapping[str, str]) -> AuthContext:
        """`headers` must be case-insensitive (Starlette's Headers is)."""
        if not self.enabled:
            return AuthContext(user_id="dev-user", tenant_id=None, group_ids=frozenset(), is_admin=True)
        api_key = headers.get("x-api-key")
        if api_key and self._api_keys:
            return self._from_api_key(api_key)
        authz = headers.get("authorization", "")
        scheme, _, token = authz.partition(" ")
        if scheme.lower() == "bearer" and token.strip() and self._jwt_key:
            return self._from_jwt(token.strip())
        logger.info("auth_failed", reason="no_usable_credentials")
        raise AuthenticationError(_GENERIC_FAILURE)

    def _from_api_key(self, key: str) -> AuthContext:
        digest = hash_api_key(key)
        match: ApiKeyRecord | None = None
        for stored, record in self._api_keys.items():  # compare against all entries: no early exit on first match
            if hmac.compare_digest(stored, digest):
                match = record
        if match is None:
            logger.info("auth_failed", reason="unknown_api_key")
            raise AuthenticationError(_GENERIC_FAILURE)
        return AuthContext(user_id=match.user_id, tenant_id=match.tenant_id, group_ids=match.groups, is_admin=match.admin)

    def _from_jwt(self, token: str) -> AuthContext:
        options: Options = {"require": ["exp", "sub"], "verify_aud": self._audience is not None}
        try:
            claims = jwt.decode(
                token, self._jwt_key, algorithms=self._algorithms, issuer=self._issuer, audience=self._audience, options=options,
            )
        except jwt.PyJWTError as exc:
            logger.info("auth_failed", reason=f"jwt:{type(exc).__name__}")
            raise AuthenticationError(_GENERIC_FAILURE) from exc
        groups = claims.get("groups", [])
        roles = claims.get("roles", [])
        if not isinstance(groups, list) or not isinstance(roles, list):
            logger.info("auth_failed", reason="jwt:malformed_groups_or_roles")
            raise AuthenticationError(_GENERIC_FAILURE)
        return AuthContext(
            user_id=str(claims["sub"]), tenant_id=claims.get("tenant_id"),
            group_ids=frozenset(str(g) for g in groups), is_admin="admin" in roles,
        )
