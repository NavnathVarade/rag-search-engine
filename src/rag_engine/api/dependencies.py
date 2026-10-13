"""FastAPI dependency providers (spec #115: routes depend on abstractions,
never construct business logic themselves)."""
from __future__ import annotations

from typing import Any

from fastapi import Depends, Request

from rag_engine.api.engine import EngineState
from rag_engine.config import Settings, get_settings
from rag_engine.exceptions import AuthenticationError, AuthorizationError, RateLimitExceededError
from rag_engine.security.authorization import AuthContext


def get_engine_state(request: Request) -> EngineState:
    return request.app.state.engine


def get_app_settings() -> Settings:
    return get_settings()


def get_auth_context(request: Request) -> AuthContext:
    """Authenticate the request (JWT bearer or API key) and return the caller's identity.

    With AUTH_ENABLED=false (development) every request is an admin so the API works out
    of the box. With AUTH_ENABLED=true invalid/missing credentials raise AuthenticationError
    (HTTP 401). Every non-health route depends on this."""
    state = request.app.state
    settings: Settings = state.settings
    client_ip = request.client.host if request.client else "unknown"
    try:
        auth: AuthContext = state.authenticator.authenticate(request.headers)
    except AuthenticationError:
        state.metrics.auth_failures.inc()
        # Failed logins are throttled per client IP (credential-guessing / fake-token floods).
        _check_limit(state, f"authfail:{client_ip}", settings.rate_limit_anonymous_per_min, "auth_failure")
        raise
    _check_limit(state, f"id:{auth.tenant_id}:{auth.user_id}", settings.rate_limit_authenticated_per_min, "identity")
    return auth


def _check_limit(state: Any, key: str, limit: int, scope: str) -> None:
    try:
        state.limiter.check(key, limit)
    except RateLimitExceededError:
        state.metrics.rate_limited.labels(scope=scope).inc()
        raise


def require_admin(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    if not auth.is_admin:
        raise AuthorizationError("administrator role required")
    return auth
