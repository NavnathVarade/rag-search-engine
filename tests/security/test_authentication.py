"""Unit tests for JWT / API-key authentication (security/authentication.py)."""
from __future__ import annotations

import json
import time

import jwt
import pytest

from rag_engine.config import Settings
from rag_engine.exceptions import AuthenticationError, ConfigurationError
from rag_engine.security.authentication import (
    Authenticator,
    generate_api_key,
    hash_api_key,
    load_api_keys,
    validate_auth_config,
)

KEY = "k" * 48
ISS, AUD = "https://issuer.test", "rag-api"


def _settings(**kw) -> Settings:
    base = {"auth_enabled": True, "jwt_key": KEY, "jwt_issuer": ISS, "jwt_audience": AUD}
    base.update(kw)
    return Settings(**base)


def _token(claims=None, key=KEY, alg="HS256", **headers) -> str:
    c = {"sub": "alice", "iss": ISS, "aud": AUD, "exp": int(time.time()) + 300, "tenant_id": "t1", "groups": ["eng"]}
    c.update(claims or {})
    return jwt.encode(c, key, algorithm=alg, headers=headers or None)


def _auth(settings=None) -> Authenticator:
    return Authenticator(settings or _settings())


def _bearer(tok: str) -> dict:
    return {"authorization": f"Bearer {tok}"}


def test_disabled_auth_grants_dev_admin():
    ctx = Authenticator(Settings(auth_enabled=False)).authenticate({})
    assert ctx.is_admin and ctx.user_id == "dev-user"


def test_valid_jwt_maps_claims_to_context():
    ctx = _auth().authenticate(_bearer(_token({"roles": ["admin"]})))
    assert (ctx.user_id, ctx.tenant_id, ctx.group_ids, ctx.is_admin) == ("alice", "t1", frozenset({"eng"}), True)


def test_admin_comes_only_from_roles_not_from_a_client_claim():
    ctx = _auth().authenticate(_bearer(_token({"is_admin": True, "admin": True})))
    assert ctx.is_admin is False


@pytest.mark.parametrize(
    "token",
    [
        _token({"exp": int(time.time()) - 10}),  # expired
        _token(key="x" * 48),  # wrong signature
        _token({"aud": "someone-else"}),
        _token({"iss": "https://evil.test"}),
        jwt.encode({"sub": "a", "iss": ISS, "aud": AUD}, KEY, algorithm="HS256"),  # no exp
        jwt.encode({"iss": ISS, "aud": AUD, "exp": int(time.time()) + 60}, KEY, algorithm="HS256"),  # no sub
        _token(alg="HS512"),  # algorithm not on the allow-list
        "not.a.jwt",
    ],
)
def test_invalid_jwts_are_rejected(token):
    with pytest.raises(AuthenticationError):
        _auth().authenticate(_bearer(token))


def test_alg_none_token_is_rejected():
    forged = jwt.encode({"sub": "mallory", "iss": ISS, "aud": AUD, "exp": int(time.time()) + 60, "roles": ["admin"]}, key=None, algorithm="none")
    with pytest.raises(AuthenticationError):
        _auth().authenticate(_bearer(forged))


def test_malformed_groups_claim_is_rejected():
    with pytest.raises(AuthenticationError):
        _auth().authenticate(_bearer(_token({"groups": "eng"})))


def test_failure_messages_do_not_reveal_which_check_failed():
    messages = set()
    for hdrs in ({}, _bearer(_token(key="x" * 48)), _bearer(_token({"aud": "no"})), {"x-api-key": "nope"}):
        with pytest.raises(AuthenticationError) as exc:
            _auth().authenticate(hdrs)
        messages.add(exc.value.message)
    assert len(messages) == 1


def test_api_key_authenticates_and_wrong_key_fails():
    key, digest = generate_api_key()
    s = _settings(jwt_key="", api_keys_json=json.dumps({digest: {"user_id": "svc", "tenant_id": "t9", "groups": ["bots"], "admin": False}}))
    auth = Authenticator(s)
    ctx = auth.authenticate({"x-api-key": key})
    assert (ctx.user_id, ctx.tenant_id, ctx.group_ids, ctx.is_admin) == ("svc", "t9", frozenset({"bots"}), False)
    with pytest.raises(AuthenticationError):
        auth.authenticate({"x-api-key": key + "x"})


def test_generated_keys_are_unique_and_hash_matches():
    (k1, h1), (k2, h2) = generate_api_key(), generate_api_key()
    assert k1 != k2 and h1 != h2 and hash_api_key(k1) == h1 and len(h1) == 64


def test_fails_closed_when_enabled_without_credentials():
    with pytest.raises(ConfigurationError):
        validate_auth_config(Settings(auth_enabled=True))


def test_short_hmac_secret_rejected():
    with pytest.raises(ConfigurationError):
        validate_auth_config(_settings(jwt_key="short"))


def test_plaintext_api_key_in_config_is_rejected():
    with pytest.raises(ConfigurationError):
        load_api_keys(json.dumps({"rag_plaintext_key": {"user_id": "x"}}))


def test_algorithm_none_cannot_be_enabled_in_config():
    with pytest.raises(ValueError):
        Settings(jwt_algorithms=("none",))
