"""Static guards for the compose files (no Docker needed). They encode bugs found by merging the
files with `docker compose config`: an empty `ports: []` in an overlay does NOT clear inherited ports,
and `replicas` with a fixed host port cannot start."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text())["services"]


def test_base_compose_publishes_no_database_or_cache_ports():
    base = _load("docker-compose.yml")
    assert "ports" not in base["postgres"]
    assert "ports" not in base["redis"]


def test_dev_ports_are_loopback_only_and_live_in_override():
    override = _load("docker-compose.override.yml")
    for svc in ("postgres", "redis"):
        assert all(p.startswith("127.0.0.1:") for p in override[svc]["ports"])


def test_prod_overlay_requires_secrets_and_has_no_replica_port_clash():
    prod = _load("docker-compose.prod.yml")
    api_env = prod["api"]["environment"]
    assert api_env["AUTH_ENABLED"] == "true"
    for var in ("API_KEYS_JSON", "METRICS_TOKEN"):
        assert ":?" in str(api_env[var]), f"{var} must be a required variable"
    assert ":?" in str(prod["postgres"]["environment"]["POSTGRES_PASSWORD"])
    assert "ports" not in prod["postgres"]
    assert not (prod["api"].get("ports") and prod["api"].get("deploy", {}).get("replicas", 1) > 1)


def test_api_service_migrates_then_uses_postgres_backend():
    api = _load("docker-compose.yml")["api"]
    assert api["environment"]["STORE_BACKEND"] == "postgres"
    assert "alembic upgrade head" in " ".join(api["command"]) and "uvicorn" in " ".join(api["command"])
