"""Shared fixtures. `pg_clean` gives real-Postgres tests a migrated-equivalent, empty database
(the same schema Alembic creates; `alembic check` guards that they agree)."""
from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine

from rag_engine.storage.models import Base

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5432/rag_test")


@pytest.fixture
def pg_url() -> str:
    return TEST_DATABASE_URL


@pytest.fixture
def pg_clean() -> str:
    sync = create_engine(TEST_DATABASE_URL.replace("+asyncpg", "+psycopg2"))
    Base.metadata.create_all(sync)
    with sync.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    yield TEST_DATABASE_URL
    with sync.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    sync.dispose()
