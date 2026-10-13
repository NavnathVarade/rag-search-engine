"""Fixtures for real-Postgres integration tests. Requires a running
Postgres reachable at TEST_DATABASE_URL (defaults to the local instance
set up for this project's own test suite)."""
import os

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.config import Settings
from rag_engine.storage.db import create_engine_from_settings, create_session_factory
from rag_engine.storage.models import Base

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5432/rag_test"
)


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    settings = Settings(_env_file=None, database_url=TEST_DATABASE_URL)
    engine = create_engine_from_settings(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        yield session
        await session.rollback()

    async with engine.begin() as conn:
        # Clean slate between tests -- truncate rather than drop/recreate
        # for speed.
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())
    await engine.dispose()
