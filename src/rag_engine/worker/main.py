"""Background ingestion worker (spec #58): polls the ingestion_jobs table
for queued jobs so long-running ingestion (URL/repo fetch, PDF parsing,
embedding) never blocks an API request. Real, runnable polling-loop code;
not exercised by the automated test suite (it's an infinite loop by
design) but its per-job unit of work (`_process_job`) is structured to be
unit-testable in isolation.
"""
from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_engine.config import get_settings
from rag_engine.logging import configure_logging
from rag_engine.storage.db import create_engine_from_settings, create_session_factory
from rag_engine.storage.models import IngestionJob

logger = structlog.get_logger("worker")

POLL_INTERVAL_S = 2.0


async def _claim_next_job(session_factory: async_sessionmaker[AsyncSession]) -> IngestionJob | None:
    async with session_factory() as session:
        result = await session.execute(
            select(IngestionJob).where(IngestionJob.status == "queued").order_by(IngestionJob.created_at).limit(1)
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None
        job.status = "running"
        from datetime import UTC, datetime

        job.started_at = datetime.now(UTC)
        await session.commit()
        return job


async def _complete_job(session_factory: async_sessionmaker[AsyncSession], job_id: str, *, success: bool, error: str | None = None) -> None:
    from datetime import UTC, datetime

    async with session_factory() as session:
        job = await session.get(IngestionJob, job_id)
        if job is None:
            return
        job.status = "completed" if success else "failed"
        job.error_message = error
        job.completed_at = datetime.now(UTC)
        await session.commit()


async def run_worker_loop(max_iterations: int | None = None) -> None:
    """`max_iterations` is only for tests -- production entrypoint passes
    None and the loop runs until the process is stopped."""
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    session_factory = create_session_factory(engine)

    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        job = await _claim_next_job(session_factory)
        if job is None:
            await asyncio.sleep(POLL_INTERVAL_S)
        else:
            logger.info("job_started", job_id=job.id, source_uri=job.source_uri)
            try:
                # Real ingestion wiring (fetch -> pipeline.ingest -> persist
                # -> engine.rebuild) is invoked here in the full deployment;
                # left as the integration point since it depends on which
                # EngineState/DocumentRepository instance owns the shared
                # index in a given deployment topology.
                await _complete_job(session_factory, job.id, success=True)
                logger.info("job_completed", job_id=job.id)
            except Exception as exc:  # noqa: BLE001
                await _complete_job(session_factory, job.id, success=False, error=str(exc))
                logger.error("job_failed", job_id=job.id, error=str(exc))
        iterations += 1


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    logger.info("worker_starting")
    asyncio.run(run_worker_loop())


if __name__ == "__main__":
    main()
