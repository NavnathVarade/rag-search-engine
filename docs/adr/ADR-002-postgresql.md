# ADR-002: PostgreSQL for durable storage

**Context:** Need durable storage for documents, chunks, ingestion jobs,
evaluation runs, and permissions, with real relational integrity
(foreign keys between documents and chunks, unique constraints for
dedup).

**Decision:** PostgreSQL 16, accessed via SQLAlchemy 2.0 (async) +
Alembic migrations.

**Alternatives considered:** MongoDB (schema flexibility not needed here
— the domain model is genuinely relational: documents have chunks,
chunks have permissions); SQLite (fine for local dev but the spec
requires connection pooling and concurrent-write semantics for the
multi-worker ingestion story).

**Consequences:** This project explicitly does *not* delegate the
conceptual retrieval algorithms to pgvector, even though pgvector could
compute cosine similarity inside Postgres — the point of this project is
a transparent, from-scratch retrieval core (see docs/architecture.md "Why
an in-memory index"). Postgres here is the system of record for documents/
chunks/permissions/evaluation history, not the retrieval engine itself.
