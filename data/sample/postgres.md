# PostgreSQL Indexing

## CREATE INDEX CONCURRENTLY

CREATE INDEX CONCURRENTLY builds a PostgreSQL index without holding a lock
that blocks writes to the table, at the cost of a slower build. It cannot
run inside a transaction block.

## B-tree vs GIN

PostgreSQL B-tree indexes support equality and range queries and are the
default index type. GIN indexes are optimized for composite values such
as arrays, jsonb, and full text search.
