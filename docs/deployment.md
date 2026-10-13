# Deployment

## Docker (single machine)
```bash
docker compose up --build                                  # dev
docker compose -f docker-compose.yml -f docker-compose.prod.yml up  # prod overlay
```
`Dockerfile` is a multi-stage build (compiler toolchain only in the build
stage; the runtime image ships only `libpq5` + the installed wheel),
non-root user, healthcheck against `/health`, no secrets baked in (all
config via environment variables, `.env` is git-ignored).

The API container runs `alembic upgrade head` and then uvicorn, with `STORE_BACKEND=postgres`, so a fresh
`docker compose up` creates the schema and stores documents in Postgres (run a single API instance, or move
the migration into a one-off job before scaling out).

`docker-compose.yml` publishes only the API port; Postgres/Redis host ports (loopback only) come from
`docker-compose.override.yml`, which plain `docker compose up` merges automatically and the
production command (explicit `-f` files) does not.

`docker-compose.prod.yml` sets `AUTH_ENABLED=true`, JSON logging, memory limits and
`restart: unless-stopped`, and **requires** `POSTGRES_PASSWORD`, `API_KEYS_JSON` and
`METRICS_TOKEN` in the environment (compose refuses to start if one is missing):
```bash
export POSTGRES_PASSWORD=$(openssl rand -hex 24)
export METRICS_TOKEN=$(openssl rand -hex 24)
uv run rag auth create-key --user-id alice --tenant-id t1   # prints the key once plus the hashed entry for API_KEYS_JSON
export API_KEYS_JSON='{"<sha256>": {"user_id": "alice", "tenant_id": "t1", "admin": false}}'
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```
Verification status: the compose files are validated with `docker compose config` (merged output
checked for published ports and required secrets) and the package was installed into a clean venv from
only the files the Dockerfile copies, then exercised end to end. **The images themselves have not
been built or run here (no Docker daemon in the dev sandbox)**; run `docker compose build` and the
healthcheck on a Docker host before relying on them.

## Database migrations
```bash
make migrate   # alembic upgrade head
```
Real migrations, generated with `alembic revision --autogenerate` against
the actual `storage/models.py` SQLAlchemy models and verified in this
project against a live PostgreSQL 16 instance (14 tables created
successfully; see `migrations/versions/`).

## Scaling beyond one process
This reference implementation's retrieval index (`EngineState`) lives in
one process's memory, rebuilt from Postgres via `rag index build`. Running
multiple API replicas means each currently holds its own independent copy
of the index -- fine for read-heavy workloads if you rebuild all replicas
after ingestion, but not a shared/consistent index out of the box. Two
realistic paths to genuinely horizontal scaling, neither implemented
here: (a) move the ANN/BM25 structures into a shared service (e.g.
pgvector + a shared BM25/OpenSearch-style service) that all API replicas
query over the network, trading some of this project's "transparent,
in-process, steppable algorithm" property for shared state; or (b) keep
the in-process design but put a a build/broadcast step in front of
deploys (rebuild once, ship the serialized index artifact to every
replica). See docs/final-report.md "Limitations".

## Worker
`Dockerfile.worker` runs `rag_engine.worker.main`, a polling loop against
`ingestion_jobs` (spec #58). The polling/job-state-transition logic is
real and tested at the unit level for its claim/complete functions; the
actual "do the ingestion work" step inside the loop is a documented
integration point rather than fully wired in this reference build --
see `src/rag_engine/worker/main.py` docstring and docs/final-report.md.
