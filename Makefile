.PHONY: dev check test test-unit test-integration lint typecheck format evaluate benchmark build up down migrate validate sync lock

# Python environment: uv. `make sync` creates .venv from uv.lock (dev tools included).
sync:
	uv sync

lock:
	uv lock

dev:
	docker compose up --build

up:
	docker compose up -d --build

down:
	docker compose down

migrate:
	uv run alembic upgrade head

check: lint typecheck test

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit tests/retrieval tests/ingestion tests/evaluation tests/security -v

test-integration:
	uv run pytest tests/integration tests/api -v -m "integration or not integration"

evaluate:
	uv run python scripts/run_evaluation.py --dataset data/evaluation/sample.jsonl

benchmark:
	uv run python scripts/benchmark.py

build:
	docker build -t rag-engine:latest .

validate:
	uv run python scripts/validate_project.py

frontend-install:
	cd frontend && npm ci

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-check:
	cd frontend && npm run typecheck && npm run lint && npm test && npm run build
