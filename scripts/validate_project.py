#!/usr/bin/env python3
"""Final validation script (spec #146). Checks environment, config,
imports, and runs a real sample ingest -> search round trip. Exits
non-zero on any failure so it's usable as a CI gate."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CHECKS_PASSED: list[str] = []
CHECKS_FAILED: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        CHECKS_PASSED.append(name)
        print(f"  [OK]   {name}")
    except Exception as exc:
        CHECKS_FAILED.append(f"{name}: {exc}")
        print(f"  [FAIL] {name}: {exc}")


def check_imports() -> None:
    import rag_engine.api.app  # noqa: F401
    import rag_engine.cli.main  # noqa: F401
    import rag_engine.storage.models  # noqa: F401


def check_config() -> None:
    from rag_engine.config import Settings

    Settings(_env_file=None)


def check_sample_pipeline() -> None:
    from rag_engine.api.engine import build_engine_state
    from rag_engine.chunking.factory import create_chunker
    from rag_engine.config import get_settings
    from rag_engine.ingestion.parsers.plaintext import MarkdownParser
    from rag_engine.ingestion.pipeline import IngestionPipeline
    from rag_engine.retrieval.dense import ChunkRecord

    settings = get_settings()
    engine = build_engine_state(settings)
    chunker = create_chunker("recursive", max_tokens=200, overlap_tokens=0)
    pipeline = IngestionPipeline(
        parser=MarkdownParser(), chunker=chunker, chunking_strategy_name="recursive",
        embedding_model=engine.embedder.model_name(), embedding_dimension=engine.embedder.dimension(),
    )
    result = pipeline.ingest(b"# Test\n\nKubernetes restarts failed containers.", source_type="local_file", source_uri="validate.md")
    assert result.chunks, "pipeline produced no chunks"

    records = [ChunkRecord(chunk_id=c.chunk_id, document_id=c.document_id, content=c.content, metadata={}) for c in result.chunks]
    engine.rebuild_from_chunks(records)
    response = engine.hybrid_retriever.search("kubernetes restart", top_k=5)
    assert response.results, "search returned no results"


def check_api_health() -> None:
    from fastapi.testclient import TestClient

    from rag_engine.api.app import create_app

    with TestClient(create_app()) as client:
        resp = client.get("/health")
        assert resp.status_code == 200


def check_tests_pass() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "not integration"],
        cwd=Path(__file__).resolve().parent.parent,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"pytest failed:\n{result.stdout[-2000:]}"


def main() -> int:
    print("Running project validation...\n")
    check("imports resolve", check_imports)
    check("configuration loads and validates", check_config)
    check("sample ingest -> index -> search pipeline", check_sample_pipeline)
    check("API health endpoint responds", check_api_health)
    check("test suite passes (non-integration)", check_tests_pass)

    print(f"\n{len(CHECKS_PASSED)} passed, {len(CHECKS_FAILED)} failed")
    if CHECKS_FAILED:
        print("\nFAILURES:")
        for f in CHECKS_FAILED:
            print(f"  - {f}")
        return 1
    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
