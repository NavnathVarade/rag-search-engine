#!/usr/bin/env python3
"""Thin script wrapper for `rag ingest`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if __name__ == "__main__":
    from rag_engine.cli.main import app

    sys.argv = [sys.argv[0], "ingest", *sys.argv[1:]]
    app()
