"""Chunking strategy factory (spec #9). Central place to construct any
supported chunker by name, used by the ingestion pipeline and the
chunking-experiment runner (evaluation/experiment.py) so a strategy can be
selected purely via configuration string.
"""
from __future__ import annotations

from typing import Any

from rag_engine.chunking.base import Chunker
from rag_engine.chunking.code import CodeChunker
from rag_engine.chunking.fixed import FixedSizeChunker
from rag_engine.chunking.markdown import MarkdownChunker
from rag_engine.chunking.parent_child import ParentChildChunker
from rag_engine.chunking.recursive import RecursiveChunker
from rag_engine.chunking.semantic import SemanticChunker
from rag_engine.chunking.sentence import SentenceChunker
from rag_engine.exceptions import ConfigurationError

_STRATEGIES = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
    "markdown": MarkdownChunker,
    "code": CodeChunker,
    "parent_child": ParentChildChunker,
    "semantic": SemanticChunker,  # requires embedder=... kwarg
}


def create_chunker(strategy: str, **kwargs: Any) -> Chunker:
    if strategy not in _STRATEGIES:
        raise ConfigurationError(
            f"unknown chunking strategy '{strategy}'. Available: {sorted(_STRATEGIES)}"
        )
    cls = _STRATEGIES[strategy]
    return cls(**kwargs)


def available_strategies() -> list[str]:
    return sorted(_STRATEGIES)
