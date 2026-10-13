"""Tests for every chunking strategy (spec #9, #52, #54)."""
import pytest

from rag_engine.chunking.code import CodeChunker
from rag_engine.chunking.factory import available_strategies, create_chunker
from rag_engine.chunking.fixed import FixedSizeChunker
from rag_engine.chunking.markdown import MarkdownChunker
from rag_engine.chunking.parent_child import ParentChildChunker
from rag_engine.chunking.recursive import RecursiveChunker
from rag_engine.chunking.semantic import SemanticChunker
from rag_engine.chunking.sentence import SentenceChunker
from rag_engine.embeddings.local import HashingEmbedder
from rag_engine.exceptions import ChunkingError, ConfigurationError

LONG_TEXT = " ".join(f"word{i}" for i in range(1000))

MARKDOWN_DOC = """# Kubernetes Pod Lifecycle

## Restart Policy

Kubernetes restarts a failed container according to restartPolicy.

## Probes

Readiness and liveness probes serve different purposes.

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
```

The above config defines a liveness probe.
"""

PYTHON_CODE = '''
import os

def restart_container(pod):
    """Restart a container in the given pod."""
    return pod.restart()


class ContainerManager:
    def __init__(self, client):
        self.client = client

    def list_pods(self):
        return self.client.pods()
'''


# --- Fixed size ----------------------------------------------------------
def test_fixed_chunker_respects_size_and_overlap():
    chunker = FixedSizeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk("doc1", LONG_TEXT)
    assert len(chunks) > 1
    for c in chunks[:-1]:
        assert c.token_count == 50


def test_fixed_chunker_overlap_actually_overlaps():
    chunker = FixedSizeChunker(chunk_size=20, overlap=5)
    chunks = chunker.chunk("doc1", LONG_TEXT)
    words0 = chunks[0].content.split()
    words1 = chunks[1].content.split()
    assert words0[-5:] == words1[:5]


def test_fixed_chunker_rejects_invalid_overlap():
    with pytest.raises(ChunkingError):
        FixedSizeChunker(chunk_size=10, overlap=10)
    with pytest.raises(ChunkingError):
        FixedSizeChunker(chunk_size=0, overlap=0)


def test_fixed_chunker_empty_text():
    assert FixedSizeChunker().chunk("doc1", "") == []


# --- Recursive -------------------------------------------------------------
def test_recursive_chunker_stays_under_budget_where_possible():
    chunker = RecursiveChunker(max_tokens=50, overlap_tokens=0)
    text = "Paragraph one. " * 20 + "\n\n" + "Paragraph two. " * 20
    chunks = chunker.chunk("doc1", text)
    assert len(chunks) > 1
    assert all(c.token_count <= 60 for c in chunks)  # small slack for merge rounding


def test_recursive_chunker_source_content_preserved():
    """Concatenated chunk words (ignoring the intentional overlap words)
    should reproduce all words from the source in order."""
    chunker = RecursiveChunker(max_tokens=30, overlap_tokens=0)
    text = "One two three four five. " * 10
    chunks = chunker.chunk("doc1", text)
    reconstructed = " ".join(c.content for c in chunks)
    original_words = text.split()
    reconstructed_words = reconstructed.split()
    assert reconstructed_words == original_words


# --- Sentence --------------------------------------------------------------
def test_sentence_chunker_does_not_split_mid_sentence():
    chunker = SentenceChunker(max_tokens=15)
    text = "Kubernetes restarts containers automatically. It uses the kubelet. Probes control readiness."
    chunks = chunker.chunk("doc1", text)
    for c in chunks:
        assert c.content.strip().endswith((".", "!", "?"))


# --- Markdown ----------------------------------------------------------
def test_markdown_chunker_preserves_code_block_intact():
    chunker = MarkdownChunker(max_tokens=400)
    chunks = chunker.chunk("doc1", MARKDOWN_DOC)
    joined = "\n".join(c.content for c in chunks)
    assert "livenessProbe:" in joined
    assert "httpGet:" in joined
    # code fence content shouldn't be split across two different chunks
    fence_chunks = [c for c in chunks if "livenessProbe" in c.content]
    assert any("port: 8080" in c.content for c in fence_chunks)


def test_markdown_chunker_tracks_heading_path():
    chunker = MarkdownChunker(max_tokens=400)
    chunks = chunker.chunk("doc1", MARKDOWN_DOC)
    probes_chunk = next(c for c in chunks if "liveness probe" in c.content.lower() and "purposes" in c.content.lower())
    assert "Probes" in probes_chunk.heading_path


# --- Code ------------------------------------------------------------------
def test_code_chunker_keeps_function_intact():
    chunker = CodeChunker(language="python", max_tokens=400)
    chunks = chunker.chunk("doc1", PYTHON_CODE)
    fn_chunk = next(c for c in chunks if c.metadata.get("code_unit") == "restart_container")
    assert "def restart_container(pod):" in fn_chunk.content
    assert "return pod.restart()" in fn_chunk.content


def test_code_chunker_separates_class_from_function():
    chunker = CodeChunker(language="python", max_tokens=400)
    chunks = chunker.chunk("doc1", PYTHON_CODE)
    units = {c.metadata.get("code_unit") for c in chunks}
    assert "restart_container" in units
    assert "ContainerManager" in units


def test_code_chunker_falls_back_on_unparseable_python():
    chunker = CodeChunker(language="python", max_tokens=50)
    chunks = chunker.chunk("doc1", "def broken(:\n  this is not python")
    assert len(chunks) >= 1  # falls back to recursive splitter, doesn't crash


# --- Parent-child ------------------------------------------------------
def test_parent_child_children_reference_parent():
    chunker = ParentChildChunker(parent_max_tokens=100, child_max_tokens=20)
    text = "Sentence about kubernetes. " * 30
    chunks = chunker.chunk("doc1", text)
    parents = [c for c in chunks if c.parent_chunk_id is None]
    children = [c for c in chunks if c.parent_chunk_id is not None]
    assert parents and children
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_chunk_id in parent_ids for c in children)


# --- Semantic ----------------------------------------------------------
def test_semantic_chunker_runs_and_respects_budget():
    embedder = HashingEmbedder(dimension=64)
    chunker = SemanticChunker(embedder=embedder, similarity_threshold=0.3, max_tokens=30)
    text = (
        "Kubernetes manages containers. Containers run in pods. "
        "PostgreSQL is a relational database. Indexes speed up queries."
    )
    chunks = chunker.chunk("doc1", text)
    assert len(chunks) >= 1
    assert all(c.token_count <= 35 for c in chunks)


# --- Factory -------------------------------------------------------
def test_factory_creates_known_strategies():
    for name in ("fixed", "recursive", "sentence", "markdown"):
        chunker = create_chunker(name)
        assert chunker is not None


def test_factory_rejects_unknown_strategy():
    with pytest.raises(ConfigurationError):
        create_chunker("not_a_real_strategy")


def test_available_strategies_lists_all():
    strategies = available_strategies()
    for expected in ("fixed", "recursive", "sentence", "markdown", "code", "parent_child", "semantic"):
        assert expected in strategies
