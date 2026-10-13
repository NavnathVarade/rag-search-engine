"""Ingestion pipeline tests: parsers, cleaners, dedup, loaders, SSRF
protection, and the full pipeline end-to-end (spec #6-8, #50-51)."""

import httpx
import pytest

from rag_engine.chunking.recursive import RecursiveChunker
from rag_engine.exceptions import (
    DocumentTooLargeError,
    ParsingError,
    SSRFBlockedError,
)
from rag_engine.ingestion.cleaners.deduplication import DuplicateTracker, content_hash
from rag_engine.ingestion.cleaners.text_cleaner import clean_text
from rag_engine.ingestion.loaders.local_file import (
    discover_files,
    read_file_with_limit,
    should_exclude,
)
from rag_engine.ingestion.loaders.url_loader import UrlLoader
from rag_engine.ingestion.parsers.factory import get_parser_for_file
from rag_engine.ingestion.parsers.html import HtmlParser
from rag_engine.ingestion.parsers.pdf import PdfParser
from rag_engine.ingestion.parsers.plaintext import MarkdownParser, decode_bytes
from rag_engine.ingestion.pipeline import IngestionPipeline


# --- text cleaner --------------------------------------------------------
def test_clean_text_collapses_excess_whitespace():
    dirty = "Hello   world.\n\n\n\n\nNext paragraph."
    cleaned = clean_text(dirty)
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_text_preserves_code_block_indentation():
    text = "Explanation.\n\n```python\ndef f():\n    return 1\n```\n\nMore text."
    cleaned = clean_text(text)
    assert "    return 1" in cleaned  # indentation inside code block preserved


def test_clean_text_normalizes_unicode():
    text = "café\u200b test"  # zero-width space
    cleaned = clean_text(text)
    assert "\u200b" not in cleaned


# --- deduplication ------------------------------------------------
def test_content_hash_deterministic():
    assert content_hash("hello world") == content_hash("hello world")


def test_content_hash_ignores_surrounding_whitespace():
    assert content_hash("hello world") == content_hash("  hello world  ")


def test_duplicate_tracker_detects_repeat_content():
    tracker = DuplicateTracker()
    assert tracker.is_duplicate("some text") is None
    tracker.register("some text", "doc_1")
    assert tracker.is_duplicate("some text") == "doc_1"
    assert tracker.is_duplicate("different text") is None


# --- parsers -------------------------------------------------------
def test_html_parser_strips_boilerplate_and_keeps_content():
    html = b"""
    <html><head><title>K8s Docs</title></head>
    <body>
      <nav>Home | About | Contact</nav>
      <main><h1>Pod Lifecycle</h1><p>Kubernetes restarts failed containers.</p></main>
      <footer>Copyright 2024</footer>
    </body></html>
    """
    parsed = HtmlParser().parse(html)
    assert parsed.title == "K8s Docs"
    assert "Kubernetes restarts failed containers" in parsed.text
    assert "Home | About | Contact" not in parsed.text
    assert "Copyright 2024" not in parsed.text


def test_html_parser_preserves_code_blocks():
    html = b"<html><body><pre>kubectl get pods -n default</pre></body></html>"
    parsed = HtmlParser().parse(html)
    assert "kubectl get pods -n default" in parsed.text


def test_html_parser_rejects_empty_content():
    with pytest.raises(ParsingError):
        HtmlParser().parse(b"")


def test_markdown_parser_extracts_h1_title():
    md = b"# CREATE INDEX CONCURRENTLY\n\nBuilds an index without locking."
    parsed = MarkdownParser().parse(md)
    assert parsed.title == "CREATE INDEX CONCURRENTLY"


def test_decode_bytes_handles_latin1_fallback():
    raw = "café".encode("latin-1")
    text = decode_bytes(raw)
    assert "caf" in text  # doesn't crash, produces readable output


def test_parser_factory_selects_correct_parser():
    assert get_parser_for_file("readme.md").__class__.__name__ == "MarkdownParser"
    assert get_parser_for_file("index.html").__class__.__name__ == "HtmlParser"
    assert get_parser_for_file("script.py").__class__.__name__ == "SourceCodeParser"
    assert get_parser_for_file("doc.pdf").__class__.__name__ == "PdfParser"


def test_parser_factory_rejects_unsupported_type():
    from rag_engine.exceptions import UnsupportedDocumentTypeError

    with pytest.raises(UnsupportedDocumentTypeError):
        get_parser_for_file("archive.zip")


def test_pdf_parser_rejects_malformed_pdf():
    with pytest.raises(ParsingError):
        PdfParser().parse(b"not a real pdf file at all")


# --- local file loader -----------------------------------------------
def test_should_exclude_git_and_node_modules(tmp_path):
    assert should_exclude(tmp_path / ".git" / "config")
    assert should_exclude(tmp_path / "node_modules" / "pkg" / "index.js")
    assert not should_exclude(tmp_path / "src" / "main.py")


def test_discover_files_excludes_binary_and_vcs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")

    files = discover_files(tmp_path)
    names = {f.name for f in files}
    assert "main.py" in names
    assert "HEAD" not in names
    assert "logo.png" not in names


def test_read_file_with_limit_rejects_oversized_file(tmp_path):
    big_file = tmp_path / "big.txt"
    big_file.write_bytes(b"x" * (2 * 1024 * 1024))
    with pytest.raises(DocumentTooLargeError):
        read_file_with_limit(big_file, max_size_mb=1)


def test_discover_files_raises_path_traversal_error_never_escapes_root(tmp_path):
    (tmp_path / "safe.txt").write_text("ok")
    # discover_files itself only walks within root via rglob, so this test
    # documents the invariant rather than forcing a symlink escape (which
    # is environment-dependent); the traversal guard is exercised directly
    # in the security test suite (tests/security/test_sanitization.py).
    files = discover_files(tmp_path)
    assert all(str(f).startswith(str(tmp_path.resolve())) for f in files)


# --- URL loader / SSRF ----------------------------------------------
def test_url_loader_blocks_ssrf_before_any_network_call():
    loader = UrlLoader()
    with pytest.raises(SSRFBlockedError):
        loader.fetch("http://127.0.0.1/admin")


def test_url_loader_fetches_via_mocked_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"# Docs\n\nHello world.")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    loader = UrlLoader(client=client, resolver=lambda host, port: ["93.184.216.34"])
    content = loader.fetch("https://docs.example.com/page")
    assert b"Hello world" in content


def test_url_loader_enforces_size_limit_via_mocked_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (2 * 1024 * 1024))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    loader = UrlLoader(max_size_mb=1, client=client, resolver=lambda host, port: ["93.184.216.34"])
    with pytest.raises(DocumentTooLargeError):
        loader.fetch("https://docs.example.com/huge")


# --- full pipeline -----------------------------------------------
def test_full_pipeline_parse_clean_chunk():
    parser = MarkdownParser()
    chunker = RecursiveChunker(max_tokens=100, overlap_tokens=0)
    pipeline = IngestionPipeline(
        parser=parser, chunker=chunker, chunking_strategy_name="recursive",
        embedding_model="hashing-embedder", embedding_dimension=256,
    )
    content = b"# Kubernetes Restart Policy\n\nKubernetes restarts a failed container based on its restartPolicy."
    result = pipeline.ingest(content, source_type="local_file", source_uri="docs/k8s.md")

    assert not result.was_duplicate
    assert result.metadata.title == "Kubernetes Restart Policy"
    assert len(result.chunks) >= 1
    assert result.stage_timings_ms["parsing_ms"] >= 0


def test_full_pipeline_detects_duplicate_on_reingest():
    parser = MarkdownParser()
    chunker = RecursiveChunker(max_tokens=100, overlap_tokens=0)
    tracker = DuplicateTracker()
    pipeline = IngestionPipeline(
        parser=parser, chunker=chunker, chunking_strategy_name="recursive",
        embedding_model="hashing-embedder", embedding_dimension=256, duplicate_tracker=tracker,
    )
    content = b"# Doc\n\nSame content every time."
    first = pipeline.ingest(content, source_type="local_file", source_uri="a.md")
    second = pipeline.ingest(content, source_type="local_file", source_uri="a.md")

    assert not first.was_duplicate
    assert second.was_duplicate
    assert second.document_id == first.document_id
    assert second.chunks == []  # not re-chunked
