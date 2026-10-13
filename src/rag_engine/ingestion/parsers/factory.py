"""Parser selection by file extension / MIME type (spec #6)."""
from __future__ import annotations

from rag_engine.exceptions import UnsupportedDocumentTypeError
from rag_engine.ingestion.parsers.base import DocumentParser
from rag_engine.ingestion.parsers.html import HtmlParser
from rag_engine.ingestion.parsers.pdf import PdfParser
from rag_engine.ingestion.parsers.plaintext import MarkdownParser, PlainTextParser
from rag_engine.ingestion.parsers.source_code import SourceCodeParser

_CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs", ".yaml", ".yml",
    ".json", ".sql", ".sh", ".rb", ".c", ".cpp", ".h", ".hpp",
}


def get_parser_for_file(filename: str, max_pdf_pages: int = 500) -> DocumentParser:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return PdfParser(max_pages=max_pdf_pages)
    if lower.endswith((".html", ".htm")):
        return HtmlParser()
    if lower.endswith(".md"):
        return MarkdownParser()
    if lower.endswith(".txt"):
        return PlainTextParser()
    if any(lower.endswith(ext) for ext in _CODE_EXTENSIONS):
        return SourceCodeParser()
    raise UnsupportedDocumentTypeError(f"no parser available for file: {filename}")


_MEDIA_TYPES = {
    "text/html": "html", "application/xhtml+xml": "html", "application/pdf": "pdf",
    "text/markdown": "markdown", "text/x-markdown": "markdown", "text/plain": "text",
}


def get_parser_for_content_type(content_type: str, url: str, max_pdf_pages: int = 500) -> DocumentParser:
    """Choose a parser for a fetched URL from its Content-Type, falling back to the URL's
    path extension (a URL like /docs/page has no extension, so the extension alone is
    not enough)."""
    kind = _MEDIA_TYPES.get(content_type.split(";")[0].strip().lower())
    if kind == "html":
        return HtmlParser()
    if kind == "pdf":
        return PdfParser(max_pages=max_pdf_pages)
    if kind == "markdown":
        return MarkdownParser()
    if kind == "text":
        from urllib.parse import urlparse

        # text/plain is frequently served for .md files (e.g. raw.githubusercontent.com)
        return MarkdownParser() if urlparse(url).path.lower().endswith(".md") else PlainTextParser()
    from urllib.parse import urlparse

    return get_parser_for_file(urlparse(url).path or url, max_pdf_pages=max_pdf_pages)
