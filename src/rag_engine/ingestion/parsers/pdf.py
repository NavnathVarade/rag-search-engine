"""PDF parsing via pypdf (mature library, per spec's allowed list).
Handles common PDF extraction artifacts (spec #8): hyphenated line-wrap
breaks are rejoined, and pages are joined with an explicit page-boundary
marker (spec #8 "page boundaries") so downstream chunking can be aware of
them without losing that information."""
from __future__ import annotations

import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_engine.exceptions import DocumentTooLargeError, ParsingError
from rag_engine.ingestion.parsers.base import ParsedDocument

_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def _dehyphenate(text: str) -> str:
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


class PdfParser:
    def __init__(self, max_pages: int = 500) -> None:
        self.max_pages = max_pages

    def parse(self, content: bytes, source_hint: str = "") -> ParsedDocument:
        if not content:
            raise ParsingError("empty document", details={"source": source_hint})
        import io

        try:
            reader = PdfReader(io.BytesIO(content))
        except PdfReadError as exc:
            raise ParsingError(f"malformed PDF: {exc}", details={"source": source_hint}) from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ParsingError(f"encrypted PDF cannot be read: {exc}", details={"source": source_hint}) from exc

        n_pages = len(reader.pages)
        if n_pages > self.max_pages:
            raise DocumentTooLargeError(
                f"PDF has {n_pages} pages, exceeding max_pdf_pages={self.max_pages}",
                details={"source": source_hint, "pages": n_pages},
            )

        page_texts = []
        for _page_index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""  # a single corrupt page shouldn't fail the whole document
            page_texts.append(_dehyphenate(text))

        full_text = "\n\n[PAGE_BREAK]\n\n".join(page_texts)
        title = None
        try:
            meta = reader.metadata
            if meta and meta.title:
                title = str(meta.title)
        except Exception:
            pass

        return ParsedDocument(text=full_text, title=title, metadata={"parser": "pdf", "page_count": n_pages})
