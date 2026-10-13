"""Plain-text and Markdown "parsing" -- decode bytes to text with a robust
fallback ladder for encoding issues (spec #6, #8: handle malformed
unicode without destroying content)."""
from __future__ import annotations

from rag_engine.exceptions import ParsingError
from rag_engine.ingestion.parsers.base import ParsedDocument

_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


def decode_bytes(content: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    # last resort: replace undecodable bytes rather than dropping the whole doc
    return content.decode("utf-8", errors="replace")


class PlainTextParser:
    def parse(self, content: bytes, source_hint: str = "") -> ParsedDocument:
        if not content:
            raise ParsingError("empty document", details={"source": source_hint})
        text = decode_bytes(content)
        title = next((line.strip() for line in text.splitlines() if line.strip()), None)
        return ParsedDocument(text=text, title=title, metadata={"parser": "plaintext"})


class MarkdownParser:
    def parse(self, content: bytes, source_hint: str = "") -> ParsedDocument:
        if not content:
            raise ParsingError("empty document", details={"source": source_hint})
        text = decode_bytes(content)
        title = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break
        return ParsedDocument(text=text, title=title, metadata={"parser": "markdown"})
