"""Parser abstraction: raw bytes + declared/sniffed content type -> plain
text + structural metadata (spec #6). Each parser raises ParsingError on
malformed input rather than silently returning garbage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ParsedDocument:
    text: str
    title: str | None = None
    language: str | None = None
    metadata: dict = field(default_factory=dict)


class DocumentParser(Protocol):
    def parse(self, content: bytes, source_hint: str = "") -> ParsedDocument: ...
