"""Source-code "parsing": decode + language detection by extension. Actual
structural understanding (functions/classes) happens later in
chunking/code.py -- this stage's job is just getting clean text +
language metadata out of the file (spec #6)."""
from __future__ import annotations

from rag_engine.ingestion.parsers.base import ParsedDocument
from rag_engine.ingestion.parsers.plaintext import decode_bytes

_EXTENSION_LANGUAGE = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".go": "go", ".java": "java", ".rs": "rust",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".sql": "sql",
    ".sh": "shell", ".md": "markdown", ".rb": "ruby", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp",
}


def detect_language(filename: str) -> str:
    for ext, lang in _EXTENSION_LANGUAGE.items():
        if filename.endswith(ext):
            return lang
    return "unknown"


class SourceCodeParser:
    def parse(self, content: bytes, source_hint: str = "") -> ParsedDocument:
        text = decode_bytes(content)
        language = detect_language(source_hint)
        return ParsedDocument(text=text, metadata={"parser": "source_code", "language": language}, language=language)
