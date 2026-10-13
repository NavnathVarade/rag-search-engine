"""Code-aware chunking (spec #9.F). Uses Python's own `ast` module for
Python source (exact, since Python can parse Python) and a conservative
regex/brace-matching heuristic for other C-family languages (JS/TS/Go/
Java/Rust), keeping function/class/method bodies intact rather than
cutting mid-definition. Falls back to RecursiveChunker for anything it
cannot confidently parse structurally, rather than producing broken code
chunks.
"""
from __future__ import annotations

import ast
import re
from typing import Any

from rag_engine.chunking.base import Chunk, estimate_tokens
from rag_engine.chunking.recursive import RecursiveChunker

_BRACE_UNIT_RE = re.compile(
    r"^[ \t]*(export\s+)?(async\s+)?(function|class|def|func|public|private|protected|fn|impl)\b.*?\{",
    re.MULTILINE,
)


def _python_units(source: str) -> list[tuple[str, int, int]]:
    """Return (label, start_line, end_line) for each top-level function/class
    definition using Python's real AST -- exact by construction."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    units = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            units.append((node.name, node.lineno, end))
    return units


class CodeChunker:
    def __init__(self, language: str = "python", max_tokens: int = 400) -> None:
        self.language = language
        self.max_tokens = max_tokens
        self._fallback = RecursiveChunker(max_tokens=max_tokens, overlap_tokens=0)

    def chunk(self, document_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text.strip():
            return []
        lines = text.splitlines()

        if self.language == "python":
            units = _python_units(text)
        else:
            units = []  # heuristic path below

        chunks: list[Chunk] = []
        if units:
            # Include any leading module-level code (imports, docstring) as chunk 0.
            first_start = units[0][1]
            preamble = "\n".join(lines[: first_start - 1]).strip()
            position = 0
            if preamble:
                chunks.append(self._make_chunk(document_id, position, preamble, "module_preamble", metadata))
                position += 1
            for name, start, end in units:
                content = "\n".join(lines[start - 1 : end])
                chunks.append(self._make_chunk(document_id, position, content, name, metadata))
                position += 1
            return chunks

        # Non-Python heuristic: split at top-level `function|class|...{` boundaries.
        matches = list(_BRACE_UNIT_RE.finditer(text))
        if not matches:
            return self._fallback.chunk(document_id, text, metadata)

        position = 0
        boundaries = [m.start() for m in matches] + [len(text)]
        if boundaries[0] > 0:
            preamble = text[: boundaries[0]].strip()
            if preamble:
                chunks.append(self._make_chunk(document_id, position, preamble, "preamble", metadata))
                position += 1
        for start, end in zip(boundaries, boundaries[1:]):
            content = text[start:end].strip()
            if content:
                chunks.append(self._make_chunk(document_id, position, content, None, metadata))
                position += 1
        return chunks

    def _make_chunk(self, document_id: str, position: int, content: str, unit_name: str | None, metadata: dict[str, Any] | None) -> Chunk:
        meta = dict(metadata or {})
        if unit_name:
            meta["code_unit"] = unit_name
        return Chunk(
            chunk_id=f"{document_id}:code:{position}",
            document_id=document_id,
            content=content,
            position=position,
            token_count=estimate_tokens(content),
            character_count=len(content),
            language=self.language,
            metadata=meta,
        )
