"""Markdown-aware chunking (spec #9.E). Uses markdown-it-py to parse the
token stream so headings, code fences, tables, and lists are recognized
structurally rather than via regex-guessing, then groups content under its
owning heading path (H1 > H2 > H3 ...) and packs it into token-budgeted
chunks, keeping code blocks and tables intact rather than splitting mid-block.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from markdown_it import MarkdownIt

from rag_engine.chunking.base import Chunk, estimate_tokens

_MD = MarkdownIt("commonmark")


@dataclass
class _Section:
    heading_path: list[str]
    blocks: list[str] = field(default_factory=list)


def _extract_sections(text: str) -> list[_Section]:
    tokens = _MD.parse(text)
    sections: list[_Section] = [_Section(heading_path=[])]
    heading_stack: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1])
            title_tok = tokens[i + 1]
            title = title_tok.content
            heading_stack = heading_stack[: level - 1] + [title]
            sections.append(_Section(heading_path=list(heading_stack)))
            i += 3  # heading_open, inline, heading_close
            continue
        if tok.type == "fence" or tok.type == "code_block":
            block = f"```{tok.info}\n{tok.content}```" if tok.type == "fence" else tok.content
            sections[-1].blocks.append(block)
        elif tok.type == "inline":
            if tok.content.strip():
                sections[-1].blocks.append(tok.content)
        elif tok.type in ("bullet_list_open", "ordered_list_open"):
            pass  # list items arrive as their own inline tokens
        i += 1
    return [s for s in sections if s.blocks]


class MarkdownChunker:
    def __init__(self, max_tokens: int = 400) -> None:
        self.max_tokens = max_tokens

    def chunk(self, document_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        sections = _extract_sections(text)
        chunks: list[Chunk] = []
        position = 0
        for section in sections:
            current_blocks: list[str] = []
            current_tokens = 0
            for block in section.blocks:
                block_tokens = estimate_tokens(block)
                if current_blocks and current_tokens + block_tokens > self.max_tokens:
                    chunks.append(self._make_chunk(document_id, position, current_blocks, section.heading_path, metadata))
                    position += 1
                    current_blocks, current_tokens = [], 0
                current_blocks.append(block)
                current_tokens += block_tokens
            if current_blocks:
                chunks.append(self._make_chunk(document_id, position, current_blocks, section.heading_path, metadata))
                position += 1
        return chunks

    def _make_chunk(self, document_id: str, position: int, blocks: list[str], heading_path: list[str], metadata: dict[str, Any] | None) -> Chunk:
        content = "\n\n".join(blocks)
        return Chunk(
            chunk_id=f"{document_id}:md:{position}",
            document_id=document_id,
            content=content,
            position=position,
            token_count=estimate_tokens(content),
            character_count=len(content),
            section=heading_path[-1] if heading_path else None,
            heading_path=heading_path,
            metadata=dict(metadata or {}),
        )
