"""Citation extraction + validation (spec #28, #140-141).

The generated answer is expected to contain bracketed markers like [1],
[2] referencing the 1-based position of a context chunk as it was given
to the LLM (see prompts.build_context_block). This module:
  * parses those markers out of the answer text
  * maps each marker back to the actual chunk_id/document/source metadata
    that was supplied
  * flags any citation marker that does NOT correspond to a chunk that was
    actually in the supplied context (a hallucinated citation) so the
    caller can reject/repair the answer instead of returning a citation
    pointing at nothing (spec #140: "never generate citations to
    nonexistent chunks")
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag_engine.generation.prompts import PromptContext

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    marker: int
    chunk_id: str
    document_id: str | None
    source_label: str
    valid: bool


@dataclass
class CitationValidationResult:
    citations: list[Citation]
    invalid_markers: list[int]
    has_any_citation: bool

    @property
    def all_valid(self) -> bool:
        return not self.invalid_markers


def extract_and_validate_citations(
    answer_text: str, supplied_contexts: list[PromptContext], document_ids: dict[str, str] | None = None
) -> CitationValidationResult:
    """`document_ids` optionally maps chunk_id -> document_id for richer
    citation metadata; omitted values default to None."""
    document_ids = document_ids or {}
    markers = sorted({int(m) for m in _CITATION_RE.findall(answer_text)})

    citations: list[Citation] = []
    invalid: list[int] = []
    for marker in markers:
        if 1 <= marker <= len(supplied_contexts):
            ctx = supplied_contexts[marker - 1]
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=ctx.chunk_id,
                    document_id=document_ids.get(ctx.chunk_id),
                    source_label=ctx.source_label,
                    valid=True,
                )
            )
        else:
            invalid.append(marker)
            citations.append(Citation(marker=marker, chunk_id="", document_id=None, source_label="", valid=False))

    return CitationValidationResult(citations=citations, invalid_markers=invalid, has_any_citation=bool(markers))


def strip_invalid_citations(answer_text: str, invalid_markers: list[int]) -> str:
    """Repair an answer by removing citation markers that don't map to any
    supplied chunk, rather than showing the user a broken/misleading
    citation (spec #140-141)."""
    result = answer_text
    for marker in invalid_markers:
        result = re.sub(rf"\[{marker}\]", "", result)
    return result
