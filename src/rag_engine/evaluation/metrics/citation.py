"""Citation correctness/completeness -- deterministic (spec #34), unlike
faithfulness/relevance which typically need an LLM judge. These are
derived directly and exactly from CitationValidationResult, so they carry
no LLM-judge uncertainty."""
from __future__ import annotations

from rag_engine.generation.citations import CitationValidationResult


def citation_correctness(validation: CitationValidationResult) -> float:
    """Fraction of citation markers in the answer that point to a chunk
    that was actually supplied as context. 1.0 if the answer cited nothing
    (vacuously correct -- see citation_completeness for whether it SHOULD
    have cited something)."""
    total = len(validation.citations)
    if total == 0:
        return 1.0
    valid = sum(1 for c in validation.citations if c.valid)
    return valid / total


def citation_completeness(validation: CitationValidationResult, expected_min_citations: int = 1) -> float:
    """1.0 if the answer cited at least `expected_min_citations` valid
    sources, else 0.0. Use this to catch answers that make claims without
    citing any evidence at all."""
    valid = sum(1 for c in validation.citations if c.valid)
    return 1.0 if valid >= expected_min_citations else 0.0
