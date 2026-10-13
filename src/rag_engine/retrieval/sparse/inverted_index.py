"""From-scratch inverted index with posting lists.

Supports incremental add, delete (tombstone-based, compacted on rebuild),
and persistence to a JSON-serializable structure. Used by both TF-IDF and
BM25 scorers, which only need term -> {doc_id: term_frequency} plus
per-document lengths / total document count -- everything both scoring
formulas require.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rag_engine.retrieval.sparse.tokenizer import Tokenizer


@dataclass
class PostingList:
    """term -> doc_id -> term_frequency in that document."""

    postings: dict[str, int] = field(default_factory=dict)

    def add(self, doc_id: str, count: int) -> None:
        self.postings[doc_id] = self.postings.get(doc_id, 0) + count

    def remove(self, doc_id: str) -> None:
        self.postings.pop(doc_id, None)

    @property
    def document_frequency(self) -> int:
        return len(self.postings)


class InvertedIndex:
    """A term -> PostingList inverted index over chunk_id documents.

    Also tracks per-document token length (for BM25 length normalization)
    and total corpus stats, and supports versioning so callers can detect
    a stale index after the tokenizer/config changes.
    """

    def __init__(self, tokenizer: Tokenizer | None = None, version: str = "v1") -> None:
        self.tokenizer = tokenizer or Tokenizer()
        self.version = version
        self._index: dict[str, PostingList] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_ids: set[str] = set()
        self._total_length = 0  # running sum so average_document_length is O(1)

    # -- mutation -----------------------------------------------------
    def add_document(self, doc_id: str, text: str) -> list[str]:
        """Tokenize `text` and add postings for `doc_id`. Idempotent per doc_id
        (re-adding the same doc_id first removes its previous postings)."""
        if doc_id in self._doc_ids:
            self.remove_document(doc_id)

        tokens = self.tokenizer.tokenize(text)
        term_counts: dict[str, int] = {}
        for tok in tokens:
            term_counts[tok] = term_counts.get(tok, 0) + 1

        for term, count in term_counts.items():
            self._index.setdefault(term, PostingList()).add(doc_id, count)

        self._doc_lengths[doc_id] = len(tokens)
        self._total_length += len(tokens)
        self._doc_ids.add(doc_id)
        return tokens

    def remove_document(self, doc_id: str) -> None:
        if doc_id not in self._doc_ids:
            return
        for posting_list in self._index.values():
            posting_list.remove(doc_id)
        self._total_length -= self._doc_lengths.pop(doc_id, 0)
        self._doc_ids.discard(doc_id)

    def compact(self) -> None:
        """Drop terms whose posting list became empty after deletions."""
        empty_terms = [t for t, pl in self._index.items() if pl.document_frequency == 0]
        for t in empty_terms:
            del self._index[t]

    # -- read -----------------------------------------------------------
    def document_frequency(self, term: str) -> int:
        pl = self._index.get(term)
        return pl.document_frequency if pl else 0

    def term_frequency(self, term: str, doc_id: str) -> int:
        pl = self._index.get(term)
        return pl.postings.get(doc_id, 0) if pl else 0

    def doc_length(self, doc_id: str) -> int:
        return self._doc_lengths.get(doc_id, 0)

    def postings_for(self, term: str) -> dict[str, int]:
        """doc_id -> term frequency for `term` (read-only view; empty if unknown)."""
        pl = self._index.get(term)
        return pl.postings if pl else {}

    def matching_documents(self, term: str) -> set[str]:
        pl = self._index.get(term)
        return set(pl.postings.keys()) if pl else set()

    @property
    def document_count(self) -> int:
        return len(self._doc_ids)

    @property
    def average_document_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return self._total_length / len(self._doc_lengths)

    @property
    def vocabulary_size(self) -> int:
        return len(self._index)

    def document_ids(self) -> set[str]:
        return set(self._doc_ids)

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "doc_lengths": self._doc_lengths,
            "postings": {term: pl.postings for term, pl in self._index.items()},
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path, tokenizer: Tokenizer | None = None) -> InvertedIndex:
        data = json.loads(path.read_text())
        idx = cls(tokenizer=tokenizer, version=data.get("version", "v1"))
        idx._doc_lengths = data["doc_lengths"]
        idx._doc_ids = set(data["doc_lengths"].keys())
        idx._index = {term: PostingList(postings=postings) for term, postings in data["postings"].items()}
        return idx
