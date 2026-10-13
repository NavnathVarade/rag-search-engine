"""Ingestion pipeline orchestrator (spec #6): parse -> clean -> extract
metadata -> deduplicate -> chunk -> (caller embeds + indexes). This module
is storage-agnostic by design (spec #117 repository pattern) -- it returns
an IngestionResult; persistence happens in storage/repositories, keeping
the pipeline itself unit-testable without a database.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag_engine.chunking.base import Chunk, Chunker
from rag_engine.exceptions import RagEngineError
from rag_engine.ingestion.cleaners.deduplication import DuplicateTracker, content_hash
from rag_engine.ingestion.cleaners.text_cleaner import clean_text
from rag_engine.ingestion.metadata import DocumentMetadata, new_document_id, now_iso
from rag_engine.ingestion.parsers.base import DocumentParser


@dataclass
class IngestionResult:
    document_id: str
    metadata: DocumentMetadata
    chunks: list[Chunk]
    was_duplicate: bool
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class IngestionError:
    source_uri: str
    stage: str
    error: str


class IngestionPipeline:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        chunking_strategy_name: str,
        embedding_model: str,
        embedding_dimension: int,
        duplicate_tracker: DuplicateTracker | None = None,
        parser_version: str = "v1",
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.chunking_strategy_name = chunking_strategy_name
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.duplicate_tracker = duplicate_tracker or DuplicateTracker()
        self.parser_version = parser_version

    def ingest(
        self,
        raw_content: bytes,
        source_type: str,
        source_uri: str,
        extra_metadata: dict | None = None,
    ) -> IngestionResult:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        parsed = self.parser.parse(raw_content, source_hint=source_uri)
        timings["parsing_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cleaned_text = clean_text(parsed.text)
        timings["cleaning_ms"] = (time.perf_counter() - t0) * 1000

        if not cleaned_text.strip():
            raise RagEngineError("document has no extractable content after cleaning", details={"source": source_uri})

        hash_value = content_hash(cleaned_text)
        existing_doc_id = self.duplicate_tracker.is_duplicate(cleaned_text)
        was_duplicate = existing_doc_id is not None
        document_id = existing_doc_id or new_document_id()

        metadata = DocumentMetadata(
            document_id=document_id,
            source_type=source_type,
            source_uri=source_uri,
            title=parsed.title,
            content_hash=hash_value,
            version_hash=hash_value,
            parser_version=self.parser_version,
            chunking_strategy=self.chunking_strategy_name,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            ingestion_timestamp=now_iso(),
            language=parsed.language,
            metadata={**parsed.metadata, **(extra_metadata or {})},
        )

        chunks: list[Chunk] = []
        if not was_duplicate:
            t0 = time.perf_counter()
            chunks = self.chunker.chunk(document_id, cleaned_text, metadata={"source": source_uri, "title": parsed.title})
            timings["chunking_ms"] = (time.perf_counter() - t0) * 1000
            self.duplicate_tracker.register(cleaned_text, document_id)

        return IngestionResult(
            document_id=document_id, metadata=metadata, chunks=chunks, was_duplicate=was_duplicate,
            stage_timings_ms=timings,
        )
