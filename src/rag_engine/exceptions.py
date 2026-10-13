"""Structured exception taxonomy for the RAG engine.

Every stage of the pipeline raises a specific subclass of RagEngineError so
that failures can be classified, logged, and surfaced to API clients
without leaking internals (see error taxonomy, docs/troubleshooting.md).
"""
from __future__ import annotations


class RagEngineError(Exception):
    """Base class for all engine errors. Carries a machine-readable code."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "details": self.details}


# --- Ingestion / parsing -----------------------------------------------
class ParsingError(RagEngineError):
    code = "parsing_failed"
    http_status = 422


class UnsupportedDocumentTypeError(RagEngineError):
    code = "unsupported_document_type"
    http_status = 422


class DocumentTooLargeError(RagEngineError):
    code = "document_too_large"
    http_status = 413


class CleaningError(RagEngineError):
    code = "cleaning_failed"
    http_status = 422


# --- Security -------------------------------------------------------
class SSRFBlockedError(RagEngineError):
    code = "ssrf_blocked"
    http_status = 400


class PathTraversalError(RagEngineError):
    code = "path_traversal_blocked"
    http_status = 400


class PromptInjectionDetectedError(RagEngineError):
    code = "prompt_injection_detected"
    http_status = 400


class AuthorizationError(RagEngineError):
    code = "forbidden"
    http_status = 403


class AuthenticationError(RagEngineError):
    code = "unauthorized"
    http_status = 401


class RateLimitExceededError(RagEngineError):
    code = "rate_limited"
    http_status = 429


# --- Chunking / embedding --------------------------------------------
class ChunkingError(RagEngineError):
    code = "chunking_failed"
    http_status = 422


class EmbeddingError(RagEngineError):
    code = "embedding_failed"
    http_status = 502


class EmbeddingDimensionMismatchError(RagEngineError):
    code = "embedding_dimension_mismatch"
    http_status = 409


# --- Retrieval ---------------------------------------------------------
class RetrievalError(RagEngineError):
    code = "retrieval_failed"
    http_status = 500


class PersistenceError(RagEngineError):
    """The durable store could not be read or written. Nothing was changed in memory."""

    code = "persistence_failed"
    http_status = 503


class IndexNotBuiltError(RagEngineError):
    code = "index_not_built"
    http_status = 409


class IndexCorruptError(RagEngineError):
    code = "index_corrupt"
    http_status = 500


# --- Reranking / generation --------------------------------------------
class RerankingError(RagEngineError):
    code = "reranking_failed"
    http_status = 502


class GenerationError(RagEngineError):
    code = "generation_failed"
    http_status = 502


class GenerationTimeoutError(GenerationError):
    code = "generation_timeout"
    http_status = 504


class CitationValidationError(RagEngineError):
    code = "invalid_citation"
    http_status = 500


# --- Configuration / storage --------------------------------------------
class ConfigurationError(RagEngineError):
    code = "configuration_error"
    http_status = 500


class RepositoryError(RagEngineError):
    code = "storage_error"
    http_status = 500


class InvalidRequestError(RagEngineError):
    code = "invalid_request"
    http_status = 422


class NotFoundError(RagEngineError):
    code = "not_found"
    http_status = 404
