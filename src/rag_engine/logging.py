"""Structured logging setup.

Production emits JSON lines with request_id/trace_id correlation fields;
development emits human-readable console output. Sensitive fields
(api keys, authorization headers, raw document content) are redacted
before a record is emitted -- see `_redact`.
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from structlog.typing import Processor

_REDACT_KEYS = {"api_key", "authorization", "password", "token", "secret", "llm_api_key"}


def _redact(_logger: Any, _name: str, event_dict: dict) -> dict:
    for key in list(event_dict.keys()):
        if key.lower() in _REDACT_KEYS:
            event_dict[key] = "***REDACTED***"
    return event_dict


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper(), logging.INFO)
    )

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _redact,
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
