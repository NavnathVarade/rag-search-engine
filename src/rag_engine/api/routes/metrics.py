"""Prometheus scrape endpoint. Mounted only when METRICS_ENABLED=true. If METRICS_TOKEN is set,
the scraper must send `Authorization: Bearer <token>` (constant-time comparison); otherwise
restrict /metrics at the network layer, since series names reveal usage patterns."""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST

from rag_engine.exceptions import AuthenticationError

router = APIRouter(tags=["observability"])


@router.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    token: str = request.app.state.settings.metrics_token
    if token:
        scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.encode(), token.encode()):
            raise AuthenticationError("invalid or missing credentials")
    return Response(content=request.app.state.metrics.render(), media_type=CONTENT_TYPE_LATEST)
