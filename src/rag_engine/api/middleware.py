"""Request correlation and timing middleware (spec #44, #89). Rate limiting lives in
api/ratelimit.py and is applied after authentication, not here."""
from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger("api.middleware")


def _route_template(request: Request) -> str:
    """Matched route template (bounded cardinality); raw paths would let a client mint unlimited label values."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        metrics = getattr(request.app.state, "metrics", None)
        try:
            response: Response = await call_next(request)
        except Exception:
            if metrics is not None:
                metrics.http_requests.labels(request.method, _route_template(request), "500").inc()
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        if metrics is not None:
            route = _route_template(request)
            metrics.http_requests.labels(request.method, route, str(response.status_code)).inc()
            metrics.http_duration.labels(request.method, route).observe(duration_ms / 1000)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_completed",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response
