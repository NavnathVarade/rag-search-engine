"""HTTP(S) URL loader with SSRF protection (spec #6, #51).

Every request -- including each redirect hop -- is validated BEFORE any network
call: the hostname is resolved, every resolved address must be public, and the
connection is then made to that exact validated IP (with the original hostname
kept for the Host header and TLS SNI/certificate verification). Connecting to the
validated address instead of resolving again closes the DNS-rebinding window.
Redirects are followed manually (max `max_redirects`), the response size is
capped, and the timeout is bounded.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from rag_engine.exceptions import DocumentTooLargeError, RagEngineError
from rag_engine.security.sanitization import Resolver, resolve_public_addresses

_REDIRECT_CODES = (301, 302, 303, 307, 308)


class TooManyRedirectsError(RagEngineError):
    pass


@dataclass(frozen=True)
class FetchedDocument:
    content: bytes
    content_type: str  # lower-cased media type without parameters, e.g. "text/html"
    final_url: str


class UrlLoader:
    def __init__(
        self,
        max_size_mb: int = 25,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
        max_redirects: int = 5,
        allow_private_hosts: bool = False,
        resolver: Resolver | None = None,
        pin_resolved_ip: bool = True,
    ) -> None:
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.timeout_s = timeout_s
        self.max_redirects = max_redirects
        self.allow_private_hosts = allow_private_hosts
        self.resolver = resolver
        self.pin_resolved_ip = pin_resolved_ip
        self._client = client or httpx.Client(follow_redirects=False)  # redirects handled (and re-validated) manually

    def _pinned_request(self, url: str) -> tuple[str, dict[str, str], dict[str, str]]:
        """Validate `url`; return (url_to_request, extra_headers, extensions)."""
        addresses = resolve_public_addresses(url, resolver=self.resolver, allow_private=self.allow_private_hosts)
        if not self.pin_resolved_ip:
            return url, {}, {}
        parsed = urlparse(url)
        ip = addresses[0]
        host_for_url = f"[{ip}]" if ":" in ip else ip
        netloc = host_for_url + (f":{parsed.port}" if parsed.port else "")
        host_header = parsed.hostname or ""
        if parsed.port:
            host_header += f":{parsed.port}"
        ext = {"sni_hostname": parsed.hostname or ""} if parsed.scheme == "https" else {}
        return urlunparse(parsed._replace(netloc=netloc)), {"Host": host_header}, ext

    def fetch_document(self, url: str) -> FetchedDocument:
        current = url
        for _ in range(self.max_redirects + 1):
            request_url, headers, extensions = self._pinned_request(current)
            with self._client.stream("GET", request_url, headers=headers, extensions=extensions, timeout=self.timeout_s) as resp:
                if resp.status_code in _REDIRECT_CODES:
                    location = resp.headers.get("location")
                    if not location:
                        raise httpx.HTTPStatusError("redirect without Location header", request=resp.request, response=resp)
                    current = urljoin(current, location)  # re-validated at the top of the next loop iteration
                    continue
                resp.raise_for_status()
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > self.max_size_bytes:
                    raise DocumentTooLargeError(f"response from {current} exceeds size limit")
                content = bytearray()
                for chunk in resp.iter_bytes():
                    content.extend(chunk)
                    if len(content) > self.max_size_bytes:
                        raise DocumentTooLargeError(f"response from {current} exceeds size limit")
                media_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                return FetchedDocument(content=bytes(content), content_type=media_type, final_url=current)
        raise TooManyRedirectsError(f"more than {self.max_redirects} redirects fetching {url}")

    def fetch(self, url: str) -> bytes:
        return self.fetch_document(url).content
