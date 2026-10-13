"""Input sanitization helpers (spec #29, #51)."""
from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from urllib.parse import urlparse

from rag_engine.exceptions import PathTraversalError, SSRFBlockedError

_BLOCKED_NETWORKS = [
    ipaddress.ip_network(n)
    for n in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
        "100.64.0.0/10", "0.0.0.0/8", "::1/128", "fc00::/7", "fe80::/10", "::/128",
    )
]

Resolver = Callable[[str, int], list[str]]


def system_resolver(host: str, port: int) -> list[str]:
    """Resolve `host` with the OS resolver. This also normalizes tricks such as decimal
    (`2130706433`), hex (`0x7f.1`) and octal IPv4 literals that a naive string check
    would treat as an ordinary hostname but the OS would connect to as 127.0.0.1."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SSRFBlockedError(f"could not resolve host '{host}': {exc}") from exc
    return sorted({str(info[4][0]) for info in infos})


def is_blocked_ip(ip_text: str) -> bool:
    ip = ipaddress.ip_address(ip_text.split("%")[0])  # strip IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 must be judged as 127.0.0.1
    return (not ip.is_global) or any(ip in net for net in _BLOCKED_NETWORKS)


def resolve_public_addresses(
    url: str,
    allowed_schemes: tuple[str, ...] = ("https", "http"),
    resolver: Resolver | None = None,
    allow_private: bool = False,
) -> list[str]:
    """Validate `url` and return the IP addresses it resolves to.

    Rejects disallowed schemes, missing hosts, and (unless `allow_private`) any host
    where ANY resolved address is loopback/private/link-local/CGNAT/reserved/multicast
    -- checking every address defeats "one public + one private A record" tricks.
    Callers must connect to one of the returned addresses (see UrlLoader) rather than
    resolving again, which closes the DNS-rebinding window."""
    parsed = urlparse(url)
    if parsed.scheme not in allowed_schemes:
        raise SSRFBlockedError(f"URL scheme '{parsed.scheme}' is not allowed")
    host = parsed.hostname
    if not host:
        raise SSRFBlockedError("URL has no hostname")
    if parsed.username or parsed.password:
        raise SSRFBlockedError("URLs with embedded credentials are not allowed")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise SSRFBlockedError(f"invalid port in URL: {exc}") from exc

    addresses = (resolver or system_resolver)(host, port)
    if not addresses:
        raise SSRFBlockedError(f"host '{host}' did not resolve to any address")
    if not allow_private:
        for addr in addresses:
            if is_blocked_ip(addr):
                raise SSRFBlockedError(f"URL resolves to a blocked private/internal address: {host} -> {addr}")
    return addresses


def validate_public_url(
    url: str,
    allowed_schemes: tuple[str, ...] = ("https", "http"),
    resolver: Resolver | None = None,
    allow_private: bool = False,
) -> str:
    """SSRF defense (spec #51): returns `url` if it is safe to fetch, else raises
    SSRFBlockedError. See resolve_public_addresses for the rules."""
    resolve_public_addresses(url, allowed_schemes, resolver, allow_private)
    return url


_SAFE_PATH_RE = re.compile(r"^[\w\-./]+$")


def validate_safe_relative_path(base_dir: str, relative_path: str) -> str:
    """Prevent path traversal (spec #51) when resolving a user-supplied
    relative path (e.g. a filename inside an uploaded archive) against a
    base directory."""
    import os

    if ".." in relative_path.split("/") or relative_path.startswith("/"):
        raise PathTraversalError(f"unsafe path: {relative_path}")

    resolved_base = os.path.realpath(base_dir)
    resolved_target = os.path.realpath(os.path.join(base_dir, relative_path))
    if not resolved_target.startswith(resolved_base + os.sep) and resolved_target != resolved_base:
        raise PathTraversalError(f"path escapes base directory: {relative_path}")
    return resolved_target
