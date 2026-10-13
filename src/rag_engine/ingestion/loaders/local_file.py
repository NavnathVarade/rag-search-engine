"""Local filesystem loader with resource limits (spec #6, #51, #121)."""
from __future__ import annotations

from pathlib import Path

from rag_engine.exceptions import DocumentTooLargeError, PathTraversalError

_EXCLUDED_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next", "target"}
_EXCLUDED_SUFFIXES = {
    ".pyc", ".so", ".dll", ".exe", ".bin", ".png", ".jpg", ".jpeg", ".gif", ".mp4",
    ".zip", ".tar", ".gz", ".woff", ".woff2", ".ttf", ".eot", ".ico",
}


def should_exclude(path: Path, excluded_dirs: set[str] = _EXCLUDED_DIR_NAMES) -> bool:
    if any(part in excluded_dirs for part in path.parts):
        return True
    return path.suffix.lower() in _EXCLUDED_SUFFIXES


def discover_files(
    root: Path,
    max_files: int = 5000,
    excluded_dirs: set[str] | None = None,
    allowed_suffixes: set[str] | None = None,
) -> list[Path]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"path does not exist: {root}")

    excluded_dirs = excluded_dirs or _EXCLUDED_DIR_NAMES
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not str(resolved).startswith(str(root)):
            raise PathTraversalError(f"discovered file escapes root: {resolved}")
        if should_exclude(path, excluded_dirs):
            continue
        if allowed_suffixes and path.suffix.lower() not in allowed_suffixes:
            continue
        files.append(path)
        if len(files) > max_files:
            raise DocumentTooLargeError(f"repository exceeds max_files_per_repo={max_files}")
    return files


def read_file_with_limit(path: Path, max_size_mb: int = 25) -> bytes:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise DocumentTooLargeError(f"{path} is {size_mb:.1f}MB, exceeding max_document_size_mb={max_size_mb}")
    return path.read_bytes()
