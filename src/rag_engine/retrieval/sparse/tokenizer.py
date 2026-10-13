"""From-scratch tokenizer + normalization for sparse retrieval.

Deliberately simple and dependency-free (no spaCy/NLTK model downloads
required) so BM25/TF-IDF are fully offline and deterministic. Handles:
  * lowercasing
  * unicode NFKC normalization
  * splitting on non-alphanumeric boundaries while keeping identifiers
    like `create_index_concurrently`, `asyncio.gather`, `k8s-v1.29` intact
    as much as reasonably possible (technical-doc friendly)
  * configurable stopword removal
  * optional Porter-style suffix stemming (lightweight, no dependency)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rag_engine.retrieval.sparse.stemmer import porter_stem

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./#+-]*[a-z0-9]|[a-z0-9]")

_DEFAULT_STOPWORDS = frozenset(
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
        "at", "for", "with", "as", "by", "from", "this", "that", "these", "those", "it", "its", "it's",
        "and", "or", "but", "if", "then", "else", "so", "not", "no", "nor", "do", "does", "did", "doing",
        "have", "has", "had", "having", "i", "you", "he", "she", "we", "they", "them", "his", "her",
        "their", "our", "your", "my", "me", "him", "us", "what", "which", "who", "whom", "whose",
        "when", "where", "why", "how", "can", "could", "will", "would", "shall", "should", "may",
        "might", "must",
    ]
)

# Very small, conservative suffix-stripping stemmer (not a full Porter
# implementation) -- reduces plural/verb-form mismatches for technical
# vocabulary without over-aggressively mangling identifiers like "postgres".
_SUFFIXES = ("ing", "edly", "edness", "ed", "ies", "es", "s")


def _light_stem(token: str) -> str:
    if len(token) <= 4:
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            if suf == "ies":
                return token[: -3] + "y"
            return token[: -len(suf)]
    return token


@dataclass(frozen=True)
class TokenizerConfig:
    lowercase: bool = True
    remove_stopwords: bool = True
    stopwords: frozenset[str] = field(default_factory=lambda: _DEFAULT_STOPWORDS)
    stem: bool = False
    stemmer: str = "light"  # "light" (conservative suffix stripper) or "porter" (Porter 1980)
    min_token_length: int = 1


class Tokenizer:
    """Deterministic tokenizer shared by TF-IDF, BM25, and the inverted index.

    Two documents/queries tokenized with the same TokenizerConfig always
    produce the same token stream -- this determinism is required for BM25
    scores to be reproducible and is asserted by unit tests.
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        normalized = unicodedata.normalize("NFKC", text)
        if self.config.lowercase:
            normalized = normalized.lower()

        raw_tokens = _TOKEN_RE.findall(normalized)

        tokens: list[str] = []
        for tok in raw_tokens:
            tok = tok.strip("_./#+-")
            if len(tok) < self.config.min_token_length:
                continue
            if self.config.remove_stopwords and tok in self.config.stopwords:
                continue
            if self.config.stem:
                tok = porter_stem(tok) if self.config.stemmer == "porter" else _light_stem(tok)
            if tok:
                tokens.append(tok)
        return tokens
