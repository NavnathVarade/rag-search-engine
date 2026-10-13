"""Text cleaning pipeline (spec #8): fixes whitespace and unicode issues
while explicitly preserving technical content (code blocks, identifiers,
version numbers) -- it normalizes structure, it does not strip meaning."""
from __future__ import annotations

import re
import unicodedata

_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_WHITESPACE_RE = re.compile(r"[ \t]+\n")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def clean_text(text: str, *, preserve_code_blocks: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)

    if preserve_code_blocks:
        # Protect fenced code blocks from whitespace collapsing, since
        # indentation inside code is semantically meaningful.
        code_blocks: list[str] = []

        def _stash(match: re.Match) -> str:
            code_blocks.append(match.group(0))
            return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

        text = re.sub(r"```.*?```", _stash, text, flags=re.DOTALL)

    text = _TRAILING_WHITESPACE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    text = text.strip()

    if preserve_code_blocks:
        for i, block in enumerate(code_blocks):
            text = text.replace(f"\x00CODEBLOCK{i}\x00", block)

    return text
