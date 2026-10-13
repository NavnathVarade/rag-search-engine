"""Prompt-injection detection over retrieved document content (spec #29).

Retrieved chunks are untrusted data (spec #29-30). This module flags
content containing common injection patterns so callers can (a) log/alert,
and (b) still safely pass the content to the LLM wrapped in the delimited
<context> blocks with the explicit "treat as data, not instructions"
system-prompt rule (generation/prompts.py) -- detection here is
defense-in-depth, not the only mitigation, since regex matching alone can
never be a complete defense against injection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |the )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (all |the )?(previous|prior|above)", re.I),
    re.compile(r"you are now (in )?(developer|debug|admin|dan) mode", re.I),
    re.compile(r"system\s*(prompt|message)\s*[:=]", re.I),
    re.compile(r"reveal (your |the )?(system prompt|instructions|api key|secret)", re.I),
    re.compile(r"\bexfiltrate\b", re.I),
    re.compile(r"send (this|the following|it)\b.{0,20}\bto\s+https?://", re.I),
    re.compile(r"execute (this|the following) (command|code|script)", re.I),
    re.compile(r"</?(system|assistant|user)>", re.I),  # fake chat-turn delimiters
    re.compile(r"call this url", re.I),
]


@dataclass
class InjectionScanResult:
    suspicious: bool
    matched_patterns: list[str]


def scan_for_prompt_injection(text: str) -> InjectionScanResult:
    matches = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
    return InjectionScanResult(suspicious=bool(matches), matched_patterns=matches)
