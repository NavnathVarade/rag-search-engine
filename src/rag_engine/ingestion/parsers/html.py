"""HTML parsing via BeautifulSoup (mature library, per spec's allowed
list) with boilerplate stripping (nav/header/footer/script/style) --
spec #8 "handle HTML boilerplate, navigation menus" while spec #8
"preserve code blocks, tables" is honored by converting <pre>/<code> and
<table> into fenced/plain-text-preserving forms before stripping tags,
instead of naively concatenating all visible text.
"""
from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup

from rag_engine.exceptions import ParsingError
from rag_engine.ingestion.parsers.base import ParsedDocument

_BOILERPLATE_TAGS = ("nav", "header", "footer", "script", "style", "noscript", "aside", "form")


def _table_to_text(table: Any) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


class HtmlParser:
    def parse(self, content: bytes, source_hint: str = "") -> ParsedDocument:
        if not content:
            raise ParsingError("empty document", details={"source": source_hint})
        try:
            soup = BeautifulSoup(content, "html.parser")
        except Exception as exc:
            raise ParsingError(f"failed to parse HTML: {exc}", details={"source": source_hint}) from exc

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        for tag_name in _BOILERPLATE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        for pre in soup.find_all("pre"):
            code_text = pre.get_text()
            pre.replace_with(f"\n```\n{code_text}\n```\n")

        for table in soup.find_all("table"):
            table.replace_with(_table_to_text(table))

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        deduped_lines = []
        prev = None
        for line in lines:
            if (line or prev) and not (line == "" and prev == ""):
                deduped_lines.append(line)
            prev = line
        cleaned = "\n".join(deduped_lines).strip()

        return ParsedDocument(text=cleaned, title=title, metadata={"parser": "html"})
