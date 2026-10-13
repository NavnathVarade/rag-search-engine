"""Rewrite data/evaluation/sample.jsonl so relevant_chunks match the current CLI index.

Run right after `rag ingest ./data/sample`. Maps each question's topic to the
ingested file whose name matches the topic.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "indexes" / "cli_state.json"
DATASET = ROOT / "data" / "evaluation" / "sample.jsonl"


def main() -> None:
    chunks = json.loads(STATE.read_text())
    by_topic: dict[str, list[dict]] = {}
    for c in chunks:
        stem = Path(str(c.get("metadata", {}).get("source", ""))).stem
        by_topic.setdefault(stem, []).append(c)

    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    for row in rows:
        matched = by_topic.get(row.get("topic", ""), [])
        row["relevant_documents"] = sorted({c["document_id"] for c in matched})
        row["relevant_chunks"] = [c["chunk_id"] for c in matched]
    DATASET.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"Rewrote {len(rows)} questions; topics indexed: {sorted(by_topic)}")


if __name__ == "__main__":
    main()
