#!/usr/bin/env python3
"""Scaffold a new evaluation dataset skeleton (spec #83): emits one JSONL
template record per topic/query-type combination for a human annotator to
fill in with real expected_answer/relevant_chunks. Never invents labels
itself -- see docs/evaluation.md 'Building the full benchmark' for the
annotation workflow.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_engine.evaluation.dataset import Difficulty, QueryType  # noqa: E402

TOPICS = ["python", "fastapi", "kubernetes", "postgresql", "aws"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/evaluation/questions_template.jsonl")
    parser.add_argument("--per-topic", type=int, default=2)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    counter = 1
    for topic in TOPICS:
        for query_type in list(QueryType)[: args.per_topic]:
            records.append(
                {
                    "id": f"q{counter:03d}",
                    "question": f"TODO: write a real {query_type.value} question about {topic}",
                    "expected_answer": "TODO: fill in from actual indexed documentation",
                    "relevant_documents": [],
                    "relevant_chunks": [],
                    "difficulty": Difficulty.medium.value,
                    "topic": topic,
                    "query_type": query_type.value,
                    "source": "manual-annotation-pending",
                }
            )
            counter += 1

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} template records to {out_path} -- fill in TODOs before use.")


if __name__ == "__main__":
    main()
