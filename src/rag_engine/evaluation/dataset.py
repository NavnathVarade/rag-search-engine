"""Evaluation dataset schema + JSONL loader/validator (spec #31-32)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rag_engine.exceptions import ConfigurationError


class QueryType(str, Enum):
    factual = "factual"
    conceptual = "conceptual"
    comparison = "comparison"
    troubleshooting = "troubleshooting"
    api_lookup = "api_lookup"
    code_related = "code_related"
    configuration = "configuration"
    version_specific = "version_specific"
    exact_match = "exact_match"
    multi_hop = "multi_hop"
    ambiguous = "ambiguous"
    unsupported = "unsupported"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


@dataclass
class EvaluationQuestion:
    id: str
    question: str
    expected_answer: str
    relevant_documents: list[str]
    relevant_chunks: list[str]
    difficulty: Difficulty
    topic: str
    query_type: QueryType
    source: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationQuestion:
        required = {"id", "question", "expected_answer", "relevant_documents", "relevant_chunks", "difficulty", "topic", "query_type"}
        missing = required - data.keys()
        if missing:
            raise ConfigurationError(f"evaluation record missing fields: {missing}", details={"record_id": data.get("id")})
        return cls(
            id=data["id"],
            question=data["question"],
            expected_answer=data["expected_answer"],
            relevant_documents=data["relevant_documents"],
            relevant_chunks=data["relevant_chunks"],
            difficulty=Difficulty(data["difficulty"]),
            topic=data["topic"],
            query_type=QueryType(data["query_type"]),
            source=data.get("source", ""),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "question": self.question, "expected_answer": self.expected_answer,
            "relevant_documents": self.relevant_documents, "relevant_chunks": self.relevant_chunks,
            "difficulty": self.difficulty.value, "topic": self.topic, "query_type": self.query_type.value,
            "source": self.source,
        }


def load_dataset(path: Path) -> list[EvaluationQuestion]:
    questions = []
    errors = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                questions.append(EvaluationQuestion.from_dict(data))
            except Exception as exc:
                errors.append(f"line {line_no}: {exc}")
    if errors:
        raise ConfigurationError(f"invalid evaluation dataset {path}", details={"errors": errors})
    return questions


def validate_dataset_file(path: Path) -> dict:
    """Returns a report dict instead of raising, for `rag documents audit`
    -style tooling that wants to show all problems at once."""
    ids_seen: set[str] = set()
    duplicates: list[str] = []
    valid_count = 0
    errors: list[str] = []

    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                q = EvaluationQuestion.from_dict(data)
                if q.id in ids_seen:
                    duplicates.append(q.id)
                ids_seen.add(q.id)
                valid_count += 1
            except Exception as exc:
                errors.append(f"line {line_no}: {exc}")

    return {
        "path": str(path),
        "valid_records": valid_count,
        "duplicate_ids": duplicates,
        "errors": errors,
        "is_valid": not errors and not duplicates,
    }
