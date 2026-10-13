"""Experiment tracking for reproducibility (spec #36-37, #81, #128).

Every experiment run is serialized with its exact configuration, dataset
path + hash, git commit (if available), python version, and timestamp, so
results can be traced back to exactly what produced them and reproduced
later. Results are written under data/evaluation/results/<experiment_id>.json.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@dataclass
class ExperimentRecord:
    experiment_id: str
    name: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    dataset_path: str | None
    dataset_hash: str | None
    git_commit: str | None
    python_version: str
    platform: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_experiment(
    name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    dataset_path: Path | None = None,
    results_dir: Path = Path("data/evaluation/results"),
) -> ExperimentRecord:
    record = ExperimentRecord(
        experiment_id=f"{name}-{uuid.uuid4().hex[:8]}",
        name=name,
        config=config,
        metrics=metrics,
        dataset_path=str(dataset_path) if dataset_path else None,
        dataset_hash=_file_hash(dataset_path) if dataset_path else None,
        git_commit=_git_commit(),
        python_version=sys.version,
        platform=platform.platform(),
        timestamp=datetime.now(UTC).isoformat(),
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{record.experiment_id}.json"
    out_path.write_text(json.dumps(record.to_dict(), indent=2))
    return record
