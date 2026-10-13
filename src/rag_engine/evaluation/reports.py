"""Comparison report generation across experiment runs (spec #37, #98-99,
#129). Reads back ExperimentRecord JSON files and renders a plain-text
comparison table -- deliberately dependency-free (no plotting library
required) so `rag experiment compare` works in any environment; a
notebook (notebooks/evaluation_analysis.ipynb) can load the same JSON
files for charts when matplotlib/pandas are available.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_experiments(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return records


def comparison_table(records: list[dict], metric_keys: list[str]) -> str:
    if not records:
        return "No experiment records found."

    headers = ["experiment_id", "name", *metric_keys]
    rows = []
    for r in records:
        row = [r.get("experiment_id", "?"), r.get("name", "?")]
        for key in metric_keys:
            value = r.get("metrics", {})
            for part in key.split("."):
                value = value.get(part, "N/A") if isinstance(value, dict) else "N/A"
            row.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        rows.append(row)

    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)
