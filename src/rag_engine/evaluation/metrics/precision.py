"""Re-exports precision_at_k for import symmetry with the spec's module
layout (docs reference `evaluation/metrics/precision.py`)."""
from rag_engine.evaluation.metrics.recall import precision_at_k

__all__ = ["precision_at_k"]
