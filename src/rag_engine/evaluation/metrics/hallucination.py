"""Hallucination rate / unsupported-claim rate (spec #34, #138).

Deterministic proxy: 1 - lexical_faithfulness_proxy (the fraction of
answer content NOT lexically grounded in the supplied context). Same
"lower bound, not ground truth" caveat as faithfulness.py applies.
"""
from __future__ import annotations

from rag_engine.evaluation.metrics.faithfulness import lexical_faithfulness_proxy


def unsupported_claim_rate_proxy(answer: str, context_texts: list[str], n: int = 3) -> float:
    return 1.0 - lexical_faithfulness_proxy(answer, context_texts, n=n)
