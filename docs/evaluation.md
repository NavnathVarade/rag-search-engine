# Evaluation

## Retrieval metrics (all implemented from scratch, `evaluation/metrics/`)
- **Recall@K**: fraction of the labeled relevant set found in the top-K.
  Limitation: needs complete relevance labels to be meaningful; with a
  partial label set it underestimates true recall.
- **Precision@K**: fraction of the top-K that are actually relevant.
  Limitation: penalizes retrieving more candidates than strictly needed,
  even when downstream reranking will filter them anyway.
- **MRR**: 1/rank of the first relevant result, averaged across queries.
  Limitation: ignores everything after the first hit -- two systems where
  one finds 1 relevant doc at rank 1 and the other finds 5 relevant docs
  at rank 1 score identically.
- **NDCG@K**: rewards relevant documents more when ranked higher, with
  logarithmic discount; supports graded (not just binary) relevance.
  Most information-rich of these metrics but also the least intuitive to
  read directly.
- **Hit Rate@K**: 1.0 if any relevant doc appears in top-K, else 0.0. Use
  for "did we find anything useful at all" style monitoring.

## Generation metrics -- deterministic vs LLM-as-judge (spec #34, explicit
   separation, not blended)
- **Deterministic**: `citation_correctness`, `citation_completeness`
  (exact -- derived directly from citation-marker validation against the
  supplied context, no ambiguity) and the lexical proxies
  (`lexical_faithfulness_proxy`, `lexical_relevance_proxy`,
  `unsupported_claim_rate_proxy`) which measure n-gram overlap between the
  answer and the cited context. These are reproducible and free but
  **under-count legitimate paraphrase** -- treat them as a fast lower-bound
  signal, not ground truth.
- **LLM-as-judge**: `llm_judge_faithfulness`, `llm_judge_relevance` prompt
  a configured LLM to score 0-1. These are the more semantically aware
  option but are **non-deterministic** (depend on judge model, sampling
  temperature, and prompt phrasing) and carry **judge-model bias** (a
  judge tends to rate answers written in a similar style to its own
  favorably). Never presented as objective truth in this codebase or its
  reports -- always labeled as an LLM judgment.

## Dataset format
JSONL, one `EvaluationQuestion` per line (`evaluation/dataset.py`):
```json
{"id": "q001", "question": "...", "expected_answer": "...",
 "relevant_documents": [...], "relevant_chunks": [...],
 "difficulty": "medium", "topic": "kubernetes", "query_type": "conceptual"}
```
`query_type` ∈ factual/conceptual/comparison/troubleshooting/api_lookup/
code_related/configuration/version_specific/exact_match/multi_hop/
ambiguous/unsupported. A working 3-question sample lives at
`data/evaluation/sample.jsonl` with real (not fabricated) chunk-id
labels matching `data/sample/*.md`; `scripts/generate_dataset.py` scaffolds
TODO-filled templates for building the full 200-500 question benchmark --
it never invents question text or labels, since that would defeat the
purpose of a benchmark.

## Running an evaluation
```bash
uv run rag evaluate --dataset data/evaluation/sample.jsonl --mode hybrid
```
Every run is recorded via `evaluation/experiment.py` with its exact
config, dataset hash, git commit, and timestamp under
`data/evaluation/results/<experiment_id>.json`, so a metric reported
today can be traced back to exactly what produced it. Compare runs with:
```python
from rag_engine.evaluation.reports import load_experiments, comparison_table
records = load_experiments(Path("data/evaluation/results"))
print(comparison_table(records, ["recall_at_k.10", "mrr"]))
```

## Honesty about this project's evaluation numbers
The only evaluation numbers this repository ships are (a) unit-test
hand-calculated examples (exact, by construction) and (b) a 3-question
sample run against the 3-document sample corpus (trivially easy -- every
metric is 1.0). Neither constitutes a claim about real-world retrieval
quality on a real 500-question technical-docs benchmark. Building that
benchmark and running the full ablation suite (spec #37, #99) is real,
non-trivial annotation work that this repository provides the tooling
for but has not itself performed -- see docs/final-report.md
"Limitations" and "Recommended next steps".
