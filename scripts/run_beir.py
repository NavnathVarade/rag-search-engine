"""Evaluate the retrieval engine on a BEIR-format dataset (e.g. SciFact, NFCorpus).

Usage:
    python scripts/run_beir.py --dataset /path/to/scifact [--split test] [--out results.json]

The dataset directory must contain corpus.jsonl, queries.jsonl and qrels/<split>.tsv
(the standard BEIR layout). Every document is indexed as ONE chunk (title + text), so
chunk ids equal BEIR document ids and the qrels apply directly.

All retrieval components are the project's own implementations (BM25, hashing or
sentence-transformer embeddings, brute-force/HNSW ANN, RRF) -- nothing is mocked.
Reported numbers are measured, never estimated.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_engine.config import RetrievalMode  # noqa: E402
from rag_engine.embeddings.local import HashingEmbedder  # noqa: E402
from rag_engine.evaluation.metrics.mrr import reciprocal_rank  # noqa: E402
from rag_engine.evaluation.metrics.ndcg import graded_ndcg_at_k  # noqa: E402
from rag_engine.evaluation.metrics.recall import recall_at_k  # noqa: E402
from rag_engine.retrieval.ann.brute_force import BruteForceIndex  # noqa: E402
from rag_engine.retrieval.ann.hnsw import HNSWIndex  # noqa: E402
from rag_engine.retrieval.dense import ChunkRecord, DenseRetriever  # noqa: E402
from rag_engine.retrieval.hybrid import HybridRetriever  # noqa: E402
from rag_engine.retrieval.sparse.bm25 import Bm25Scorer  # noqa: E402
from rag_engine.retrieval.sparse.inverted_index import InvertedIndex  # noqa: E402
from rag_engine.retrieval.sparse.tokenizer import Tokenizer, TokenizerConfig  # noqa: E402
from rag_engine.retrieval.sparse_retriever import Bm25Retriever  # noqa: E402


def load_beir(root: Path, split: str) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    corpus: dict[str, str] = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            corpus[str(row["_id"])] = f"{row.get('title', '')}. {row.get('text', '')}".strip(". ")
    queries: dict[str, str] = {}
    with (root / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    qrels: dict[str, dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    queries = {q: t for q, t in queries.items() if q in qrels}
    return corpus, queries, qrels


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    ndcg10, rec100, rec10, mrr10 = [], [], [], []
    for qid, ranked in run.items():
        rel = qrels[qid]
        rel_set = set(rel)
        ndcg10.append(graded_ndcg_at_k(ranked, {k: float(v) for k, v in rel.items()}, 10))
        rec10.append(recall_at_k(ranked, rel_set, 10))
        rec100.append(recall_at_k(ranked, rel_set, 100))
        mrr10.append(reciprocal_rank(ranked[:10], rel_set))
    return {
        "ndcg@10": round(float(np.mean(ndcg10)), 4),
        "recall@10": round(float(np.mean(rec10)), 4),
        "recall@100": round(float(np.mean(rec100)), 4),
        "mrr@10": round(float(np.mean(mrr10)), 4),
    }


def build_embedder(name: str, dimension: int):
    if name == "hashing":
        return HashingEmbedder(dimension=dimension)
    from rag_engine.embeddings.local import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name=name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--embedder", default="hashing", help="'hashing' or a sentence-transformers model name")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--hnsw-m", type=int, default=16)
    ap.add_argument("--hnsw-ef-construction", type=int, default=100)
    ap.add_argument("--hnsw-ef-search", type=int, default=64)
    ap.add_argument("--skip-hnsw", action="store_true")
    ap.add_argument("--stem", choices=["none", "light", "porter"], default="none", help="BM25 tokenizer stemming")
    ap.add_argument("--bm25-only", action="store_true", help="evaluate only BM25 (fast; skips embeddings/HNSW)")
    ap.add_argument("--tune-split", default=None, help="e.g. 'train': choose the hybrid dense weight on this split")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    corpus, queries, qrels = load_beir(args.dataset, args.split)
    print(f"dataset={args.dataset.name} docs={len(corpus)} queries={len(queries)} split={args.split}")

    records = {did: ChunkRecord(chunk_id=did, document_id=did, content=text, metadata={}) for did, text in corpus.items()}
    ids = list(records)

    t0 = time.perf_counter()
    cfg = TokenizerConfig(stem=args.stem != "none", stemmer=args.stem if args.stem != "none" else "light")
    inv = InvertedIndex(tokenizer=Tokenizer(cfg))
    for did in ids:
        inv.add_document(did, records[did].content)
    bm25_build_s = time.perf_counter() - t0

    if args.bm25_only:
        bm25_only = Bm25Retriever(Bm25Scorer(inv), records)
        run = {q: [r.chunk_id for r in bm25_only.retrieve(t, top_k=100)] for q, t in queries.items()}
        print(f"bm25 stem={args.stem} {evaluate_run(run, qrels)}")
        return

    embedder = build_embedder(args.embedder, args.dim)
    t0 = time.perf_counter()
    vectors = embedder.embed_documents([records[d].content for d in ids])
    embed_s = time.perf_counter() - t0
    dim = embedder.dimension()

    brute = BruteForceIndex(dimension=dim)
    brute.add(ids, vectors)

    hnsw = None
    hnsw_build_s = None
    if not args.skip_hnsw:
        hnsw = HNSWIndex(dimension=dim, m=args.hnsw_m, ef_construction=args.hnsw_ef_construction, ef_search=args.hnsw_ef_search)
        t0 = time.perf_counter()
        hnsw.add(ids, vectors)
        hnsw_build_s = time.perf_counter() - t0

    bm25 = Bm25Retriever(Bm25Scorer(inv), records)

    def make_hybrid(ann) -> HybridRetriever:
        return HybridRetriever(dense_retriever=DenseRetriever(embedder, ann, records), bm25_retriever=bm25)

    configs = {
        "bm25": (make_hybrid(brute), RetrievalMode.bm25),
        "dense_bruteforce": (make_hybrid(brute), RetrievalMode.dense),
        "hybrid_bruteforce": (make_hybrid(brute), RetrievalMode.hybrid),
    }
    tuned_weight = None
    tuning_curve: dict[str, float] = {}
    if args.tune_split:
        # Pick the dense weight on a DIFFERENT split than the one we report, so the
        # reported test numbers are not tuned on the test queries.
        _, tune_queries, tune_qrels = load_beir(args.dataset, args.tune_split)
        for w in (0.0, 0.1, 0.2, 0.3, 0.5):
            hr = HybridRetriever(DenseRetriever(embedder, brute, records), bm25, dense_weight=w, sparse_weight=1.0 - w)
            tune_run = {q: [r.chunk_id for r in hr.search(t, mode=RetrievalMode.hybrid_weighted, top_k=100).results] for q, t in tune_queries.items()}
            tuning_curve[str(w)] = evaluate_run(tune_run, tune_qrels)["ndcg@10"]
        tuned_weight = float(max(tuning_curve, key=lambda k: tuning_curve[k]))
        print(f"weight tuning on '{args.tune_split}' (nDCG@10 by dense weight): {tuning_curve} -> best dense_weight={tuned_weight}")
        tuned = HybridRetriever(DenseRetriever(embedder, brute, records), bm25, dense_weight=tuned_weight, sparse_weight=1.0 - tuned_weight)
        configs["hybrid_weighted_tuned"] = (tuned, RetrievalMode.hybrid_weighted)
    if hnsw is not None:
        configs["dense_hnsw"] = (make_hybrid(hnsw), RetrievalMode.dense)
        configs["hybrid_hnsw"] = (make_hybrid(hnsw), RetrievalMode.hybrid)

    results: dict[str, dict] = {}
    for name, (retriever, mode) in configs.items():
        run: dict[str, list[str]] = {}
        lat: list[float] = []
        for qid, text in queries.items():
            t = time.perf_counter()
            resp = retriever.search(text, mode=mode, top_k=100)
            lat.append((time.perf_counter() - t) * 1000)
            run[qid] = [r.chunk_id for r in resp.results]
        metrics = evaluate_run(run, qrels)
        metrics["latency_p50_ms"] = round(statistics.median(lat), 2)
        metrics["latency_p95_ms"] = round(float(np.percentile(lat, 95)), 2)
        results[name] = metrics
        print(f"{name:20s} {metrics}")

    ann_recall = None
    if hnsw is not None:
        hits = []
        for text in queries.values():
            qv = embedder.embed_query(text)
            exact = {i for i, _ in brute.search(qv, top_k=10)}
            approx = {i for i, _ in hnsw.search(qv, top_k=10)}
            hits.append(len(exact & approx) / max(len(exact), 1))
        ann_recall = round(float(np.mean(hits)), 4)
        print(f"HNSW recall@10 vs brute force: {ann_recall}")

    report = {
        "dataset": args.dataset.name,
        "split": args.split,
        "num_docs": len(corpus),
        "num_queries": len(queries),
        "embedder": embedder.model_name(),
        "hnsw_params": {"m": args.hnsw_m, "ef_construction": args.hnsw_ef_construction, "ef_search": args.hnsw_ef_search},
        "build_seconds": {"bm25": round(bm25_build_s, 2), "embedding": round(embed_s, 2), "hnsw": None if hnsw_build_s is None else round(hnsw_build_s, 2)},
        "hnsw_recall_at_10_vs_bruteforce": ann_recall,
        "weight_tuning": {"split": args.tune_split, "ndcg@10_by_dense_weight": tuning_curve, "chosen_dense_weight": tuned_weight},
        "results": results,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
