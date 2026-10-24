"""Check the embedding backend selected by your settings (.env / environment).

    uv run python scripts/check_embeddings.py

Loads the embedder exactly the way the API and CLI do (`build_embedder`), then reports:
  * which model loaded, its vector size, and how long loading/encoding took;
  * whether vectors are unit length (the dense index assumes cosine == dot product);
  * a semantic sanity test: for each triple, a *related* text must score higher against the query than an
    *unrelated* one. A real sentence-embedding model passes all of them; the offline hashing embedder
    cannot, because it only matches shared words. That failure is expected and not a bug.
  * the similarity range of related vs unrelated pairs, to help you choose MIN_DENSE_SIMILARITY
    (the chat evidence gate; it should sit between the two).

Exit code 0 = every triple ordered correctly, 1 = at least one did not, 2 = the embedder failed to load.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_engine.api.engine import build_embedder  # noqa: E402
from rag_engine.config import Settings  # noqa: E402
from rag_engine.exceptions import RagEngineError  # noqa: E402

# (query, related passage, unrelated passage)
TRIPLES = [
    ("my car will not start", "The automobile engine failed to turn over this morning.",
     "A recipe for banana bread with walnuts."),
    ("How do I restart a failed container?", "Kubernetes restarts a crashed pod according to its restartPolicy.",
     "The museum opens at nine and closes at five."),
    ("speed up slow database queries", "Adding an index lets PostgreSQL find rows without scanning the whole table.",
     "The football team won the championship last night."),
    ("how to reset my password", "Click 'Forgot password' and we will email you a link to choose a new one.",
     "Glaciers are retreating because of rising temperatures."),
]


def main() -> int:
    settings = Settings()
    print(f"EMBEDDING_PROVIDER={settings.embedding_provider}  EMBEDDING_MODEL={settings.embedding_model}  DEVICE={settings.embedding_device}")
    t0 = time.perf_counter()
    try:
        emb = build_embedder(settings)
        emb.embed_query("warm up")  # first call can be slow (lazy init)
    except RagEngineError as exc:
        print(f"FAILED to load embedder: {exc.message}")
        return 2
    print(f"loaded: {emb.model_name()}  dimension={emb.dimension()}  load+warmup={time.perf_counter() - t0:.1f}s")

    texts = [t for triple in TRIPLES for t in triple]
    t0 = time.perf_counter()
    vectors = emb.embed_documents(texts)
    per_text_ms = (time.perf_counter() - t0) * 1000 / len(texts)
    norms = np.linalg.norm(vectors, axis=1)
    print(f"encoded {len(texts)} texts at ~{per_text_ms:.1f} ms each; norms min/max = {norms.min():.4f}/{norms.max():.4f}")
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("WARNING: vectors are not unit length; cosine scores from the index will not be meaningful.")

    related, unrelated, failures = [], [], 0
    print(f"\n{'related':>8} {'unrelated':>9}  result  query")
    for i, (query, _good, _bad) in enumerate(TRIPLES):
        q, g, b = vectors[3 * i], vectors[3 * i + 1], vectors[3 * i + 2]
        s_good, s_bad = float(q @ g), float(q @ b)
        related.append(s_good)
        unrelated.append(s_bad)
        ok = s_good > s_bad
        failures += not ok
        print(f"{s_good:8.3f} {s_bad:9.3f}  {'ok  ' if ok else 'FAIL'}   {query}")

    print(f"\nrelated similarity   min={min(related):.3f}  max={max(related):.3f}")
    print(f"unrelated similarity min={min(unrelated):.3f}  max={max(unrelated):.3f}")
    print(f"current MIN_DENSE_SIMILARITY={settings.min_dense_similarity}")
    if max(unrelated) < min(related):
        print(f"-> a threshold between {max(unrelated):.2f} and {min(related):.2f} separates these examples "
              "(only 4 pairs: validate on your own documents).")
    else:
        print("-> related and unrelated scores overlap here, so no threshold separates them on these examples.")

    if failures:
        if settings.embedding_provider.lower() == "hashing":
            hint = "Expected for the hashing embedder (no semantics): use EMBEDDING_PROVIDER=sentence_transformers."
        else:
            hint = "Check the model name and that it is a sentence-embedding model."
        print(f"\n{failures}/{len(TRIPLES)} triples ordered wrongly. {hint}")
        return 1
    print(f"\nall {len(TRIPLES)} triples ordered correctly: this embedder captures meaning beyond shared words.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
