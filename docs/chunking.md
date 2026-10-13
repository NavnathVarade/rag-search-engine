# Chunking

## Strategies implemented (`src/rag_engine/chunking/`)
- **fixed**: sliding window of N words with configurable overlap. Simplest,
  cheapest, but ignores document structure entirely -- can split a
  sentence or code block in half.
- **recursive**: splits by a separator hierarchy (paragraphs → lines →
  sentences → words), recursing into a piece only if it's still over
  budget. The general-purpose default.
- **sentence**: packs whole sentences up to a token budget, never
  splitting mid-sentence.
- **markdown**: parses actual Markdown structure (via markdown-it-py) so
  headings, code fences, and tables are recognized and kept intact,
  tagged with their heading path (H1 > H2 > H3).
- **code**: uses Python's real `ast` module for Python source (exact --
  functions/classes are never split mid-definition) and a brace-matching
  heuristic for C-family languages, falling back to `recursive` for
  anything unparseable.
- **parent_child**: large parent sections (context) + small child chunks
  (the actual retrieval unit, carrying `parent_chunk_id`). Retrieve
  children for precision, optionally expand to the parent for more
  surrounding context before generation.
- **semantic**: embeds each sentence and inserts a boundary wherever
  consecutive-sentence cosine similarity drops below a threshold (a topic
  shift). Most expensive (one embedding call per sentence) -- reach for
  it only when topic drift *within* documents is a known problem, not as
  a default.

## Why large chunks can hurt retrieval
A large chunk mixes multiple topics/facts into one embedding, diluting
the vector toward an average that matches *none* of the specific things a
query might ask about. BM25 similarly loses precision because the
term-frequency signal for any one topic gets diluted by the surrounding
unrelated text, and length normalization (the `b` parameter) further
penalizes long documents relative to focused ones.

## Why tiny chunks can hurt generation
A chunk too small to contain a complete thought forces the LLM to
reconstruct meaning from fragments, increases the total chunk *count*
needed to cover an answer (more context-budget pressure, more chances for
near-duplicate near-misses to crowd out something better), and can strip
away the surrounding qualifier/exception that changes a fact's meaning.

## Why overlap exists, and when it becomes harmful
Overlap ensures a fact or sentence that falls near a chunk boundary still
appears *whole* in at least one chunk. Too much overlap (e.g. 50% of
`chunk_size`) means near-duplicate content gets indexed multiple times,
inflating the index, wasting context-budget slots on redundant text, and
requiring the deduplication stage (`context/deduplicator.py`) to do more
work to avoid showing the LLM the same fact five times.
