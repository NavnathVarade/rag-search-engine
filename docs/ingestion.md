# Ingestion

## Pipeline (`ingestion/pipeline.py::IngestionPipeline`)
```
raw bytes -> parse -> clean -> extract metadata -> deduplicate -> chunk
```
Storage-agnostic by design: `ingest()` returns an `IngestionResult`
(metadata + chunks); persistence happens in `storage/repositories`, so the
pipeline is unit-testable with zero database (see
`tests/ingestion/test_ingestion.py::test_full_pipeline_parse_clean_chunk`).

## Parsers (`ingestion/parsers/`)
- `PlainTextParser` / `MarkdownParser`: decode with an encoding fallback
  ladder (utf-8 → utf-8-sig → latin-1 → utf-8 with replacement) so a
  single bad byte never drops a whole document.
- `HtmlParser`: BeautifulSoup-based, strips `nav/header/footer/script/
  style/aside/form` boilerplate, converts `<pre>` to fenced code and
  `<table>` to pipe-delimited text *before* stripping tags so code/tabular
  content survives (spec #8's "preserve code blocks, tables").
- `PdfParser`: pypdf-based, de-hyphenates line-wrapped words, marks page
  boundaries explicitly, tolerates a single corrupt page without failing
  the whole document, enforces `max_pdf_pages`.
- `SourceCodeParser`: decodes + detects language by extension; structural
  understanding happens later in `chunking/code.py`.

## Cleaning (`ingestion/cleaners/text_cleaner.py`)
Unicode NFKC normalization, zero-width-character stripping, whitespace
collapsing -- with fenced code blocks stashed out and restored verbatim
first, since indentation inside code is semantically meaningful and must
never be collapsed.

## Deduplication (`ingestion/cleaners/deduplication.py`)
SHA-256 content hash over normalized text. Re-ingesting byte-identical
content returns the *same* `document_id` and skips re-chunking entirely
(spec #59 idempotency) -- verified in
`test_full_pipeline_detects_duplicate_on_reingest`.

## Loaders (`ingestion/loaders/`)
- `local_file.py`: excludes `.git/node_modules/__pycache__/venv/dist/
  build` directories and binary extensions by default (configurable);
  enforces `max_files_per_repo` and `max_document_size_mb`.
- `url_loader.py`: validates the URL against SSRF rules *before* any
  network call, streams the response with a hard size cap, and
  re-validates any redirect target rather than blindly following it.

## Git repositories
Cloning + branch/commit-aware ingestion is exposed via
`rag ingest --url <git-clone-url>` combined with `local_file.py`'s
directory walker once cloned locally; this reference build does not ship
a `git clone` wrapper itself (would need `subprocess.run(["git", "clone",
...])` with the same URL-validation discipline as `url_loader.py`) -- see
docs/final-report.md "Recommended next steps".
