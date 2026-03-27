# prod-rag-chunker

Production-ready Python package + FastAPI service for document chunking optimized for small-context LLMs (Llama, Mistral) and financial documents.

## Why this design works better for small LLM RAG

Small LLMs need:
- **token-safe chunk boundaries**
- **high semantic density per chunk**
- **strict lineage metadata** for retrieval and audit trails
- **strategy flexibility** by document shape (narrative, table-heavy, sectioned reports)

This package enforces all four through configurable token-aware strategies and metadata-rich output.

## Deep review findings (before refactor)

1. Chunking output used weak metadata lineage (limited source/page/parent linkage).
2. Strategy behavior was mostly hierarchical and not cleanly configurable.
3. HTML/Markdown ingestion was missing.
4. Service mode was missing (CLI only).
5. Tests focused on a single happy path and skipped service + lineage checks.

## Refactoring summary

- Added strategy-aware chunking (`sentence_aware`, `section_aware`, `semantic`, `hierarchical`, `table_aware`)
- Added richer chunk metadata (`source`, `section`, `chunk_id`, `parent_id`, `token_count`)
- Added page lineage extraction from PDF page markers
- Added HTML + Markdown loaders
- Added FastAPI service with `/health` and `/v1/chunk`
- Added metrics model (`avg/max/min tokens`) and output hooks for chunk quality benchmarking
- Kept library-first API while enabling service mode

## Clean package structure

```text
src/prod_rag/
  chunking/
    parser.py
    tokenizer.py
    chunker.py
    pipeline.py
  loaders/
    base.py
    text_loader.py
    pdf_loader.py
    docx_loader.py
    html_loader.py
    markdown_loader.py
    factory.py
  service/
    api.py
  models.py
  cli.py
```

## Features

- PDF, DOCX, TXT, HTML, Markdown support
- Token-aware overlap control
- Table-aware chunk isolation
- Semantic merge mode for dense sections
- Parent-child hierarchical chunks for expandable retrieval
- FastAPI service layer
- Typed Pydantic request/response/config models
- Tests for chunking and service behavior

## Install

```bash
pip install -e .
```

## CLI usage

```bash
prod-rag serve --host 0.0.0.0 --port 8000
prod-rag index examples/config.yaml examples/sample.txt
prod-rag ask examples/config.yaml "What is the renewal obligation?"
```

## FastAPI usage

```bash
uvicorn prod_rag.service.api:app --host 0.0.0.0 --port 8000
```

### Chunk inline text

```bash
curl -X POST http://localhost:8000/v1/chunk \
  -H 'Content-Type: application/json' \
  -d '{
    "document_id": "annual-2025",
    "text": "# Annual Report\n\nRevenue increased 10% year over year.",
    "strategy": "section_aware"
  }'
```

### Chunk a file

```bash
curl -X POST http://localhost:8000/v1/chunk \
  -H 'Content-Type: application/json' \
  -d '{"source": "examples/sample.txt", "strategy": "hierarchical"}'
```

## Chunk quality benchmark hooks

Response includes:
- `stats.document_tokens_est`
- `stats.num_child_chunks`
- `metrics.avg_tokens`
- `metrics.min_tokens`
- `metrics.max_tokens`

These can be used to measure retrieval efficiency across financial sources like passports, account summaries, annual reports, Moody’s, and Orbis exports.
