# prod-rag-chunker

Production-grade, adaptive chunking framework for RAG/LAG and long-context GenAI systems.

## 1) Proposed architecture

```text
                 ┌──────────────────────────┐
                 │ Ingestion Layer          │
                 │ loaders/*                │
                 └────────────┬─────────────┘
                              │ text + source metadata
                 ┌────────────▼─────────────┐
                 │ Structure Parsing Layer   │
                 │ chunking/parser.py        │
                 └────────────┬─────────────┘
                              │ DocumentSection[]
                 ┌────────────▼─────────────┐
                 │ Type+Signal Analysis      │
                 │ chunking/classifier.py    │
                 └────────────┬─────────────┘
                              │ doc_type + structural signals
                 ┌────────────▼─────────────┐
                 │ Strategy Selector         │
                 │ chunking/strategy_selector│
                 └────────────┬─────────────┘
                              │ chosen strategy
                 ┌────────────▼─────────────┐
                 │ Adaptive Chunker          │
                 │ chunking/chunker.py       │
                 └────────────┬─────────────┘
                              │ Chunk[] + ParentChunk[]
                 ┌────────────▼─────────────┐
                 │ Metadata Enrichment       │
                 │ chunking/metadata.py      │
                 └────────────┬─────────────┘
                              │ lineage + CPS + confidence
                 ┌────────────▼─────────────┐
                 │ Embedding Layer           │
                 │ embeddings.py             │
                 └────────────┬─────────────┘
                              │ vectors
                 ┌────────────▼────────────────────────────────┐
                 │ Vector Adapter Layer                        │
                 │ FAISS/Qdrant/PGVector/Elastic              │
                 └─────────────────────────────────────────────┘
```

## 2) Refactoring plan (implemented)

1. Extend domain models for document type, chunk lineage, and quality metrics.
2. Add document-type classifier and structural signal extraction.
3. Add strategy selector using signals + optional config rules.
4. Add strategy-specific unit segmentation (legal/FAQ/code/chat/table-aware).
5. Add metadata enricher with context-preservation score.
6. Add orchestration service for chunk->embed->persist.
7. Add PGVector and Elasticsearch vector adapters.
8. Expand tests and examples.

## 3) Final code implementation

### Core modules
- `src/prod_rag/chunking/classifier.py`
- `src/prod_rag/chunking/strategy_selector.py`
- `src/prod_rag/chunking/chunker.py`
- `src/prod_rag/chunking/metadata.py`
- `src/prod_rag/chunking/pipeline.py`
- `src/prod_rag/chunking/service.py`

### Pipeline stages
1. normalize input text
2. parse sections / block types
3. infer `DocumentType` + structural signals
4. choose chunking strategy dynamically
5. chunk with token budget + overlap
6. enrich chunk metadata (lineage, keyword/entity hints, CPS)
7. optional embeddings + vector persistence

## 4) Config structure

```yaml
chunking:
  strategy: hierarchical
  enable_dynamic_strategy_selection: true
  target_chunk_tokens: 700
  overlap_tokens: 90
  context_window_sentences: 2
  strategy_rules: []

embedder:
  model_name: sentence-transformers/all-MiniLM-L6-v2

vector_store:
  type: faiss   # faiss | qdrant | pgvector | elastic
  postgres_dsn: postgresql://postgres:postgres@localhost:5432/postgres
  postgres_table: rag_chunks
  elastic_url: http://localhost:9200
  elastic_index: prod_rag_chunks
```

## 5) PGVector adapter

Implemented at `src/prod_rag/vectorstores/pgvector_store.py`.

- Auto-creates `vector` extension/table
- Upserts `id`, `embedding`, `text`, `metadata`
- Uses cosine distance operator for KNN retrieval

## 6) Elasticsearch adapter

Implemented at `src/prod_rag/vectorstores/elastic_store.py`.

- Creates index with `dense_vector`
- Bulk upsert for chunk vectors
- KNN search endpoint-ready for hybrid extension

## 7) Unit tests

See `tests/test_chunking.py` for adaptive behavior coverage.

## 8) Sample input/output examples

- `examples/sample_legal.txt`
- `examples/sample_output.json`

## 9) Why this design is differentiated / patent-oriented

This design introduces a **multi-signal adaptive chunking control loop**:

- **Signal-driven strategy routing** (document type + structure + density signals)
- **Boundary-preserving chunk units** per content class (clauses, Q/A, turns, code blocks, tables)
- **Context Preservation Score (CPS)** for each chunk to quantify lineage/adjacency/cohesion quality
- **Link-rich metadata graph** (`prev/next/parent/section_path`) to improve retrieval and reduce hallucinations

The combination of dynamic strategy routing + quality-scored chunks + lineage graph is materially different from fixed-size or naive recursive chunking.

## 10) Suggested next enhancements

1. Add spaCy NER and noun-phrase extraction for stronger metadata quality.
2. Add hybrid retrieval adapter (BM25 + vector + reranker).
3. Add diagram/figure reference linking for technical PDFs.
4. Add online quality telemetry dashboard (CPS drift, chunk size drift).
5. Add active-learning feedback loop from retrieval misses.

## Quick start

```bash
pip install -e .
uvicorn prod_rag.service.api:app --reload
```

```bash
curl -X POST http://localhost:8000/v1/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "msa-001",
    "source": "examples/sample_legal.txt",
    "strategy": "hierarchical"
  }'
```

> Dynamic strategy selection may override the static strategy when enabled.
