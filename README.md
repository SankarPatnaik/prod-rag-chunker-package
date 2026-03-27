# prod-rag-chunker

Production-ready package for large-document processing with small models such as Llama and Mistral.

## Features
- PDF and DOCX loaders
- Structure-aware, token-aware hierarchical chunking
- FAISS and Qdrant vector store adapters
- Embedding wrapper using Sentence Transformers
- Llama.cpp and vLLM inference wrappers
- End-to-end RAG pipeline with map-reduce answering
- CLI for indexing and querying

## Install
```bash
pip install -e .[full]
```

For a lighter install:
```bash
pip install -e .[faiss,embeddings,llama_cpp]
```

## Quick start
```bash
prod-rag index examples/config.yaml examples/sample.docx
prod-rag ask examples/config.yaml "What are the renewal obligations?"
```

## Config highlights
- `loader.type`: `auto`, `pdf`, `docx`, or `text`
- `embedder.model_name`: sentence-transformers model
- `vector_store.type`: `faiss` or `qdrant`
- `llm.backend`: `llama_cpp` or `vllm`
- `chunking.*`: token budgets and overlap settings

## Architecture
1. Load document
2. Normalize and parse structure
3. Create child and parent chunks
4. Embed child chunks
5. Store vectors in FAISS or Qdrant
6. Retrieve top-k child chunks for a query
7. Expand to parent chunks when needed
8. Run map step over relevant chunks
9. Run reduce step for final answer

## Notes
- The PDF loader uses `pypdf` text extraction. For scanned PDFs, add OCR before this pipeline.
- For legal and policy documents, parent expansion often improves answer completeness.
