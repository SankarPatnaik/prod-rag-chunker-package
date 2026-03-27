from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from prod_rag.chunking.pipeline import ChunkingPipeline
from prod_rag.loaders.factory import get_loader
from prod_rag.models import ChunkRequest, ChunkResponse, ChunkingConfig, LoaderConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="prod-rag-chunker", version="0.2.0")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "prod-rag-chunker API is running",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chunk", response_model=ChunkResponse)
def chunk_document(request: ChunkRequest) -> ChunkResponse:
    config = ChunkingConfig()
    if request.strategy:
        config.strategy = request.strategy
    pipeline = ChunkingPipeline(config)

    if request.text:
        document_id = request.document_id or "inline-document"
        return pipeline.chunk_document(document_text=request.text, document_id=document_id, source=request.source or "inline")

    if request.source:
        path = Path(request.source)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Source file not found: {request.source}")
        loader = get_loader(path, LoaderConfig(type="auto"))
        doc = loader.load(path)
        logger.info("Loaded document %s using %s", doc.document_id, loader.__class__.__name__)
        return pipeline.chunk_document(document_text=doc.text, document_id=request.document_id or doc.document_id, source=str(path))

    raise HTTPException(status_code=400, detail="Provide either text or source")
