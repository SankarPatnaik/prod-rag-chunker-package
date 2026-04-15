from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prod_rag.chunking.pipeline import ChunkingPipeline
from prod_rag.embeddings import SentenceTransformerEmbedder
from prod_rag.models import AppConfig, ChunkResponse
from prod_rag.vectorstores.factory import get_vector_store


@dataclass
class ProcessResult:
    chunk_response: ChunkResponse
    vector_records: int


class AdaptiveChunkingService:
    """Enterprise chunking service orchestrating parsing, adaptive chunking, embeddings, and vector persistence."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.chunk_pipeline = ChunkingPipeline(config.chunking)
        self.embedder = SentenceTransformerEmbedder(config.embedder)
        self.vector_store = None

    def process_document(self, document_id: str, source: str, text: str, store_vectors: bool = True) -> ProcessResult:
        chunked = self.chunk_pipeline.chunk_document(
            document_text=text,
            document_id=document_id,
            source=source,
            embedding_model=self.config.embedder.model_name,
        )

        if not store_vectors:
            return ProcessResult(chunk_response=chunked, vector_records=0)

        texts = [c.text for c in chunked.child_chunks]
        ids = [c.chunk_id for c in chunked.child_chunks]
        metadatas: list[dict[str, Any]] = [{k: v for k, v in c.model_dump().items() if k != "text"} for c in chunked.child_chunks]
        vectors = self.embedder.embed_texts(texts)

        if self.vector_store is None:
            self.vector_store = get_vector_store(self.config.vector_store, dim=vectors.shape[1])
        self.vector_store.upsert(ids, vectors, texts, metadatas)
        self.vector_store.persist()

        return ProcessResult(chunk_response=chunked, vector_records=len(ids))
