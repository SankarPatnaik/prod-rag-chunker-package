from __future__ import annotations

from prod_rag.models import VectorStoreConfig
from prod_rag.vectorstores.faiss_store import FAISSVectorStore
from prod_rag.vectorstores.qdrant_store import QdrantVectorStore


def get_vector_store(config: VectorStoreConfig, dim: int):
    if config.type == "qdrant":
        return QdrantVectorStore(config, dim)
    return FAISSVectorStore(config, dim)
