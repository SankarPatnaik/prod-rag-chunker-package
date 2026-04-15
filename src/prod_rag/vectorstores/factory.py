from __future__ import annotations

from prod_rag.models import VectorBackendType, VectorStoreConfig
from prod_rag.vectorstores.elastic_store import ElasticVectorStore
from prod_rag.vectorstores.faiss_store import FAISSVectorStore
from prod_rag.vectorstores.pgvector_store import PGVectorStore
from prod_rag.vectorstores.qdrant_store import QdrantVectorStore


def get_vector_store(config: VectorStoreConfig, dim: int):
    if config.type == VectorBackendType.QDRANT:
        return QdrantVectorStore(config, dim)
    if config.type == VectorBackendType.PGVECTOR:
        return PGVectorStore(config, dim)
    if config.type == VectorBackendType.ELASTIC:
        return ElasticVectorStore(config, dim)
    return FAISSVectorStore(config, dim)
