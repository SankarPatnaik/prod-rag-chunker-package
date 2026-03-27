from __future__ import annotations

from typing import List

import numpy as np

from prod_rag.models import SearchResult, VectorStoreConfig
from prod_rag.vectorstores.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, config: VectorStoreConfig, dim: int) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http.models import Distance, PointStruct, VectorParams  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Install qdrant extras to use QdrantVectorStore") from exc
        self.PointStruct = PointStruct
        self.client = QdrantClient(host=config.host, port=config.port)
        self.collection = config.collection_name
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def upsert(self, ids: List[str], vectors: np.ndarray, texts: List[str], metadatas: List[dict]) -> None:
        points = []
        for i, (id_, vector, text, meta) in enumerate(zip(ids, vectors, texts, metadatas)):
            payload = {"text": text, **meta}
            points.append(self.PointStruct(id=i if id_.isdigit() else abs(hash(id_)) % (10**9), vector=vector.tolist(), payload=payload))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        hits = self.client.search(collection_name=self.collection, query_vector=query_vector.tolist(), limit=top_k)
        return [SearchResult(id=str(hit.id), score=float(hit.score), text=hit.payload.get("text", ""), metadata=dict(hit.payload)) for hit in hits]

    def persist(self) -> None:
        return
