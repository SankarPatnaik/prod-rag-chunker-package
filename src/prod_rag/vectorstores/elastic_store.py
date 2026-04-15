from __future__ import annotations

from typing import List

import numpy as np

from prod_rag.models import SearchResult, VectorStoreConfig
from prod_rag.vectorstores.base import BaseVectorStore


class ElasticVectorStore(BaseVectorStore):
    def __init__(self, config: VectorStoreConfig, dim: int) -> None:
        try:
            from elasticsearch import Elasticsearch
        except Exception as exc:  # pragma: no cover
            raise ImportError("Install elasticsearch to use ElasticVectorStore") from exc

        self.client = Elasticsearch(config.elastic_url)
        self.index = config.elastic_index
        self.dim = dim

        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(
                index=self.index,
                mappings={
                    "properties": {
                        "id": {"type": "keyword"},
                        "text": {"type": "text"},
                        "metadata": {"type": "object", "enabled": True},
                        "embedding": {"type": "dense_vector", "dims": self.dim, "index": True, "similarity": "cosine"},
                    }
                },
            )

    def upsert(self, ids: List[str], vectors: np.ndarray, texts: List[str], metadatas: List[dict]) -> None:
        operations = []
        for id_, vector, text, meta in zip(ids, vectors, texts, metadatas):
            operations.append({"index": {"_index": self.index, "_id": id_}})
            operations.append({"id": id_, "text": text, "metadata": meta, "embedding": vector.tolist()})
        self.client.bulk(operations=operations, refresh=True)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        response = self.client.search(
            index=self.index,
            size=top_k,
            knn={"field": "embedding", "query_vector": query_vector.tolist(), "k": top_k, "num_candidates": max(50, top_k * 4)},
        )
        hits = response.get("hits", {}).get("hits", [])
        return [
            SearchResult(
                id=hit.get("_id", ""),
                score=float(hit.get("_score", 0.0)),
                text=hit.get("_source", {}).get("text", ""),
                metadata=hit.get("_source", {}).get("metadata", {}),
            )
            for hit in hits
        ]

    def persist(self) -> None:
        self.client.indices.refresh(index=self.index)
