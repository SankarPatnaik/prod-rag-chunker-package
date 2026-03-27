from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from prod_rag.models import SearchResult, VectorStoreConfig
from prod_rag.utils.io import ensure_dir
from prod_rag.vectorstores.base import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    def __init__(self, config: VectorStoreConfig, dim: int) -> None:
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Install faiss extras to use FAISSVectorStore") from exc
        self.faiss = faiss
        self.config = config
        self.path = ensure_dir(config.path)
        self.index_path = self.path / "index.faiss"
        self.meta_path = self.path / "metadata.json"
        self.dim = dim
        self.records: List[dict] = []
        if self.index_path.exists() and self.meta_path.exists():
            self.index = self.faiss.read_index(str(self.index_path))
            self.records = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self.index = self.faiss.IndexFlatIP(dim)

    def upsert(self, ids: List[str], vectors: np.ndarray, texts: List[str], metadatas: List[dict]) -> None:
        self.index.add(vectors)
        for id_, text, meta in zip(ids, texts, metadatas):
            self.records.append({"id": id_, "text": text, "metadata": meta})

    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.records):
                continue
            record = self.records[idx]
            results.append(SearchResult(id=record["id"], score=float(score), text=record["text"], metadata=record["metadata"]))
        return results

    def persist(self) -> None:
        self.faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
