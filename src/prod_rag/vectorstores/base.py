from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np

from prod_rag.models import SearchResult


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: List[str], vectors: np.ndarray, texts: List[str], metadatas: List[dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def persist(self) -> None:
        raise NotImplementedError
