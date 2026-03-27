from __future__ import annotations

from typing import Iterable, List

import numpy as np

from prod_rag.models import EmbedderConfig


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbedderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Install sentence-transformers extras to use embeddings") from exc
        self.model = SentenceTransformer(config.model_name, device=config.device)
        self.batch_size = config.batch_size

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], batch_size=1, normalize_embeddings=True)
        return np.asarray(vec[0], dtype="float32")
