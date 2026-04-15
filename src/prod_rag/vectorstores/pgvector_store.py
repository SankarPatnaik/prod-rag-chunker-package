from __future__ import annotations

import json
from typing import List

import numpy as np

from prod_rag.models import SearchResult, VectorStoreConfig
from prod_rag.vectorstores.base import BaseVectorStore


class PGVectorStore(BaseVectorStore):
    def __init__(self, config: VectorStoreConfig, dim: int) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except Exception as exc:  # pragma: no cover
            raise ImportError("Install psycopg and pgvector to use PGVectorStore") from exc

        self.dim = dim
        self.table = config.postgres_table
        self.conn = psycopg.connect(config.postgres_dsn)
        register_vector(self.conn)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id TEXT PRIMARY KEY,
                embedding vector({self.dim}),
                text TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb
            )
            """
        )
        self.conn.commit()

    def upsert(self, ids: List[str], vectors: np.ndarray, texts: List[str], metadatas: List[dict]) -> None:
        sql = f"""
        INSERT INTO {self.table}(id, embedding, text, metadata)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id)
        DO UPDATE SET embedding = EXCLUDED.embedding, text = EXCLUDED.text, metadata = EXCLUDED.metadata
        """
        with self.conn.cursor() as cur:
            for id_, vector, text, meta in zip(ids, vectors, texts, metadatas):
                cur.execute(sql, (id_, vector.tolist(), text, json.dumps(meta)))
        self.conn.commit()

    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        sql = f"""
            SELECT id, text, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self.conn.cursor() as cur:
            vec = query_vector.tolist()
            cur.execute(sql, (vec, vec, top_k))
            rows = cur.fetchall()
        return [SearchResult(id=row[0], text=row[1], metadata=row[2] or {}, score=float(row[3])) for row in rows]

    def persist(self) -> None:
        self.conn.commit()
