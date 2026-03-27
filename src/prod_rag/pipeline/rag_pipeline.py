from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

from prod_rag.chunking.pipeline import ChunkingPipeline
from prod_rag.embeddings import SentenceTransformerEmbedder
from prod_rag.llm.factory import get_llm
from prod_rag.loaders.factory import get_loader
from prod_rag.models import AppConfig, PipelineAnswer, SearchResult
from prod_rag.pipeline.prompts import build_map_prompt, build_reduce_prompt
from prod_rag.utils.io import ensure_dir, write_json
from prod_rag.vectorstores.factory import get_vector_store


class RAGPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embedder = SentenceTransformerEmbedder(config.embedder)
        self.llm = get_llm(config.llm)
        self.chunk_pipeline = ChunkingPipeline(config.chunking)
        self.index_dir = ensure_dir(config.vector_store.path)
        self.parent_lookup_path = self.index_dir / "parents.json"
        self.child_lookup_path = self.index_dir / "children.json"
        self.parents: Dict[str, dict] = self._read_json(self.parent_lookup_path)
        self.children: Dict[str, dict] = self._read_json(self.child_lookup_path)
        self.bm25 = None
        self.bm25_ids: List[str] = []
        if self.children:
            self._init_bm25()
        self.vector_store = None

    def index_document(self, path: str) -> dict:
        loader = get_loader(path, self.config.loader)
        doc = loader.load(path)
        chunked_response = self.chunk_pipeline.chunk_document(doc.text, doc.document_id, source=str(path))
        chunked = chunked_response.model_dump()
        child_chunks = chunked["child_chunks"]
        parent_chunks = chunked["parent_chunks"]

        texts = [c["text"] for c in child_chunks]
        ids = [c["chunk_id"] for c in child_chunks]
        metadatas = [{k: v for k, v in c.items() if k != "text"} for c in child_chunks]
        vectors = self.embedder.embed_texts(texts)
        if self.vector_store is None:
            self.vector_store = get_vector_store(self.config.vector_store, dim=vectors.shape[1])
        self.vector_store.upsert(ids, vectors, texts, metadatas)
        self.vector_store.persist()

        for p in parent_chunks:
            self.parents[p["parent_chunk_id"]] = p
        for c in child_chunks:
            self.children[c["chunk_id"]] = c
        write_json(self.parent_lookup_path, self.parents)
        write_json(self.child_lookup_path, self.children)
        self._init_bm25()
        return chunked["stats"]

    def ask(self, query: str) -> PipelineAnswer:
        retrieved = self.retrieve(query)
        mapped = []
        for item in retrieved:
            prompt = build_map_prompt(query, item.text, item.metadata.get("heading_path", []), item.id)
            response = self.llm.generate(prompt)
            if response.strip() == "IRRELEVANT":
                continue
            mapped.append({
                "chunk_id": item.id,
                "heading": item.metadata.get("heading", ""),
                "score": item.score,
                "evidence": response,
            })
        final_prompt = build_reduce_prompt(query, mapped)
        answer = self.llm.generate(final_prompt)
        return PipelineAnswer(query=query, answer=answer, mapped_evidence=mapped, retrieved_chunks=retrieved)

    def retrieve(self, query: str) -> List[SearchResult]:
        if self.vector_store is None and self.children:
            sample_vec = self.embedder.embed_query("hello")
            self.vector_store = get_vector_store(self.config.vector_store, dim=sample_vec.shape[0])
        dense_results: List[SearchResult] = []
        if self.vector_store is not None:
            qv = self.embedder.embed_query(query)
            dense_results = self.vector_store.search(qv, self.config.retrieval.top_k)

        if self.config.retrieval.use_bm25_fallback and self.bm25 is not None:
            bm25_hits = self._bm25_search(query, self.config.retrieval.top_k)
            merged = self._merge_results(dense_results, bm25_hits)
        else:
            merged = dense_results

        if self.config.retrieval.use_parent_expansion:
            return self._expand_to_parent_context(merged)
        return merged[: self.config.retrieval.top_k]

    def _expand_to_parent_context(self, results: List[SearchResult]) -> List[SearchResult]:
        expanded: List[SearchResult] = []
        seen = set()
        for result in results[: self.config.retrieval.parent_expansion_limit]:
            child = self.children.get(result.id)
            if not child:
                expanded.append(result)
                continue
            parent_text = result.text
            for parent in self.parents.values():
                if result.id in parent.get("child_chunk_ids", []):
                    parent_id = parent["parent_chunk_id"]
                    if parent_id in seen:
                        continue
                    seen.add(parent_id)
                    expanded.append(SearchResult(
                        id=parent_id,
                        score=result.score,
                        text=parent["text"],
                        metadata={"heading": " > ".join(parent.get("heading_path", [])), "heading_path": parent.get("heading_path", [])},
                    ))
                    break
            else:
                expanded.append(result)
        if not expanded:
            return results
        return expanded

    def _merge_results(self, dense: List[SearchResult], bm25: List[SearchResult]) -> List[SearchResult]:
        merged = {}
        for result in dense + bm25:
            if result.id not in merged or result.score > merged[result.id].score:
                merged[result.id] = result
        return sorted(merged.values(), key=lambda x: x.score, reverse=True)[: self.config.retrieval.top_k]

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.bm25_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [SearchResult(id=id_, score=float(score), text=self.children[id_]["text"], metadata=self.children[id_]) for id_, score in ranked if id_ in self.children]

    def _init_bm25(self) -> None:
        self.bm25_ids = list(self.children.keys())
        corpus = [self.children[id_]["text"].lower().split() for id_ in self.bm25_ids]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def _read_json(self, path: Path) -> Dict[str, dict]:
        if not path.exists():
            return {}
        import json
        return json.loads(path.read_text(encoding="utf-8"))
