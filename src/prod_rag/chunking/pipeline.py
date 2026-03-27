from __future__ import annotations

from prod_rag.chunking.chunker import HierarchicalChunker
from prod_rag.chunking.parser import StructureParser, TextNormalizer
from prod_rag.chunking.tokenizer import TokenCounter
from prod_rag.models import ChunkingConfig


class ChunkingPipeline:
    def __init__(self, config: ChunkingConfig, tokenizer_name: str | None = None) -> None:
        self.config = config
        self.normalizer = TextNormalizer()
        self.parser = StructureParser()
        self.token_counter = TokenCounter(tokenizer_name=tokenizer_name)
        self.chunker = HierarchicalChunker(self.token_counter, config)

    def chunk_document(self, document_text: str, document_id: str) -> dict:
        normalized = self.normalizer.normalize(document_text)
        sections = self.parser.parse(document_id=document_id, text=normalized)
        child_chunks, parent_chunks = self.chunker.chunk_sections(document_id=document_id, sections=sections)
        return {
            "document_id": document_id,
            "sections": [s.model_dump() for s in sections],
            "child_chunks": [c.model_dump() for c in child_chunks],
            "parent_chunks": [p.model_dump() for p in parent_chunks],
            "stats": {
                "document_tokens_est": self.token_counter.count(normalized),
                "num_sections": len(sections),
                "num_child_chunks": len(child_chunks),
                "num_parent_chunks": len(parent_chunks),
            },
        }
