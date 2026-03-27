from __future__ import annotations

from statistics import mean

from prod_rag.chunking.chunker import HierarchicalChunker
from prod_rag.chunking.parser import StructureParser, TextNormalizer
from prod_rag.chunking.tokenizer import TokenCounter
from prod_rag.models import ChunkMetrics, ChunkResponse, ChunkingConfig


class ChunkingPipeline:
    def __init__(self, config: ChunkingConfig, tokenizer_name: str | None = None) -> None:
        self.config = config
        self.normalizer = TextNormalizer()
        self.parser = StructureParser()
        self.token_counter = TokenCounter(tokenizer_name=tokenizer_name)
        self.chunker = HierarchicalChunker(self.token_counter, config)

    def chunk_document(self, document_text: str, document_id: str, source: str = "inline") -> ChunkResponse:
        normalized = self.normalizer.normalize(document_text)
        sections = self.parser.parse(document_id=document_id, text=normalized)
        child_chunks, parent_chunks = self.chunker.chunk_sections(document_id=document_id, source=source, sections=sections)

        token_counts = [c.token_count for c in child_chunks] or [0]
        metrics = ChunkMetrics(
            total_chunks=len(child_chunks),
            avg_tokens=float(mean(token_counts)),
            max_tokens=max(token_counts),
            min_tokens=min(token_counts),
        )

        return ChunkResponse(
            document_id=document_id,
            sections=sections,
            child_chunks=child_chunks,
            parent_chunks=parent_chunks,
            stats={
                "document_tokens_est": self.token_counter.count(normalized),
                "num_sections": len(sections),
                "num_child_chunks": len(child_chunks),
                "num_parent_chunks": len(parent_chunks),
                "strategy": self.config.strategy.value,
            },
            metrics=metrics,
        )
