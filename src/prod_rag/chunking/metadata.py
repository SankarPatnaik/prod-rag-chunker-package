from __future__ import annotations

import math
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Sequence

from prod_rag.models import Chunk, ChunkingStrategy, DocumentType


class ChunkMetadataEnricher:
    WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")
    ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

    def enrich(
        self,
        chunks: Sequence[Chunk],
        document_type: DocumentType,
        strategy: ChunkingStrategy,
        embedding_model: str | None,
    ) -> list[Chunk]:
        now = datetime.now(UTC).isoformat()
        out = list(chunks)

        for idx, chunk in enumerate(out):
            chunk.chunk_index = idx
            chunk.document_type = document_type
            chunk.strategy_used = strategy.value
            chunk.section_title = chunk.heading_path[-1] if chunk.heading_path else None
            chunk.section_path = " > ".join(chunk.heading_path)
            chunk.normalized_text = " ".join(chunk.text.split())
            chunk.sentence_count = max(1, len(re.findall(r"[.!?](?:\s|$)", chunk.text)))
            chunk.semantic_keywords = self._keywords(chunk.text)
            chunk.entities = self._entities(chunk.text)
            chunk.table_flag = chunk.block_type == "table" or "|" in chunk.text
            chunk.code_flag = chunk.block_type == "code" or "```" in chunk.text or "def " in chunk.text
            chunk.citation_flag = bool(re.search(r"\[\d+\]|\([A-Z][A-Za-z]+,\s*\d{4}\)", chunk.text))
            chunk.previous_chunk_id = out[idx - 1].chunk_id if idx > 0 else None
            chunk.next_chunk_id = out[idx + 1].chunk_id if idx < len(out) - 1 else None
            chunk.embedding_model = embedding_model
            chunk.created_at = now
            chunk.confidence_score = round(self._segmentation_confidence(chunk), 4)
            chunk.context_preservation_score = round(self._context_score(chunk), 4)
            chunk.metadata.update(
                {
                    "section_title": chunk.section_title,
                    "section_path": chunk.section_path,
                    "strategy_used": chunk.strategy_used,
                    "document_type": chunk.document_type.value,
                    "semantic_keywords": chunk.semantic_keywords,
                    "entities": chunk.entities,
                    "confidence_score": chunk.confidence_score,
                    "context_preservation_score": chunk.context_preservation_score,
                }
            )
        return out

    def _keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        tokens = [w.lower() for w in self.WORD_RE.findall(text)]
        if not tokens:
            return []
        counts = Counter(tokens)
        stop = {"the", "and", "for", "with", "that", "from", "this", "are", "was", "were", "has", "have", "into"}
        ranked = [w for w, _ in counts.most_common(max_keywords * 2) if w not in stop]
        return ranked[:max_keywords]

    def _entities(self, text: str, max_entities: int = 8) -> list[str]:
        ents = list(dict.fromkeys(self.ENTITY_RE.findall(text)))
        return ents[:max_entities]

    def _segmentation_confidence(self, chunk: Chunk) -> float:
        size_score = 1.0 - min(1.0, abs(chunk.token_count - 350) / 700)
        sentence_score = min(1.0, chunk.sentence_count / 6)
        section_score = 1.0 if chunk.section_path else 0.6
        return (0.45 * size_score) + (0.3 * sentence_score) + (0.25 * section_score)

    def _context_score(self, chunk: Chunk) -> float:
        lineage = 1.0 if chunk.section_path else 0.5
        adjacency = 1.0 if (chunk.previous_chunk_id or chunk.next_chunk_id) else 0.4
        lexical = min(1.0, math.log2(max(2, chunk.token_count)) / 8)
        boundary = 1.0
        if chunk.table_flag or chunk.code_flag or chunk.citation_flag:
            boundary = 1.0
        elif chunk.token_count < 80:
            boundary = 0.6
        return 0.35 * lineage + 0.25 * adjacency + 0.2 * lexical + 0.2 * boundary
