from __future__ import annotations

from prod_rag.chunking.classifier import DocumentSignals
from prod_rag.models import ChunkingConfig, ChunkingStrategy, DocumentType


class StrategySelector:
    def choose(self, doc_type: DocumentType, signals: DocumentSignals, config: ChunkingConfig) -> ChunkingStrategy:
        if not config.enable_dynamic_strategy_selection:
            return config.strategy

        for rule in config.strategy_rules:
            if rule.document_type != doc_type:
                continue
            if signals.structural_richness < rule.min_structural_richness:
                continue
            if signals.heading_frequency < rule.min_heading_frequency:
                continue
            if signals.table_density < rule.min_table_density:
                continue
            if signals.code_density < rule.min_code_density:
                continue
            if signals.citation_density < rule.min_citation_density:
                continue
            return rule.strategy

        if doc_type in {DocumentType.LEGAL, DocumentType.CONTRACT, DocumentType.POLICY}:
            return ChunkingStrategy.LEGAL_CLAUSE_AWARE
        if doc_type in {DocumentType.SOURCE_CODE}:
            return ChunkingStrategy.CODE_AWARE
        if doc_type in {DocumentType.FAQ, DocumentType.KNOWLEDGE_ARTICLE}:
            return ChunkingStrategy.FAQ_AWARE
        if doc_type in {DocumentType.EMAIL, DocumentType.CHAT_TRANSCRIPT}:
            return ChunkingStrategy.CHAT_AWARE
        if signals.table_density > 0.25:
            return ChunkingStrategy.TABLE_AWARE
        if signals.heading_frequency > 0.15:
            return ChunkingStrategy.SECTION_AWARE
        if signals.avg_paragraph_length > 120:
            return ChunkingStrategy.SEMANTIC
        return config.strategy
