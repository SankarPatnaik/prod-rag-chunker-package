from __future__ import annotations

from prod_rag.chunking.pipeline import ChunkingPipeline
from prod_rag.models import ChunkingConfig, ChunkingStrategy


def test_chunking_pipeline_basic():
    text = "TITLE\n\nThis is sentence one. This is sentence two. This is sentence three."
    pipeline = ChunkingPipeline(
        ChunkingConfig(
            target_chunk_tokens=12,
            reserved_prompt_tokens=10,
            reserved_output_tokens=10,
            max_input_tokens=512,
            overlap_tokens=4,
        )
    )
    result = pipeline.chunk_document(text, "doc1")
    assert result.stats["num_child_chunks"] >= 1
    assert result.child_chunks[0].document_id == "doc1"


def test_table_aware_chunking_preserves_table_header():
    text = "Revenue | 2023 | 2024\nRetail | 10 | 11\nEnterprise | 20 | 25"
    pipeline = ChunkingPipeline(
        ChunkingConfig(
            strategy=ChunkingStrategy.TABLE_AWARE,
            enable_dynamic_strategy_selection=False,
            target_chunk_tokens=10,
            reserved_prompt_tokens=10,
            reserved_output_tokens=10,
            max_input_tokens=512,
        )
    )
    result = pipeline.chunk_document(text, "finance-table")
    assert result.child_chunks
    assert "Revenue" in result.child_chunks[0].text


def test_semantic_strategy_merges_related_chunks():
    text = (
        "Cash and cash equivalents increased in Q4. "
        "Cash and cash equivalents increased due to refinancing. "
        "Unrelated sentence about weather forecasts."
    )
    pipeline = ChunkingPipeline(
        ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            enable_dynamic_strategy_selection=False,
            target_chunk_tokens=14,
            reserved_prompt_tokens=10,
            reserved_output_tokens=10,
            max_input_tokens=512,
            semantic_similarity_threshold=0.3,
        )
    )
    result = pipeline.chunk_document(text, "doc2")
    assert result.stats["num_child_chunks"] >= 1
    assert result.metrics.max_tokens >= result.metrics.min_tokens


def test_chunk_metadata_contains_lineage_fields():
    text = "# Annual Report\n\nNet Income was stable."
    pipeline = ChunkingPipeline(ChunkingConfig(max_input_tokens=512, reserved_prompt_tokens=10, reserved_output_tokens=10))
    result = pipeline.chunk_document(text, "doc3", source="annual-report.md")
    first = result.child_chunks[0]
    assert first.metadata["source"] == "annual-report.md"
    assert "parent_id" in first.metadata
    assert "token_count" in first.metadata
    assert first.context_preservation_score > 0.0


def test_dynamic_strategy_chooses_legal_clause_aware_for_contract_like_text():
    text = """
MASTER SERVICE AGREEMENT
1. Definitions
1.1 Affiliate means controlled entity.
2. Data Protection
2.1 Provider shall encrypt data at rest.
Annexure A Compliance Controls.
"""
    pipeline = ChunkingPipeline(ChunkingConfig(max_input_tokens=512, reserved_prompt_tokens=10, reserved_output_tokens=10))
    result = pipeline.chunk_document(text, "legal-1", source="msa.txt")
    assert result.stats["document_type"] == "legal"
    assert result.stats["strategy"] == "legal_clause_aware"


def test_faq_detection_routes_to_faq_aware_strategy():
    text = "Q: What is RTO?\nA: Recovery time objective.\nQ: What is RPO?\nA: Recovery point objective."
    pipeline = ChunkingPipeline(ChunkingConfig(max_input_tokens=512, reserved_prompt_tokens=10, reserved_output_tokens=10))
    result = pipeline.chunk_document(text, "faq-1")
    assert result.stats["document_type"] == "faq"
    assert result.stats["strategy"] == "faq_aware"
