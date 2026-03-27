from prod_rag.chunking.pipeline import ChunkingPipeline
from prod_rag.models import ChunkingConfig


def test_chunking_pipeline_basic():
    text = "TITLE\n\nThis is sentence one. This is sentence two. This is sentence three."
    pipeline = ChunkingPipeline(ChunkingConfig(target_chunk_tokens=12, reserved_prompt_tokens=10, reserved_output_tokens=10, max_input_tokens=256))
    result = pipeline.chunk_document(text, "doc1")
    assert result["stats"]["num_child_chunks"] >= 1
    assert result["child_chunks"][0]["document_id"] == "doc1"
