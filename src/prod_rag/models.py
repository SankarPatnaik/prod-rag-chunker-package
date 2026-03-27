from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    section_id: str
    heading_path: List[str] = Field(default_factory=list)
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    block_type: str = "body"
    char_start: int = 0
    char_end: int = 0


class Chunk(BaseModel):
    chunk_id: str
    parent_section_id: str
    document_id: str
    heading_path: List[str] = Field(default_factory=list)
    text: str
    token_count: int
    sequence_no: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: int = 0
    char_end: int = 0
    overlap_prev_chars: int = 0
    overlap_next_chars: int = 0
    block_type: str = "body"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParentChunk(BaseModel):
    parent_chunk_id: str
    document_id: str
    heading_path: List[str] = Field(default_factory=list)
    text: str
    token_count: int
    child_chunk_ids: List[str] = Field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LoadedDocument(BaseModel):
    document_id: str
    source_path: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineAnswer(BaseModel):
    query: str
    answer: str
    mapped_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: List[SearchResult] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    model_name: str = "small-model"
    max_input_tokens: int = 4096
    reserved_prompt_tokens: int = 700
    reserved_output_tokens: int = 512
    target_chunk_tokens: int = 700
    min_chunk_tokens: int = 180
    overlap_tokens: int = 90
    parent_chunk_target_tokens: int = 1800
    hard_max_chunk_tokens: Optional[int] = None


class LoaderConfig(BaseModel):
    type: Literal["auto", "pdf", "docx", "text"] = "auto"


class EmbedderConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 16


class VectorStoreConfig(BaseModel):
    type: Literal["faiss", "qdrant"] = "faiss"
    path: str = ".rag_index"
    collection_name: str = "prod_rag_chunks"
    host: str = "localhost"
    port: int = 6333


class LLMConfig(BaseModel):
    backend: Literal["llama_cpp", "vllm"] = "llama_cpp"
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    n_ctx: int = 4096
    gpu_layers: int = 0
    top_p: float = 0.95


class RetrievalConfig(BaseModel):
    top_k: int = 6
    use_parent_expansion: bool = True
    parent_expansion_limit: int = 3
    use_bm25_fallback: bool = True


class AppConfig(BaseModel):
    loader: LoaderConfig = LoaderConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
