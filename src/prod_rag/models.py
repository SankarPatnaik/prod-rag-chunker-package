from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChunkingStrategy(str, Enum):
    SENTENCE_AWARE = "sentence_aware"
    SECTION_AWARE = "section_aware"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    TABLE_AWARE = "table_aware"
    LEGAL_CLAUSE_AWARE = "legal_clause_aware"
    FAQ_AWARE = "faq_aware"
    CODE_AWARE = "code_aware"
    CHAT_AWARE = "chat_aware"


class DocumentType(str, Enum):
    LEGAL = "legal"
    POLICY = "policy"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    EMAIL = "email"
    MARKDOWN = "markdown"
    WIKI = "wiki"
    PDF_EXTRACTED = "pdf_extracted"
    TABULAR = "tabular"
    KNOWLEDGE_ARTICLE = "knowledge_article"
    SOP = "sop"
    CONTRACT = "contract"
    SOURCE_CODE = "source_code"
    CHAT_TRANSCRIPT = "chat_transcript"
    FAQ = "faq"
    GENERIC = "generic"


class DocumentSection(BaseModel):
    section_id: str
    heading_path: List[str] = Field(default_factory=list)
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    block_type: str = "body"
    char_start: int = 0
    char_end: int = 0
    hierarchy_level: int = 0


class ChunkLineage(BaseModel):
    source: str
    document_id: str
    section_id: str
    parent_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class Chunk(BaseModel):
    chunk_id: str
    parent_section_id: str
    document_id: str
    heading_path: List[str] = Field(default_factory=list)
    text: str
    normalized_text: Optional[str] = None
    section_title: Optional[str] = None
    section_path: str = ""
    token_count: int
    sentence_count: int = 0
    sequence_no: int
    chunk_index: int = 0
    parent_chunk_id: Optional[str] = None
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: int = 0
    char_end: int = 0
    overlap_prev_chars: int = 0
    overlap_next_chars: int = 0
    block_type: str = "body"
    document_type: DocumentType = DocumentType.GENERIC
    strategy_used: str = ChunkingStrategy.HIERARCHICAL.value
    semantic_keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    table_flag: bool = False
    code_flag: bool = False
    citation_flag: bool = False
    confidence_score: float = 0.0
    context_preservation_score: float = 0.0
    embedding_model: Optional[str] = None
    created_at: str = ""
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


class StrategyRule(BaseModel):
    document_type: DocumentType
    min_structural_richness: float = 0.0
    min_heading_frequency: float = 0.0
    min_table_density: float = 0.0
    min_code_density: float = 0.0
    min_citation_density: float = 0.0
    strategy: ChunkingStrategy


class ChunkingConfig(BaseModel):
    model_name: str = "small-model"
    strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
    enable_dynamic_strategy_selection: bool = True
    max_input_tokens: int = 4096
    reserved_prompt_tokens: int = 700
    reserved_output_tokens: int = 512
    target_chunk_tokens: int = 700
    min_chunk_tokens: int = 120
    overlap_tokens: int = 90
    parent_chunk_target_tokens: int = 1800
    hard_max_chunk_tokens: Optional[int] = None
    semantic_similarity_threshold: float = 0.72
    enable_table_isolation: bool = True
    preserve_section_boundaries: bool = True
    context_window_sentences: int = 2
    include_parent_summary_hint: bool = True
    enable_entity_extraction: bool = True
    enable_keywords: bool = True
    max_keywords: int = 10
    strategy_rules: List[StrategyRule] = Field(default_factory=list)

    @field_validator("overlap_tokens")
    @classmethod
    def validate_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("overlap_tokens cannot be negative")
        return value


class LoaderConfig(BaseModel):
    type: Literal["auto", "pdf", "docx", "text", "html", "markdown"] = "auto"


class EmbedderConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 16


class VectorBackendType(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"
    ELASTIC = "elastic"


class VectorStoreConfig(BaseModel):
    type: VectorBackendType = VectorBackendType.FAISS
    path: str = ".rag_index"
    collection_name: str = "prod_rag_chunks"
    host: str = "localhost"
    port: int = 6333
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/postgres"
    postgres_table: str = "rag_chunks"
    elastic_url: str = "http://localhost:9200"
    elastic_index: str = "prod_rag_chunks"


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


class ChunkRequest(BaseModel):
    document_id: Optional[str] = None
    source: Optional[str] = None
    text: Optional[str] = None
    strategy: Optional[ChunkingStrategy] = None


class ChunkMetrics(BaseModel):
    total_chunks: int
    avg_tokens: float
    max_tokens: int
    min_tokens: int


class ChunkResponse(BaseModel):
    document_id: str
    child_chunks: List[Chunk]
    parent_chunks: List[ParentChunk]
    sections: List[DocumentSection]
    stats: Dict[str, Any] = Field(default_factory=dict)
    metrics: ChunkMetrics
