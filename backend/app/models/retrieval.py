"""Pydantic schemas and models for retrieval, re-ranking, context building, and citations."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Document chunk candidate retrieved from vector store."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original source document filename")
    page: Union[int, str] = Field(default=1, description="Page number or section")
    chunk_index: int = Field(..., description="Original index within document")
    content: str = Field(..., description="Text content of the chunk")
    source: str = Field(default="", description="Filesystem source path")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    distance: float = Field(..., description="Raw vector distance from ChromaDB")
    initial_rank: int = Field(..., description="Rank before re-ranking (1-based)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Complete chunk metadata")


class RerankedChunk(BaseModel):
    """Document chunk scored and ordered by the Cross-Encoder."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original source document filename")
    page: Union[int, str] = Field(default=1, description="Page number or section")
    chunk_index: int = Field(..., description="Original index within document")
    content: str = Field(..., description="Text content of the chunk")
    source: str = Field(default="", description="Filesystem source path")
    original_vector_score: float = Field(
        ..., description="Original vector similarity score before re-ranking"
    )
    reranker_score: float = Field(..., description="Cross-encoder relevance logit/score")
    initial_rank: int = Field(..., description="Vector similarity rank before re-ranking")
    reranked_rank: int = Field(..., description="Final rank after cross-encoder scoring")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Complete chunk metadata")


class Citation(BaseModel):
    """Source citation object mapping text chunks to original document locations."""

    citation_number: int = Field(..., description="Numbered index of citation [1], [2], etc.")
    document_id: str = Field(..., description="Referenced document UUID")
    filename: str = Field(..., description="Referenced document filename")
    page: Union[int, str] = Field(default=1, description="Page or section number")
    chunk_index: int = Field(..., description="Referenced chunk index")
    formatted_citation: str = Field(..., description="Formatted string e.g. '[1] doc.pdf — Page 4'")
    source: str = Field(default="", description="Source file path")


class ContextResult(BaseModel):
    """Optimized context package prepared for downstream reasoning/synthesis."""

    context_text: str = Field(..., description="Deduplicated and structured context string")
    selected_chunks: List[RerankedChunk] = Field(..., description="Chunks included in context")
    total_characters: int = Field(..., description="Total character length of context text")
    estimated_tokens: int = Field(..., description="Heuristic token estimation (~chars / 4)")


class RagSearchRequest(BaseModel):
    """Request schema for RAG retrieval and re-ranking debug endpoint."""

    query: str = Field(..., description="Natural language search query")
    top_k: Optional[int] = Field(
        default=None, description="Number of candidate vector chunks to retrieve"
    )
    similarity_threshold: Optional[float] = Field(
        default=None, description="Minimum vector similarity threshold"
    )
    rerank_top_n: Optional[int] = Field(
        default=None, description="Number of top chunks to retain after re-ranking"
    )
    use_mmr: bool = Field(
        default=True, description="Whether to apply MMR diversity selection"
    )
    mmr_lambda: Optional[float] = Field(
        default=None, description="MMR trade-off between relevance and diversity (0.0 to 1.0)"
    )


class RagSearchResponse(BaseModel):
    """Complete retrieval, re-ranking, and context debug response."""

    query: str = Field(..., description="Original raw query")
    normalized_query: str = Field(..., description="Sanitized and normalized query")
    retrieved_candidates: List[RetrievedChunk] = Field(
        ..., description="Candidate chunks retrieved by semantic search"
    )
    reranked_documents: List[RerankedChunk] = Field(
        ..., description="All candidates ordered by cross-encoder scores"
    )
    selected_documents: List[RerankedChunk] = Field(
        ..., description="Final subset selected for LLM context"
    )
    context: str = Field(..., description="Optimized context string")
    citations: List[Citation] = Field(
        ..., description="Exact citations for all selected chunks"
    )
    context_characters: int = Field(..., description="Character count of generated context")
    estimated_tokens: int = Field(..., description="Token estimate for generated context")
    execution_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Latencies and filtering statistics"
    )
