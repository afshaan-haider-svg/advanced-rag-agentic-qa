"""Pydantic schemas and models for document ingestion, registry, and management."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for an ingested document in the registry."""

    document_id: str = Field(..., description="Unique UUID for the document")
    filename: str = Field(..., description="Original filename uploaded by the user")
    file_type: str = Field(..., description="Normalized file extension (.pdf, .txt, .docx)")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    upload_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of upload",
    )
    total_pages: int = Field(default=1, description="Total pages or structural items in document")
    num_chunks: int = Field(default=0, description="Number of text chunks created and indexed")
    status: str = Field(
        default="processing",
        description="Current processing status: processing | indexed | failed",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Details of any processing failure",
    )


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful document upload and indexing."""

    document_id: str = Field(..., description="Unique UUID assigned to the document")
    filename: str = Field(..., description="Name of the uploaded file")
    file_type: str = Field(..., description="Detected file type")
    total_pages: int = Field(..., description="Total pages or sections extracted")
    num_chunks: int = Field(..., description="Total vector chunks generated and indexed")
    status: str = Field(..., description="Ingestion status")
    message: str = Field(..., description="Human-readable status message")


class DocumentListResponse(BaseModel):
    """Response containing list of all registered documents."""

    documents: List[DocumentMetadata] = Field(
        default_factory=list,
        description="List of registered documents",
    )
    total: int = Field(..., description="Total count of documents in registry")


class DocumentDeleteResponse(BaseModel):
    """Response returned after deleting a document."""

    document_id: str = Field(..., description="ID of the deleted document")
    filename: str = Field(..., description="Filename of the deleted document")
    status: str = Field(default="deleted", description="Deletion status")
    deleted_chunks: int = Field(default=0, description="Number of vector chunks removed")
    message: str = Field(..., description="Human-readable deletion result message")


class DocumentChunk(BaseModel):
    """Representation of an individual text chunk ready for embedding."""

    chunk_id: str = Field(..., description="Unique ID for this chunk (document_id_chunk_N)")
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Parent document filename")
    page: Union[int, str] = Field(default=1, description="Page number or section")
    chunk_index: int = Field(..., description="Zero-based index of this chunk")
    content: str = Field(..., description="Raw text content of chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Complete chunk metadata")
