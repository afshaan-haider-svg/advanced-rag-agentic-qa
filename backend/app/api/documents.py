"""API endpoints for document ingestion, listing, and deletion."""

from fastapi import APIRouter, File, Path as PathParam, UploadFile, status

try:
    from backend.app.core.logging import get_logger
    from backend.app.models.document import (
        DocumentDeleteResponse,
        DocumentListResponse,
        DocumentUploadResponse,
    )
    from backend.app.services.document_service import document_service
except ImportError:
    from app.core.logging import get_logger
    from app.models.document import (
        DocumentDeleteResponse,
        DocumentListResponse,
        DocumentUploadResponse,
    )
    from app.services.document_service import document_service

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Index Document",
    description=(
        "Upload a document (PDF, TXT, or DOCX) to be cleaned, chunked, embedded, "
        "and indexed into the persistent Chroma vector store."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (.pdf, .txt, .docx)")
) -> DocumentUploadResponse:
    """Handle document upload and initiate ingestion pipeline."""
    return await document_service.process_and_index_document(file)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List All Documents",
    description="Retrieve metadata for all uploaded and indexed documents in the registry.",
)
async def list_documents() -> DocumentListResponse:
    """Return all registered documents."""
    return document_service.list_documents()


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete Document",
    description=(
        "Delete a document by ID, removing its metadata, vector chunks from ChromaDB, "
        "and local uploaded file."
    ),
)
async def delete_document(
    document_id: str = PathParam(..., description="Unique ID of document to delete")
) -> DocumentDeleteResponse:
    """Delete a document and its associated vectors and storage."""
    return document_service.delete_document(document_id)
