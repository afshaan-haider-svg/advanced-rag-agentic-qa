"""Document ingestion, registry management, and lifecycle service."""

import json
from pathlib import Path
import shutil
import threading
from typing import Dict, List, Optional
import uuid
from fastapi import HTTPException, UploadFile, status

try:
    from backend.app.core.config import settings
    from backend.app.core.logging import get_logger
    from backend.app.models.document import (
        DocumentDeleteResponse,
        DocumentListResponse,
        DocumentMetadata,
        DocumentUploadResponse,
    )
    from backend.app.rag.chunking import chunk_documents
    from backend.app.rag.cleaning import clean_documents
    from backend.app.rag.loaders import load_document
    from backend.app.rag.vector_store import delete_document_chunks, index_chunks
except ImportError:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.models.document import (
        DocumentDeleteResponse,
        DocumentListResponse,
        DocumentMetadata,
        DocumentUploadResponse,
    )
    from app.rag.chunking import chunk_documents
    from app.rag.cleaning import clean_documents
    from app.rag.loaders import load_document
    from app.rag.vector_store import delete_document_chunks, index_chunks

logger = get_logger(__name__)

_registry_lock = threading.Lock()


class DocumentRegistry:
    """Thread-safe JSON-backed document registry."""

    def __init__(self, file_path: Path):
        self.file_path = file_path.resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._save({})

    def _load(self) -> Dict[str, Dict]:
        with _registry_lock:
            if not self.file_path.exists():
                return {}
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to load document registry from %s: %s", self.file_path, e)
                return {}

    def _save(self, data: Dict[str, Dict]) -> None:
        with _registry_lock:
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error("Failed to save document registry to %s: %s", self.file_path, e)

    def get_all(self) -> List[DocumentMetadata]:
        data = self._load()
        return [DocumentMetadata(**item) for item in data.values()]

    def get_by_id(self, document_id: str) -> Optional[DocumentMetadata]:
        data = self._load()
        record = data.get(document_id)
        return DocumentMetadata(**record) if record else None

    def upsert(self, metadata: DocumentMetadata) -> None:
        data = self._load()
        data[metadata.document_id] = metadata.model_dump()
        self._save(data)

    def delete(self, document_id: str) -> bool:
        data = self._load()
        if document_id in data:
            del data[document_id]
            self._save(data)
            return True
        return False


registry = DocumentRegistry(settings.REGISTRY_FILE_PATH)


class DocumentService:
    """Service handling ingestion, parsing, chunking, indexing, and deletion."""

    def __init__(self):
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def validate_file(self, filename: Optional[str]) -> str:
        """Validate filename and ensure extension is supported."""
        if not filename or not filename.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename cannot be empty.",
            )

        suffix = Path(filename).suffix.lower()
        if suffix not in settings.SUPPORTED_EXTENSIONS:
            supported_str = ", ".join(sorted(settings.SUPPORTED_EXTENSIONS))
            logger.warning(
                "Rejected upload with unsupported extension '%s'. Supported: %s",
                suffix,
                supported_str,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file format '{suffix}'. "
                    f"Supported file types are: {supported_str}."
                ),
            )
        return suffix

    async def save_upload_file(self, upload_file: UploadFile, target_path: Path) -> int:
        """Stream upload file to disk and return byte count."""
        size = 0
        with open(target_path, "wb") as f:
            while chunk := await upload_file.read(1024 * 1024):  # 1MB chunks
                size += len(chunk)
                if size > settings.MAX_UPLOAD_SIZE_BYTES:
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"File size exceeds maximum allowed limit of "
                            f"{settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
                        ),
                    )
                f.write(chunk)
        return size

    async def process_and_index_document(self, upload_file: UploadFile) -> DocumentUploadResponse:
        """Orchestrate the end-to-end ingestion pipeline for an uploaded file."""
        original_filename = Path(upload_file.filename or "unknown").name
        file_extension = self.validate_file(original_filename)
        document_id = str(uuid.uuid4())

        # Save file to disk
        safe_filename = f"{document_id}_{original_filename}"
        saved_path = settings.UPLOAD_DIR / safe_filename

        logger.info(
            "Starting ingestion for '%s' (assigned ID: %s)",
            original_filename,
            document_id,
        )

        # Initialize registry entry with status 'processing'
        doc_meta = DocumentMetadata(
            document_id=document_id,
            filename=original_filename,
            file_type=file_extension,
            status="processing",
        )
        registry.upsert(doc_meta)

        try:
            # 1. Save file to disk
            file_size = await self.save_upload_file(upload_file, saved_path)
            doc_meta.file_size_bytes = file_size

            # 2. Document Loading
            raw_documents = load_document(saved_path, document_id, original_filename)
            if not raw_documents:
                raise ValueError("No extractable content found in document.")

            # 3. Document Cleaning & Normalization
            cleaned_documents = clean_documents(raw_documents)
            if not cleaned_documents:
                raise ValueError("Document contains only blank or unprocessable content.")

            total_pages = len(cleaned_documents)
            doc_meta.total_pages = total_pages

            # 4. Chunking
            chunks = chunk_documents(cleaned_documents)
            if not chunks:
                raise ValueError("Could not generate text chunks from cleaned content.")

            num_chunks = len(chunks)
            doc_meta.num_chunks = num_chunks

            # 5 & 6. Embeddings and Vector Store Indexing
            indexed_count = index_chunks(chunks)
            logger.info(
                "Indexed %d chunks in ChromaDB for document '%s'",
                indexed_count,
                original_filename,
            )

            # 7. Update Registry
            doc_meta.status = "indexed"
            registry.upsert(doc_meta)

            return DocumentUploadResponse(
                document_id=document_id,
                filename=original_filename,
                file_type=file_extension,
                total_pages=total_pages,
                num_chunks=num_chunks,
                status="indexed",
                message=f"Document '{original_filename}' successfully processed and indexed into vector store.",
            )

        except HTTPException:
            saved_path.unlink(missing_ok=True)
            doc_meta.status = "failed"
            registry.upsert(doc_meta)
            raise

        except Exception as e:
            logger.exception("Ingestion failed for document '%s': %s", original_filename, e)
            saved_path.unlink(missing_ok=True)
            doc_meta.status = "failed"
            doc_meta.error_message = str(e)
            registry.upsert(doc_meta)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process and index document '{original_filename}'. Please check file integrity.",
            )

    def list_documents(self) -> DocumentListResponse:
        """List all registered documents."""
        docs = registry.get_all()
        return DocumentListResponse(documents=docs, total=len(docs))

    def delete_document(self, document_id: str) -> DocumentDeleteResponse:
        """Delete document from registry, Chroma vector store, and filesystem."""
        doc_meta = registry.get_by_id(document_id)
        if not doc_meta:
            logger.warning("Attempted to delete non-existent document ID: %s", document_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' does not exist.",
            )

        logger.info("Deleting document ID: %s ('%s')", document_id, doc_meta.filename)

        # 1. Delete chunks from ChromaDB
        deleted_chunks = delete_document_chunks(document_id)

        # 2. Delete local uploaded file
        safe_filename = f"{document_id}_{doc_meta.filename}"
        file_path = settings.UPLOAD_DIR / safe_filename
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info("Deleted local file: %s", file_path)
            except Exception as e:
                logger.warning("Could not delete file %s: %s", file_path, e)

        # 3. Delete from registry
        registry.delete(document_id)

        return DocumentDeleteResponse(
            document_id=document_id,
            filename=doc_meta.filename,
            status="deleted",
            deleted_chunks=deleted_chunks,
            message=f"Document '{doc_meta.filename}' (ID: {document_id}) and its {deleted_chunks} vector chunks were successfully deleted.",
        )


document_service = DocumentService()
