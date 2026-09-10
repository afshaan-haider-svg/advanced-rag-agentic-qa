"""ChromaDB persistent vector store integration for document chunks."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

try:
    from backend.app.core.config import settings
    from backend.app.core.logging import get_logger
    from backend.app.rag.embeddings import get_embeddings_client
except ImportError:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.rag.embeddings import get_embeddings_client

logger = get_logger(__name__)

_chroma_client: Optional[chromadb.PersistentClient] = None
_chunks_collection: Optional[Collection] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Return or initialize the persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        persist_dir = settings.CHROMA_PERSIST_DIR.resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing ChromaDB PersistentClient at: %s", persist_dir)
        _chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    return _chroma_client


def get_chunks_collection() -> Collection:
    """Return or create the primary document chunks collection."""
    global _chunks_collection
    if _chunks_collection is None:
        client = get_chroma_client()
        logger.info(
            "Accessing Chroma collection '%s' (cosine distance)",
            settings.CHROMA_COLLECTION_NAME,
        )
        _chunks_collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _chunks_collection


def index_chunks(chunks: List[Document]) -> int:
    """Embed and index document chunks into persistent ChromaDB collection.

    Does not recreate the collection, ensuring existing data survives restarts.
    """
    if not chunks:
        logger.warning("No chunks provided for vector indexing.")
        return 0

    collection = get_chunks_collection()
    embeddings_client = get_embeddings_client()

    texts: List[str] = [chunk.page_content for chunk in chunks]
    ids: List[str] = [
        str(
            chunk.metadata.get(
                "chunk_id",
                f"{chunk.metadata.get('document_id', 'doc')}_chunk_{idx}",
            )
        )
        for idx, chunk in enumerate(chunks)
    ]

    # Sanitize metadata values so Chroma accepts only primitives (str, int, float, bool)
    metadatas: List[Dict[str, Any]] = []
    for chunk in chunks:
        clean_meta = {}
        for k, v in chunk.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        metadatas.append(clean_meta)

    logger.info("Generating embeddings for %d chunks...", len(texts))
    embeddings = embeddings_client.embed_documents(texts)

    # Upsert in batches of 100 to handle large documents comfortably
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        collection.upsert(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
        )

    logger.info("Successfully indexed %d chunks in ChromaDB.", len(chunks))
    return len(chunks)


def delete_document_chunks(document_id: str) -> int:
    """Delete all indexed chunks associated with a specific document_id."""
    collection = get_chunks_collection()
    # Check existing count for this document
    existing = collection.get(where={"document_id": document_id})
    existing_ids = existing.get("ids", [])

    if existing_ids:
        collection.delete(ids=existing_ids)
        logger.info(
            "Deleted %d vector chunks for document_id '%s'",
            len(existing_ids),
            document_id,
        )
        return len(existing_ids)

    logger.info("No vector chunks found to delete for document_id '%s'", document_id)
    return 0


def count_document_chunks(document_id: str) -> int:
    """Count number of vector chunks stored for a specific document_id."""
    collection = get_chunks_collection()
    result = collection.get(where={"document_id": document_id})
    return len(result.get("ids", []))


def get_total_chunks() -> int:
    """Return total number of vector chunks currently in the collection."""
    collection = get_chunks_collection()
    return collection.count()
