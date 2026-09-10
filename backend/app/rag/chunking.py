"""Document chunking using LangChain RecursiveCharacterTextSplitter."""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from backend.app.core.config import settings
    from backend.app.core.logging import get_logger
except ImportError:
    from app.core.config import settings
    from app.core.logging import get_logger

logger = get_logger(__name__)


def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """Create a configured RecursiveCharacterTextSplitter instance.

    Allows overriding chunk parameters for experimental evaluation or specialized workflows.
    """
    effective_chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    effective_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    )

    return RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split documents into chunks while preserving and enriching metadata on each chunk.

    Each output chunk retains:
    - document_id
    - filename
    - page
    - chunk_index
    - chunk_id (unique string identifier)
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_chunks: List[Document] = []
    global_chunk_index = 0

    for doc in documents:
        doc_chunks = splitter.split_text(doc.page_content)
        for sub_idx, chunk_text in enumerate(doc_chunks):
            chunk_metadata = dict(doc.metadata)
            chunk_metadata["chunk_index"] = global_chunk_index
            chunk_metadata["sub_chunk_index"] = sub_idx
            chunk_id = f"{chunk_metadata.get('document_id', 'doc')}_chunk_{global_chunk_index}"
            chunk_metadata["chunk_id"] = chunk_id

            split_chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
            )
            global_chunk_index += 1

    logger.info(
        "Split %d documents into %d chunks (chunk_size=%d, chunk_overlap=%d)",
        len(documents),
        len(split_chunks),
        chunk_size or settings.CHUNK_SIZE,
        chunk_overlap or settings.CHUNK_OVERLAP,
    )
    return split_chunks
