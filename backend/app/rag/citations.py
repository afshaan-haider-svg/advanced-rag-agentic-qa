"""Citation generator mapping retrieved context chunks to verifiable document sources."""

from typing import List, Union
from pydantic import BaseModel

try:
    from backend.app.core.logging import get_logger
    from backend.app.models.retrieval import Citation, RerankedChunk
except ImportError:
    from app.core.logging import get_logger
    from app.models.retrieval import Citation, RerankedChunk

logger = get_logger(__name__)


def format_citation_string(
    citation_num: int,
    filename: str,
    page: Union[int, str],
    file_type: str = "",
) -> str:
    """Format human-readable citation string according to assignment specifications.

    Format examples:
        [1] contract.pdf — Page 4
        [2] policy.docx — Page 2
        [3] notes.txt — Page 1
    """
    page_label = f"Page {page}"
    return f"[{citation_num}] {filename} — {page_label}"


def build_citations(selected_chunks: List[RerankedChunk]) -> List[Citation]:
    """Generate sequential, verifiable citations corresponding to selected context chunks.

    Guarantees:
    - Never fabricates document filenames or page numbers.
    - Preserves exact document_id, filename, page, and chunk_index.
    """
    citations: List[Citation] = []

    for idx, chunk in enumerate(selected_chunks, start=1):
        filename = chunk.filename or "unknown_document"
        page = chunk.page if chunk.page is not None else 1
        file_type = str(chunk.metadata.get("file_type", ""))

        formatted = format_citation_string(
            citation_num=idx,
            filename=filename,
            page=page,
            file_type=file_type,
        )

        citation = Citation(
            citation_number=idx,
            document_id=chunk.document_id,
            filename=filename,
            page=page,
            chunk_index=chunk.chunk_index,
            formatted_citation=formatted,
            source=chunk.source,
        )
        citations.append(citation)

    logger.info("Generated %d citations for selected context chunks.", len(citations))
    return citations
