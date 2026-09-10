"""Document text cleaning, sanitization, and metadata normalization."""

import hashlib
import re
import unicodedata
from typing import List, Set
from langchain_core.documents import Document

try:
    from backend.app.core.logging import get_logger
except ImportError:
    from app.core.logging import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace, sanitize control characters, and clean broken unicode."""
    if not text:
        return ""

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove null bytes and non-printable control characters (except newline, carriage return, tab)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Standardize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace runs of horizontal whitespace (spaces, tabs) with a single space
    lines = text.split("\n")
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    # Rejoin lines
    text = "\n".join(cleaned_lines)

    # Collapse 3 or more consecutive newlines into 2 (preserves paragraph separation)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def compute_content_hash(text: str) -> str:
    """Compute sha256 hash of normalized text for duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_documents(documents: List[Document]) -> List[Document]:
    """Preprocess a list of documents: remove empty pages, duplicates, and clean text."""
    cleaned_docs: List[Document] = []
    seen_hashes: Set[str] = set()

    for doc in documents:
        cleaned_content = clean_text(doc.page_content)

        # Skip empty or whitespace-only pages
        if not cleaned_content:
            logger.debug(
                "Skipping empty page %s from %s",
                doc.metadata.get("page", "unknown"),
                doc.metadata.get("filename", "unknown"),
            )
            continue

        # Check for duplicate page content within the same document
        content_hash = compute_content_hash(cleaned_content)
        if len(cleaned_content) > 50 and content_hash in seen_hashes:
            logger.info(
                "Skipping duplicate content page %s in %s",
                doc.metadata.get("page", "unknown"),
                doc.metadata.get("filename", "unknown"),
            )
            continue

        seen_hashes.add(content_hash)

        # Normalize metadata
        normalized_metadata = {
            "document_id": str(doc.metadata.get("document_id", "")),
            "filename": str(doc.metadata.get("filename", "unknown")),
            "file_type": str(doc.metadata.get("file_type", "")),
            "page": int(doc.metadata.get("page", 1)),
            "source": str(doc.metadata.get("source", "")),
        }

        cleaned_docs.append(
            Document(page_content=cleaned_content, metadata=normalized_metadata)
        )

    logger.info(
        "Cleaned %d input pages down to %d valid documents",
        len(documents),
        len(cleaned_docs),
    )
    return cleaned_docs
