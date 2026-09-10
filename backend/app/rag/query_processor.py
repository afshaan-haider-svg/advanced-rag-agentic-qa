"""Query normalization and preprocessing module."""

import re
import unicodedata

try:
    from backend.app.core.logging import get_logger
except ImportError:
    from app.core.logging import get_logger

logger = get_logger(__name__)


def process_query(raw_query: str) -> str:
    """Validate, clean, and normalize a user query without altering semantic meaning.

    Steps:
    1. Null byte and control character sanitization.
    2. Unicode NFKC normalization.
    3. Collapse multiple whitespace and newline characters to a single space.
    4. Validation: reject empty or whitespace-only queries.
    """
    if raw_query is None:
        raise ValueError("Query cannot be None.")

    # Remove null bytes and non-printable control characters
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", raw_query)

    # Unicode NFKC normalization
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # Collapse consecutive whitespace, tabs, and newlines to a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        logger.warning("Rejected empty or whitespace-only query.")
        raise ValueError("Query cannot be empty or whitespace-only.")

    logger.debug("Processed query: '%s' -> '%s'", raw_query, cleaned)
    return cleaned
