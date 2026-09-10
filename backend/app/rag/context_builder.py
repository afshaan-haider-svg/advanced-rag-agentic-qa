"""Context optimization, deduplication, and budget management."""

import hashlib
import re
from typing import List, Optional, Set
from pydantic import BaseModel

try:
    from backend.app.core.config import settings
    from backend.app.core.logging import get_logger
    from backend.app.models.retrieval import ContextResult, RerankedChunk
except ImportError:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.models.retrieval import ContextResult, RerankedChunk

logger = get_logger(__name__)


def compute_text_hash(text: str) -> str:
    """Compute sha256 hash of normalized text."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_word_jaccard(text1: str, text2: str) -> float:
    """Compute word-level Jaccard similarity between two texts."""
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def build_optimized_context(
    reranked_chunks: List[RerankedChunk],
    max_chunks: Optional[int] = None,
    max_characters: Optional[int] = None,
    near_duplicate_threshold: float = 0.70,
) -> ContextResult:
    """Filter, deduplicate, and compile chunks into an optimized context package.

    Enforces:
    1. Exact duplicate removal via text hash.
    2. Near-duplicate removal using Jaccard word similarity.
    3. Maximum chunk count limit (RERANK_TOP_N).
    4. Strict character budget constraint (CONTEXT_MAX_CHARS).
    5. Deterministic rank ordering with metadata headers.
    """
    limit_chunks = max_chunks if max_chunks is not None else settings.RERANK_TOP_N
    char_budget = max_characters if max_characters is not None else settings.CONTEXT_MAX_CHARS

    selected_chunks: List[RerankedChunk] = []
    seen_hashes: Set[str] = set()
    accepted_texts: List[str] = []
    formatted_blocks: List[str] = []
    current_char_count = 0

    for chunk in reranked_chunks:
        if len(selected_chunks) >= limit_chunks:
            break

        # 1. Exact duplicate check
        text_hash = compute_text_hash(chunk.content)
        if text_hash in seen_hashes:
            logger.debug("Skipping exact duplicate chunk: %s", chunk.chunk_id)
            continue

        # 2. Near-duplicate check against previously accepted chunks
        is_near_dup = False
        for accepted in accepted_texts:
            if compute_word_jaccard(chunk.content, accepted) >= near_duplicate_threshold:
                logger.debug(
                    "Skipping near-duplicate chunk %s (high Jaccard overlap)", chunk.chunk_id
                )
                is_near_dup = True
                break

        if is_near_dup:
            continue

        # 3. Format block with source metadata header
        header = f"[Source: {chunk.filename} | Page: {chunk.page} | Chunk: {chunk.chunk_index}]"
        block = f"{header}\n{chunk.content}"
        block_length = len(block) + (len("\n\n---\n\n") if formatted_blocks else 0)

        # 4. Context character budget check
        if current_char_count + block_length > char_budget:
            # Check if we can fit a truncated version or if budget is reached
            remaining_budget = char_budget - current_char_count - len(header) - 10
            if remaining_budget > 150:
                truncated_content = chunk.content[:remaining_budget] + "... [truncated]"
                block = f"{header}\n{truncated_content}"
                formatted_blocks.append(block)
                selected_chunks.append(chunk)
                current_char_count += len(block)
                logger.info(
                    "Chunk %s truncated to fit remaining character budget.", chunk.chunk_id
                )
            else:
                logger.info("Context character budget (%d) reached. Halting chunk inclusion.", char_budget)
            break

        formatted_blocks.append(block)
        selected_chunks.append(chunk)
        seen_hashes.add(text_hash)
        accepted_texts.append(chunk.content)
        current_char_count += block_length

    full_context = "\n\n---\n\n".join(formatted_blocks)
    estimated_tokens = max(1, len(full_context) // 4)

    logger.info(
        "Context optimized: %d chunks selected, %d characters, ~%d tokens",
        len(selected_chunks),
        len(full_context),
        estimated_tokens,
    )

    return ContextResult(
        context_text=full_context,
        selected_chunks=selected_chunks,
        total_characters=len(full_context),
        estimated_tokens=estimated_tokens,
    )
