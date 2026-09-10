"""Cross-Encoder re-ranking module using cross-encoder/ms-marco-MiniLM-L-6-v2."""

import time
from typing import List, Optional
from sentence_transformers import CrossEncoder

try:
    from backend.app.core.config import settings
    from backend.app.core.logging import get_logger
    from backend.app.models.retrieval import RerankedChunk, RetrievedChunk
except ImportError:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.models.retrieval import RerankedChunk, RetrievedChunk

logger = get_logger(__name__)

_cross_encoder_instance: Optional[CrossEncoder] = None


def get_cross_encoder() -> CrossEncoder:
    """Retrieve or initialize the cached singleton CrossEncoder model on CPU."""
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        logger.info(
            "Loading local Cross-Encoder model '%s' on %s...",
            settings.RERANKER_MODEL_NAME,
            settings.RERANKER_DEVICE,
        )
        _cross_encoder_instance = CrossEncoder(
            settings.RERANKER_MODEL_NAME,
            device=settings.RERANKER_DEVICE,
        )
        logger.info("Local Cross-Encoder model loaded successfully.")
    return _cross_encoder_instance


def rerank_chunks(
    query: str,
    candidates: List[RetrievedChunk],
    top_n: Optional[int] = None,
) -> List[RerankedChunk]:
    """Score and reorder candidate chunks using the Cross-Encoder.

    Preserves the original vector similarity score alongside the cross-encoder score
    and tracks both initial and final rank positions.
    """
    if not candidates:
        return []

    model = get_cross_encoder()

    # Build query-document pairs
    sentence_pairs = [[query, chunk.content] for chunk in candidates]

    start_time = time.time()
    scores = model.predict(sentence_pairs, show_progress_bar=False)
    rerank_latency_ms = (time.time() - start_time) * 1000.0

    scored_items = []
    for chunk, score in zip(candidates, scores):
        scored_items.append((chunk, float(score)))

    # Sort in descending order of cross-encoder score
    scored_items.sort(key=lambda x: x[1], reverse=True)

    limit = top_n if top_n is not None else len(scored_items)
    reranked_chunks: List[RerankedChunk] = []

    for rank, (chunk, score) in enumerate(scored_items[:limit], start=1):
        reranked_chunk = RerankedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            source=chunk.source,
            original_vector_score=chunk.similarity_score,
            reranker_score=round(score, 4),
            initial_rank=chunk.initial_rank,
            reranked_rank=rank,
            metadata=chunk.metadata,
        )
        reranked_chunks.append(reranked_chunk)

    logger.info(
        "Re-ranking completed in %.2f ms for %d candidates. Retained %d chunks.",
        rerank_latency_ms,
        len(candidates),
        len(reranked_chunks),
    )
    return reranked_chunks
