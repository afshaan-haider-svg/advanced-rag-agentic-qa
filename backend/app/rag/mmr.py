"""Maximal Marginal Relevance (MMR) for diversity-aware document selection."""

from typing import List, Sequence, Tuple
import numpy as np

try:
    from backend.app.core.logging import get_logger
    from backend.app.models.retrieval import RetrievedChunk
except ImportError:
    from app.core.logging import get_logger
    from app.models.retrieval import RetrievedChunk

logger = get_logger(__name__)


def maximal_marginal_relevance(
    query_embedding: Sequence[float],
    candidate_embeddings: Sequence[Sequence[float]],
    candidates: List[RetrievedChunk],
    top_n: int = 4,
    lambda_mult: float = 0.7,
) -> List[RetrievedChunk]:
    """Select a diverse and relevant subset of candidates using Maximal Marginal Relevance.

    Formula:
        MMR = argmax_{d_i in R \\ S} [ lambda * sim(d_i, q) - (1 - lambda) * max_{d_j in S} sim(d_i, d_j) ]

    Args:
        query_embedding: Vector embedding of the query.
        candidate_embeddings: 2D array of embeddings for candidate chunks.
        candidates: List of RetrievedChunk metadata corresponding to candidate_embeddings.
        top_n: Number of diverse candidates to return.
        lambda_mult: Diversity trade-off (1.0 = pure semantic similarity, 0.0 = maximal diversity).

    Returns:
        List of selected RetrievedChunk objects with diversity applied.
    """
    if not candidates or not candidate_embeddings:
        return []

    if len(candidates) <= top_n:
        return list(candidates)

    q_vec = np.array(query_embedding, dtype=np.float32)
    cand_matrix = np.array(candidate_embeddings, dtype=np.float32)

    # Normalize vectors to ensure cosine similarity via dot product
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    cand_norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
    cand_norms[cand_norms == 0] = 1.0
    cand_matrix = cand_matrix / cand_norms

    # Similarity to query for all candidates
    query_similarities = np.dot(cand_matrix, q_vec)

    # Precompute candidate-candidate pairwise cosine similarity matrix
    candidate_similarities = np.dot(cand_matrix, cand_matrix.T)

    selected_indices: List[int] = []
    unselected_indices: List[int] = list(range(len(candidates)))

    # Select the single most relevant chunk first
    first_idx = int(np.argmax(query_similarities))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)

    target_count = min(top_n, len(candidates))

    while len(selected_indices) < target_count and unselected_indices:
        best_score = -float("inf")
        best_idx = None

        for unselected in unselected_indices:
            relevance = query_similarities[unselected]
            # Max similarity to any already selected candidate
            max_diversity_penalty = np.max(candidate_similarities[unselected, selected_indices])

            mmr_score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * max_diversity_penalty)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = unselected

        if best_idx is not None:
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
        else:
            break

    selected_chunks = [candidates[i] for i in selected_indices]
    logger.debug(
        "MMR selected %d chunks from %d candidates (lambda=%.2f)",
        len(selected_chunks),
        len(candidates),
        lambda_mult,
    )
    return selected_chunks
