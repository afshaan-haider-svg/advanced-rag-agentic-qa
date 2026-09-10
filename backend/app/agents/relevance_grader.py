"""Retrieval relevance grader."""

from typing import Dict, List


def grade_relevance(selected_documents: List[dict], context: str) -> Dict:
    if not selected_documents or not context.strip():
        return {
            "relevant": False,
            "score": 0.0,
            "reason": "No usable document context was retrieved.",
        }

    scores = []

    for d in selected_documents:
        score = d.get(
            "reranker_score",
            d.get("original_vector_score", 0.0),
        )

        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            pass

    # Convert cross-encoder score into a 0-1 relevance score.
    best = max(scores) if scores else 0.0

    normalized = (
        1.0 / (1.0 + pow(2.718281828, -best))
        if scores
        else 0.0
    )

    # Low relevance should trigger the LangGraph query-rewrite/retry path.
    relevance_threshold = 0.50
    is_relevant = normalized >= relevance_threshold

    if is_relevant:
        reason = (
            "Retrieved context passed the relevance threshold "
            "after semantic retrieval and reranking."
        )
    else:
        reason = (
            "Retrieved context scored below the relevance threshold "
            "and should be rewritten and retrieved again."
        )

    return {
        "relevant": is_relevant,
        "score": round(normalized, 4),
        "reason": reason,
    }
