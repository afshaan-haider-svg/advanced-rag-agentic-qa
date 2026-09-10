"""Semantic vector retriever using persistent ChromaDB and diversity filtering."""

import time
from typing import List, Optional, Tuple

try:
    from backend.app.core.config import settings
    from backend.app.core.logging import get_logger
    from backend.app.models.retrieval import RetrievedChunk
    from backend.app.rag.embeddings import get_embeddings_client
    from backend.app.rag.mmr import maximal_marginal_relevance
    from backend.app.rag.vector_store import get_chunks_collection
except ImportError:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.models.retrieval import RetrievedChunk
    from app.rag.embeddings import get_embeddings_client
    from app.rag.mmr import maximal_marginal_relevance
    from app.rag.vector_store import get_chunks_collection

logger = get_logger(__name__)


def retrieve_from_chroma(
    query: str,
    top_k: int,
    similarity_threshold: float,
) -> Tuple[List[RetrievedChunk], List[List[float]]]:
    """Retrieve nearest neighbor document chunks from ChromaDB.

    Calculates cosine similarity from distance and filters candidates by threshold.
    Returns list of candidate chunks and their corresponding embedding vectors.
    """
    collection = get_chunks_collection()
    total_docs = collection.count()

    if total_docs == 0:
        logger.info("Chroma collection '%s' is empty. Returning 0 candidates.", settings.CHROMA_COLLECTION_NAME)
        return [], []

    fetch_k = min(top_k, total_docs)
    embeddings_client = get_embeddings_client()
    query_embedding = embeddings_client.embed_query(query)

    start_time = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    query_latency_ms = (time.time() - start_time) * 1000.0

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    embeddings_raw = results.get("embeddings")
    embeddings = embeddings_raw[0] if embeddings_raw is not None and len(embeddings_raw) > 0 else None

    candidates: List[RetrievedChunk] = []
    retained_embeddings: List[List[float]] = []

    for idx, (chunk_id, doc_text, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances)
    ):
        # In Chroma with cosine space, distance = 1 - cosine_similarity
        # Clamp similarity between 0.0 and 1.0
        similarity = max(0.0, min(1.0, 1.0 - float(dist)))

        if similarity < similarity_threshold:
            logger.debug(
                "Filtered out candidate '%s' (similarity %.4f < threshold %.4f)",
                chunk_id,
                similarity,
                similarity_threshold,
            )
            continue

        chunk = RetrievedChunk(
            chunk_id=str(chunk_id),
            document_id=str(meta.get("document_id", "")),
            filename=str(meta.get("filename", "unknown")),
            page=meta.get("page", 1),
            chunk_index=int(meta.get("chunk_index", idx)),
            content=str(doc_text),
            source=str(meta.get("source", "")),
            similarity_score=round(similarity, 4),
            distance=round(float(dist), 4),
            initial_rank=len(candidates) + 1,
            metadata=meta or {},
        )
        candidates.append(chunk)

        if embeddings is not None and len(embeddings) > idx:
            emb_item = embeddings[idx]
            retained_embeddings.append(emb_item.tolist() if hasattr(emb_item, "tolist") else emb_item)

    logger.info(
        "Semantic search completed in %.2f ms. Retrieved %d candidates after threshold filtering (threshold=%.2f)",
        query_latency_ms,
        len(candidates),
        similarity_threshold,
    )
    return candidates, retained_embeddings


def retrieve_candidates(
    query: str,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    use_mmr: bool = True,
    mmr_lambda: Optional[float] = None,
) -> List[RetrievedChunk]:
    """Execute complete candidate retrieval pipeline with optional MMR diversity selection."""
    effective_top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
    effective_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.SIMILARITY_THRESHOLD
    )
    effective_lambda = mmr_lambda if mmr_lambda is not None else settings.MMR_LAMBDA

    # When MMR is enabled, fetch a wider pool of candidates first
    pool_size = max(effective_top_k, settings.MMR_CANDIDATES) if use_mmr else effective_top_k

    candidates, candidate_embeddings = retrieve_from_chroma(
        query=query,
        top_k=pool_size,
        similarity_threshold=effective_threshold,
    )

    if not candidates:
        return []

    if use_mmr and len(candidates) > effective_top_k and len(candidate_embeddings) > 0:
        query_embedding = get_embeddings_client().embed_query(query)
        selected_candidates = maximal_marginal_relevance(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidates=candidates,
            top_n=effective_top_k,
            lambda_mult=effective_lambda,
        )
        # Update initial ranks sequentially
        for rank, c in enumerate(selected_candidates, start=1):
            c.initial_rank = rank
        return selected_candidates

    return candidates[:effective_top_k]
