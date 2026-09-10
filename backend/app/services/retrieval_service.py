"""High-level retrieval service orchestrating the complete Advanced RAG retrieval flow."""

import time
from typing import Any, Dict, List, Optional

try:
    from backend.app.core.config import settings
    from backend.app.core.logging import get_logger
    from backend.app.models.retrieval import (
        ContextResult,
        RagSearchRequest,
        RagSearchResponse,
        RerankedChunk,
        RetrievedChunk,
    )
    from backend.app.rag.citations import build_citations
    from backend.app.rag.context_builder import build_optimized_context
    from backend.app.rag.query_processor import process_query
    from backend.app.rag.reranker import rerank_chunks
    from backend.app.rag.retriever import retrieve_candidates
except ImportError:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.models.retrieval import (
        ContextResult,
        RagSearchRequest,
        RagSearchResponse,
        RerankedChunk,
        RetrievedChunk,
    )
    from app.rag.citations import build_citations
    from app.rag.context_builder import build_optimized_context
    from app.rag.query_processor import process_query
    from app.rag.reranker import rerank_chunks
    from app.rag.retriever import retrieve_candidates

logger = get_logger(__name__)


class RetrievalService:
    """Orchestrates query processing, semantic search, MMR, re-ranking, and context building."""

    def search_and_build_context(self, request: RagSearchRequest) -> RagSearchResponse:
        """Execute full Advanced RAG retrieval pipeline and assemble optimized context."""
        total_start_time = time.time()

        # 1. Query Processing
        t0 = time.time()
        normalized_query = process_query(request.query)
        query_proc_ms = (time.time() - t0) * 1000.0

        # 2 & 3. Semantic Retrieval & MMR Diversity Selection
        t1 = time.time()
        candidates: List[RetrievedChunk] = retrieve_candidates(
            query=normalized_query,
            top_k=request.top_k or settings.RETRIEVAL_TOP_K,
            similarity_threshold=request.similarity_threshold or settings.SIMILARITY_THRESHOLD,
            use_mmr=request.use_mmr,
            mmr_lambda=request.mmr_lambda or settings.MMR_LAMBDA,
        )
        retrieval_ms = (time.time() - t1) * 1000.0

        if not candidates:
            total_ms = (time.time() - total_start_time) * 1000.0
            logger.info("No candidate chunks found for query: '%s'", normalized_query)
            return RagSearchResponse(
                query=request.query,
                normalized_query=normalized_query,
                retrieved_candidates=[],
                reranked_documents=[],
                selected_documents=[],
                context="",
                citations=[],
                context_characters=0,
                estimated_tokens=0,
                execution_stats={
                    "query_processing_ms": round(query_proc_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "reranking_ms": 0.0,
                    "total_latency_ms": round(total_ms, 2),
                    "candidates_retrieved": 0,
                    "candidates_reranked": 0,
                    "chunks_selected": 0,
                },
            )

        # 4. Cross-Encoder Re-ranking
        t2 = time.time()
        reranked_documents: List[RerankedChunk] = rerank_chunks(
            query=normalized_query,
            candidates=candidates,
            top_n=request.rerank_top_n or len(candidates),
        )
        rerank_ms = (time.time() - t2) * 1000.0

        # 5. Context Optimization & Deduplication
        context_result: ContextResult = build_optimized_context(
            reranked_chunks=reranked_documents,
            max_chunks=request.rerank_top_n or settings.RERANK_TOP_N,
            max_characters=settings.CONTEXT_MAX_CHARS,
        )

        # 6. Citation Generation
        citations = build_citations(context_result.selected_chunks)

        total_ms = (time.time() - total_start_time) * 1000.0

        stats: Dict[str, Any] = {
            "query_processing_ms": round(query_proc_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "reranking_ms": round(rerank_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "candidates_retrieved": len(candidates),
            "candidates_reranked": len(reranked_documents),
            "chunks_selected": len(context_result.selected_chunks),
        }

        logger.info(
            "Retrieval pipeline finished in %.2f ms (Retrieved: %d, Reranked: %d, Selected: %d)",
            total_ms,
            len(candidates),
            len(reranked_documents),
            len(context_result.selected_chunks),
        )

        return RagSearchResponse(
            query=request.query,
            normalized_query=normalized_query,
            retrieved_candidates=candidates,
            reranked_documents=reranked_documents,
            selected_documents=context_result.selected_chunks,
            context=context_result.context_text,
            citations=citations,
            context_characters=context_result.total_characters,
            estimated_tokens=context_result.estimated_tokens,
            execution_stats=stats,
        )


retrieval_service = RetrievalService()
