"""Debug and inspection API router for Advanced RAG retrieval and re-ranking."""

from fastapi import APIRouter, HTTPException, status

try:
    from backend.app.core.logging import get_logger
    from backend.app.models.retrieval import RagSearchRequest, RagSearchResponse
    from backend.app.services.retrieval_service import retrieval_service
except ImportError:
    from app.core.logging import get_logger
    from app.models.retrieval import RagSearchRequest, RagSearchResponse
    from app.services.retrieval_service import retrieval_service

logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG Retrieval"])


@router.post(
    "/search",
    response_model=RagSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Advanced RAG Search & Re-ranking (Debug Endpoint)",
    description=(
        "Executes query processing, semantic search from ChromaDB, MMR diversity selection, "
        "Cross-Encoder re-ranking, context optimization, and citation generation. "
        "NOTE: This is a retrieval-debug endpoint; final LLM synthesis will be integrated via LangGraph."
    ),
)
async def search_and_rerank(request: RagSearchRequest) -> RagSearchResponse:
    """Execute retrieval and re-ranking pipeline and return detailed diagnostic breakdown."""
    try:
        return retrieval_service.search_and_build_context(request)
    except ValueError as ve:
        logger.warning("Invalid retrieval request: %s", ve)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.exception("Unexpected error during retrieval/re-ranking: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete retrieval pipeline. Please check server logs.",
        )
