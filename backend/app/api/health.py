"""Health check API router."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

try:
    from backend.app.core.config import settings
except ImportError:
    from app.core.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(default="healthy", description="Service health status")
    service: str = Field(default="advanced-rag-api", description="Service identifier")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API service.",
)
async def get_health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
    )


@router.get("/health/chat", summary="Agentic Chat Health")
async def get_chat_health():
    """Report graph/vector/LLM configuration without exposing secrets."""

    try:
        from backend.app.graph.workflow import agent_graph
    except ImportError:
        from app.graph.workflow import agent_graph

    provider = (settings.LLM_PROVIDER or "").lower()

    if provider == "gemini":
        llm_configured = bool(settings.GOOGLE_API_KEY)

    elif provider == "openai":
        llm_configured = bool(settings.OPENAI_API_KEY)

    else:
        llm_configured = bool(
            settings.GOOGLE_API_KEY or settings.OPENAI_API_KEY
        )

    return {
        "status": "healthy",
        "graph_available": agent_graph is not None,
        "vector_store_configured": bool(settings.CHROMA_PERSIST_DIR),
        "llm_configured": llm_configured,
    }