"""Agentic chat endpoints powered by LangGraph."""

import uuid

from fastapi import APIRouter, HTTPException

try:
    from backend.app.core.config import settings
    from backend.app.models.chat import (
        ChatRequest,
        ChatResponse,
        ChatHistoryResponse,
    )
    from backend.app.graph.workflow import agent_graph
    from backend.app.services.chat_history_service import chat_history_service
except ImportError:
    from app.core.config import settings
    from app.models.chat import (
        ChatRequest,
        ChatResponse,
        ChatHistoryResponse,
    )
    from app.graph.workflow import agent_graph
    from app.services.chat_history_service import chat_history_service


router = APIRouter(tags=["Chat"])


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()

    return any(
        phrase in text
        for phrase in [
            "resource_exhausted",
            "rate limit",
            "ratelimit",
            "too many requests",
            "quota exceeded",
            "429",
        ]
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())

    try:
        result = agent_graph.invoke(
            {
                "question": req.message,
                "retry_count": 0,
                "max_retries": settings.MAX_RETRIEVAL_RETRIES,
                "citations": [],
                "unsupported_claims": [],
            }
        )

    except Exception as exc:
        error_text = str(exc)

        if "LLM is not configured" in error_text:
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM provider is not configured. "
                    "Please configure the required API key."
                ),
            ) from exc

        if _is_rate_limit_error(exc):
            raise HTTPException(
                status_code=429,
                detail=(
                    "The AI provider rate limit or quota has been reached. "
                    "Please wait and try again later."
                ),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the chat request.",
        ) from exc

    answer = result.get("answer", "")
    citations = result.get("citations", [])

    chat_history_service.append(
        sid,
        req.message,
        answer,
        citations,
    )

    return ChatResponse(
        session_id=sid,
        answer=answer,
        citations=citations,
        route=result.get("route", "document_qa"),
        relevance_score=result.get("relevance_score", 0.0),
        retry_count=result.get("retry_count", 0),
    )


@router.get(
    "/chat/history/{session_id}",
    response_model=ChatHistoryResponse,
)
def history(session_id: str):
    return ChatHistoryResponse(
        session_id=session_id,
        messages=chat_history_service.get(session_id),
    )