"""Chat API schemas."""
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Dict[str, Any]] = []
    route: str
    relevance_score: float = 0.0
    retry_count: int = 0

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
