"""Shared LangGraph state for agentic document QA."""
from typing import Any, Dict, List, Optional, TypedDict

class AgentState(TypedDict, total=False):
    question: str
    normalized_query: str
    rewritten_query: str
    query_type: str
    should_retrieve: bool
    documents: List[Dict[str, Any]]
    selected_documents: List[Dict[str, Any]]
    context: str
    relevance_score: float
    is_relevant: bool
    relevance_reason: str
    answer: str
    verified_answer: str
    citations: List[Dict[str, Any]]
    citation_check_passed: bool
    unsupported_claims: List[str]
    retry_count: int
    max_retries: int
    route: str
    error: Optional[str]
