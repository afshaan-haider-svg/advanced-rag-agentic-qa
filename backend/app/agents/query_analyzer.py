"""Query classification agent."""
import re
from typing import Dict
try:
    from backend.app.services.llm_service import llm_service
except ImportError:
    from app.services.llm_service import llm_service

GENERAL_PATTERNS = (r"^(hi|hello|hey)\b", r"\bwho are you\b", r"\bwhat can you do\b", r"\bthank(s| you)\b")
DOCUMENT_HINTS = ("document", "file", "pdf", "uploaded", "according to", "source", "policy", "report", "page")

def analyze_query(question: str) -> Dict:
    q = " ".join(question.split()).strip()
    if not q:
        return {"query_type": "ambiguous", "should_retrieve": False, "normalized_query": q}
    low = q.lower()
    if any(re.search(p, low) for p in GENERAL_PATTERNS):
        return {"query_type": "general", "should_retrieve": False, "normalized_query": q}
    # In a document-QA product, factual questions default to retrieval; explicit general chit-chat bypasses it.
    return {"query_type": "document_qa", "should_retrieve": True, "normalized_query": q}
