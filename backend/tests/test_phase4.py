"""Focused Phase 4 unit tests that do not require a paid LLM API."""
from backend.app.agents.query_analyzer import analyze_query
from backend.app.agents.query_rewriter import rewrite_query
from backend.app.agents.relevance_grader import grade_relevance
from backend.app.agents.citation_checker import verify_answer

def test_document_route(): assert analyze_query("What database is used?")["should_retrieve"] is True
def test_general_route(): assert analyze_query("Hello")["query_type"] == "general"
def test_relevance_empty(): assert grade_relevance([],"")["relevant"] is False
def test_rewriter_preserves_intent(): assert "database" in rewrite_query("What database?","What database?",1).lower()
def test_fake_citation_rejected(): assert verify_answer("Claim [9]","ctx",[{"citation_number":1}],True)["citation_check_passed"] is False
def test_valid_citation(): assert verify_answer("Claim [1]","ctx",[{"citation_number":1}],True)["citation_check_passed"] is True
def test_general_strips_citations(): assert "[1]" not in verify_answer("Hello [1]","",[],False)["verified_answer"]
