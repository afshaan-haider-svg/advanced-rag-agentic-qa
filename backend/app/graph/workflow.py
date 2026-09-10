"""Real LangGraph workflow for agentic document QA."""
from langgraph.graph import StateGraph, START, END
try:
    from backend.app.graph.state import AgentState
    from backend.app.agents.query_analyzer import analyze_query
    from backend.app.agents.relevance_grader import grade_relevance
    from backend.app.agents.query_rewriter import rewrite_query
    from backend.app.agents.answer_generator import generate_document_answer, generate_general_answer, INSUFFICIENT
    from backend.app.agents.citation_checker import verify_answer
    from backend.app.models.retrieval import RagSearchRequest
    from backend.app.services.retrieval_service import retrieval_service
except ImportError:
    from app.graph.state import AgentState
    from app.agents.query_analyzer import analyze_query
    from app.agents.relevance_grader import grade_relevance
    from app.agents.query_rewriter import rewrite_query
    from app.agents.answer_generator import generate_document_answer, generate_general_answer, INSUFFICIENT
    from app.agents.citation_checker import verify_answer
    from app.models.retrieval import RagSearchRequest
    from app.services.retrieval_service import retrieval_service

def _dump(x): return x.model_dump() if hasattr(x, "model_dump") else dict(x)

def analyze_node(state: AgentState): return analyze_query(state["question"])
def retrieve_node(state: AgentState):
    q=state.get("rewritten_query") or state.get("normalized_query") or state["question"]
    r=retrieval_service.search_and_build_context(RagSearchRequest(query=q))
    return {"documents":[_dump(x) for x in r.reranked_documents],"selected_documents":[_dump(x) for x in r.selected_documents],"context":r.context,"citations":[_dump(x) for x in r.citations]}
def grade_node(state: AgentState):
    g=grade_relevance(state.get("selected_documents",[]),state.get("context","")); return {"is_relevant":g["relevant"],"relevance_score":g["score"],"relevance_reason":g["reason"]}
def rewrite_node(state: AgentState):
    n=state.get("retry_count",0)+1; q=rewrite_query(state["question"],state.get("rewritten_query") or state.get("normalized_query",""),n); return {"rewritten_query":q,"retry_count":n}
def answer_node(state: AgentState): return {"answer":generate_document_answer(state["question"],state.get("context",""),state.get("citations",[])),"route":"document_qa"}
def general_node(state: AgentState): return {"answer":generate_general_answer(state["question"]),"citations":[],"route":"general"}
def insufficient_node(state: AgentState): return {"answer":INSUFFICIENT,"verified_answer":INSUFFICIENT,"citations":[],"route":"insufficient_context","citation_check_passed":True}
def check_node(state: AgentState):
    v=verify_answer(state.get("answer",""),state.get("context",""),state.get("citations",[]),True); return {**v,"answer":v["verified_answer"]}
def final_node(state: AgentState): return {"answer":state.get("verified_answer") or state.get("answer","")}
def route_after_analyze(s): return "retrieve" if s.get("should_retrieve") else "general"
def route_after_grade(s):
    if s.get("is_relevant"): return "answer"
    return "rewrite" if s.get("retry_count",0) < s.get("max_retries",2) else "insufficient"

def build_graph():
    g=StateGraph(AgentState)
    for name,fn in [("analyze",analyze_node),("retrieve",retrieve_node),("grade",grade_node),("rewrite",rewrite_node),("answer",answer_node),("general",general_node),("check",check_node),("insufficient",insufficient_node),("final",final_node)]: g.add_node(name,fn)
    g.add_edge(START,"analyze"); g.add_conditional_edges("analyze",route_after_analyze,{"retrieve":"retrieve","general":"general"}); g.add_edge("retrieve","grade"); g.add_conditional_edges("grade",route_after_grade,{"answer":"answer","rewrite":"rewrite","insufficient":"insufficient"}); g.add_edge("rewrite","retrieve"); g.add_edge("answer","check"); g.add_edge("check","final"); g.add_edge("general","final"); g.add_edge("insufficient","final"); g.add_edge("final",END)
    return g.compile()

agent_graph=build_graph()
