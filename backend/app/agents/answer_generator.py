"""Grounded answer generation agent."""
try:
    from backend.app.services.llm_service import llm_service
except ImportError:
    from app.services.llm_service import llm_service

INSUFFICIENT = "I don't have enough relevant information in the uploaded documents to answer that question reliably."

def generate_document_answer(question: str, context: str, citations: list) -> str:
    if not context.strip(): return INSUFFICIENT
    prompt = f'''You are a grounded document QA assistant. Answer ONLY from the supplied context.\nIf the context is insufficient, say so. Cite factual claims with only the available citation numbers such as [1]. Never invent citations.\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nAVAILABLE CITATIONS:\n{[c.get("formatted_citation") for c in citations]}\n\nANSWER:'''
    return llm_service.invoke_text(prompt)

def generate_general_answer(question: str) -> str:
    return llm_service.invoke_text(f"Answer this general question concisely. Do not use document citations like [1]:\n{question}")
