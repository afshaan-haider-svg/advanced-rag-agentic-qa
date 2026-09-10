"""Intent-preserving retrieval query rewriter."""

try:
    from backend.app.services.llm_service import llm_service
except ImportError:
    from app.services.llm_service import llm_service


def rewrite_query(question: str, current_query: str, retry_count: int) -> str:
    base = " ".join((current_query or question).split()).strip()

    prompt = f"""
You are a query rewriting agent for a document retrieval system.

Rewrite the user's query into a short, clear, standalone search query that
will retrieve the most relevant document chunks.

Rules:
- Preserve the original meaning and intent.
- Replace vague wording with clearer technical terms when appropriate.
- Do not answer the question.
- Do not add unsupported details.
- Return only the rewritten search query.
- Do not use quotation marks or explanations.

Original user question:
{question}

Current retrieval query:
{base}

Retry number:
{retry_count}

Rewritten search query:
""".strip()

    try:
        rewritten = llm_service.invoke_text(prompt).strip()

        if rewritten:
            rewritten = rewritten.strip('"').strip("'").strip()
            return rewritten

    except Exception:
        pass

    # Safe deterministic fallback if the LLM is unavailable.
    prefixes = (
        "according to the documents,",
        "in the uploaded documents,",
        "please tell me",
        "can you tell me",
        "tell me about",
    )

    cleaned = base
    low = cleaned.lower()

    for prefix in prefixes:
        if low.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" ,:?")
            break

    return cleaned or question