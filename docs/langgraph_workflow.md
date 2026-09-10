# LangGraph Agentic Workflow

```mermaid
flowchart TD
    S([START]) --> A[Query Analyzer]
    A -->|Document QA| R[Retriever: Semantic + MMR + Reranker]
    A -->|General| G[General LLM Answer]
    R --> V[Relevance Grader]
    V -->|Relevant| AN[Grounded Answer Generator]
    V -->|Not relevant; retries remain| RW[Query Rewriter]
    RW --> R
    V -->|Retry limit reached| I[Insufficient Context]
    AN --> C[Citation / Hallucination Checker]
    C --> F[Final Response]
    G --> F
    I --> F
    F --> E([END])
```

The workflow is implemented with a real `langgraph.graph.StateGraph` and conditional edges. Retrieval reuses the Phase 3 `RetrievalService`; it is not duplicated inside the graph.
