# Advanced RAG & Agentic Document QA System

A production-oriented **Retrieval-Augmented Generation (RAG) and Agentic Document Question Answering system** built with LangChain, LangGraph, FastAPI, Next.js, ChromaDB, PostgreSQL, Redis, Docker, local Hugging Face embeddings, Cross-Encoder reranking, and Google Gemini.

The application allows users to upload PDF, TXT, and DOCX documents and ask natural-language questions. The system retrieves relevant document chunks, reranks them, builds optimized context, generates grounded answers, and returns source citations through an agentic LangGraph workflow.

---

## Key Features

- PDF, TXT, and DOCX document ingestion
- Text cleaning and normalization
- Experimental chunk-size optimization
- RecursiveCharacterTextSplitter
- Local Hugging Face embeddings
- Persistent ChromaDB vector database
- Semantic similarity retrieval
- Maximal Marginal Relevance (MMR)
- Cross-Encoder reranking
- Context optimization and deduplication
- LangGraph agentic workflow
- Query analysis and conditional routing
- Relevance grading
- Query rewriting and bounded retrieval retries
- Gemini-powered answer generation
- Source citations
- Citation verification
- Persistent chat history
- PostgreSQL database
- Redis caching
- FastAPI REST API
- Next.js responsive frontend
- Docker multi-container deployment
- RAG evaluation framework
- Retrieval and reranking experiments
- Graceful AI-provider quota/rate-limit handling

---

## System Architecture

```text
                         USER
                           |
                           v
                    Next.js Frontend
                           |
                           v
                     FastAPI Backend
                           |
                           v
                  LangGraph Workflow
                           |
             +-------------+-------------+
             |                           |
             v                           v
       Query Analyzer              General Answer
             |
             v
       Query Processing
             |
             v
     Semantic Retrieval
             |
             v
        ChromaDB Search
             |
             v
       MMR Selection
             |
             v
   Cross-Encoder Reranking
             |
             v
      Relevance Grader
             |
       +-----+------+
       |            |
   Relevant      Not Relevant
       |            |
       |       Query Rewriter
       |            |
       |       Retrieval Retry
       |            |
       +------------+
             |
             v
      Context Optimization
             |
             v
      Gemini Answer Generator
             |
             v
       Citation Checker
             |
             v
       Grounded Response
             |
             v
        Next.js Frontend

Supporting Infrastructure:
- ChromaDB   -> Vector persistence
- PostgreSQL -> Persistent chat history
- Redis      -> Chat-history cache
- Docker     -> Multi-service containerization
```

---

## LangGraph Agentic Workflow

The QA pipeline is implemented as a real LangGraph `StateGraph`.

```text
START
  |
  v
Query Analyzer
  |
  +--------------------+
  |                    |
General Query      Document Query
  |                    |
  v                    v
General Answer       Retriever
                       |
                       v
                Relevance Grader
                       |
              +--------+--------+
              |                 |
           Relevant         Not Relevant
              |                 |
              |            Query Rewriter
              |                 |
              |            Retry Retrieval
              |                 |
              +-----------------+
                       |
                       v
                Answer Generator
                       |
                       v
                Citation Checker
                       |
                       v
                      END
```

Retrieval retries are bounded by `MAX_RETRIEVAL_RETRIES` to prevent infinite loops.

See:

```text
docs/langgraph_workflow.md
```

for additional workflow documentation.

---

## Advanced RAG Pipeline

The retrieval pipeline consists of the following stages:

1. **Query Processing**
   - Whitespace normalization
   - Unicode normalization
   - Input validation

2. **Semantic Retrieval**
   - Local SentenceTransformer embeddings
   - ChromaDB cosine similarity search
   - Similarity threshold filtering

3. **MMR Diversity Selection**
   - Maximal Marginal Relevance reduces redundant retrieval results.

4. **Cross-Encoder Reranking**
   - Query-document pairs are rescored using:
   - `cross-encoder/ms-marco-MiniLM-L-6-v2`

5. **Relevance Grading**
   - Retrieved context is checked before answer generation.

6. **Query Rewriting**
   - Weak retrieval can trigger query rewriting and another retrieval attempt.

7. **Context Optimization**
   - Duplicate removal
   - Near-duplicate filtering
   - Context-size budgeting

8. **Answer Generation**
   - Google Gemini generates an answer from selected document context.

9. **Citation Generation and Verification**
   - Source metadata is returned with the answer.
   - Citation checks are performed before final response delivery.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, React, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11 |
| Agent Workflow | LangGraph |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Google Gemini |
| Relational Database | PostgreSQL |
| Cache | Redis |
| Validation | Pydantic |
| Containerization | Docker, Docker Compose |
| API Documentation | Swagger / OpenAPI |

---

## Document Ingestion Pipeline

```text
Uploaded Document
       |
       v
File Validation
       |
       v
PDF / TXT / DOCX Loader
       |
       v
Cleaning & Normalization
       |
       v
RecursiveCharacterTextSplitter
       |
       v
Local Embeddings
       |
       v
Persistent ChromaDB
```

Document metadata and chunk metadata are preserved during ingestion.

---

## Chunking Experiment

Chunk size and overlap were not selected arbitrarily.

Three configurations were experimentally evaluated using the project's real documents, 15 evaluation questions, and the same local embedding model used by the production RAG pipeline.

| Chunk Size / Overlap | Chunks | MRR | Recall@4 | Avg. Relevant Rank | Avg. Query Latency |
|---|---:|---:|---:|---:|---:|
| 300 / 50 | 176 | 0.3338 | 53.33% | 24.67 | 15.84 ms |
| 500 / 100 | 102 | 0.5115 | 66.67% | 5.67 | 13.21 ms |
| **800 / 150** | **64** | **0.6502** | **73.33%** | **3.87** | **16.84 ms** |

### Selected Configuration

```env
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

The `800/150` configuration achieved the highest MRR and Recall@4 and the best average relevant rank among the tested configurations.

The production documents were subsequently re-indexed using this configuration.

Experiment outputs:

```text
evaluation/chunking_experiment_results.json
evaluation/chunking_experiment_summary.json
```

---

## Reranking Evaluation

The project also evaluates retrieval ranking before and after Cross-Encoder reranking.

Chunk-level evaluation used expected-answer lexical overlap as a proxy for identifying a relevant chunk. This is an evaluation heuristic rather than a human relevance judgment.

### Results

| Metric | Before Reranking | After Reranking |
|---|---:|---:|
| MRR | 0.6262 | 0.6433 |
| Average Relevant Rank | 3.00 | 3.00 |
| Recall@4 | 73.33% | 66.67% |

Additional observations:

```text
Queries improved : 4
Queries unchanged: 7
Queries worsened : 4

Average retrieval latency : 54.91 ms
Average reranking latency : 214.98 ms
Average total latency     : 273.29 ms
```

The experiment shows a modest MRR improvement after reranking while also demonstrating the latency and Recall@4 trade-off. Therefore, reranking is treated as a ranking-quality component rather than assumed to improve every query.

Results:

```text
evaluation/reranking_chunk_results.json
evaluation/reranking_chunk_results.csv
evaluation/reranking_chunk_summary.json
```

---

## Final RAG Evaluation

A 15-question evaluation dataset was created across two document domains:

- Artificial Intelligence in Education
- Artificial Intelligence in Healthcare

The final production configuration was evaluated through the complete `/chat` pipeline.

### Final Results

| Metric | Result |
|---|---:|
| Successful Requests | **15 / 15** |
| Retrieval Hit Rate | **100.00%** |
| Average Answer F1 | **0.4932** |
| Faithfulness Proxy | **1.0000** |
| Average Latency | **21.66 s** |
| Minimum Latency | **3.76 s** |
| Maximum Latency | **205.27 s** |

The first request produced a large latency outlier, consistent with cold-start/provider delay. Most subsequent requests completed substantially faster.

### Faithfulness Metric Note

`Faithfulness Proxy` is a lightweight project-specific check based on the presence of inline citation markers together with returned citation metadata.

It should **not** be interpreted as a full RAGAS semantic faithfulness score.

Evaluation artifacts:

```text
evaluation/eval_dataset.json
evaluation/evaluation_results.json
evaluation/evaluation_results.csv
evaluation/evaluation_summary.json
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend health check |
| GET | `/health/chat` | Chat/RAG service health |
| GET | `/docs` | Swagger/OpenAPI documentation |
| POST | `/documents/upload` | Upload and index PDF/TXT/DOCX |
| GET | `/documents` | List indexed documents |
| GET | `/documents/{id}` | Retrieve document information |
| DELETE | `/documents/{id}` | Delete document and vector chunks |
| POST | `/rag/search` | Debug advanced retrieval pipeline |
| POST | `/chat` | Run agentic document QA |
| GET | `/chat/history/{session_id}` | Retrieve persistent chat history |

---

## Example Chat Request

```json
{
  "message": "What are the main challenges of Artificial Intelligence in education?",
  "session_id": "demo-session"
}
```

The response contains the generated answer together with source citations and workflow information.

---

## Persistent Chat History

Chat history uses:

```text
Redis Cache
     |
     v
PostgreSQL
     |
     v
JSON fallback
```

PostgreSQL provides durable storage, while Redis reduces repeated history lookups.

A JSON fallback is available for graceful degradation.

---

## Project Structure

```text
advanced-rag-agentic-qa/
|
|-- backend/
|   |-- app/
|   |   |-- api/
|   |   |-- agents/
|   |   |-- core/
|   |   |-- graph/
|   |   |-- models/
|   |   |-- rag/
|   |   |-- services/
|   |   `-- main.py
|   |
|   |-- tests/
|   |-- requirements.txt
|   `-- Dockerfile
|
|-- frontend/
|   |-- src/
|   |-- public/
|   |-- package.json
|   |-- next.config.ts
|   `-- Dockerfile
|
|-- data/
|   |-- chroma/
|   |-- uploads/
|   `-- document_registry.json
|
|-- evaluation/
|   |-- eval_dataset.json
|   |-- run_evaluation.py
|   |-- run_reranking_evaluation.py
|   |-- run_chunking_experiment.py
|   `-- evaluation artifacts
|
|-- docs/
|   `-- langgraph_workflow.md
|
|-- docker-compose.yml
|-- .dockerignore
|-- .env.example
|-- .gitignore
`-- README.md
```

---

## Environment Configuration

Create `.env` from `.env.example`.

Example:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key
LLM_MODEL=gemini-3.6-flash
LLM_TEMPERATURE=0

CHUNK_SIZE=800
CHUNK_OVERLAP=150

DATABASE_URL=postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db
REDIS_URL=redis://localhost:6379/0
```

Never commit a real API key or production secret to Git.

---

## Run Locally

### Backend

Create/activate the Python 3.11 virtual environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Start FastAPI:

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:3000
```

---

## Docker Deployment

The project includes multi-stage Docker builds and Docker Compose orchestration.

Services:

```text
frontend   -> Next.js
backend    -> FastAPI + LangGraph + RAG
postgres   -> PostgreSQL
redis      -> Redis
```

Build and start the complete stack:

```powershell
docker compose up -d --build
```

Check containers:

```powershell
docker compose ps
```

Expected local services:

```text
Frontend : http://localhost:3000
Backend  : http://localhost:8000
Swagger  : http://localhost:8000/docs
```

Stop services:

```powershell
docker compose down
```

Persistent Docker volumes are used for PostgreSQL and Redis, while application data is mounted for document/vector persistence.

---

## Testing

Run backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -v
```

Run final RAG evaluation:

```powershell
.\.venv\Scripts\python.exe evaluation\run_evaluation.py
```

Run reranking evaluation:

```powershell
.\.venv\Scripts\python.exe evaluation\run_reranking_evaluation.py
```

Run chunking experiment:

```powershell
.\.venv\Scripts\python.exe evaluation\run_chunking_experiment.py
```

---

## Reliability and Error Handling

The backend includes:

- Pydantic request validation
- File-format validation
- File-size validation
- Structured logging
- Retrieval retry limits
- Graceful insufficient-context handling
- LLM configuration checks
- Provider rate-limit detection
- HTTP `429` response for exhausted AI-provider quota
- Persistent storage
- Redis cache fallback
- API health endpoints

---

## Security

Sensitive values are loaded from environment variables.

The repository must not contain:

- Google/Gemini API keys
- OpenAI API keys
- `.env`
- passwords
- private deployment credentials

Production deployments should use the hosting provider's secret/environment-variable management system.

---

## Current Status

The application has been validated locally as a complete Dockerized stack:

```text
Next.js Frontend
       |
       v
FastAPI
       |
       v
LangGraph
       |
       v
Advanced Retrieval + Reranking
       |
       v
Gemini
       |
       v
Grounded Answer + Citations
```

PostgreSQL and Redis are integrated for production-oriented persistence and caching.

---

## Repository

GitHub repository:

```text
To be added after final repository push.
```

## Live Deployment

Deployed application:

```text
To be added after cloud deployment.
```

---

## Academic Project

**Advanced AI Engineering**

This project demonstrates the design and implementation of an advanced, production-oriented RAG application rather than a basic tutorial retrieval pipeline.

Major areas covered include:

- LangChain
- LangGraph
- Advanced RAG
- Vector databases
- Retrieval evaluation
- Cross-Encoder reranking
- FastAPI
- React / Next.js
- PostgreSQL
- Redis
- Docker
- Evaluation and testing
- Production-oriented deployment architecture

---

## Limitations

- Final answer quality depends on retrieved document quality and the configured LLM provider.
- Provider rate limits can increase response latency.
- Cross-Encoder reranking improves some rankings but does not improve every query.
- The current faithfulness score is a lightweight citation-based proxy rather than full semantic RAGAS evaluation.
- The current evaluation dataset is intentionally small and domain-specific for assignment validation.

---

## Future Improvements

Potential extensions include:

- Hybrid BM25 + dense retrieval
- Streaming responses
- Semantic hallucination verification
- Full RAGAS evaluation
- Authentication and authorization
- Expanded evaluation datasets
- CI/CD
- Monitoring and observability
- Production secret management
- Additional LLM providers