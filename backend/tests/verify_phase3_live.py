"""End-to-end live server verification for Phase 3 Advanced Retrieval & Re-ranking."""

import io
import json
from pathlib import Path
import sys
import threading
import time
import httpx
import uvicorn

# Setup sys.path
_current = Path(__file__).resolve()
_root = _current.parent.parent.parent
_backend = _root / "backend"
for p in (_root, _backend):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.app.main import app
from backend.app.core.config import settings


def main():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    time.sleep(2)
    base_url = "http://127.0.0.1:8000"
    created_doc_ids = []

    try:
        with httpx.Client(base_url=base_url, timeout=60.0) as client:
            # 1. Verify GET /health
            health_resp = client.get("/health")
            print("1. [GET /health]:", health_resp.status_code, health_resp.json())
            assert health_resp.status_code == 200

            # 2. Verify GET /docs
            docs_resp = client.get("/docs")
            print("2. [GET /docs]:", docs_resp.status_code, "Swagger UI in HTML:", "swagger-ui" in docs_resp.text.lower())
            assert docs_resp.status_code == 200

            # 3. Ingest Knowledge Documents (A, B, C)
            test_corpus = [
                ("document_a.txt", "The production database is PostgreSQL. PostgreSQL is used for persistent storage and relational data management."),
                ("document_b.txt", "Redis is used for caching and session storage."),
                ("document_c.txt", "The frontend framework is Next.js."),
            ]

            print("\n3. Ingesting test documents...")
            for fname, text in test_corpus:
                files = {"file": (fname, text.encode("utf-8"), "text/plain")}
                up_resp = client.post("/documents/upload", files=files)
                assert up_resp.status_code == 201
                data = up_resp.json()
                created_doc_ids.append(data["document_id"])
                print(f"   Indexed '{fname}' -> ID: {data['document_id']} ({data['num_chunks']} chunk)")

            # 4. Query 1: Database
            print("\n4. Testing Query 1: 'What database is used for persistent storage?'")
            q1_resp = client.post("/rag/search", json={"query": "What database is used for persistent storage?"})
            assert q1_resp.status_code == 200
            q1_data = q1_resp.json()
            print("   Top Reranked Chunk Content:", q1_data["reranked_documents"][0]["content"][:80], "...")
            print("   Top Reranker Score:", q1_data["reranked_documents"][0]["reranker_score"])
            print("   Top Citation:", q1_data["citations"][0]["formatted_citation"])
            assert "PostgreSQL" in q1_data["reranked_documents"][0]["content"]

            # 5. Query 2: Caching
            print("\n5. Testing Query 2: 'What technology handles caching?'")
            q2_resp = client.post("/rag/search", json={"query": "What technology handles caching?"})
            assert q2_resp.status_code == 200
            q2_data = q2_resp.json()
            print("   Top Reranked Chunk Content:", q2_data["reranked_documents"][0]["content"][:80], "...")
            print("   Top Reranker Score:", q2_data["reranked_documents"][0]["reranker_score"])
            assert "Redis" in q2_data["reranked_documents"][0]["content"]

            # 6. Query 3: Frontend
            print("\n6. Testing Query 3: 'What frontend framework is used?'")
            q3_resp = client.post("/rag/search", json={"query": "What frontend framework is used?"})
            assert q3_resp.status_code == 200
            q3_data = q3_resp.json()
            print("   Top Reranked Chunk Content:", q3_data["reranked_documents"][0]["content"][:80], "...")
            print("   Top Reranker Score:", q3_data["reranked_documents"][0]["reranker_score"])
            assert "Next.js" in q3_data["reranked_documents"][0]["content"]

            # 7. Query 4: Unrelated Query Filtering
            print("\n7. Testing Unrelated Query with Similarity Threshold Filtering")
            unrelated_query = "Quantum chromodynamics of quark-gluon plasma in particle accelerators"
            unrelated_resp = client.post(
                "/rag/search",
                json={"query": unrelated_query, "similarity_threshold": 0.35},
            )
            assert unrelated_resp.status_code == 200
            unrelated_data = unrelated_resp.json()
            print(f"   Unrelated Query Candidates Retrieved: {len(unrelated_data['retrieved_candidates'])}")
            print(f"   Context Length: {len(unrelated_data['context'])}")
            assert len(unrelated_data["retrieved_candidates"]) == 0
            assert unrelated_data["context"] == ""

            # 8. Cleanup test documents
            print("\n8. Cleaning up test documents...")
            for doc_id in created_doc_ids:
                del_resp = client.delete(f"/documents/{doc_id}")
                assert del_resp.status_code == 200
                print(f"   Deleted document: {doc_id}")

            # 9. Verify documents list is empty
            list_after = client.get("/documents")
            assert list_after.json()["total"] == 0
            print("9. [GET /documents after cleanup]: Total =", list_after.json()["total"])

        print("\nAll Phase 3 Live Server Verification Checks Passed Successfully!")

    finally:
        server.should_exit = True
        server_thread.join(timeout=3)
        print("Server stopped cleanly.")


if __name__ == "__main__":
    main()
