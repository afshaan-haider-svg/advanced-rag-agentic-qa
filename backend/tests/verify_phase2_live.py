"""End-to-end live server test for Phase 2 Document Ingestion Pipeline."""

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

    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            # 1. Verify GET /health
            health_resp = client.get("/health")
            print("1. [GET /health]:", health_resp.status_code, health_resp.json())
            assert health_resp.status_code == 200

            # 2. Verify GET /docs
            docs_resp = client.get("/docs")
            print("2. [GET /docs]:", docs_resp.status_code, "Swagger UI in HTML:", "swagger-ui" in docs_resp.text.lower())
            assert docs_resp.status_code == 200

            # 3. Upload a small TXT file
            txt_content = (
                "Deep Learning and Information Retrieval\n\n"
                "Dense retrieval maps queries and documents into a shared continuous embedding space, "
                "enabling semantic matching beyond exact keyword overlaps.\n\n"
                "Combined with vector indexing techniques like HNSW, dense retrieval enables high-throughput "
                "nearest neighbor searches across millions of document chunks."
            )
            files = {"file": ("dense_retrieval.txt", txt_content.encode("utf-8"), "text/plain")}
            upload_resp = client.post("/documents/upload", files=files)
            print("3. [POST /documents/upload]:", upload_resp.status_code)
            upload_data = upload_resp.json()
            print("   Upload Response:", json.dumps(upload_data, indent=2))
            assert upload_resp.status_code == 201
            doc_id = upload_data["document_id"]
            num_chunks = upload_data["num_chunks"]
            assert num_chunks >= 1
            assert upload_data["status"] == "indexed"

            # 4. Verify GET /documents
            list_resp = client.get("/documents")
            print("4. [GET /documents]:", list_resp.status_code)
            list_data = list_resp.json()
            print("   Document Count:", list_data["total"])
            print("   Documents Listed:", [d["filename"] for d in list_data["documents"]])
            assert list_resp.status_code == 200
            assert any(d["document_id"] == doc_id for d in list_data["documents"])

            # 5. Verify Chroma persistence directory
            chroma_path = settings.CHROMA_PERSIST_DIR.resolve()
            print(f"5. [Chroma Persistence Dir]: {chroma_path} exists? {chroma_path.exists()}")
            assert chroma_path.exists()

            # 6. Verify DELETE /documents/{id}
            del_resp = client.delete(f"/documents/{doc_id}")
            print(f"6. [DELETE /documents/{doc_id}]:", del_resp.status_code)
            del_data = del_resp.json()
            print("   Delete Response:", json.dumps(del_data, indent=2))
            assert del_resp.status_code == 200
            assert del_data["status"] == "deleted"

            # 7. Verify GET /documents is empty again
            list_resp_after = client.get("/documents")
            print("7. [GET /documents after delete]: total =", list_resp_after.json()["total"])
            assert not any(d["document_id"] == doc_id for d in list_resp_after.json()["documents"])

        print("\nAll Phase 2 live verification checks passed successfully!")

    finally:
        server.should_exit = True
        server_thread.join(timeout=3)
        print("Server stopped.")


if __name__ == "__main__":
    main()
