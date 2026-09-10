"""Integration and unit tests for document ingestion, chunking, and persistence."""

import io
import json
from pathlib import Path
import sys
import unittest

# Ensure workspace root and backend directory are in sys.path
_current_file = Path(__file__).resolve()
_root_dir = _current_file.parent.parent.parent
_backend_dir = _root_dir / "backend"

for p in (_root_dir, _backend_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.rag.vector_store import get_chunks_collection


class TestDocumentIngestion(unittest.TestCase):
    """Test suite for Phase 2 Document Ingestion Pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.created_doc_ids = []

    @classmethod
    def tearDownClass(cls):
        # Cleanup any remaining test documents
        for doc_id in cls.created_doc_ids:
            try:
                cls.client.delete(f"/documents/{doc_id}")
            except Exception:
                pass

    def test_01_health_check(self):
        """Verify Phase 1 /health endpoint continues to function."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "advanced-rag-api")

    def test_02_reject_unsupported_file_type(self):
        """Verify rejection of files with unsupported extensions (e.g. .py, .exe, .json)."""
        fake_file = io.BytesIO(b"import os\nprint('malicious')")
        response = self.client.post(
            "/documents/upload",
            files={"file": ("malicious_script.py", fake_file, "text/x-python")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file format", response.json()["detail"])

    def test_03_txt_ingestion(self):
        """Verify uploading, chunking, embedding, and indexing of a TXT file."""
        sample_text = (
            "Artificial Intelligence and Retrieval-Augmented Generation (RAG)\n\n"
            "Retrieval-Augmented Generation (RAG) is an architecture that optimizes the output "
            "of a Large Language Model by referencing an authoritative external knowledge base "
            "outside of its training data sources before generating a response.\n\n"
            "Large Language Models (LLMs) can be inconsistent, out-of-date, or lack domain-specific "
            "expertise. RAG solves this by retrieving relevant context and injecting it into the prompt.\n\n"
            "Agentic RAG takes this further by equipping autonomous agents with tools, decision loops, "
            "and dynamic query routing to execute multi-step document investigations."
        )

        file_bytes = io.BytesIO(sample_text.encode("utf-8"))
        response = self.client.post(
            "/documents/upload",
            files={"file": ("rag_overview.txt", file_bytes, "text/plain")},
        )

        self.assertEqual(response.status_code, 201)
        data = response.json()
        doc_id = data["document_id"]
        self.created_doc_ids.append(doc_id)

        self.assertEqual(data["filename"], "rag_overview.txt")
        self.assertEqual(data["file_type"], ".txt")
        self.assertEqual(data["status"], "indexed")
        self.assertGreaterEqual(data["num_chunks"], 1)
        self.assertEqual(data["total_pages"], 1)

    def test_04_list_documents(self):
        """Verify GET /documents returns registered document metadata."""
        response = self.client.get("/documents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("documents", data)
        self.assertGreaterEqual(data["total"], 1)

        filenames = [d["filename"] for d in data["documents"]]
        self.assertIn("rag_overview.txt", filenames)

    def test_05_persistence_verification(self):
        """Verify Chroma collection and registry files exist on disk."""
        self.assertTrue(settings.CHROMA_PERSIST_DIR.exists())
        self.assertTrue(settings.REGISTRY_FILE_PATH.exists())

        collection = get_chunks_collection()
        self.assertGreater(collection.count(), 0)

    def test_06_delete_document(self):
        """Verify DELETE /documents/{id} removes document, vectors, and returns 404 subsequently."""
        self.assertGreater(len(self.created_doc_ids), 0)
        target_id = self.created_doc_ids.pop(0)

        # 1. Delete document
        del_response = self.client.delete(f"/documents/{target_id}")
        self.assertEqual(del_response.status_code, 200)
        del_data = del_response.json()
        self.assertEqual(del_data["document_id"], target_id)
        self.assertEqual(del_data["status"], "deleted")
        self.assertGreaterEqual(del_data["deleted_chunks"], 1)

        # 2. Deleting again should return 404
        second_del_response = self.client.delete(f"/documents/{target_id}")
        self.assertEqual(second_del_response.status_code, 404)

        # 3. Document should no longer be listed
        list_response = self.client.get("/documents")
        remaining_ids = [d["document_id"] for d in list_response.json()["documents"]]
        self.assertNotIn(target_id, remaining_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
