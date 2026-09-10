"""Comprehensive test suite for Phase 3 Advanced Retrieval, Re-ranking, Context Optimization, and Citations."""

import io
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
from backend.app.models.retrieval import RerankedChunk, RetrievedChunk
from backend.app.rag.citations import build_citations, format_citation_string
from backend.app.rag.context_builder import build_optimized_context
from backend.app.rag.mmr import maximal_marginal_relevance
from backend.app.rag.query_processor import process_query
from backend.app.rag.reranker import rerank_chunks
from backend.app.rag.retriever import retrieve_candidates
from backend.app.services.retrieval_service import retrieval_service


class TestPhase3Retrieval(unittest.TestCase):
    """Unit and integration tests for the Advanced Retrieval pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.doc_ids = []

        # Ingest 3 distinct knowledge documents for retrieval testing
        docs = [
            ("database_architecture.txt", "The production database is PostgreSQL. It stores relational data and handles ACID transactions."),
            ("cache_architecture.txt", "Redis is used for caching and session storage to provide sub-millisecond data access."),
            ("frontend_framework.txt", "The frontend framework is Next.js with React components and TypeScript."),
        ]

        for filename, content in docs:
            file_bytes = io.BytesIO(content.encode("utf-8"))
            resp = cls.client.post(
                "/documents/upload",
                files={"file": (filename, file_bytes, "text/plain")},
            )
            assert resp.status_code == 201, f"Failed to ingest test document: {filename}"
            cls.doc_ids.append(resp.json()["document_id"])

    @classmethod
    def tearDownClass(cls):
        # Clean up test documents
        for doc_id in cls.doc_ids:
            try:
                cls.client.delete(f"/documents/{doc_id}")
            except Exception:
                pass

    def test_01_health_and_ingestion_stability(self):
        """J & I: Verify Phase 1 /health and Phase 2 /documents remain fully functional."""
        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(health_resp.json()["status"], "healthy")

        docs_resp = self.client.get("/documents")
        self.assertEqual(docs_resp.status_code, 200)
        self.assertGreaterEqual(docs_resp.json()["total"], 3)

    def test_02_query_processor(self):
        """1: Verify query processing, normalization, and empty query rejection."""
        # Whitespace and unicode normalization
        processed = process_query("  What   database is \t used for  persistent storage? \n\n")
        self.assertEqual(processed, "What database is used for persistent storage?")

        # Empty queries must raise ValueError
        with self.assertRaises(ValueError):
            process_query("")
        with self.assertRaises(ValueError):
            process_query("   \t \n  ")

    def test_03_semantic_retrieval_and_threshold_filtering(self):
        """A & B: Verify semantic retrieval returns relevant chunks and threshold filters noise."""
        # Targeted query matching PostgreSQL document
        candidates = retrieve_candidates(
            query="What database is used for persistent storage?",
            top_k=4,
            similarity_threshold=0.30,
            use_mmr=False,
        )
        self.assertGreater(len(candidates), 0)
        top_candidate = candidates[0]
        self.assertIn("PostgreSQL", top_candidate.content)
        self.assertGreaterEqual(top_candidate.similarity_score, 0.30)
        self.assertEqual(top_candidate.initial_rank, 1)

        # High threshold filtering out unrelated queries
        high_threshold_candidates = retrieve_candidates(
            query="Deep sea underwater volcanic marine biology and submarine navigation",
            top_k=4,
            similarity_threshold=0.85,  # Very high threshold
            use_mmr=False,
        )
        self.assertEqual(len(high_threshold_candidates), 0)

    def test_04_mmr_diversity_selection(self):
        """C: Verify MMR/diversity prevents redundant candidates."""
        # Create synthetic candidates where c0 and c1 are identical (redundant)
        # while c2 covers a distinct aspect of the query
        q_emb = [1.0, 0.5, 0.0]
        c0_emb = [1.0, 0.0, 0.0]     # high relevance, aspect X
        c1_emb = [1.0, 0.0, 0.0]     # duplicate of c0 (aspect X)
        c2_emb = [0.0, 1.0, 0.0]     # moderate relevance, aspect Y (diverse)

        c0 = RetrievedChunk(
            chunk_id="c0", document_id="d1", filename="f1.txt", page=1, chunk_index=0,
            content="Aspect X topic detail 1", source="", similarity_score=0.89, distance=0.11,
            initial_rank=1, metadata={}
        )
        c1 = RetrievedChunk(
            chunk_id="c1", document_id="d1", filename="f1.txt", page=1, chunk_index=1,
            content="Aspect X topic detail 2 (duplicate)", source="", similarity_score=0.89, distance=0.11,
            initial_rank=2, metadata={}
        )
        c2 = RetrievedChunk(
            chunk_id="c2", document_id="d2", filename="f2.txt", page=1, chunk_index=0,
            content="Aspect Y distinct topic detail", source="", similarity_score=0.45, distance=0.55,
            initial_rank=3, metadata={}
        )

        # With pure similarity (lambda=1.0), top 2 are c0, c1
        sim_selected = maximal_marginal_relevance(
            query_embedding=q_emb,
            candidate_embeddings=[c0_emb, c1_emb, c2_emb],
            candidates=[c0, c1, c2],
            top_n=2,
            lambda_mult=1.0,
        )
        self.assertEqual([c.chunk_id for c in sim_selected], ["c0", "c1"])

        # With balanced MMR (lambda=0.5), diversity penalizes c1, selecting c0 and c2
        mmr_selected = maximal_marginal_relevance(
            query_embedding=q_emb,
            candidate_embeddings=[c0_emb, c1_emb, c2_emb],
            candidates=[c0, c1, c2],
            top_n=2,
            lambda_mult=0.5,
        )
        self.assertEqual([c.chunk_id for c in mmr_selected], ["c0", "c2"])

    def test_05_cross_encoder_reranking_and_comparison(self):
        """D & E: Verify cross-encoder scores and compare Before-vs-After ranking."""
        query = "What technology handles caching and session storage?"

        candidates = retrieve_candidates(query=query, top_k=3, similarity_threshold=0.1, use_mmr=False)
        self.assertGreater(len(candidates), 0)

        reranked = rerank_chunks(query=query, candidates=candidates)
        self.assertEqual(len(reranked), len(candidates))

        # Check fields and scores
        print("\n" + "=" * 65)
        print("BEFORE vs AFTER RE-RANKING COMPARISON:")
        print(f"Query: '{query}'")
        print("-" * 65)
        print(f"{'Chunk ID':<15} | {'Initial Rank':<12} | {'Vector Sim':<10} | {'Reranked':<10} | {'CE Score':<10}")
        print("-" * 65)
        for r in reranked:
            self.assertIsNotNone(r.original_vector_score)
            self.assertIsNotNone(r.reranker_score)
            self.assertGreaterEqual(r.reranked_rank, 1)
            print(f"{r.chunk_id:<15} | #{r.initial_rank:<11} | {r.original_vector_score:<10.4f} | #{r.reranked_rank:<9} | {r.reranker_score:<10.4f}")
        print("=" * 65 + "\n")

        # The top reranked document for caching MUST be Redis
        top_reranked = reranked[0]
        self.assertIn("Redis", top_reranked.content)

    def test_06_context_builder_deduplication(self):
        """F: Verify context builder removes exact and near duplicate chunks."""
        c1 = RerankedChunk(
            chunk_id="c1", document_id="d1", filename="doc.txt", page=1, chunk_index=0,
            content="PostgreSQL handles relational persistent storage safely.",
            source="", original_vector_score=0.85, reranker_score=2.5, initial_rank=1, reranked_rank=1
        )
        c2_exact = RerankedChunk(
            chunk_id="c2", document_id="d1", filename="doc.txt", page=1, chunk_index=1,
            content="PostgreSQL handles relational persistent storage safely.",  # exact duplicate
            source="", original_vector_score=0.85, reranker_score=2.4, initial_rank=2, reranked_rank=2
        )
        c3_near = RerankedChunk(
            chunk_id="c3", document_id="d1", filename="doc.txt", page=1, chunk_index=2,
            content="PostgreSQL handles relational persistent storage safely and reliably.",  # near duplicate (>85% overlap)
            source="", original_vector_score=0.84, reranker_score=2.3, initial_rank=3, reranked_rank=3
        )
        c4_distinct = RerankedChunk(
            chunk_id="c4", document_id="d2", filename="cache.txt", page=1, chunk_index=0,
            content="Redis operates in-memory for sub-millisecond retrieval.",
            source="", original_vector_score=0.75, reranker_score=1.8, initial_rank=4, reranked_rank=4
        )

        res = build_optimized_context([c1, c2_exact, c3_near, c4_distinct], max_chunks=4)
        # Should keep c1 and c4_distinct, dropping c2_exact and c3_near
        selected_ids = [c.chunk_id for c in res.selected_chunks]
        self.assertIn("c1", selected_ids)
        self.assertNotIn("c2", selected_ids)
        self.assertNotIn("c3", selected_ids)
        self.assertIn("c4", selected_ids)

    def test_07_context_builder_max_chars_budget(self):
        """G: Verify context builder respects CONTEXT_MAX_CHARS constraint."""
        chunks = [
            RerankedChunk(
                chunk_id=f"c{i}", document_id="d1", filename="doc.txt", page=1, chunk_index=i,
                content=f"Paragraph {i}: " + ("x" * 200),
                source="", original_vector_score=0.8, reranker_score=2.0 - i * 0.1,
                initial_rank=i+1, reranked_rank=i+1
            )
            for i in range(10)
        ]

        # Limit budget to 400 characters
        res = build_optimized_context(chunks, max_chunks=10, max_characters=400)
        self.assertLessEqual(res.total_characters, 400)
        self.assertLess(len(res.selected_chunks), 10)

    def test_08_citations_exact_match(self):
        """H: Verify citations correspond exactly to retrieved metadata without hallucination."""
        sample_chunks = [
            RerankedChunk(
                chunk_id="c1", document_id="doc-uuid-1", filename="architecture_spec.pdf", page=4,
                chunk_index=2, content="System overview", source="/path/to/spec.pdf",
                original_vector_score=0.9, reranker_score=3.2, initial_rank=1, reranked_rank=1
            ),
            RerankedChunk(
                chunk_id="c2", document_id="doc-uuid-2", filename="security_policy.docx", page=2,
                chunk_index=0, content="Encryption standards", source="/path/to/policy.docx",
                original_vector_score=0.85, reranker_score=2.8, initial_rank=2, reranked_rank=2
            ),
            RerankedChunk(
                chunk_id="c3", document_id="doc-uuid-3", filename="notes.txt", page=1,
                chunk_index=5, content="Developer notes", source="/path/to/notes.txt",
                original_vector_score=0.80, reranker_score=2.1, initial_rank=3, reranked_rank=3
            ),
        ]

        citations = build_citations(sample_chunks)
        self.assertEqual(len(citations), 3)

        self.assertEqual(citations[0].citation_number, 1)
        self.assertEqual(citations[0].formatted_citation, "[1] architecture_spec.pdf — Page 4")
        self.assertEqual(citations[0].filename, "architecture_spec.pdf")
        self.assertEqual(citations[0].page, 4)

        self.assertEqual(citations[1].citation_number, 2)
        self.assertEqual(citations[1].formatted_citation, "[2] security_policy.docx — Page 2")

        self.assertEqual(citations[2].citation_number, 3)
        self.assertEqual(citations[2].formatted_citation, "[3] notes.txt — Page 1")

    def test_09_rag_search_api_endpoint(self):
        """9: Verify POST /rag/search debug endpoint returns complete structured breakdown."""
        response = self.client.post(
            "/rag/search",
            json={"query": "What frontend framework is used in the project?"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("query", data)
        self.assertIn("normalized_query", data)
        self.assertIn("retrieved_candidates", data)
        self.assertIn("reranked_documents", data)
        self.assertIn("selected_documents", data)
        self.assertIn("context", data)
        self.assertIn("citations", data)
        self.assertIn("execution_stats", data)

        # Next.js must be in top selected document and context
        self.assertIn("Next.js", data["context"])
        self.assertGreaterEqual(len(data["citations"]), 1)

        # Empty query should return 400 Bad Request
        empty_resp = self.client.post("/rag/search", json={"query": "   "})
        self.assertEqual(empty_resp.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
