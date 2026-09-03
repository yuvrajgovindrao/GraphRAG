"""
Integration tests for the complete REST API surface of GraphRAG.
Tests /upload, /documents, /documents/{id}, /documents/{id}/chunks,
cascade deletion, /graph/summary, /graph/global, /graph/document/{id},
and /query execution with mocked external services.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app
import backend.main as main_module
from backend.models import QueryResponse, QueryMode, SourceCitation, GraphFact
from backend.database import insert_chunks


@pytest.fixture(scope="module")
def client():
    """FastAPI test client instance."""
    with TestClient(app) as test_client:
        yield test_client


# ── Document Management & Upload Tests ─────────────────────────────────

def test_upload_and_list_documents(client):
    """Test POST /upload for multiple files and verify registration in GET /documents."""
    with patch("backend.main._process_document", new=AsyncMock()):
        files = [
            ("files", ("test_sample_1.txt", b"First test file content about ecology.", "text/plain")),
            ("files", ("test_sample_2.txt", b"Second test file content about physics.", "text/plain")),
        ]
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["documents"]) == 2
        doc_id_1 = data["documents"][0]["doc_id"]
        doc_id_2 = data["documents"][1]["doc_id"]
        assert data["documents"][0]["filename"] == "test_sample_1.txt"
        assert data["documents"][1]["filename"] == "test_sample_2.txt"

    # Verify documents appear in GET /documents
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()
    doc_ids = [d["id"] for d in docs]
    assert doc_id_1 in doc_ids
    assert doc_id_2 in doc_ids

    # Verify GET /documents/{id}
    detail_resp = client.get(f"/documents/{doc_id_1}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["filename"] == "test_sample_1.txt"

    # Verify GET /documents/{id}/status
    status_resp = client.get(f"/documents/{doc_id_1}/status")
    assert status_resp.status_code == 200
    assert "status" in status_resp.json()

    # Verify 404 for non-existent document
    assert client.get("/documents/non-existent-uuid").status_code == 404
    assert client.get("/documents/non-existent-uuid/status").status_code == 404
    assert client.get("/documents/non-existent-uuid/chunks").status_code == 404


@pytest.mark.asyncio
async def test_get_document_chunks(client):
    """Test GET /documents/{id}/chunks retrieves persisted chunk records."""
    with patch("backend.main._process_document", new=AsyncMock()):
        files = [("files", ("chunks_test.txt", b"Sample content for chunk inspection.", "text/plain"))]
        res = client.post("/upload", files=files)
        doc_id = res.json()["documents"][0]["doc_id"]

    # Insert mock chunks directly into SQLite
    await insert_chunks([
        {"id": "c-1", "doc_id": doc_id, "text": "Chunk text 1", "page_number": 1, "chunk_index": 0},
        {"id": "c-2", "doc_id": doc_id, "text": "Chunk text 2", "page_number": 2, "chunk_index": 1},
    ])

    response = client.get(f"/documents/{doc_id}/chunks")
    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Chunk text 1"
    assert chunks[1]["text"] == "Chunk text 2"


# ── Cascade Deletion Tests ─────────────────────────────────────────────

def test_cascade_deletion(client):
    """Test DELETE /documents/{id} cascades deletion to SQLite, Qdrant vectors, and Neo4j graph nodes."""
    with patch("backend.main._process_document", new=AsyncMock()):
        files = [("files", ("delete_target.txt", b"Temporary file to delete.", "text/plain"))]
        res = client.post("/upload", files=files)
        doc_id = res.json()["documents"][0]["doc_id"]

    # Setup mocks for Qdrant and Neo4j clients
    mock_qdrant = MagicMock()
    mock_neo4j = MagicMock()
    original_qdrant = main_module.qdrant
    original_neo4j = main_module.neo4j_client

    try:
        main_module.qdrant = mock_qdrant
        main_module.neo4j_client = mock_neo4j

        del_resp = client.delete(f"/documents/{doc_id}")
        assert del_resp.status_code == 200
        assert "deleted successfully" in del_resp.json()["message"]

        # 1. Verify document is deleted from SQLite
        assert client.get(f"/documents/{doc_id}").status_code == 404

        # 2. Verify Qdrant delete was invoked
        mock_qdrant.delete.assert_called_once()

        # 3. Verify Neo4j subgraph deletion was invoked
        mock_neo4j.delete_document_graph.assert_called_once_with(doc_id)

        # Deleting already deleted document should return 404
        assert client.delete(f"/documents/{doc_id}").status_code == 404

    finally:
        main_module.qdrant = original_qdrant
        main_module.neo4j_client = original_neo4j


# ── Knowledge Graph API Tests ─────────────────────────────────────────

def test_graph_summary_endpoint(client):
    """Test GET /graph/summary returns entity and relationship counts."""
    mock_neo4j = MagicMock()
    mock_neo4j.get_graph_summary.return_value = {
        "total_entities": 42,
        "total_relationships": 65,
        "entity_types": {"CONCEPT": 30, "TECHNOLOGY": 12},
    }
    original_neo4j = main_module.neo4j_client

    try:
        main_module.neo4j_client = mock_neo4j
        resp = client.get("/graph/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entities"] == 42
        assert data["total_relationships"] == 65
    finally:
        main_module.neo4j_client = original_neo4j


def test_graph_global_endpoint(client):
    """Test GET /graph/global returns the complete connected knowledge graph."""
    mock_neo4j = MagicMock()
    mock_neo4j.get_global_subgraph.return_value = {
        "nodes": [{"id": "n1", "label": "Solar Energy", "group": "CONCEPT"}],
        "edges": [{"from": "n1", "to": "n2", "label": "REDUCES"}],
        "node_count": 1,
        "edge_count": 1,
    }
    original_neo4j = main_module.neo4j_client

    try:
        main_module.neo4j_client = mock_neo4j
        resp = client.get("/graph/global")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1
        assert data["nodes"][0]["label"] == "Solar Energy"
    finally:
        main_module.neo4j_client = original_neo4j


def test_graph_document_subgraph(client):
    """Test GET /graph/document/{id} returns document-specific subgraph."""
    with patch("backend.main._process_document", new=AsyncMock()):
        files = [("files", ("subgraph_doc.txt", b"Doc for subgraph.", "text/plain"))]
        res = client.post("/upload", files=files)
        doc_id = res.json()["documents"][0]["doc_id"]

    mock_neo4j = MagicMock()
    mock_neo4j.get_document_subgraph.return_value = {
        "nodes": [{"id": "n1", "label": "Stem Cells"}],
        "edges": [],
        "node_count": 1,
        "edge_count": 0,
    }
    original_neo4j = main_module.neo4j_client

    try:
        main_module.neo4j_client = mock_neo4j
        resp = client.get(f"/graph/document/{doc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_count"] == 1
        assert data["nodes"][0]["label"] == "Stem Cells"

        # Non-existent doc returns 404
        assert client.get("/graph/document/missing-doc-id").status_code == 404
    finally:
        main_module.neo4j_client = original_neo4j


# ── Query API Tests ───────────────────────────────────────────────────

def test_query_endpoint_execution_and_429_handling(client):
    """Test POST /query handles vector and graph modes and maps 429 quota exceptions cleanly."""
    mock_llm = MagicMock()
    mock_qdrant = MagicMock()
    mock_neo4j = MagicMock()

    original_llm = main_module.llm
    original_qdrant = main_module.qdrant
    original_neo4j = main_module.neo4j_client

    try:
        main_module.llm = mock_llm
        main_module.qdrant = mock_qdrant
        main_module.neo4j_client = mock_neo4j

        # 1. Test successful query
        fake_response = QueryResponse(
            question="What is photosynthesis?",
            answer="Photosynthesis creates glucose.",
            mode=QueryMode.VECTOR,
            sources=[SourceCitation(chunk_id="c1", text="Sample text", filename="bio.pdf", page_number=1, relevance_score=0.9)],
            graph_facts=[],
        )

        with patch("backend.main.vector_only_query", new=AsyncMock(return_value=fake_response)):
            resp = client.post("/query", json={"question": "What is photosynthesis?", "mode": "vector", "top_k": 3})
            assert resp.status_code == 200
            assert resp.json()["answer"] == "Photosynthesis creates glucose."
            assert len(resp.json()["sources"]) == 1

        # 2. Test 429 Rate Limit error handling
        with patch("backend.main.graph_enhanced_query", new=AsyncMock(side_effect=Exception("429 RESOURCE_EXHAUSTED"))):
            err_resp = client.post("/query", json={"question": "Heavy load query?", "mode": "graph"})
            assert err_resp.status_code == 429
            assert "Gemini API quota or rate limit reached" in err_resp.json()["detail"]

    finally:
        main_module.llm = original_llm
        main_module.qdrant = original_qdrant
        main_module.neo4j_client = original_neo4j
