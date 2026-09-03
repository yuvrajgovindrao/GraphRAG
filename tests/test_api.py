"""
Unit and Integration Tests for GraphRAG API.
Tests endpoints, schemas, chunking, and entity resolution with ZERO external API keys required.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models import (
    QueryRequest, QueryMode, DocumentResponse, DocumentStatus,
    ExtractionResult, ExtractedEntity, ExtractedRelationship,
)
from backend.ingestion.parser import ParsedDocument, PageText
from backend.ingestion.chunker import chunk_document
from backend.graph.entity_resolution import resolve_entities


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """Test the /health endpoint returns 200 and expected schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "llm_provider" in data
    assert "sqlite" in data
    assert "qdrant" in data
    assert "neo4j" in data


def test_query_validation_error(client):
    """Test /query validation fails properly with 422 when question is missing."""
    response = client.post("/query", json={})
    assert response.status_code == 422
    errors = response.json()
    assert "detail" in errors


def test_models_validation():
    """Test Pydantic models validate default values and types correctly."""
    req = QueryRequest(question="What is climate change?")
    assert req.question == "What is climate change?"
    assert req.mode == QueryMode.GRAPH
    assert req.top_k == 5

    doc_resp = DocumentResponse(
        id="test-id",
        filename="test.pdf",
        file_type="pdf",
        status=DocumentStatus.READY,
        total_chunks=10,
        processed_chunks=10,
    )
    assert doc_resp.id == "test-id"
    assert doc_resp.total_chunks == 10
    assert doc_resp.status == DocumentStatus.READY


def test_chunk_document_logic():
    """Test sentence-boundary chunking logic without external calls."""
    parsed = ParsedDocument(
        filename="sample.txt",
        pages=[
            PageText(
                page_number=1,
                text="The greenhouse effect is a natural process that warms the Earth. Greenhouse gases absorb infrared radiation.",
            ),
            PageText(
                page_number=2,
                text="Human activities such as burning fossil fuels and deforestation increase greenhouse gas concentrations.",
            ),
        ],
    )
    chunks = chunk_document(parsed, doc_id="doc-123", chunk_size=100, chunk_overlap=10)
    assert len(chunks) >= 1
    assert chunks[0].doc_id == "doc-123"
    assert chunks[0].source_filename == "sample.txt"
    assert chunks[0].text is not None


def test_entity_resolution_logic():
    """Test entity deduplication and name normalization."""
    extraction_results = [
        ExtractionResult(
            chunk_id="chunk-1",
            entities=[
                ExtractedEntity(name="Global Warming", type="CONCEPT", description="Rising temperatures across the globe."),
                ExtractedEntity(name="Carbon Dioxide", type="CHEMICAL", description="A primary greenhouse gas."),
            ],
            relationships=[
                ExtractedRelationship(source="Global Warming", target="Carbon Dioxide", relation="DRIVEN_BY", description="Driven by emissions."),
            ],
        ),
        ExtractionResult(
            chunk_id="chunk-2",
            entities=[
                ExtractedEntity(name="global warming", type="CONCEPT", description="Planetary heating effect."),
            ],
            relationships=[],
        ),
    ]

    resolved_entities, resolved_relationships = resolve_entities(extraction_results)

    names = [e["name"] for e in resolved_entities]
    # "Global Warming" and "global warming" should resolve into 1 canonical entity
    assert len(resolved_entities) == 2
    assert "Global Warming" in names
    assert len(resolved_relationships) == 1
    assert resolved_relationships[0]["relation"] == "DRIVEN_BY"
