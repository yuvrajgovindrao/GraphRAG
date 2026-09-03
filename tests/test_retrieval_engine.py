"""
Unit tests for the Hybrid Retrieval Engine, Vector Search, and Graph Traversal.
Validates search_similar, find_seed_entities, get_graph_context (multi-doc balancing),
and graph_enhanced_query / vector_only_query execution with mocked external services.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.retrieval.vector_search import search_similar, delete_doc_vectors, index_chunks
from backend.graph.traversal import find_seed_entities, get_graph_context, format_graph_facts
from backend.retrieval.graph_rag import graph_enhanced_query, vector_only_query
from backend.models import QueryMode


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Create a mock LLM provider for embeddings and completions."""
    llm = MagicMock()
    # Mock embedding: returns a 768-dimensional vector
    llm.embed = AsyncMock(return_value=[[0.1] * 768])
    # Mock generation
    llm.generate = AsyncMock(return_value="Based on the context, climate change causes rising temperatures.")
    return llm


@pytest.fixture
def mock_qdrant():
    """Create a mock Qdrant client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_neo4j():
    """Create a mock Neo4j client."""
    neo4j = MagicMock()
    neo4j.find_entities_by_names = MagicMock(return_value=[])
    neo4j.find_entities_by_chunk_ids = MagicMock(return_value=[])
    neo4j.expand_from_entities = MagicMock(return_value=[])
    return neo4j


# ── Vector Search Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_similar_payload_mapping(mock_llm, mock_qdrant):
    """Test search_similar correctly extracts and maps hit payloads."""
    fake_hit = MagicMock()
    fake_hit.id = "chunk-uuid-1"
    fake_hit.score = 0.92
    fake_hit.payload = {
        "doc_id": "doc-uuid-1",
        "text": "Greenhouse gases trap heat in Earth's atmosphere.",
        "filename": "climate_report.pdf",
        "page_number": 4,
    }

    mock_results = MagicMock()
    mock_results.points = [fake_hit]
    mock_qdrant.query_points.return_value = mock_results

    results = await search_similar(mock_qdrant, mock_llm, query="What is the greenhouse effect?", top_k=3)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-uuid-1"
    assert results[0]["doc_id"] == "doc-uuid-1"
    assert results[0]["text"] == "Greenhouse gases trap heat in Earth's atmosphere."
    assert results[0]["filename"] == "climate_report.pdf"
    assert results[0]["page_number"] == 4
    assert results[0]["score"] == 0.92


@pytest.mark.asyncio
async def test_index_chunks(mock_llm, mock_qdrant):
    """Test batch chunk indexing into Qdrant."""
    chunks = [
        {"id": "c1", "doc_id": "d1", "text": "Text 1", "page_number": 1, "chunk_index": 0, "source_filename": "a.txt"},
        {"id": "c2", "doc_id": "d1", "text": "Text 2", "page_number": 1, "chunk_index": 1, "source_filename": "a.txt"},
    ]
    mock_llm.embed = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])

    await index_chunks(mock_qdrant, mock_llm, chunks)

    mock_llm.embed.assert_awaited_once_with(["Text 1", "Text 2"])
    mock_qdrant.upsert.assert_called_once()


def test_delete_doc_vectors(mock_qdrant):
    """Test deleting document vectors with field condition filter."""
    delete_doc_vectors(mock_qdrant, "doc-to-delete")
    mock_qdrant.delete.assert_called_once()


# ── Graph Traversal Tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_find_seed_entities_multi_doc(mock_llm, mock_neo4j):
    """Test find_seed_entities combines question extraction with chunk entity lookups."""
    # Mock LLM question extraction
    with patch("backend.graph.traversal.extract_query_entities", new=AsyncMock(return_value=["Climate Change"])):
        # Match from question
        mock_neo4j.find_entities_by_names.return_value = [{"name": "Climate Change", "doc_ids": ["doc-climate"]}]

        # Vector results spanning two documents
        vector_results = [
            {"chunk_id": "chunk-1", "doc_id": "doc-climate"},
            {"chunk_id": "chunk-2", "doc_id": "doc-biology"},
        ]
        # Match chunk entities (guarantees cross-document seeds)
        mock_neo4j.find_entities_by_chunk_ids.return_value = [
            {"name": "Climate Change"},
            {"name": "Stem Cells"},
        ]

        seeds = await find_seed_entities(
            question="What is climate change and what are stem cells?",
            llm=mock_llm,
            neo4j=mock_neo4j,
            vector_results=vector_results,
        )

        assert "Climate Change" in seeds
        assert "Stem Cells" in seeds
        assert len(seeds) == 2


def test_get_graph_context_multi_doc_balancing(mock_neo4j):
    """Test that multi-document vector results trigger balanced per-document expansion."""
    vector_results = [
        {"chunk_id": "c-climate", "doc_id": "doc-climate"},
        {"chunk_id": "c-bio", "doc_id": "doc-biology"},
    ]

    mock_neo4j.find_entities_by_chunk_ids.side_effect = [
        [{"name": "Global Warming"}],
        [{"name": "Stem Cells"}],
    ]

    mock_neo4j.expand_from_entities.side_effect = [
        [{"source_entity": "Global Warming", "relation": "CAUSES", "target_entity": "Heatwaves", "description": ""}],
        [{"source_entity": "Stem Cells", "relation": "DIFFERENTIATES_INTO", "target_entity": "Tissue", "description": ""}],
    ]

    facts = get_graph_context(
        neo4j=mock_neo4j,
        seed_entities=["Global Warming", "Stem Cells"],
        max_hops=2,
        max_facts=45,
        vector_results=vector_results,
    )

    # Both documents must have facts returned
    assert len(facts) == 2
    sources = [f["source_entity"] for f in facts]
    assert "Global Warming" in sources
    assert "Stem Cells" in sources
    # expand_from_entities must have been called twice (once per doc)
    assert mock_neo4j.expand_from_entities.call_count == 2


def test_format_graph_facts():
    """Test formatting structured graph facts into prompt text."""
    facts = [
        {"source_entity": "Carbon Dioxide", "relation": "TRAPS", "target_entity": "Heat", "description": "Greenhouse effect."},
        {"source_entity": "Methane", "relation": "EMITTED_BY", "target_entity": "Livestock", "description": ""},
    ]
    formatted = format_graph_facts(facts)
    assert "- Carbon Dioxide —[TRAPS]→ Heat (Greenhouse effect.)" in formatted
    assert "- Methane —[EMITTED_BY]→ Livestock" in formatted

    empty_formatted = format_graph_facts([])
    assert empty_formatted == "No graph facts available."


# ── Hybrid Retrieval Q&A Tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_enhanced_query_end_to_end(mock_llm, mock_qdrant, mock_neo4j):
    """Test full execution of graph_enhanced_query with synthesized answer and citations."""
    fake_hit = {
        "chunk_id": "cid-1",
        "doc_id": "did-1",
        "text": "Rising sea levels threaten coastal cities.",
        "filename": "sea_levels.pdf",
        "page_number": 3,
        "score": 0.88,
    }

    with patch("backend.retrieval.graph_rag.search_similar", new=AsyncMock(return_value=[fake_hit])), \
         patch("backend.retrieval.graph_rag.find_seed_entities", new=AsyncMock(return_value=["Sea Levels"])), \
         patch("backend.retrieval.graph_rag.get_graph_context", return_value=[
             {"source_entity": "Sea Levels", "relation": "THREATENS", "target_entity": "Coastal Cities", "description": "Flooding risks"}
         ]):

        mock_llm.generate = AsyncMock(return_value="Sea level rise causes severe coastal flooding risks.")

        response = await graph_enhanced_query(
            question="What is the impact of sea level rise?",
            llm=mock_llm,
            qdrant_client=mock_qdrant,
            neo4j=mock_neo4j,
            top_k=3,
        )

        assert response.mode == QueryMode.GRAPH
        assert "coastal flooding risks" in response.answer
        assert len(response.sources) == 1
        assert response.sources[0].filename == "sea_levels.pdf"
        assert response.sources[0].page_number == 3
        assert response.sources[0].relevance_score == 0.88
        assert len(response.graph_facts) == 1
        assert response.graph_facts[0].source_entity == "Sea Levels"
        assert response.graph_facts[0].target_entity == "Coastal Cities"


@pytest.mark.asyncio
async def test_vector_only_query(mock_llm, mock_qdrant):
    """Test vector-only query mode produces citations and empty graph facts."""
    fake_hit = {
        "chunk_id": "cid-2",
        "doc_id": "did-2",
        "text": "Photosynthesis converts solar energy into chemical energy.",
        "filename": "botany.pdf",
        "page_number": 12,
        "score": 0.94,
    }

    with patch("backend.retrieval.graph_rag.search_similar", new=AsyncMock(return_value=[fake_hit])):
        mock_llm.generate = AsyncMock(return_value="Photosynthesis converts solar energy.")

        response = await vector_only_query(
            question="How does photosynthesis work?",
            llm=mock_llm,
            qdrant_client=mock_qdrant,
            top_k=2,
        )

        assert response.mode == QueryMode.VECTOR
        assert len(response.graph_facts) == 0
        assert len(response.sources) == 1
        assert response.sources[0].filename == "botany.pdf"
        assert response.sources[0].page_number == 12
