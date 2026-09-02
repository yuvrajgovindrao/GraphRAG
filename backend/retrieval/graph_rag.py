"""
Hybrid retrieval combining vector search with graph traversal.
This is the core Graph-RAG retrieval pipeline.
"""

from __future__ import annotations

import logging

from backend.llm_provider import BaseLLMProvider
from backend.graph.neo4j_client import Neo4jClient
from backend.graph.traversal import find_seed_entities, get_graph_context, format_graph_facts
from backend.retrieval.vector_search import search_similar
from backend.models import QueryResponse, QueryMode, SourceCitation, GraphFact

logger = logging.getLogger(__name__)


VECTOR_RAG_PROMPT = """You are a helpful assistant that answers questions based on the provided source documents.
Use ONLY the information from the source passages below to answer the question.
If the answer cannot be found in the sources, say "I don't have enough information to answer this question."
Always cite which source passage(s) you used.

## Source Passages
{passages}

## Question
{question}

## Instructions
Provide a clear, well-structured answer based on the source passages above. Reference specific passages when possible."""


GRAPH_RAG_PROMPT = """You are a helpful assistant that answers questions using both structured knowledge graph facts and source document passages.
Use the information from BOTH the graph facts AND source passages to construct your answer.
The graph facts show relationships between entities. The source passages provide detailed context.
If the answer cannot be found in either source, say "I don't have enough information to answer this question."

## Graph Facts (Structured Knowledge)
{graph_facts}

## Source Passages (Detailed Context)
{passages}

## Question
{question}

## Instructions
Synthesize information from both the graph facts and source passages to provide a comprehensive answer.
When connecting multiple facts, explain the reasoning chain.
Reference specific sources when possible."""


def _format_passages(results: list[dict]) -> str:
    """Format vector search results into a prompt-ready string."""
    if not results:
        return "No source passages available."

    lines = []
    for i, r in enumerate(results, 1):
        filename = r.get("filename", "unknown")
        page = r.get("page_number", "?")
        text = r.get("text", "")
        lines.append(f"[Source {i}] (File: {filename}, Page: {page})\n{text}")

    return "\n\n".join(lines)


async def vector_only_query(
    question: str,
    llm: BaseLLMProvider,
    qdrant_client,
    top_k: int = 5,
) -> QueryResponse:
    """Plain vector RAG query (Phase 2 baseline)."""
    # Retrieve similar chunks
    results = await search_similar(qdrant_client, llm, question, top_k=top_k)

    # Build prompt
    passages = _format_passages(results)
    prompt = VECTOR_RAG_PROMPT.format(passages=passages, question=question)

    # Generate answer
    answer = await llm.generate(prompt)

    # Build citations
    sources = [
        SourceCitation(
            chunk_id=r["chunk_id"],
            text=r["text"],
            filename=r.get("filename"),
            page_number=r.get("page_number"),
            relevance_score=r.get("score"),
        )
        for r in results
    ]

    return QueryResponse(
        answer=answer,
        mode=QueryMode.VECTOR,
        sources=sources,
        question=question,
    )


async def graph_enhanced_query(
    question: str,
    llm: BaseLLMProvider,
    qdrant_client,
    neo4j: Neo4jClient,
    top_k: int = 5,
    max_hops: int = 2,
) -> QueryResponse:
    """Graph-enhanced hybrid retrieval query (Phase 4)."""
    # Step 1: Vector search for passages
    vector_results = await search_similar(qdrant_client, llm, question, top_k=top_k)

    # Step 2: Find seed entities and expand graph
    seed_entities = await find_seed_entities(question, llm, neo4j, qdrant_client)
    graph_facts_raw = get_graph_context(neo4j, seed_entities, max_hops=max_hops)

    # Step 3: Build hybrid prompt
    passages = _format_passages(vector_results)
    graph_facts_text = format_graph_facts(graph_facts_raw)

    prompt = GRAPH_RAG_PROMPT.format(
        graph_facts=graph_facts_text,
        passages=passages,
        question=question,
    )

    # Step 4: Generate answer
    answer = await llm.generate(prompt)

    # Build citations
    sources = [
        SourceCitation(
            chunk_id=r["chunk_id"],
            text=r["text"],
            filename=r.get("filename"),
            page_number=r.get("page_number"),
            relevance_score=r.get("score"),
        )
        for r in vector_results
    ]

    graph_facts = [
        GraphFact(
            source_entity=f.get("source_entity", ""),
            source_type=f.get("source_type", "Concept"),
            relation=f.get("relation", ""),
            target_entity=f.get("target_entity", ""),
            target_type=f.get("target_type", "Entity"),
            source_chunk_ids=f.get("source_chunk_ids", []),
        )
        for f in graph_facts_raw
    ]

    return QueryResponse(
        answer=answer,
        mode=QueryMode.GRAPH,
        sources=sources,
        graph_facts=graph_facts,
        question=question,
    )
