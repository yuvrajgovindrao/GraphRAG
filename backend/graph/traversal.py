"""
Graph traversal utilities for seed node identification and expansion.
"""

from __future__ import annotations

import logging

from backend.graph.neo4j_client import Neo4jClient
from backend.llm_provider import BaseLLMProvider
from backend.extraction.extractor import extract_query_entities
from backend.retrieval.vector_search import search_similar

logger = logging.getLogger(__name__)


async def find_seed_entities(
    question: str,
    llm: BaseLLMProvider,
    neo4j: Neo4jClient,
    qdrant_client=None,
) -> list[str]:
    """
    Find seed entities for graph traversal from a question.

    Strategy:
    1. Extract entities from the question via LLM.
    2. Match them against the graph.
    3. Fallback: use vector search on chunks to find mentioned entities.
    """
    # Step 1: LLM entity extraction from question
    query_entities = await extract_query_entities(llm, question)
    logger.info("Query entity extraction: %s", query_entities)

    # Step 2: Match against graph
    matched = neo4j.find_entities_by_names(query_entities)
    seed_names = list(set(e["name"] for e in matched))

    # Step 3: If no matches, fall back to vector search on chunks
    if not seed_names and qdrant_client:
        vector_results = await search_similar(qdrant_client, llm, question, top_k=3)
        # Extract entity names from the chunk texts
        chunk_texts = " ".join([r["text"] for r in vector_results])
        if chunk_texts:
            fallback_entities = await extract_query_entities(llm, chunk_texts[:2000])
            matched = neo4j.find_entities_by_names(fallback_entities)
            seed_names = list(set(e["name"] for e in matched))

    logger.info("Seed entities for traversal: %s", seed_names[:10])
    return seed_names


def get_graph_context(
    neo4j: Neo4jClient,
    seed_entities: list[str],
    max_hops: int = 2,
    max_facts: int = 20,
) -> list[dict]:
    """
    Expand from seed entities and return structured graph facts.
    """
    facts = neo4j.expand_from_entities(
        entity_names=seed_entities,
        max_hops=max_hops,
        max_facts=max_facts,
    )
    return facts


def format_graph_facts(facts: list[dict]) -> str:
    """Format graph facts into a readable string for the LLM prompt."""
    if not facts:
        return "No graph facts available."

    lines = []
    for fact in facts:
        source = fact.get("source_entity", "?")
        relation = fact.get("relation", "RELATED_TO")
        target = fact.get("target_entity", "?")
        desc = fact.get("description", "")
        line = f"- {source} —[{relation}]→ {target}"
        if desc:
            line += f" ({desc})"
        lines.append(line)

    return "\n".join(lines)
