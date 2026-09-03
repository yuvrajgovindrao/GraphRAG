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
    vector_results: list[dict] | None = None,
) -> list[str]:
    """
    Find seed entities for graph traversal from both question and retrieved passages.

    Strategy:
    1. Extract entities from the user's question via LLM and match against the graph.
    2. Extract entities directly linked to the top vector search chunks across all documents.
       This guarantees that compound multi-document questions include seed nodes for all documents.
    3. Fallback: vector search on chunks if no matches found.
    """
    # Step 1: LLM entity extraction from question
    query_entities = await extract_query_entities(llm, question)
    logger.info("Query entity extraction: %s", query_entities)

    # Step 2: Match question entities against graph
    matched = neo4j.find_entities_by_names(query_entities)
    seed_names = list(set(e["name"] for e in matched))

    # Step 3: Link entities from retrieved vector chunks (ensures cross-document representation)
    if vector_results:
        chunk_ids = [r.get("chunk_id") for r in vector_results if r.get("chunk_id")]
        chunk_entities = neo4j.find_entities_by_chunk_ids(chunk_ids, limit=25)
        for ce in chunk_entities:
            name = ce.get("name")
            if name and name not in seed_names:
                seed_names.append(name)

    # Step 4: If still no matches, fall back to vector search on chunks
    if not seed_names and qdrant_client:
        fallback_results = await search_similar(qdrant_client, llm, question, top_k=3)
        chunk_texts = " ".join([r["text"] for r in fallback_results])
        if chunk_texts:
            fallback_entities = await extract_query_entities(llm, chunk_texts[:2000])
            matched = neo4j.find_entities_by_names(fallback_entities)
            seed_names = list(set(e["name"] for e in matched))

    logger.info("Total seed entities for traversal: %s (total=%d)", seed_names[:10], len(seed_names))
    return seed_names


def get_graph_context(
    neo4j: Neo4jClient,
    seed_entities: list[str],
    max_hops: int = 2,
    max_facts: int = 45,
    vector_results: list[dict] | None = None,
) -> list[dict]:
    """
    Expand from seed entities and return structured graph facts.
    If vector_results span multiple documents, balance facts across all documents
    to guarantee multi-topic queries include nodes and relationships for every document.
    """
    if vector_results:
        from collections import defaultdict
        chunks_by_doc = defaultdict(list)
        for r in vector_results:
            did = r.get("doc_id")
            cid = r.get("chunk_id")
            if did and cid:
                chunks_by_doc[did].append(cid)

        if len(chunks_by_doc) > 1:
            all_facts = []
            per_doc_limit = max(15, max_facts // len(chunks_by_doc))
            for did, cids in chunks_by_doc.items():
                doc_entities = neo4j.find_entities_by_chunk_ids(cids, limit=20)
                doc_seeds = [e["name"] for e in doc_entities]
                for s in seed_entities:
                    if s not in doc_seeds:
                        doc_seeds.append(s)
                doc_facts = neo4j.expand_from_entities(
                    entity_names=doc_seeds,
                    max_hops=max_hops,
                    max_facts=per_doc_limit,
                    per_seed_limit=6,
                )
                all_facts.extend(doc_facts)
            return all_facts

    facts = neo4j.expand_from_entities(
        entity_names=seed_entities,
        max_hops=max_hops,
        max_facts=max_facts,
        per_seed_limit=10,
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
