"""
LLM-based knowledge graph extraction from text chunks.
Runs extraction per chunk with concurrency control and retries.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from backend.llm_provider import BaseLLMProvider
from backend.extraction.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    get_extraction_prompt,
    get_query_entity_prompt,
)
from backend.models import (
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
)

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 5
MAX_RETRIES = 3


async def extract_from_chunk(
    llm: BaseLLMProvider,
    chunk_id: str,
    chunk_text: str,
) -> ExtractionResult:
    """Extract entities and relationships from a single chunk."""
    prompt = get_extraction_prompt(chunk_text)

    for attempt in range(MAX_RETRIES):
        try:
            result = await llm.generate_structured(
                prompt=prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
            )

            entities = [
                ExtractedEntity(
                    name=e.get("name", "").strip(),
                    type=e.get("type", "Concept").strip(),
                    description=e.get("description", "").strip(),
                )
                for e in result.get("entities", [])
                if e.get("name", "").strip()
            ]

            relationships = [
                ExtractedRelationship(
                    source=r.get("source", "").strip(),
                    relation=r.get("relation", "RELATED_TO").strip(),
                    target=r.get("target", "").strip(),
                    description=r.get("description", "").strip(),
                )
                for r in result.get("relationships", [])
                if r.get("source", "").strip() and r.get("target", "").strip()
            ]

            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                chunk_id=chunk_id,
            )

        except Exception as e:
            logger.warning(
                "Extraction attempt %d/%d failed for chunk %s: %s",
                attempt + 1, MAX_RETRIES, chunk_id, str(e),
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)  # exponential backoff
            else:
                logger.error("All retries exhausted for chunk %s", chunk_id)
                return ExtractionResult(chunk_id=chunk_id)


async def extract_from_chunks(
    llm: BaseLLMProvider,
    chunks: list[dict],
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> list[ExtractionResult]:
    """
    Extract entities/relationships from multiple chunks with concurrency control.

    Args:
        llm: The LLM provider to use.
        chunks: List of chunk dicts with 'id' and 'text' keys.
        on_progress: Optional async callback(processed_count, total_count).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results: list[ExtractionResult] = []
    processed = 0
    total = len(chunks)

    async def _extract_one(chunk: dict) -> ExtractionResult:
        nonlocal processed
        async with semaphore:
            result = await extract_from_chunk(llm, chunk["id"], chunk["text"])
            processed += 1
            if on_progress:
                await on_progress(processed, total)
            return result

    tasks = [_extract_one(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)

    total_entities = sum(len(r.entities) for r in results)
    total_rels = sum(len(r.relationships) for r in results)
    logger.info(
        "Extraction complete: %d chunks → %d entities, %d relationships",
        total, total_entities, total_rels,
    )

    return list(results)


async def extract_query_entities(llm: BaseLLMProvider, question: str) -> list[str]:
    """Extract entity names from a user question for graph seed lookup."""
    prompt = get_query_entity_prompt(question)
    try:
        result = await llm.generate_structured(prompt=prompt)
        entities = result.get("entities", [])
        return [e.strip() for e in entities if isinstance(e, str) and e.strip()]
    except Exception as e:
        logger.warning("Failed to extract query entities: %s", str(e))
        return []
