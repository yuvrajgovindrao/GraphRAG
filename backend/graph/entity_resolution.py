"""
Entity resolution: normalize and deduplicate entities across chunks.
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict

from backend.models import ExtractionResult, ExtractedEntity, ExtractedRelationship

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Normalize entity name for comparison."""
    # Lowercase, strip, remove punctuation for comparison key
    normalized = name.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _pick_best_name(names: list[str]) -> str:
    """Pick the most descriptive (longest, properly cased) name variant."""
    # Prefer the longest name, then the one with most uppercase letters (proper casing)
    return max(names, key=lambda n: (len(n), sum(1 for c in n if c.isupper())))


def _merge_descriptions(descriptions: list[str]) -> str:
    """Combine unique descriptions."""
    unique = []
    seen = set()
    for desc in descriptions:
        desc = desc.strip()
        if desc and desc.lower() not in seen:
            unique.append(desc)
            seen.add(desc.lower())
    return " ".join(unique[:3])  # cap at 3 descriptions to avoid bloat


def resolve_entities(
    extraction_results: list[ExtractionResult],
) -> tuple[list[dict], list[dict]]:
    """
    Deduplicate entities and update relationship references.

    Returns:
        (resolved_entities, resolved_relationships)
        Each entity dict: {name, type, description, chunk_ids}
        Each relationship dict: {source, relation, target, description, chunk_ids}
    """
    # Group entities by normalized name
    entity_groups: dict[str, list[tuple[ExtractedEntity, str]]] = defaultdict(list)
    for result in extraction_results:
        for entity in result.entities:
            key = _normalize_name(entity.name)
            if key:
                entity_groups[key].append((entity, result.chunk_id))

    # Resolve: pick best name, merge descriptions, collect chunk_ids
    resolved_entities: list[dict] = []
    name_mapping: dict[str, str] = {}  # normalized_name → canonical_name

    for norm_name, entries in entity_groups.items():
        entities, chunk_ids = zip(*entries)
        canonical_name = _pick_best_name([e.name for e in entities])
        # Pick the most common type
        type_counts: dict[str, int] = defaultdict(int)
        for e in entities:
            type_counts[e.type] += 1
        best_type = max(type_counts, key=type_counts.get)
        merged_desc = _merge_descriptions([e.description for e in entities])

        name_mapping[norm_name] = canonical_name
        resolved_entities.append({
            "name": canonical_name,
            "type": best_type,
            "description": merged_desc,
            "chunk_ids": list(set(chunk_ids)),
        })

    # Resolve relationships: update names to canonical forms
    relationship_groups: dict[tuple[str, str, str], list[tuple[ExtractedRelationship, str]]] = defaultdict(list)

    for result in extraction_results:
        for rel in result.relationships:
            source_norm = _normalize_name(rel.source)
            target_norm = _normalize_name(rel.target)

            # Map to canonical names (skip if entity not found)
            source_canonical = name_mapping.get(source_norm, rel.source)
            target_canonical = name_mapping.get(target_norm, rel.target)

            key = (source_canonical, rel.relation, target_canonical)
            relationship_groups[key].append((rel, result.chunk_id))

    resolved_relationships: list[dict] = []
    for (source, relation, target), entries in relationship_groups.items():
        rels, chunk_ids = zip(*entries)
        merged_desc = _merge_descriptions([r.description for r in rels])
        resolved_relationships.append({
            "source": source,
            "relation": relation,
            "target": target,
            "description": merged_desc,
            "chunk_ids": list(set(chunk_ids)),
        })

    logger.info(
        "Entity resolution: %d raw → %d resolved entities, %d resolved relationships",
        sum(len(r.entities) for r in extraction_results),
        len(resolved_entities),
        len(resolved_relationships),
    )

    return resolved_entities, resolved_relationships
