"""
Entity and relationship extraction prompts.
Designed for structured JSON output from any LLM provider.
"""

ENTITY_TYPES = [
    "Person",
    "Organization",
    "Concept",
    "Event",
    "Location",
    "Technology",
    "Document",
    "Date",
    "Metric",
]

RELATIONSHIP_TYPES = [
    "WORKS_FOR",
    "RELATED_TO",
    "PART_OF",
    "LOCATED_IN",
    "CREATED_BY",
    "MENTIONS",
    "DEPENDS_ON",
    "FUNDED_BY",
    "PRECEDED_BY",
    "FOLLOWED_BY",
    "COLLABORATED_WITH",
    "PUBLISHED_IN",
    "ACQUIRED_BY",
    "COMPETED_WITH",
    "USED_BY",
    "CONTAINS",
    "CAUSES",
    "MEASURES",
]

EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph extraction system. 
Your task is to extract entities and relationships from text to build a knowledge graph.

You MUST follow these rules:
1. Extract only entities and relationships that are explicitly stated or strongly implied in the text.
2. Each entity must have a name, type, and brief description.
3. Each relationship must have a source entity, relationship type, target entity, and brief description.
4. Use the standardized entity types and relationship types provided.
5. Normalize entity names: use proper capitalization, full names (not abbreviations).
6. If an entity doesn't fit the predefined types, use the closest match or "Concept".
7. Do NOT invent or hallucinate information not present in the text.
8. Keep descriptions concise (1-2 sentences max).
"""

EXTRACTION_PROMPT_TEMPLATE = """Extract all entities and relationships from the following text chunk.

## Allowed Entity Types
{entity_types}

## Allowed Relationship Types
{relationship_types}

## Text to Extract From
---
{text}
---

## Output Format
Return a JSON object with this exact structure:
{{
    "entities": [
        {{
            "name": "Entity Name",
            "type": "EntityType",
            "description": "Brief description of the entity"
        }}
    ],
    "relationships": [
        {{
            "source": "Source Entity Name",
            "relation": "RELATIONSHIP_TYPE",
            "target": "Target Entity Name",
            "description": "Brief description of the relationship"
        }}
    ]
}}

If no entities or relationships are found, return empty arrays.
Extract entities and relationships now:"""


QUERY_ENTITY_EXTRACTION_PROMPT = """Extract the key entities mentioned in this question. 
Return a JSON object with an "entities" array containing entity names.

Question: {question}

Return JSON:
{{
    "entities": ["entity1", "entity2"]
}}"""


def get_extraction_prompt(text: str) -> str:
    """Build the extraction prompt for a given text chunk."""
    return EXTRACTION_PROMPT_TEMPLATE.format(
        entity_types=", ".join(ENTITY_TYPES),
        relationship_types=", ".join(RELATIONSHIP_TYPES),
        text=text,
    )


def get_query_entity_prompt(question: str) -> str:
    """Build the entity extraction prompt for a user question."""
    return QUERY_ENTITY_EXTRACTION_PROMPT.format(question=question)
