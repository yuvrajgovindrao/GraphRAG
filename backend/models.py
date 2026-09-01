"""
Pydantic request/response schemas for the GraphRAG API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    READY = "ready"
    FAILED = "failed"


class QueryMode(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"


# ── Documents ──────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    status: DocumentStatus
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: str | None = None
    uploaded_at: str | None = None
    completed_at: str | None = None


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    message: str = "Document uploaded and processing started."


# ── Chunks ─────────────────────────────────────────────────────────────

class ChunkResponse(BaseModel):
    id: str
    doc_id: str
    text: str
    page_number: int | None = None
    chunk_index: int


# ── Query ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    mode: QueryMode = QueryMode.GRAPH
    top_k: int = Field(default=5, ge=1, le=20)


class SourceCitation(BaseModel):
    chunk_id: str
    text: str
    filename: str | None = None
    page_number: int | None = None
    relevance_score: float | None = None


class GraphFact(BaseModel):
    source_entity: str
    relation: str
    target_entity: str
    source_chunk_ids: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    mode: QueryMode
    sources: list[SourceCitation] = []
    graph_facts: list[GraphFact] = []
    question: str


# ── Health ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    neo4j: str = "connected"
    qdrant: str = "connected"
    sqlite: str = "connected"
    llm_provider: str = ""


# ── Extraction (internal) ─────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: str = ""


class ExtractedRelationship(BaseModel):
    source: str
    relation: str
    target: str
    description: str = ""


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = []
    relationships: list[ExtractedRelationship] = []
    chunk_id: str = ""
