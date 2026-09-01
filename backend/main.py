"""
GraphRAG FastAPI Application.
Document upload → knowledge graph extraction → graph-enhanced Q&A.
Serves both REST API and modern HTML/CSS/JS web dashboard.
"""

from __future__ import annotations

import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient

from backend.config import get_settings
from backend.database import (
    init_db, insert_document, update_document_status,
    get_document, get_all_documents, delete_document,
    insert_chunks, get_chunks_by_doc,
)
from backend.models import (
    HealthResponse, UploadResponse, DocumentResponse, ChunkResponse,
    QueryRequest, QueryResponse, QueryMode,
)
from backend.llm_provider import create_llm_provider, BaseLLMProvider
from backend.graph.neo4j_client import Neo4jClient
from backend.ingestion.parser import parse_document
from backend.ingestion.chunker import chunk_document
from backend.retrieval.vector_search import (
    create_qdrant_client, ensure_collection, index_chunks, delete_doc_vectors,
)
from backend.extraction.extractor import extract_from_chunks
from backend.graph.entity_resolution import resolve_entities
from backend.retrieval.graph_rag import vector_only_query, graph_enhanced_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ───────────────────────────────────────────────────────

neo4j_client: Neo4jClient | None = None
qdrant: QdrantClient | None = None
llm: BaseLLMProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown services."""
    global neo4j_client, qdrant, llm
    settings = get_settings()

    # Initialize data directory
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SQLite
    await init_db(settings.db_path)
    logger.info("SQLite initialized at %s", settings.db_path)

    # Initialize Neo4j AuraDB (graceful if credentials not yet configured)
    try:
        if settings.neo4j_password:
            neo4j_client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
            if neo4j_client.verify_connection():
                neo4j_client.setup_indexes()
                logger.info("Neo4j AuraDB connected successfully!")
            else:
                logger.warning("Neo4j connection test failed. Verify NEO4J_URI and credentials in .env")
        else:
            logger.warning("NEO4J_PASSWORD is not set in .env. AuraDB features will be disabled until configured.")
            neo4j_client = None
    except Exception as e:
        logger.warning("Neo4j AuraDB initialization error: %s", e)
        neo4j_client = None

    # Initialize Qdrant (Embedded Local Storage or Cloud)
    try:
        qdrant = create_qdrant_client(settings)
        qdrant.get_collections()
        logger.info("Qdrant vector store connected")
    except Exception as e:
        logger.warning("Qdrant initialization error: %s", e)
        qdrant = None

    # Initialize LLM provider (graceful if no API key)
    try:
        llm = create_llm_provider(settings)
        logger.info("LLM provider: %s", settings.llm_provider.value)
    except Exception as e:
        logger.warning("LLM provider not available: %s", e)
        llm = None

    # Ensure Qdrant collection exists
    if qdrant and llm:
        try:
            ensure_collection(qdrant, llm.embedding_dimension())
        except Exception as e:
            logger.warning("Could not ensure Qdrant collection: %s", e)

    yield

    # Cleanup
    if neo4j_client:
        neo4j_client.close()
    if qdrant:
        qdrant.close()


# ── App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GraphRAG API",
    description="Document upload → Knowledge Graph extraction → Graph-enhanced Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check connectivity to all services."""
    settings = get_settings()
    result = HealthResponse(llm_provider=settings.llm_provider.value)

    # Check Neo4j AuraDB
    if neo4j_client and neo4j_client.verify_connection():
        result.neo4j = "connected"
    else:
        result.neo4j = "disconnected"

    # Check Qdrant
    try:
        if qdrant:
            qdrant.get_collections()
            result.qdrant = "connected"
        else:
            result.qdrant = "disconnected"
    except Exception:
        result.qdrant = "disconnected"

    # SQLite is always available if init succeeded
    result.sqlite = "connected"

    all_connected = all(
        v == "connected" for v in [result.neo4j, result.qdrant, result.sqlite]
    )
    result.status = "ok" if all_connected else "degraded"

    return result


# ── Document Upload ────────────────────────────────────────────────────

async def _process_document(doc_id: str, file_path: Path, filename: str):
    """Background task: parse → chunk → embed → extract → store graph."""
    try:
        # Step 1: Parse document
        logger.info("Parsing document: %s", filename)
        parsed = parse_document(file_path)
        if not parsed.pages:
            await update_document_status(doc_id, "failed", error_message="No text content found in document")
            return

        # Step 2: Chunk
        chunks = chunk_document(parsed, doc_id)
        chunk_dicts = [c.to_dict() for c in chunks]
        await insert_chunks(chunk_dicts)
        await update_document_status(doc_id, "processing", total_chunks=len(chunks))
        logger.info("Created %d chunks for %s", len(chunks), filename)

        # Step 3: Embed and index in Qdrant
        if qdrant and llm:
            chunk_data_for_index = [
                {
                    "id": c.id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "source_filename": c.source_filename,
                }
                for c in chunks
            ]
            await index_chunks(qdrant, llm, chunk_data_for_index)
            logger.info("Indexed %d chunks in Qdrant for %s", len(chunks), filename)

        # Step 4: Extract entities and relationships
        await update_document_status(doc_id, "extracting")

        async def on_progress(processed: int, total: int):
            await update_document_status(
                doc_id, "extracting", processed_chunks=processed
            )

        if llm:
            extraction_results = await extract_from_chunks(
                llm,
                [{"id": c.id, "text": c.text} for c in chunks],
                on_progress=on_progress,
            )

            # Step 5: Entity resolution
            resolved_entities, resolved_relationships = resolve_entities(extraction_results)

            # Step 6: Write to Neo4j
            if neo4j_client:
                neo4j_client.merge_entities(resolved_entities, doc_id)
                neo4j_client.create_relationships(resolved_relationships, doc_id)
                logger.info("Entities & relations saved to Neo4j AuraDB for %s", filename)
            else:
                logger.warning("Neo4j client not connected, skipping graph database write")

        await update_document_status(
            doc_id, "ready",
            total_chunks=len(chunks),
            processed_chunks=len(chunks),
        )
        logger.info("Document processing complete: %s", filename)

    except Exception as e:
        logger.exception("Failed to process document %s: %s", filename, str(e))
        await update_document_status(doc_id, "failed", error_message=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a PDF or text file for processing."""
    # Validate file type
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".text", ".md"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md",
        )

    # Save file
    doc_id = str(uuid.uuid4())
    settings = get_settings()
    file_path = settings.data_dir / f"{doc_id}_{filename}"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    file_path.write_bytes(content)

    # Create document record
    await insert_document(doc_id, filename, suffix.lstrip("."))

    # Start background processing
    background_tasks.add_task(_process_document, doc_id, file_path, filename)

    return UploadResponse(doc_id=doc_id, filename=filename)


# ── Documents ──────────────────────────────────────────────────────────

@app.get("/documents", response_model=list[DocumentResponse])
async def list_documents():
    """List all uploaded documents."""
    docs = await get_all_documents()
    return [DocumentResponse(**d) for d in docs]


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document_info(doc_id: str):
    """Get document details and processing status."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(**doc)


@app.get("/documents/{doc_id}/status")
async def get_document_status(doc_id: str):
    """Get processing status for polling."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id": doc_id,
        "status": doc["status"],
        "total_chunks": doc.get("total_chunks", 0),
        "processed_chunks": doc.get("processed_chunks", 0),
        "error_message": doc.get("error_message"),
    }


@app.get("/documents/{doc_id}/chunks", response_model=list[ChunkResponse])
async def list_chunks(doc_id: str):
    """List all chunks for a document (debug/inspection)."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = await get_chunks_by_doc(doc_id)
    return [ChunkResponse(**c) for c in chunks]


@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str):
    """Delete a document and cascade-delete its chunks, vectors, and graph nodes."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from Qdrant
    if qdrant:
        try:
            delete_doc_vectors(qdrant, doc_id)
        except Exception as e:
            logger.warning("Failed to delete Qdrant vectors for %s: %s", doc_id, str(e))

    # Delete from Neo4j
    if neo4j_client:
        try:
            neo4j_client.delete_document_graph(doc_id)
        except Exception as e:
            logger.warning("Failed to delete Neo4j graph for %s: %s", doc_id, str(e))

    # Delete from SQLite (cascades to chunks)
    await delete_document(doc_id)

    # Delete raw file
    settings = get_settings()
    for f in settings.data_dir.glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)

    return {"message": f"Document {doc_id} deleted successfully"}


# ── Graph Info ─────────────────────────────────────────────────────────

@app.get("/graph/summary")
async def graph_summary():
    """Get graph statistics."""
    if not neo4j_client:
        return {"entity_count": 0, "relationship_count": 0, "status": "Neo4j AuraDB not connected"}
    return neo4j_client.get_graph_summary()


@app.get("/graph/document/{doc_id}")
async def document_graph(doc_id: str, limit: int = 150):
    """Get the entity-relationship subgraph for a document."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not neo4j_client:
        return {"nodes": [], "edges": [], "status": "Neo4j AuraDB not connected"}
    return neo4j_client.get_document_subgraph(doc_id, limit=limit)


# ── Query ──────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question using vector or graph-enhanced retrieval."""
    if not llm or not qdrant:
        raise HTTPException(
            status_code=503,
            detail="Services not ready. Ensure LLM API key and vector store are configured.",
        )
    if request.mode == QueryMode.VECTOR:
        return await vector_only_query(
            question=request.question,
            llm=llm,
            qdrant_client=qdrant,
            top_k=request.top_k,
        )
    else:
        if not neo4j_client:
            # Fallback to vector search if Neo4j is not connected
            logger.info("Neo4j not connected, falling back to vector query")
            return await vector_only_query(
                question=request.question,
                llm=llm,
                qdrant_client=qdrant,
                top_k=request.top_k,
            )
        response = await graph_enhanced_query(
            question=request.question,
            llm=llm,
            qdrant_client=qdrant,
            neo4j=neo4j_client,
            top_k=request.top_k,
        )
        return response


# ── Evaluation (optional endpoint) ────────────────────────────────────

@app.post("/evaluate")
async def run_evaluation(eval_file: str = "eval/test_questions.json"):
    """Run evaluation comparing vector vs graph retrieval."""
    from backend.evaluation.evaluator import evaluate

    eval_path = Path(eval_file)
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail=f"Eval file not found: {eval_file}")

    report = await evaluate(eval_path, llm, qdrant, neo4j_client)
    return {
        "summary": {
            "vector_avg_score": report.vector_avg_score,
            "graph_avg_score": report.graph_avg_score,
            "vector_avg_latency_ms": report.vector_avg_latency_ms,
            "graph_avg_latency_ms": report.graph_avg_latency_ms,
        },
        "markdown_report": report.to_markdown(),
        "num_questions": len(report.results),
    }


# ── Static Files (Frontend) ───────────────────────────────────────────

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")
