"""
SQLite database for document and chunk metadata.
Uses aiosqlite for async operations.
"""

from __future__ import annotations

import aiosqlite
from pathlib import Path

DB_PATH: Path | None = None


async def init_db(db_path: Path) -> None:
    """Create tables if they don't exist."""
    global DB_PATH
    DB_PATH = db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'processing',
                total_chunks INTEGER DEFAULT 0,
                processed_chunks INTEGER DEFAULT 0,
                error_message TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        await db.commit()


def _get_db_path() -> str:
    if DB_PATH is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return str(DB_PATH)


async def insert_document(doc_id: str, filename: str, file_type: str) -> None:
    async with aiosqlite.connect(_get_db_path()) as db:
        await db.execute(
            "INSERT INTO documents (id, filename, file_type) VALUES (?, ?, ?)",
            (doc_id, filename, file_type),
        )
        await db.commit()


async def update_document_status(
    doc_id: str,
    status: str,
    *,
    total_chunks: int | None = None,
    processed_chunks: int | None = None,
    error_message: str | None = None,
) -> None:
    async with aiosqlite.connect(_get_db_path()) as db:
        fields = ["status = ?"]
        values: list = [status]

        if total_chunks is not None:
            fields.append("total_chunks = ?")
            values.append(total_chunks)
        if processed_chunks is not None:
            fields.append("processed_chunks = ?")
            values.append(processed_chunks)
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)
        if status in ("ready", "failed"):
            fields.append("completed_at = CURRENT_TIMESTAMP")

        values.append(doc_id)
        await db.execute(
            f"UPDATE documents SET {', '.join(fields)} WHERE id = ?", values
        )
        await db.commit()


async def get_document(doc_id: str) -> dict | None:
    async with aiosqlite.connect(_get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_all_documents() -> list[dict]:
    async with aiosqlite.connect(_get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def delete_document(doc_id: str) -> bool:
    async with aiosqlite.connect(_get_db_path()) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()
        return cursor.rowcount > 0


async def insert_chunks(chunks: list[dict]) -> None:
    async with aiosqlite.connect(_get_db_path()) as db:
        await db.executemany(
            "INSERT INTO chunks (id, doc_id, text, page_number, chunk_index) "
            "VALUES (:id, :doc_id, :text, :page_number, :chunk_index)",
            chunks,
        )
        await db.commit()


async def get_chunks_by_doc(doc_id: str) -> list[dict]:
    async with aiosqlite.connect(_get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    async with aiosqlite.connect(_get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
