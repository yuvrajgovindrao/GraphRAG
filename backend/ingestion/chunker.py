"""
Text chunking with metadata.
Uses LlamaIndex SentenceSplitter for intelligent sentence-boundary chunking.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from llama_index.core.node_parser import SentenceSplitter

from backend.ingestion.parser import ParsedDocument


@dataclass
class Chunk:
    """A text chunk with metadata."""
    id: str
    doc_id: str
    text: str
    page_number: int | None
    chunk_index: int
    source_filename: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }


def chunk_document(
    parsed_doc: ParsedDocument,
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Split a parsed document into chunks with metadata.

    Strategy: concatenate all page texts, then split using SentenceSplitter.
    We track page boundaries to assign correct page numbers to each chunk.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Build a mapping of character offset → page number
    page_offsets: list[tuple[int, int, int]] = []  # (start, end, page_num)
    offset = 0
    for page in parsed_doc.pages:
        page_len = len(page.text)
        page_offsets.append((offset, offset + page_len, page.page_number))
        offset += page_len + 2  # +2 for the "\n\n" separator

    full_text = "\n\n".join(p.text for p in parsed_doc.pages)

    if not full_text.strip():
        return []

    # Split text into chunks
    chunk_texts = splitter.split_text(full_text)

    chunks: list[Chunk] = []
    search_start = 0

    for idx, chunk_text in enumerate(chunk_texts):
        # Find where this chunk starts in the full text
        chunk_start = full_text.find(chunk_text[:100], search_start)
        if chunk_start == -1:
            chunk_start = search_start  # fallback

        # Determine which page this chunk belongs to (by its start position)
        page_num = None
        for start, end, pn in page_offsets:
            if start <= chunk_start < end:
                page_num = pn
                break
        if page_num is None and page_offsets:
            page_num = page_offsets[-1][2]  # default to last page

        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=chunk_text,
            page_number=page_num,
            chunk_index=idx,
            source_filename=parsed_doc.filename,
        ))

        search_start = chunk_start + 1

    return chunks
