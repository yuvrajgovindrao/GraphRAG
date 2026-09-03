"""
PDF and text file parsing.
Extracts raw text from uploaded documents using PyMuPDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf  # PyMuPDF (formerly fitz)


@dataclass
class PageText:
    """Text content from a single page."""
    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """Full parsed document with per-page text."""
    filename: str
    pages: list[PageText] = field(default_factory=list)
    full_text: str = ""

    def __post_init__(self):
        if not self.full_text and self.pages:
            self.full_text = "\n\n".join(p.text for p in self.pages if p.text.strip())


def _clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove control chars."""
    # Remove control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Normalize multiple spaces to single
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple newlines to double
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Remove common header/footer patterns (page numbers)
    text = re.sub(r"^\s*-?\s*\d+\s*-?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    return text.strip()


def parse_pdf(file_path: Path) -> ParsedDocument:
    """Extract text from a PDF file, page by page."""
    pages: list[PageText] = []

    with pymupdf.open(str(file_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            try:
                raw_text = page.get_text("text")
                cleaned = _clean_text(raw_text)
                if cleaned:
                    pages.append(PageText(page_number=page_num, text=cleaned))
            except Exception as e:
                # Log page error but keep parsing remaining pages
                continue

    return ParsedDocument(filename=file_path.name, pages=pages)


def parse_txt(file_path: Path) -> ParsedDocument:
    """Read a plain text file."""
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    cleaned = _clean_text(raw_text)
    pages = [PageText(page_number=1, text=cleaned)] if cleaned else []
    return ParsedDocument(filename=file_path.name, pages=pages)


def parse_document(file_path: Path) -> ParsedDocument:
    """Auto-detect file type and parse accordingly."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix in (".txt", ".text", ".md"):
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md")
