"""Split extracted documents into overlapping chunks with source metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from document_loader import ExtractedDocument, PageContent

WORD_BOUNDARY = re.compile(r"\s+")


@dataclass
class TextChunk:
    document_id: str
    filename: str
    file_type: str
    page_number: int | None
    section_heading: str | None
    chunk_id: str
    text: str
    source_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_metadata": self.source_metadata,
        }


def _is_heading_line(line: str, known_headings: list[str]) -> bool:
    stripped = line.strip()
    if stripped in known_headings:
        return True
    if stripped.startswith("#") and len(stripped) < 120:
        return True
    return False


def _segments_from_page(page: PageContent) -> list[tuple[str | None, str]]:
    """Return (section_heading, text) segments while preserving heading context."""
    lines = (page.text or "").split("\n")
    current_heading: str | None = None
    buffer: list[str] = []
    segments: list[tuple[str | None, str]] = []
    known = page.headings or []

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            segments.append((current_heading, text))
        buffer = []

    for line in lines:
        if _is_heading_line(line, known) and line.strip():
            flush()
            current_heading = line.strip().lstrip("#").strip() or current_heading
            continue
        buffer.append(line)
    flush()
    if not segments and (page.text or "").strip():
        heading = known[0] if known else None
        segments.append((heading, page.text.strip()))
    return segments


def _window_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = WORD_BOUNDARY.sub(" ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            space = text.rfind(" ", start + max(chunk_size // 2, 1), end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(0, end - overlap)
        if start >= end:
            start = end
    return chunks


def chunk_document(
    extracted: ExtractedDocument,
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    if chunk_size < 50:
        chunk_size = 50
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks: list[TextChunk] = []
    counter = 0
    for page in extracted.pages:
        for heading, body in _segments_from_page(page):
            for piece in _window_chunks(body, chunk_size, overlap):
                counter += 1
                chunks.append(
                    TextChunk(
                        document_id=document_id,
                        filename=extracted.filename,
                        file_type=extracted.file_type,
                        page_number=page.page_number,
                        section_heading=heading,
                        chunk_id=f"chunk_{counter:03d}",
                        text=piece,
                        source_metadata=dict(extracted.source_metadata or {}),
                    )
                )
    return chunks


def overlap_is_respected(chunks: list[str], overlap: int) -> bool:
    """Utility used by tests: consecutive overlapping windows share characters when possible."""
    if len(chunks) < 2 or overlap <= 0:
        return True
    for left, right in zip(chunks, chunks[1:]):
        shared = 0
        max_check = min(len(left), len(right), overlap + 20)
        for size in range(max_check, 0, -1):
            if left[-size:] == right[:size] or left[-size:].strip() in right[: size + 10]:
                shared = size
                break
        if shared == 0 and overlap > 0:
            tail = left[-overlap:]
            if tail.strip() and tail.strip() not in right:
                return False
    return True
