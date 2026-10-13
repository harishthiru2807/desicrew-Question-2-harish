"""Citation validation and formatting grounded in indexed chunk metadata."""

from __future__ import annotations

from dataclasses import dataclass

from retriever import RetrievedPassage
from text_chunker import TextChunk
from vector_store import VectorStore


@dataclass
class Citation:
    filename: str
    page_number: int | None
    section_heading: str | None
    chunk_id: str
    document_id: str
    passage: str
    score: float | None = None

    def format_label(self) -> str:
        parts = [f"Source: {self.filename}"]
        if self.page_number is not None:
            parts.append(f"Page: {self.page_number}")
        if self.section_heading:
            parts.append(f"Section: {self.section_heading}")
        parts.append(f"Chunk: {self.chunk_id}")
        return "[" + " | ".join(parts) + "]"


def citation_from_chunk(chunk: TextChunk, score: float | None = None) -> Citation:
    return Citation(
        filename=chunk.filename,
        page_number=chunk.page_number,
        section_heading=chunk.section_heading,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        passage=chunk.text,
        score=score,
    )


def citations_from_passages(passages: list[RetrievedPassage]) -> list[Citation]:
    seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []
    for passage in passages:
        key = (passage.chunk.document_id, passage.chunk.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation_from_chunk(passage.chunk, score=passage.score))
    return citations


def validate_citations(
    citations: list[Citation],
    passages: list[RetrievedPassage],
    store: VectorStore | None = None,
) -> list[Citation]:
    """Keep only citations that match retrieved (and still-indexed) chunks."""
    allowed = {(p.chunk.document_id, p.chunk.chunk_id): p for p in passages}
    valid: list[Citation] = []
    for citation in citations:
        key = (citation.document_id, citation.chunk_id)
        passage = allowed.get(key)
        if passage is None:
            continue
        if store is not None and not store.contains_chunk(citation.document_id, citation.chunk_id):
            continue
        if store is not None and citation.document_id not in store.documents:
            continue
        # Prefer live indexed text over any model-emitted paraphrase.
        valid.append(citation_from_chunk(passage.chunk, score=passage.score))
    return valid


def filter_stale_passages(passages: list[RetrievedPassage], store: VectorStore) -> list[RetrievedPassage]:
    live: list[RetrievedPassage] = []
    for passage in passages:
        if store.contains_chunk(passage.chunk.document_id, passage.chunk.chunk_id):
            live.append(passage)
    return live
