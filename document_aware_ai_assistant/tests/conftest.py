from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig
from embedding_service import DeterministicEmbeddingService
from runtime import AssistantRuntime


@pytest.fixture
def embeddings():
    return DeterministicEmbeddingService(dim=96)


@pytest.fixture
def config():
    return AppConfig(
        llm_api_key="",
        chunk_size=800,
        chunk_overlap=150,
        top_k=5,
        min_relevance_score=0.05,
    )


@pytest.fixture
def runtime(embeddings, config):
    return AssistantRuntime(embeddings=embeddings, config=config)


def make_pdf_bytes(pages: list[str], *, image_only: bool = False) -> bytes:
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if image_only:
            # Blank page with no text layer: stands in for a scanned PDF.
            continue
        if text:
            page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_docx_bytes(paragraphs: list[tuple[str, str]]) -> bytes:
    from docx import Document

    document = Document()
    for style, text in paragraphs:
        para = document.add_paragraph(text)
        if style:
            try:
                para.style = style
            except ValueError:
                para.style = document.styles["Normal"]
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
