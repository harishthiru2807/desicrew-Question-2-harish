from __future__ import annotations

import pytest

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from document_loader import DocumentLoadError, load_document
from text_chunker import chunk_document
from tests.conftest import make_docx_bytes, make_pdf_bytes


def test_txt_extraction():
    data = b"Title Line That Looks Heading\n\nBody paragraph about annual leave."
    extracted = load_document(data, "policy.txt")
    assert extracted.ok
    assert extracted.file_type == "txt"
    assert extracted.pages[0].page_number is None
    assert "annual leave" in extracted.full_text
    assert extracted.filename == "policy.txt"


def test_docx_extraction_and_headings():
    data = make_docx_bytes(
        [
            ("Heading 1", "Leave Eligibility"),
            ("Normal", "Employees receive 18 days after probation."),
        ]
    )
    extracted = load_document(data, "policy.docx")
    assert extracted.ok
    assert extracted.file_type == "docx"
    assert extracted.pages[0].page_number is None
    headings = extracted.pages[0].headings
    assert any("Leave Eligibility" in h for h in headings)
    assert "18 days" in extracted.full_text


def test_pdf_extraction_with_page_metadata():
    data = make_pdf_bytes(["Page one policy text about probation.", "Page two remaining details."])
    extracted = load_document(data, "policy.pdf")
    assert extracted.ok
    assert extracted.file_type == "pdf"
    assert [p.page_number for p in extracted.pages] == [1, 2]
    assert extracted.source_metadata.get("page_count") == 2
    assert "probation" in extracted.full_text.lower()


def test_empty_txt_is_not_indexed_as_success():
    extracted = load_document(b"   \n", "empty.txt")
    assert extracted.is_empty
    assert not extracted.ok
    assert extracted.error


def test_unsupported_format():
    with pytest.raises(DocumentLoadError):
        load_document(b"hello", "notes.xlsx")


def test_invalid_docx_content():
    extracted = load_document(b"this is not a docx", "broken.docx")
    assert not extracted.ok
    assert extracted.error


def test_invalid_pdf_content():
    extracted = load_document(b"%PDF-1.4 not-really-valid", "broken.pdf")
    assert not extracted.ok


def test_scanned_pdf_without_text_requests_ocr_without_claiming_it_ran():
    data = make_pdf_bytes(["ignored"], image_only=True)
    extracted = load_document(data, "scan.pdf")
    assert extracted.needs_ocr
    assert not extracted.ok
    assert "OCR was not performed" in (extracted.error or "")


def test_heading_detection_in_txt():
    text = "# Leave Eligibility\nEmployees receive 18 days.\n\n## Work From Home\nTwo days after probation."
    extracted = load_document(text.encode("utf-8"), "headings.txt")
    chunks = chunk_document(extracted, "doc1", chunk_size=120, overlap=20)
    headings = {c.section_heading for c in chunks}
    assert "Leave Eligibility" in headings or any(c.section_heading and "Leave" in c.section_heading for c in chunks)


def test_chunk_size_and_overlap():
    body = " ".join([f"sentence number {i} about workplace policy." for i in range(80)])
    extracted = load_document(body.encode("utf-8"), "long.txt")
    chunks = chunk_document(extracted, "doc1", chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
    assert len(chunks) >= 2
    assert all(len(c.text) <= DEFAULT_CHUNK_SIZE + 20 for c in chunks)
    left, right = chunks[0].text, chunks[1].text
    assert left[-40:].split()[-1] in right or any(word in right for word in left[-DEFAULT_CHUNK_OVERLAP:].split()[-4:])


def test_metadata_preservation_on_chunks():
    data = make_pdf_bytes(["LEAVE POLICY\nAnnual leave is 18 days after probation."])
    extracted = load_document(data, "meta.pdf")
    chunks = chunk_document(extracted, "abc123", chunk_size=200, overlap=20)
    assert chunks
    chunk = chunks[0]
    assert chunk.document_id == "abc123"
    assert chunk.filename == "meta.pdf"
    assert chunk.file_type == "pdf"
    assert chunk.page_number == 1
    assert chunk.chunk_id.startswith("chunk_")
    assert chunk.source_metadata.get("filename") == "meta.pdf"


def test_empty_file_bytes():
    extracted = load_document(b"", "blank.txt")
    assert extracted.is_empty
    assert not extracted.ok
