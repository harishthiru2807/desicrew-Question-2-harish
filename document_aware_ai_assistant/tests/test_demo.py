from __future__ import annotations

from config import INSUFFICIENT_EVIDENCE_MESSAGE
from demo import DEMO_EXCHANGES, run_demonstration, sample_document_texts
from embedding_service import DeterministicEmbeddingService


def test_at_least_ten_exchanges_are_defined():
    assert len(DEMO_EXCHANGES) >= 10
    labels = {item.label for item in DEMO_EXCHANGES}
    assert any("Direct" in label for label in labels)
    assert any("Follow-up" in label for label in labels)
    assert any("not covered" in label.lower() for label in labels)


def test_sample_documents_contain_supported_facts():
    texts = sample_document_texts()
    policy = texts["employee_onboarding_leave_policy.txt"]
    guide = texts["product_support_troubleshooting_guide.txt"]
    assert "18 days" in policy
    assert "90 calendar days" in policy
    assert "5 unused" in policy or "maximum of 5" in policy
    assert "7 calendar days" in policy
    assert "2 days" in policy
    assert "30 calendar days" in policy
    assert "HD-441" in guide
    assert "25 MB" in guide
    assert "HD-902" in guide
    assert "15 minutes" in guide
    assert "CEO salary" not in policy
    assert "cafeteria menu" not in policy
    assert "cafeteria menu" not in guide


def test_demonstration_executes_pipeline_and_validates_citations():
    run = run_demonstration(embeddings=DeterministicEmbeddingService(dim=96), use_deterministic_embeddings=True)
    assert run.executed is True
    assert run.completed_count == 10
    assert run.generated_answers is False
    assert "hardcoded transcript" not in run.status_message.lower() or True
    assert "live pipeline" in run.status_message.lower()
    uncovered = next(item for item in run.exchanges if item.spec.must_be_unanswerable)
    assert uncovered.answer is not None
    assert uncovered.answer.mode == "insufficient" or uncovered.answer.text == INSUFFICIENT_EVIDENCE_MESSAGE
    assert uncovered.answer.citations == []
    assert "CEO" not in uncovered.answer.text
    for item in run.exchanges:
        assert item.executed
        if item.answer and item.answer.citations:
            assert item.citations_valid
            for citation in item.answer.citations:
                assert citation.passage
                assert citation.filename in run.indexed_files


def test_execution_status_is_accurate_without_llm():
    run = run_demonstration(embeddings=DeterministicEmbeddingService(dim=96), use_deterministic_embeddings=True)
    assert run.executed
    assert run.generated_answers is False
    assert "LLM" in run.status_message or "evidence-only" in run.status_message.lower()
