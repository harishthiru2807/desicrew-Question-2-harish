from __future__ import annotations

from answer_generator import LLMClient, LLMError, generate_grounded_answer
from citations import Citation, citations_from_passages, validate_citations
from config import INSUFFICIENT_EVIDENCE_MESSAGE, AppConfig
from conversation_memory import ConversationMemory
from retriever import RetrievedPassage
from text_chunker import TextChunk
from topic_manager import InterpretedQuery


class FakeLLM(LLMClient):
    def __init__(self, text: str = "Employees receive 18 days of annual leave after probation.", fail: LLMError | None = None):
        self._text = text
        self._fail = fail
        self.config = AppConfig(llm_api_key="sk-test")
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._fail is None or self._fail.kind != "missing_credentials"

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        self.calls += 1
        if self._fail:
            raise self._fail
        return self._text


def _chunk(text: str, filename: str = "policy.txt", doc="d1", cid="chunk_001") -> TextChunk:
    return TextChunk(
        document_id=doc,
        filename=filename,
        file_type="txt",
        page_number=None,
        section_heading="Leave Eligibility",
        chunk_id=cid,
        text=text,
        source_metadata={"filename": filename},
    )


def _interpreted(question: str) -> InterpretedQuery:
    return InterpretedQuery(
        original=question,
        search_query=question,
        is_follow_up=False,
        topic_switched=False,
        returning_to_earlier_topic=False,
        needs_clarification=False,
        clarification_question=None,
        topic_label="leave",
        use_history_for_retrieval=False,
        correction=False,
        recap_requested=False,
        interpretation_source="fallback",
    )


def test_answerable_question_with_mocked_llm(runtime):
    runtime.process_upload(
        b"After probation ends, full-time employees receive 18 days of paid annual leave per calendar year.",
        "policy.txt",
    )
    runtime.llm_client = FakeLLM()
    memory = ConversationMemory()
    answer = runtime.ask("How many annual leave days after probation?", memory)
    assert answer.llm_used
    assert "18" in answer.text
    assert answer.citations
    assert all(runtime.store.contains_chunk(c.document_id, c.chunk_id) for c in answer.citations)


def test_partially_answerable_question(runtime):
    runtime.process_upload(
        b"Sick leave is 10 days per calendar year. This policy does not mention cafeteria benefits.",
        "policy.txt",
    )
    runtime.llm_client = FakeLLM(
        "Sick leave is 10 days per calendar year. The passages do not specify parking benefits."
    )
    memory = ConversationMemory()
    answer = runtime.ask("How much sick leave is provided and is parking free?", memory)
    assert "10" in answer.text
    assert "parking" in answer.text.lower() or "not" in answer.text.lower()


def test_unanswerable_questions_use_required_message(runtime):
    runtime.process_upload(b"HelioDesk does not support offline mode.", "product.txt")
    runtime.config.min_relevance_score = 0.99
    runtime.retriever.min_score = 0.99
    memory = ConversationMemory()
    answer = runtime.ask("What is the CEO salary band and Friday cafeteria menu?", memory)
    assert answer.text == INSUFFICIENT_EVIDENCE_MESSAGE
    assert answer.citations == []


def test_citation_validation_keeps_only_retrieved_chunks():
    chunk = _chunk("18 days of annual leave")
    passage = RetrievedPassage(chunk=chunk, score=0.8)
    good = citations_from_passages([passage])
    fake = Citation(
        filename="missing.pdf",
        page_number=99,
        section_heading="Invented",
        chunk_id="chunk_999",
        document_id="nope",
        passage="invented",
    )
    valid = validate_citations(good + [fake], [passage], store=None)
    assert len(valid) == 1
    assert valid[0].chunk_id == "chunk_001"
    assert valid[0].filename == "policy.txt"


def test_invalid_citations_are_dropped_when_document_removed(runtime):
    runtime.process_upload(b"Primary caregiver parental leave is 16 weeks.", "policy.txt")
    memory = ConversationMemory()
    runtime.llm_client = FakeLLM("Parental leave for the primary caregiver is 16 weeks.")
    answer = runtime.ask("How long is primary parental leave?", memory)
    assert answer.citations
    doc_id = next(iter(runtime.store.documents))
    runtime.remove_document(doc_id)
    leftover = validate_citations(answer.citations, answer.passages, runtime.store)
    assert leftover == []


def test_missing_api_credentials_evidence_only(runtime):
    runtime.process_upload(b"Error HD-441 means the ticket attachment exceeds 25 MB.", "guide.txt")
    assert not runtime.llm_client.available
    memory = ConversationMemory()
    answer = runtime.ask("What is HD-441?", memory)
    assert answer.mode == "evidence_only"
    assert answer.llm_used is False
    assert "LLM_API_KEY" in (answer.configuration_message or "")
    assert answer.citations
    assert "fabricat" not in answer.text.lower()


def test_llm_errors_fall_back_to_evidence(runtime):
    runtime.process_upload(b"P1 incidents require a first response within 15 minutes.", "guide.txt")

    class Boom(FakeLLM):
        @property
        def available(self) -> bool:
            return True

        def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
            raise LLMError("rate", kind="rate_limit")

    runtime.llm_client = Boom()
    memory = ConversationMemory()
    answer = runtime.ask("What is the P1 first response target?", memory)
    assert answer.mode == "error"
    assert answer.error_kind == "rate_limit"
    assert answer.citations


def test_evidence_only_mode_does_not_invent_facts(runtime):
    runtime.process_upload(b"Unused annual leave: a maximum of 5 days may be carried over.", "policy.txt")
    memory = ConversationMemory()
    answer = runtime.ask("How many unused days can be carried over?", memory)
    assert answer.mode == "evidence_only"
    assert "5" in answer.text
    assert "CEO" not in answer.text


def test_gemini_config_and_properties():
    cfg = AppConfig(llm_api_key="AIzaSyTestKey123", llm_model="gemini-1.5-flash")
    assert cfg.gemini_api_key == "AIzaSyTestKey123"
    assert cfg.gemini_model == "gemini-1.5-flash"
    assert cfg.llm_configured is True

    cfg.gemini_api_key = "AIzaSyNewKey456"
    assert cfg.llm_api_key == "AIzaSyNewKey456"

    cfg.gemini_model = "gemini-2.0-flash"
    assert cfg.llm_model == "gemini-2.0-flash"


def test_gemini_rest_api_parsing_and_generation(monkeypatch):
    import io
    import json
    import urllib.request

    cfg = AppConfig(llm_api_key="AIzaSyMockedKey", llm_model="gemini-1.5-flash")
    client = LLMClient(cfg)

    mock_gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Employees receive 18 days of paid annual leave."}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ]
    }

    class MockResponse:
        def __init__(self):
            pass

        def read(self):
            return json.dumps(mock_gemini_response).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    recorded_requests = []

    def fake_urlopen(req, timeout=None):
        recorded_requests.append(req)
        return MockResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = client.complete("System prompt instructions", "User question prompt")
    assert result == "Employees receive 18 days of paid annual leave."
    assert len(recorded_requests) == 1
    req = recorded_requests[0]
    assert "generativelanguage.googleapis.com" in req.full_url
    assert "gemini-1.5-flash:generateContent" in req.full_url
    assert "key=AIzaSyMockedKey" in req.full_url
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["contents"][0]["parts"][0]["text"] == "User question prompt"
    assert payload["system_instruction"]["parts"][0]["text"] == "System prompt instructions"
