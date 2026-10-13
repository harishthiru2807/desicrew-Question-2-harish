from __future__ import annotations

from embedding_service import DeterministicEmbeddingService
from runtime import AssistantRuntime


def _index(runtime: AssistantRuntime, name: str, text: str, replace: bool = False):
    return runtime.process_upload(text.encode("utf-8"), name, replace=replace)


def test_embedding_generation_shape():
    service = DeterministicEmbeddingService(dim=32)
    vectors = service.embed_texts(["alpha leave policy", "beta product error"])
    assert vectors.shape == (2, 32)
    # Unit-ish vectors after L2 normalize
    assert abs(float((vectors[0] ** 2).sum()) - 1.0) < 1e-5


def test_faiss_index_creation_and_retrieval(runtime):
    _index(runtime, "leave.txt", "Annual leave is 18 days after the 90 day probation period ends.")
    _index(runtime, "product.txt", "Error HD-441 means the attachment is larger than 25 MB.")
    assert runtime.store.size >= 2
    hits = runtime.retriever.retrieve("How many annual leave days are provided after probation?", top_k=3)
    assert hits
    assert "18" in hits[0].text or "leave" in hits[0].text.lower()


def test_empty_knowledge_base_retrieval(runtime):
    assert runtime.store.is_empty
    assert runtime.retriever.retrieve("anything", top_k=5) == []


def test_metadata_preserved_during_retrieval(runtime):
    _index(runtime, "leave.txt", "Work from home is allowed for 2 days each week after probation.")
    hits = runtime.retriever.retrieve("work from home days after probation", top_k=3)
    assert hits
    chunk = hits[0].chunk
    assert chunk.filename == "leave.txt"
    assert chunk.document_id in runtime.store.documents
    assert chunk.chunk_id


def test_document_addition(runtime):
    first = _index(runtime, "a.txt", "Alpha onboarding buddy is assigned in week 1.")
    second = _index(runtime, "b.txt", "Beta SLA first response for P1 is 15 minutes.")
    assert first.status == "indexed"
    assert second.status == "indexed"
    assert len(runtime.store.documents) == 2


def test_document_removal_prevents_stale_sources(runtime):
    _index(runtime, "leave.txt", "Notice period for individual contributors is 30 calendar days.")
    _index(runtime, "product.txt", "HelioDesk P1 first response is 15 minutes.")
    leave = runtime.store.document_by_filename("leave.txt")
    assert leave is not None
    runtime.remove_document(leave.document_id)
    hits = runtime.retriever.retrieve("notice period 30 calendar days individual contributors", top_k=5)
    assert all(h.filename != "leave.txt" for h in hits)
    assert "leave.txt" not in runtime.store.filename_to_id


def test_document_replacement_invalidates_old_chunks(runtime):
    _index(runtime, "policy.txt", "Annual leave is 99 days according to the obsolete draft.")
    replaced = _index(runtime, "policy.txt", "Annual leave is 18 days after probation.", replace=True)
    assert replaced.replaced
    assert replaced.status == "indexed"
    texts = " ".join(c.text for c in runtime.store.all_chunks())
    assert "99 days" not in texts
    assert "18 days" in texts
    hits = runtime.retriever.retrieve("how many annual leave days after probation", top_k=4)
    assert hits
    assert all("99 days" not in h.text for h in hits)


def test_duplicate_upload_is_prevented(runtime):
    _index(runtime, "policy.txt", "Probation lasts 90 calendar days.")
    again = _index(runtime, "policy.txt", "Probation lasts 90 calendar days.")
    assert again.status == "duplicate"
    assert len(runtime.store.documents) == 1


def test_index_rebuilding(runtime):
    _index(runtime, "a.txt", "Security training must be finished by day 14.")
    before = runtime.store.size
    rebuilt = runtime.rebuild_index()
    assert rebuilt == before
    hits = runtime.retriever.retrieve("security training day 14", top_k=3)
    assert hits


def test_empty_document_not_added(runtime):
    result = runtime.process_upload(b"   ", "empty.txt")
    assert result.status in {"empty", "rejected"}
    assert runtime.store.is_empty


def test_stale_source_prevention_after_clear(runtime):
    _index(runtime, "a.txt", "Zendesk migrator copies open tickets and contacts.")
    runtime.clear_knowledge_base()
    hits = runtime.retriever.retrieve("zendesk migrator copies open tickets", top_k=5)
    assert hits == []
    assert runtime.store.is_empty
