from __future__ import annotations

from conversation_memory import ConversationMemory
from topic_manager import TopicManager


def test_follow_up_question_resolution(runtime):
    runtime.process_upload(
        b"After probation ends, full-time employees receive 18 days of paid annual leave. "
        b"A maximum of 5 unused annual leave days may be carried over into the next calendar year.",
        "policy.txt",
    )
    memory = ConversationMemory()
    runtime.ask("How many annual leave days do employees receive after probation?", memory)
    follow = runtime.ask("Can unused leave be carried over to the next year?", memory)
    assert follow.passages or follow.mode in {"evidence_only", "generated", "insufficient"}
    combined = follow.text.lower() + " ".join(p.text.lower() for p in follow.passages)
    assert "5" in combined or "carry" in combined


def test_references_to_earlier_responses():
    memory = ConversationMemory()
    memory.add_user("What is HD-441?")
    memory.add_assistant("HD-441 means the attachment exceeds 25 MB.")
    assert memory.looks_like_follow_up("Can you explain that more simply?")
    recap = memory.recap_text()
    assert "HD-441" in recap or "attachment" in recap.lower() or "discussed" in recap.lower()


def test_topic_switching(runtime):
    runtime.process_upload(b"Annual leave is 18 days after probation.", "policy.txt")
    runtime.process_upload(b"Error HD-441 means the ticket attachment exceeds 25 MB.", "guide.txt")
    memory = ConversationMemory()
    runtime.ask("How many annual leave days after probation?", memory)
    switched = TopicManager().interpret("What does error HD-441 mean in HelioDesk?", memory, ["policy.txt", "guide.txt"])
    assert switched.topic_switched or "HD-441" in switched.search_query
    answer = runtime.ask("What does error HD-441 mean in HelioDesk?", memory)
    blob = answer.text + " ".join(p.text for p in answer.passages)
    assert "25" in blob or "HD-441" in blob


def test_returning_to_earlier_topic():
    memory = ConversationMemory()
    memory.add_user("Tell me about leave", topic="leave")
    memory.add_assistant("18 days", topic="leave")
    memory.add_user("What is HD-441?", topic="product")
    memory.add_assistant("25 MB", topic="product")
    interpreted = TopicManager().interpret("Going back to the leave policy, how far in advance is leave requested?", memory)
    assert interpreted.returning_to_earlier_topic


def test_repetition_handling(runtime):
    runtime.process_upload(b"After probation ends, full-time employees receive 18 days of paid annual leave.", "policy.txt")
    memory = ConversationMemory()
    q = "How many annual leave days do full-time employees receive after probation?"
    first = runtime.ask(q, memory)
    second = runtime.ask(q, memory)
    assert second.text
    assert first.text
    # The second turn should still be grounded rather than empty.
    assert second.mode in {"evidence_only", "generated", "insufficient"}


def test_explicit_summary_request():
    memory = ConversationMemory()
    memory.add_user("What is the notice period?")
    memory.add_assistant("Individual contributors provide 30 calendar days of notice.")
    assert memory.wants_recap("Please summarize what we have discussed so far.")
    recap = memory.recap_text()
    assert "30" in recap or "notice" in recap.lower()


def test_conversation_reset_preserves_knowledge_base(runtime):
    runtime.process_upload(b"Security awareness course must be finished by day 14.", "policy.txt")
    memory = ConversationMemory()
    runtime.ask("When is security training due?", memory)
    assert memory.messages
    memory.clear_conversation()
    assert memory.messages == []
    assert memory.summary == ""
    assert not runtime.store.is_empty
    assert runtime.store.document_by_filename("policy.txt") is not None


def test_new_chat_does_not_share_history():
    first = ConversationMemory()
    first.add_user("secret topic unique to session one")
    second = first.new_chat()
    assert second.session_id != first.session_id
    assert second.messages == []
    assert first.messages[0].content == "secret topic unique to session one"


def test_session_isolation_between_runtimes(embeddings, config):
    from runtime import AssistantRuntime

    a = AssistantRuntime(embeddings=embeddings, config=config)
    b = AssistantRuntime(embeddings=embeddings, config=config)
    a.process_upload(b"Only runtime A has the garden leave rule.", "a.txt")
    mem_a = ConversationMemory()
    mem_b = ConversationMemory()
    a.ask("What is garden leave?", mem_a)
    b.ask("What is garden leave?", mem_b)
    assert mem_a.session_id != mem_b.session_id
    assert b.store.is_empty
    assert not a.store.is_empty
