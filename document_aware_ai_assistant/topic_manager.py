"""Topic switching, follow-up interpretation, and reference resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass

from conversation_memory import ConversationMemory

RETURN_PATTERNS = (
    "go back",
    "going back",
    "return to",
    "earlier topic",
    "previous topic",
    "back to",
    "returning to",
)
CLARIFY_PATTERNS = (
    "more simply",
    "explain that",
    "what does that mean",
    "clarify",
    "in other words",
    "can you explain",
)
SWITCH_HINTS = (
    "unrelated",
    "different document",
    "instead",
    "switching",
    "another document",
    "now about",
    "on a different note",
)


@dataclass
class InterpretedQuery:
    original: str
    search_query: str
    is_follow_up: bool
    topic_switched: bool
    returning_to_earlier_topic: bool
    needs_clarification: bool
    clarification_question: str | None
    topic_label: str
    use_history_for_retrieval: bool
    correction: bool
    recap_requested: bool
    interpretation_source: str  # "llm" or "fallback"


class TopicManager:
    """LLM-backed interpretation with a deterministic fallback."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def interpret(self, question: str, memory: ConversationMemory, document_names: list[str] | None = None) -> InterpretedQuery:
        document_names = document_names or memory.document_mentions
        if self.llm_client is not None and getattr(self.llm_client, "available", False):
            try:
                return self._interpret_with_llm(question, memory, document_names)
            except Exception:
                return self._fallback(question, memory, document_names)
        return self._fallback(question, memory, document_names)

    def _interpret_with_llm(self, question: str, memory: ConversationMemory, document_names: list[str]) -> InterpretedQuery:
        prompt = (
            "Rewrite the user question into a standalone search query for a document knowledge base. "
            "Also classify: follow_up (yes/no), topic_switch (yes/no), return_to_earlier (yes/no), "
            "needs_clarification (yes/no), recap (yes/no), correction (yes/no). "
            "If the reference is ambiguous across multiple documents or topics, needs_clarification=yes. "
            "Reply as lines: SEARCH: ...\nFOLLOW_UP: ...\nTOPIC_SWITCH: ...\nRETURN: ...\nCLARIFY: ...\nRECAP: ...\nCORRECTION: ...\nTOPIC: ...\nCLARIFY_Q: ...\n"
            f"Documents: {', '.join(document_names) or 'none'}\n"
            f"History:\n{memory.context_for_rewrite()}\n"
            f"Question: {question}"
        )
        text = self.llm_client.complete(
            system_prompt="You interpret conversational questions. Do not answer the user's factual question.",
            user_prompt=prompt,
            temperature=0.0,
        )
        parsed = _parse_llm_interpretation(text, question)
        parsed.interpretation_source = "llm"
        return parsed

    def _fallback(self, question: str, memory: ConversationMemory, document_names: list[str]) -> InterpretedQuery:
        lowered = question.lower()
        recap = memory.wants_recap(question)
        returning = any(p in lowered for p in RETURN_PATTERNS)
        clarification = any(p in lowered for p in CLARIFY_PATTERNS)
        correction = any(p in lowered for p in ("actually", "i meant", "correction", "not that", "wrong"))
        follow = memory.looks_like_follow_up(question) or clarification or returning
        explicit_switch = any(p in lowered for p in SWITCH_HINTS)
        mentioned_docs = [name for name in document_names if name.lower().split(".")[0] in lowered]
        last_topic = memory.topics[-1] if memory.topics else "general"
        topic_label = last_topic
        topic_switched = False

        if returning and len(memory.topics) >= 2:
            topic_label = memory.topics[-2]
        elif explicit_switch:
            topic_switched = True
            topic_label = _guess_topic(question)
        elif mentioned_docs and last_topic and mentioned_docs[0].lower().split(".")[0] not in last_topic.lower():
            topic_switched = True
            topic_label = mentioned_docs[0]
        elif not follow and memory.messages:
            overlap = _token_overlap(question, last_topic + " " + (memory.last_user().content if memory.last_user() else ""))
            if overlap < 0.12:
                topic_switched = True
                topic_label = _guess_topic(question)

        ambiguous = _is_ambiguous(question, memory)
        search_query = question
        use_history = follow and not topic_switched
        if use_history and memory.last_user():
            search_query = _expand_with_history(question, memory)
        if returning and memory.topics:
            search_query = f"{question} {topic_label}"
        if recap:
            use_history = True

        clarification_question = None
        if ambiguous:
            clarification_question = (
                "I can interpret that in more than one way from our earlier discussion. "
                "Which document or topic should I use: "
                + _ambiguity_options(memory)
                + "?"
            )

        return InterpretedQuery(
            original=question,
            search_query=search_query,
            is_follow_up=follow and not topic_switched,
            topic_switched=topic_switched and not returning,
            returning_to_earlier_topic=returning,
            needs_clarification=ambiguous,
            clarification_question=clarification_question,
            topic_label=topic_label,
            use_history_for_retrieval=use_history,
            correction=correction,
            recap_requested=recap,
            interpretation_source="fallback",
        )


def _guess_topic(question: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question) if len(w) > 3]
    return " ".join(words[:6]) or "general"


def _token_overlap(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-z0-9]+", a.lower()))
    sb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _expand_with_history(question: str, memory: ConversationMemory) -> str:
    last_user = memory.last_user().content if memory.last_user() else ""
    last_assistant = memory.last_assistant().content if memory.last_assistant() else ""
    entities = " ".join(memory.entities[-8:])
    docs = " ".join(memory.document_mentions)
    return f"{question}\nPrior question: {last_user}\nPrior answer excerpt: {last_assistant[:400]}\nEntities: {entities}\nDocuments: {docs}"


def _is_ambiguous(question: str, memory: ConversationMemory) -> bool:
    lowered = question.lower()
    vague = any(p in lowered for p in ("that document", "that section", "the other one", "the second option"))
    if not vague:
        return False
    if "second option" in lowered and len(memory.messages) == 0:
        return True
    if "that document" in lowered and len(memory.document_mentions) > 1:
        return True
    if lowered.strip() in {"what about that?", "and that?", "that one?"}:
        return True
    return False


def _ambiguity_options(memory: ConversationMemory) -> str:
    options = memory.document_mentions[-3:] or memory.topics[-3:] or ["the current topic"]
    return " or ".join(options)


def _parse_llm_interpretation(text: str, original: str) -> InterpretedQuery:
    def grab(key: str, default: str = "") -> str:
        match = re.search(rf"{key}:\s*(.+)", text, flags=re.I)
        return match.group(1).strip() if match else default

    def flag(key: str) -> bool:
        return grab(key, "no").lower().startswith("y")

    search = grab("SEARCH", original)
    return InterpretedQuery(
        original=original,
        search_query=search or original,
        is_follow_up=flag("FOLLOW_UP"),
        topic_switched=flag("TOPIC_SWITCH"),
        returning_to_earlier_topic=flag("RETURN"),
        needs_clarification=flag("CLARIFY"),
        clarification_question=grab("CLARIFY_Q") or None,
        topic_label=grab("TOPIC") or _guess_topic(original),
        use_history_for_retrieval=flag("FOLLOW_UP") and not flag("TOPIC_SWITCH"),
        correction=flag("CORRECTION"),
        recap_requested=flag("RECAP"),
        interpretation_source="llm",
    )
