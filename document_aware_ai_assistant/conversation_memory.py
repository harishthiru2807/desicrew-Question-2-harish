"""Per-session conversation history, compact summaries, and remembered entities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import uuid4

from citations import Citation

FOLLOW_UP_HINTS = (
    "the second",
    "the first",
    "the earlier",
    "previous answer",
    "you said",
    "more simply",
    "explain that",
    "what about",
    "which section",
    "return to",
    "go back",
    "as you mentioned",
    "that mean",
    "unused leave",
    "carry over",
)


@dataclass
class ChatMessage:
    role: str
    content: str
    citations: list[Citation] = field(default_factory=list)
    topic: str | None = None
    follow_ups: list[str] = field(default_factory=list)
    mode: str | None = None


@dataclass
class ConversationMemory:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[ChatMessage] = field(default_factory=list)
    summary: str = ""
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    document_mentions: list[str] = field(default_factory=list)
    facts_explained: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def add_user(self, content: str, topic: str | None = None) -> ChatMessage:
        message = ChatMessage(role="user", content=content, topic=topic)
        self.messages.append(message)
        self._observe_text(content)
        if topic:
            self._remember_topic(topic)
        return message

    def add_assistant(
        self,
        content: str,
        citations: list[Citation] | None = None,
        topic: str | None = None,
        follow_ups: list[str] | None = None,
        mode: str | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            role="assistant",
            content=content,
            citations=citations or [],
            topic=topic,
            follow_ups=follow_ups or [],
            mode=mode,
        )
        self.messages.append(message)
        if topic:
            self._remember_topic(topic)
        for citation in message.citations:
            if citation.filename not in self.document_mentions:
                self.document_mentions.append(citation.filename)
            if citation.section_heading and citation.section_heading not in self.entities:
                self.entities.append(citation.section_heading)
        self.facts_explained.append(content[:240])
        if len(self.facts_explained) > 12:
            self.facts_explained = self.facts_explained[-12:]
        self._refresh_summary()
        return message

    def clear_conversation(self) -> None:
        """Remove conversational state while leaving session identity intact."""
        self.messages = []
        self.summary = ""
        self.entities = []
        self.topics = []
        self.document_mentions = []
        self.facts_explained = []
        self.constraints = []

    def new_chat(self) -> ConversationMemory:
        """Fresh conversational state; caller keeps the knowledge base separately."""
        return ConversationMemory(session_id=uuid4().hex)

    def last_assistant(self) -> ChatMessage | None:
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message
        return None

    def last_user(self) -> ChatMessage | None:
        for message in reversed(self.messages):
            if message.role == "user":
                return message
        return None

    def looks_like_follow_up(self, question: str) -> bool:
        lowered = question.lower().strip()
        if not self.messages:
            return False
        if re.fullmatch(r"(and|also|what about|why|how|yes|no|ok)[?]?", lowered):
            return True
        return any(hint in lowered for hint in FOLLOW_UP_HINTS)

    def wants_recap(self, question: str) -> bool:
        lowered = question.lower()
        return any(
            phrase in lowered
            for phrase in (
                "summarize",
                "summary",
                "recap",
                "what did we discuss",
                "repeat what you said",
                "what have we covered",
            )
        )

    def wants_repeat(self, question: str) -> bool:
        lowered = question.lower()
        return any(
            phrase in lowered
            for phrase in ("what did you say", "repeat that", "you already told", "earlier you said")
        )

    def similar_to_previous_question(self, question: str) -> ChatMessage | None:
        normalized = _normalize(question)
        for message in self.messages:
            if message.role != "user":
                continue
            if _normalize(message.content) == normalized:
                return message
        return None

    def previous_answer_for_question(self, question: str) -> ChatMessage | None:
        match = self.similar_to_previous_question(question)
        if match is None:
            return None
        try:
            index = self.messages.index(match)
        except ValueError:
            return None
        for message in self.messages[index + 1 :]:
            if message.role == "assistant":
                return message
        return None

    def context_for_rewrite(self, limit: int = 6) -> str:
        recent = self.messages[-limit:]
        lines = []
        for message in recent:
            lines.append(f"{message.role}: {message.content}")
        if self.summary:
            lines.insert(0, f"running_summary: {self.summary}")
        if self.topics:
            lines.insert(0, f"topics: {', '.join(self.topics[-5:])}")
        if self.document_mentions:
            lines.insert(0, f"documents: {', '.join(self.document_mentions)}")
        return "\n".join(lines)

    def recap_text(self) -> str:
        if self.summary:
            body = self.summary
        elif self.messages:
            pairs = []
            question = None
            for message in self.messages:
                if message.role == "user":
                    question = message.content
                elif message.role == "assistant" and question:
                    pairs.append(f"- Q: {question}\n  A: {message.content[:400]}")
                    question = None
            body = "\n".join(pairs[-8:]) if pairs else "No prior discussion to recap."
        else:
            body = "No prior discussion to recap."
        return f"Here is a recap of our conversation so far:\n{body}"

    def _observe_text(self, text: str) -> None:
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text):
            if token.lower() in {"what", "how", "which", "when", "does", "can"}:
                continue
            if token not in self.entities:
                self.entities.append(token)
        self.entities = self.entities[-40:]

    def _remember_topic(self, topic: str) -> None:
        topic = topic.strip()
        if not topic:
            return
        if self.topics and self.topics[-1] == topic:
            return
        self.topics.append(topic)
        self.topics = self.topics[-20:]

    def _refresh_summary(self) -> None:
        user_turns = [m.content for m in self.messages if m.role == "user"]
        if not user_turns:
            self.summary = ""
            return
        topics = ", ".join(self.topics[-4:]) if self.topics else "general document questions"
        docs = ", ".join(self.document_mentions) if self.document_mentions else "the uploaded knowledge base"
        last_q = user_turns[-1]
        self.summary = (
            f"The user has asked {len(user_turns)} question(s) about {topics}. "
            f"Documents referenced: {docs}. Most recent question: {last_q}"
        )


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
