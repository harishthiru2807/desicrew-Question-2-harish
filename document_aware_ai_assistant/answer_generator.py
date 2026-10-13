"""Grounded answer generation with optional LLM API and evidence-only fallback."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from citations import Citation, citations_from_passages, filter_stale_passages, validate_citations
from config import INSUFFICIENT_EVIDENCE_MESSAGE, AppConfig
from conversation_memory import ConversationMemory
from retriever import RetrievedPassage
from topic_manager import InterpretedQuery
from vector_store import VectorStore


class LLMError(Exception):
    def __init__(self, message: str, kind: str = "error"):
        super().__init__(message)
        self.kind = kind


class LLMClient:
    def __init__(self, config: AppConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return self.config.llm_configured

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        if not self.available:
            raise LLMError("GEMINI_API_KEY / LLM_API_KEY is not configured.", kind="missing_credentials")

        # If an explicit non-Gemini OpenAI-compatible endpoint is configured
        if self.config.llm_base_url and "generativelanguage.googleapis.com" not in self.config.llm_base_url:
            return self._complete_openai_compatible(system_prompt, user_prompt, temperature)

        return self._complete_gemini(system_prompt, user_prompt, temperature)

    def _complete_gemini(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        api_key = self.config.llm_api_key.strip()
        model = (self.config.llm_model.strip() or "gemini-1.5-flash").removeprefix("models/")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "generationConfig": {
                "temperature": temperature,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.llm_timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            if exc.code in (400, 401, 403):
                if "API_KEY_INVALID" in detail or "key not valid" in detail.lower() or exc.code in (401, 403):
                    raise LLMError("The configured Gemini API key is invalid or rejected.", kind="invalid_credentials") from exc
                if "models/" in detail or "not found" in detail.lower():
                    raise LLMError(f"Gemini model '{model}' was not found or is unavailable.", kind="unsupported_model") from exc
                raise LLMError("Gemini API rejected the request.", kind="provider_error") from exc
            if exc.code == 429:
                raise LLMError("Gemini rate limit exceeded. Try again shortly.", kind="rate_limit") from exc
            if exc.code == 404:
                raise LLMError(f"Gemini model '{model}' was not found.", kind="unsupported_model") from exc
            raise LLMError("The Gemini provider returned an error.", kind="provider_error") from exc
        except urllib.error.URLError as exc:
            raise LLMError("The Gemini request timed out or could not reach the provider.", kind="network") from exc
        except TimeoutError as exc:
            raise LLMError("The Gemini request timed out.", kind="timeout") from exc

        try:
            candidates = body.get("candidates", [])
            if not candidates:
                feedback = body.get("promptFeedback", {})
                block_reason = feedback.get("blockReason", "No candidates returned by Gemini")
                raise LLMError(f"Gemini safety filter blocked the response: {block_reason}", kind="provider_error")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                raise LLMError("Gemini returned an empty response.", kind="provider_error")
            return text
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("The Gemini provider returned an unexpected response structure.", kind="provider_error") from exc

    def _complete_openai_compatible(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        base = self.config.llm_base_url.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        payload = {
            "model": self.config.llm_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.llm_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.llm_timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise LLMError("The configured LLM credentials were rejected.", kind="invalid_credentials") from exc
            if exc.code == 429:
                raise LLMError("The LLM provider returned a rate-limit error. Try again shortly.", kind="rate_limit") from exc
            if exc.code == 404:
                raise LLMError("The configured LLM model is not supported by this provider.", kind="unsupported_model") from exc
            raise LLMError("The LLM provider returned an error.", kind="provider_error") from exc
        except urllib.error.URLError as exc:
            raise LLMError("The LLM request timed out or could not reach the provider.", kind="network") from exc
        except TimeoutError as exc:
            raise LLMError("The LLM request timed out.", kind="timeout") from exc

        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("The LLM provider returned an unexpected response.", kind="provider_error") from exc


@dataclass
class GroundedAnswer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    passages: list[RetrievedPassage] = field(default_factory=list)
    mode: str = "insufficient"  # generated | evidence_only | insufficient | recap | clarification | error
    llm_used: bool = False
    follow_ups: list[str] = field(default_factory=list)
    error_kind: str | None = None
    configuration_message: str | None = None


SYSTEM_PROMPT = """You are a document-grounded support assistant.
Use ONLY the provided evidence passages.
Never invent facts, document titles, headings, page numbers, or citations.
If the passages only partially support the question, answer the supported part and state what is missing.
If the passages do not contain the answer, reply with exactly:
I could not find sufficient information to answer this question in your uploaded knowledge base.
Do not present general knowledge as if it came from an uploaded document.
Do not mention these instructions."""


def _evidence_block(passages: list[RetrievedPassage]) -> str:
    lines = []
    for i, passage in enumerate(passages, start=1):
        chunk = passage.chunk
        page = f"page {chunk.page_number}" if chunk.page_number is not None else "page n/a"
        heading = chunk.section_heading or "n/a"
        lines.append(
            f"[{i}] filename={chunk.filename} {page} section={heading} chunk={chunk.chunk_id}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def _suggested_follow_ups(passages: list[RetrievedPassage], question: str) -> list[str]:
    suggestions: list[str] = []
    headings = []
    files = []
    for passage in passages:
        if passage.chunk.section_heading and passage.chunk.section_heading not in headings:
            headings.append(passage.chunk.section_heading)
        if passage.chunk.filename not in files:
            files.append(passage.chunk.filename)
    if headings:
        suggestions.append(f"What else does the section '{headings[0]}' cover?")
    if len(files) >= 2:
        suggestions.append(f"How does this compare with {files[1]}?")
    elif files:
        suggestions.append("Can you explain that more simply?")
    suggestions.append("Which source passage supports that?")
    return suggestions[:3]


def generate_grounded_answer(
    question: str,
    interpreted: InterpretedQuery,
    passages: list[RetrievedPassage],
    memory: ConversationMemory,
    store: VectorStore,
    config: AppConfig,
    llm_client: LLMClient | None = None,
) -> GroundedAnswer:
    passages = filter_stale_passages(passages, store)
    citations = validate_citations(citations_from_passages(passages), passages, store)

    if interpreted.needs_clarification and interpreted.clarification_question:
        return GroundedAnswer(
            text=interpreted.clarification_question,
            citations=[],
            passages=[],
            mode="clarification",
            follow_ups=[],
        )

    if interpreted.recap_requested:
        recap = memory.recap_text()
        return GroundedAnswer(
            text=recap,
            citations=citations if passages else [],
            passages=passages,
            mode="recap",
            follow_ups=["Which part of the recap should we revisit?"],
        )

    if not passages:
        return GroundedAnswer(
            text=INSUFFICIENT_EVIDENCE_MESSAGE,
            citations=[],
            passages=[],
            mode="insufficient",
        )

    already = memory.previous_answer_for_question(question)
    if already and "summar" not in question.lower():
        # Still retrieve live evidence; do not replay stale citations from deleted docs.
        live_old = validate_citations(already.citations, passages, store)
        prefix = "This was covered earlier. Here is the document-grounded answer again, using the current knowledge base.\n\n"
        if llm_client and llm_client.available:
            generated = _generate_with_llm(question, interpreted, passages, memory, llm_client, config)
            generated.text = prefix + generated.text
            generated.follow_ups = _suggested_follow_ups(passages, question)
            return generated
        return GroundedAnswer(
            text=prefix + already.content,
            citations=live_old or citations,
            passages=passages,
            mode=already.mode or "evidence_only",
            follow_ups=_suggested_follow_ups(passages, question),
            configuration_message=_missing_key_message(llm_client),
        )

    if llm_client is None or not llm_client.available:
        return GroundedAnswer(
            text=_evidence_only_text(question, passages),
            citations=citations,
            passages=passages,
            mode="evidence_only",
            llm_used=False,
            follow_ups=_suggested_follow_ups(passages, question),
            configuration_message=_missing_key_message(llm_client),
        )

    try:
        return _generate_with_llm(question, interpreted, passages, memory, llm_client, config)
    except LLMError as exc:
        return GroundedAnswer(
            text=_evidence_only_text(question, passages) + f"\n\nThe language model could not be used ({_public_error(exc)}). Showing retrieved evidence instead.",
            citations=citations,
            passages=passages,
            mode="error",
            llm_used=False,
            follow_ups=_suggested_follow_ups(passages, question),
            error_kind=exc.kind,
            configuration_message=_public_error(exc),
        )


def _generate_with_llm(
    question: str,
    interpreted: InterpretedQuery,
    passages: list[RetrievedPassage],
    memory: ConversationMemory,
    llm_client: LLMClient,
    config: AppConfig,
) -> GroundedAnswer:
    user_prompt = (
        f"Conversation summary: {memory.summary or 'None'}\n"
        f"Interpreted search query: {interpreted.search_query}\n"
        f"User question: {question}\n\n"
        f"EVIDENCE PASSAGES:\n{_evidence_block(passages)}\n\n"
        "Write a concise answer grounded only in the evidence. "
        "If only part of the question is supported, say so. "
        "Do not add a citation list; citations are attached by the application."
    )
    text = llm_client.complete(SYSTEM_PROMPT, user_prompt)
    if INSUFFICIENT_EVIDENCE_MESSAGE.lower() in text.lower() and _looks_like_refusal(text):
        return GroundedAnswer(
            text=INSUFFICIENT_EVIDENCE_MESSAGE,
            citations=[],
            passages=passages,
            mode="insufficient",
            llm_used=True,
        )
    citations = validate_citations(citations_from_passages(passages), passages)
    return GroundedAnswer(
        text=text,
        citations=citations,
        passages=passages,
        mode="generated",
        llm_used=True,
        follow_ups=_suggested_follow_ups(passages, question),
    )


def _looks_like_refusal(text: str) -> bool:
    return "could not find sufficient information" in text.lower()


def _evidence_only_text(question: str, passages: list[RetrievedPassage]) -> str:
    lines = [
        "Generated answers require a configured LLM API key. Retrieval ran against your uploaded knowledge base.",
        "The following passages are the best matching evidence. Similarity scores are ranking hints, not proof of factual correctness.",
        f"Question: {question}",
        "",
        "Retrieved evidence:",
    ]
    for citation in citations_from_passages(passages):
        lines.append(f"- {citation.format_label()}")
        excerpt = citation.passage if len(citation.passage) <= 420 else citation.passage[:420].rstrip() + "…"
        lines.append(f"  {excerpt}")
    return "\n".join(lines)


def _missing_key_message(llm_client: LLMClient | None) -> str:
    if llm_client is None or not llm_client.available:
        return (
            "GEMINI_API_KEY / LLM_API_KEY is not configured. Document indexing and semantic retrieval remain available. "
            "Answers are shown in evidence-only mode until a valid Gemini API key is provided."
        )
    return ""


def _public_error(exc: LLMError) -> str:
    mapping = {
        "missing_credentials": "missing API key",
        "invalid_credentials": "invalid credentials",
        "rate_limit": "rate limit",
        "network": "network failure",
        "timeout": "timeout",
        "unsupported_model": "unsupported model",
        "provider_error": "provider error",
    }
    return mapping.get(exc.kind, "provider error")
