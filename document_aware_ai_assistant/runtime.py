"""Application runtime: document lifecycle plus grounded question answering."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from answer_generator import GroundedAnswer, LLMClient, generate_grounded_answer
from config import AppConfig, get_config
from conversation_memory import ConversationMemory
from document_loader import DocumentLoadError, ExtractedDocument, load_document
from embedding_service import EmbeddingService
from retriever import Retriever
from text_chunker import chunk_document
from topic_manager import TopicManager
from vector_store import StoredDocument, VectorStore


@dataclass
class ProcessResult:
    filename: str
    status: str
    message: str
    document_id: str | None = None
    chunk_count: int = 0
    preview_text: str = ""
    replaced: bool = False


class AssistantRuntime:
    def __init__(
        self,
        embeddings: EmbeddingService,
        config: AppConfig | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.config = config or get_config()
        self.config.validate_chunking()
        self.embeddings = embeddings
        self.store = VectorStore(dim=embeddings.dim)
        self.retriever = Retriever(self.store, embeddings, min_score=self.config.min_relevance_score)
        self.llm_client = llm_client if llm_client is not None else LLMClient(self.config)
        self.topic_manager = TopicManager(self.llm_client)
        self.previews: dict[str, str] = {}
        self.extracted: dict[str, ExtractedDocument] = {}

    def process_upload(
        self,
        file_bytes: bytes,
        filename: str,
        replace: bool = False,
    ) -> ProcessResult:
        try:
            extracted = load_document(file_bytes, filename, max_bytes=self.config.max_upload_bytes)
        except DocumentLoadError as exc:
            return ProcessResult(filename=filename, status="rejected", message=str(exc))

        if extracted.needs_ocr:
            return ProcessResult(filename=extracted.filename, status="needs_ocr", message=extracted.error or "OCR required.")
        if extracted.is_empty or not extracted.ok:
            return ProcessResult(
                filename=extracted.filename,
                status="empty",
                message=extracted.error or "The document is empty and was not indexed.",
            )

        existing = self.store.document_by_filename(extracted.filename)
        if existing is not None and not replace:
            return ProcessResult(
                filename=extracted.filename,
                status="duplicate",
                message=f"'{extracted.filename}' is already in the knowledge base. Enable replace to overwrite it.",
                document_id=existing.document_id,
            )

        document_id = uuid4().hex
        chunks = chunk_document(
            extracted,
            document_id=document_id,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        if not chunks:
            return ProcessResult(
                filename=extracted.filename,
                status="empty",
                message="No indexable chunks could be created from this document.",
            )

        vectors = self.embeddings.embed_texts([c.text for c in chunks])
        stored = StoredDocument(
            document_id=document_id,
            filename=extracted.filename,
            file_type=extracted.file_type,
            status="indexed",
            chunk_count=len(chunks),
            preview_text=extracted.preview_text,
            source_metadata=extracted.source_metadata,
            content_hash=extracted.content_hash,
        )
        old_id = existing.document_id if existing is not None else None
        if replace or existing is not None:
            self.store.replace_document(old_id, stored, chunks, vectors)
            replaced = True
        else:
            self.store.add_document(stored, chunks, vectors)
            replaced = False
        self.previews[document_id] = extracted.preview_text
        self.extracted[document_id] = extracted
        if old_id:
            self.previews.pop(old_id, None)
            self.extracted.pop(old_id, None)
        return ProcessResult(
            filename=extracted.filename,
            status="indexed",
            message=f"Indexed {len(chunks)} chunk(s).",
            document_id=document_id,
            chunk_count=len(chunks),
            preview_text=extracted.preview_text,
            replaced=replaced,
        )

    def remove_document(self, document_id: str) -> bool:
        self.previews.pop(document_id, None)
        self.extracted.pop(document_id, None)
        return self.store.remove_document(document_id)

    def rebuild_index(self) -> int:
        self.store.rebuild()
        return self.store.size

    def clear_knowledge_base(self) -> None:
        self.store.clear()
        self.previews = {}
        self.extracted = {}

    def ask(self, question: str, memory: ConversationMemory) -> GroundedAnswer:
        names = [doc.filename for doc in self.store.documents.values()]
        interpreted = self.topic_manager.interpret(question, memory, document_names=names)
        memory.add_user(question, topic=interpreted.topic_label)

        if interpreted.recap_requested:
            answer = generate_grounded_answer(
                question,
                interpreted,
                [],
                memory,
                self.store,
                self.config,
                self.llm_client,
            )
            memory.add_assistant(answer.text, answer.citations, interpreted.topic_label, answer.follow_ups, answer.mode)
            return answer

        passages = self.retriever.retrieve(
            interpreted.search_query,
            top_k=self.config.top_k,
            min_score=self.config.min_relevance_score,
        )
        if not passages and interpreted.search_query != question:
            passages = self.retriever.retrieve(question, top_k=self.config.top_k)

        answer = generate_grounded_answer(
            question,
            interpreted,
            passages,
            memory,
            self.store,
            self.config,
            self.llm_client,
        )
        memory.add_assistant(answer.text, answer.citations, interpreted.topic_label, answer.follow_ups, answer.mode)
        return answer

    def knowledge_status(self) -> str:
        count = len(self.store.documents)
        chunks = self.store.size
        if count == 0:
            return "Knowledge base is empty. Upload PDF, DOCX, or TXT files to begin."
        return f"{count} document(s) indexed · {chunks} chunk(s) searchable"
