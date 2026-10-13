"""Semantic retrieval with relevance filtering. Scores are not calibrated probabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from embedding_service import EmbeddingService
from text_chunker import TextChunk
from vector_store import VectorStore

_STOP = {
    "what",
    "which",
    "when",
    "where",
    "does",
    "have",
    "with",
    "from",
    "that",
    "this",
    "your",
    "about",
    "please",
    "could",
    "would",
    "should",
    "into",
    "than",
    "then",
    "them",
    "they",
    "been",
    "were",
    "their",
    "there",
    "company",
    "companies",
    "document",
    "policy",
    "guide",
    "question",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3 and t not in _STOP}


@dataclass
class RetrievedPassage:
    chunk: TextChunk
    score: float

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def filename(self) -> str:
        return self.chunk.filename

    @property
    def text(self) -> str:
        return self.chunk.text


class Retriever:
    def __init__(self, store: VectorStore, embeddings: EmbeddingService, min_score: float = 0.28):
        self.store = store
        self.embeddings = embeddings
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        exclude_document_ids: set[str] | None = None,
    ) -> list[RetrievedPassage]:
        if not query.strip() or self.store.is_empty:
            return []
        threshold = self.min_score if min_score is None else min_score
        vector = self.embeddings.embed_query(query)
        raw = self.store.search(vector, top_k=max(top_k * 2, top_k))
        passages: list[RetrievedPassage] = []
        for chunk, score in raw:
            if exclude_document_ids and chunk.document_id in exclude_document_ids:
                continue
            if score < threshold:
                continue
            q_tokens = _content_tokens(query)
            p_tokens = _content_tokens(chunk.text)
            overlap = q_tokens & p_tokens
            if q_tokens and p_tokens and len(overlap) == 0:
                continue
            if q_tokens and len(q_tokens) >= 4 and len(overlap) < 2:
                continue
            passages.append(RetrievedPassage(chunk=chunk, score=score))
            if len(passages) >= top_k:
                break
        return passages

    def evidence_is_sufficient(self, passages: list[RetrievedPassage]) -> bool:
        return bool(passages)

    def query_embedding(self, query: str) -> np.ndarray:
        return self.embeddings.embed_query(query)
