"""Local FAISS index management synchronized with chunk metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from text_chunker import TextChunk

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only when faiss is missing
    faiss = None


@dataclass
class StoredDocument:
    document_id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int = 0
    preview_text: str = ""
    source_metadata: dict = field(default_factory=dict)
    content_hash: str = ""
    error: str | None = None


class VectorStoreError(Exception):
    pass


class VectorStore:
    """In-memory FAISS store. Removing a document rebuilds the index without its chunks."""

    def __init__(self, dim: int):
        if faiss is None:
            raise VectorStoreError("faiss-cpu is required for vector indexing.")
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[TextChunk] = []
        self._embeddings: np.ndarray | None = None
        self.documents: dict[str, StoredDocument] = {}
        self.filename_to_id: dict[str, str] = {}

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def document_by_filename(self, filename: str) -> StoredDocument | None:
        doc_id = self.filename_to_id.get(filename.lower())
        if not doc_id:
            return None
        return self.documents.get(doc_id)

    def has_filename(self, filename: str) -> bool:
        return filename.lower() in self.filename_to_id

    def add_document(
        self,
        document: StoredDocument,
        chunks: list[TextChunk],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk count does not match embedding count.")
        if embeddings.size and embeddings.shape[1] != self.dim:
            raise VectorStoreError(f"Expected embedding dimension {self.dim}, got {embeddings.shape[1]}.")
        if not chunks:
            raise VectorStoreError("Refusing to index a document with no chunks.")

        self.documents[document.document_id] = document
        self.filename_to_id[document.filename.lower()] = document.document_id
        self._chunks.extend(chunks)
        if self._embeddings is None or self._embeddings.size == 0:
            self._embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            self._embeddings = np.vstack([self._embeddings, np.asarray(embeddings, dtype=np.float32)])
        document.chunk_count = len(chunks)
        document.status = "indexed"
        self._rebuild_index()

    def remove_document(self, document_id: str) -> bool:
        stored = self.documents.get(document_id)
        if stored is None:
            return False
        keep_idx = [i for i, chunk in enumerate(self._chunks) if chunk.document_id != document_id]
        self._chunks = [self._chunks[i] for i in keep_idx]
        if self._embeddings is not None and len(keep_idx) > 0:
            self._embeddings = self._embeddings[keep_idx]
        else:
            self._embeddings = None
        self.filename_to_id.pop(stored.filename.lower(), None)
        del self.documents[document_id]
        self._rebuild_index()
        return True

    def replace_document(
        self,
        old_document_id: str | None,
        document: StoredDocument,
        chunks: list[TextChunk],
        embeddings: np.ndarray,
    ) -> None:
        if old_document_id:
            self.remove_document(old_document_id)
        existing = self.document_by_filename(document.filename)
        if existing is not None:
            self.remove_document(existing.document_id)
        self.add_document(document, chunks, embeddings)

    def rebuild(self) -> None:
        self._rebuild_index()

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = None
        self.documents = {}
        self.filename_to_id = {}
        self._rebuild_index()

    def all_chunks(self) -> list[TextChunk]:
        return list(self._chunks)

    def chunks_for_document(self, document_id: str) -> list[TextChunk]:
        return [c for c in self._chunks if c.document_id == document_id]

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[TextChunk, float]]:
        if self.is_empty or self._index.ntotal == 0:
            return []
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vector, k)
        results: list[tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[int(idx)], float(score)))
        return results

    def contains_chunk(self, document_id: str, chunk_id: str) -> bool:
        return any(c.document_id == document_id and c.chunk_id == chunk_id for c in self._chunks)

    def _rebuild_index(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        if self._embeddings is not None and len(self._chunks) > 0:
            vectors = np.ascontiguousarray(self._embeddings.astype(np.float32))
            self._index.add(vectors)
        elif len(self._chunks) != 0:
            raise VectorStoreError("Chunk metadata and embeddings are out of sync.")
