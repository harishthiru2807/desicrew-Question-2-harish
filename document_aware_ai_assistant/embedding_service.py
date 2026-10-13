"""Embedding generation with a local Sentence Transformers default and test doubles."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from config import DEFAULT_EMBEDDING_MODEL

TOKEN_RE = re.compile(r"[a-z0-9]+")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


class EmbeddingService(ABC):
    dim: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class DeterministicEmbeddingService(EmbeddingService):
    """Hash/bag-of-tokens embeddings for tests. Not used for production retrieval quality."""

    def __init__(self, dim: int = 96):
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = TOKEN_RE.findall((text or "").lower())
            if not tokens:
                digest = hashlib.sha256((text or "").encode("utf-8")).digest()
                for j, byte in enumerate(digest):
                    vectors[i, j % self.dim] += (byte / 255.0)
                continue
            for token in tokens:
                bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vectors[i, bucket] += 1.0
            # Light character n-gram signal to keep short queries aligned with passages.
            compact = "".join(tokens)
            for n in range(0, max(0, len(compact) - 2), 3):
                gram = compact[n : n + 3]
                bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self.dim
                vectors[i, bucket] += 0.15
        return l2_normalize(vectors)


class SentenceTransformerEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self.dim = 384

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def create_embedding_service(model_name: str | None = None, use_deterministic: bool = False) -> EmbeddingService:
    if use_deterministic:
        return DeterministicEmbeddingService()
    return SentenceTransformerEmbeddingService(model_name or DEFAULT_EMBEDDING_MODEL)
