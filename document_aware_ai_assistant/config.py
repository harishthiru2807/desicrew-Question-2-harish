"""Configuration and environment settings for the Document-Aware AI Support Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per file
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.28
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gemini-1.5-flash"
SUPPORTED_GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-2.5-flash",
]
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I could not find sufficient information to answer this question in your uploaded knowledge base."
)


def _resolve_gemini_api_key() -> str:
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"):
        val = os.getenv(key, "").strip()
        if val:
            return val
    return ""


def _resolve_gemini_model() -> str:
    for key in ("GEMINI_MODEL", "LLM_MODEL"):
        val = os.getenv(key, "").strip()
        if val:
            return val
    return DEFAULT_LLM_MODEL


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class AppConfig:
    """Runtime configuration. Values may be overridden from the Streamlit sidebar."""

    llm_api_key: str = field(default_factory=_resolve_gemini_api_key)
    llm_model: str = field(default_factory=_resolve_gemini_model)
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "").strip())
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", DEFAULT_TOP_K))
    min_relevance_score: float = field(default_factory=lambda: _env_float("MIN_RELEVANCE_SCORE", DEFAULT_MIN_SCORE))
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    llm_timeout_seconds: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 45))
    max_context_chunks_chars: int = 6000

    @property
    def gemini_api_key(self) -> str:
        return self.llm_api_key

    @gemini_api_key.setter
    def gemini_api_key(self, value: str) -> None:
        self.llm_api_key = value

    @property
    def gemini_model(self) -> str:
        return self.llm_model

    @gemini_model.setter
    def gemini_model(self, value: str) -> None:
        self.llm_model = value

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    def validate_chunking(self) -> None:
        if self.chunk_size < 100:
            self.chunk_size = 100
        if self.chunk_overlap < 0:
            self.chunk_overlap = 0
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 5)
        if self.top_k < 1:
            self.top_k = 1
        if self.top_k > 20:
            self.top_k = 20


def get_config() -> AppConfig:
    cfg = AppConfig()
    cfg.validate_chunking()
    return cfg
