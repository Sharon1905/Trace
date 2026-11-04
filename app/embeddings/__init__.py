"""Embedding adapters used by TRACE retrieval."""

from app.embeddings.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingError,
    EmbeddingSettings,
    SentenceTransformerEmbedder,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingError",
    "EmbeddingSettings",
    "SentenceTransformerEmbedder",
]
