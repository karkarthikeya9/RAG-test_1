"""
pipeline/embeddings.py — local embedding wrapper (sentence-transformers).

Kept behind one tiny class so a different embedding model (or an API embedding
service) can be swapped in later without touching ingest/chat.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

import config


class LocalEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        print(f"[EMBED] loading {self.model_name} ...")
        self.model = SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return v[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [row.tolist() for row in v]