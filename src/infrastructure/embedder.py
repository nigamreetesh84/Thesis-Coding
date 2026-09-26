"""Sentence-BERT embedding wrapper.

Wraps `sentence-transformers` to provide L2-normalized, 384-dimensional,
float32 embeddings suitable for cosine-similarity search via a FAISS
IndexFlatIP. When the Sentence-BERT weights are not available locally and
network access is disabled, the GPU-free STuBLM path falls back to a
deterministic local hashing embedder with the same shape/dtype/norm
contract so demos and tests still run end-to-end without external calls.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class Embedder:
    """Thin wrapper around a local Sentence-BERT model.

    The underlying model is loaded lazily on first use so importing this
    module or constructing an `Embedder` never triggers a network call or
    a model download by itself.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, allow_fallback: bool = True) -> None:
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self._model: SentenceTransformer | None = None
        self.using_fallback = False

    @property
    def model(self) -> SentenceTransformer | None:
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
            except OSError:
                if not self.allow_fallback:
                    raise
                self.using_fallback = True
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized float32 vectors of shape (n, 384)."""
        model = self.model
        if self.using_fallback or model is None:
            return self._hash_embeddings(texts)

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def _hash_embeddings(texts: list[str]) -> np.ndarray:
        """Deterministic local fallback embeddings with the Sentence-BERT contract."""
        vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = text.lower().split()
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little", signed=False)
                col_idx = value % EMBEDDING_DIM
                sign = 1.0 if (value >> 8) & 1 else -1.0
                vectors[row_idx, col_idx] += sign
            norm = np.linalg.norm(vectors[row_idx])
            if norm == 0.0:
                vectors[row_idx, 0] = 1.0
            else:
                vectors[row_idx] /= norm
        return vectors.astype(np.float32)