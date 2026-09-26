"""FAISS-backed vector store wrapper.

Loads a persisted ``IndexIDMap2``-wrapped ``IndexFlatIP`` FAISS index together
with the parquet file describing each stored chunk, and exposes a simple
top-k similarity search returning plain dictionaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd


class VectorStore:
    """Loads and searches a FAISS index of embedded rule-document chunks."""

    def __init__(self, index_path: str | Path, chunks_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self._index: faiss.Index | None = None
        self._chunks: pd.DataFrame | None = None

    def load(self) -> "VectorStore":
        """Load the FAISS index and chunk table from disk. Returns self."""
        self._index = faiss.read_index(str(self.index_path))
        self._chunks = pd.read_parquet(self.chunks_path)
        return self

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            self.load()
        assert self._index is not None
        return self._index

    @property
    def chunks(self) -> pd.DataFrame:
        if self._chunks is None:
            self.load()
        assert self._chunks is not None
        return self._chunks

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """Return up to ``top_k`` chunk dicts, ranked by inner-product score.

        Chunk ids returned by FAISS as -1 (meaning "no result", which can
        happen when the index holds fewer than ``top_k`` vectors) are
        silently skipped.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        scores, ids = self.index.search(query_vector.astype(np.float32), top_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0], strict=True):
            if idx == -1:
                continue
            row = self.chunks.iloc[int(idx)]
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "bank": row["bank"],
                    "card": row["card"],
                    "text": row["text"],
                    "score": float(score),
                }
            )
        return results