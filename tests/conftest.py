from __future__ import annotations

import numpy as np
import pytest

from src.domain.entities import UserProfile


class UnitVectorEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for row_index, text in enumerate(texts):
            column_index = sum(text.encode("utf-8")) % vectors.shape[1]
            vectors[row_index, column_index] = 1.0
        return vectors


@pytest.fixture
def example_profile() -> UserProfile:
    return UserProfile(
        annual_income=800_000,
        age=27,
        cibil=760,
        employment="salaried",
        existing_relationship=False,
        banks=["SBI", "HDFC"],
    )


@pytest.fixture
def unit_vector_embedder() -> UnitVectorEmbedder:
    return UnitVectorEmbedder()