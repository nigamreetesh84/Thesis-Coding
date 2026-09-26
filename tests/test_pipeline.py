from __future__ import annotations

import faiss
import numpy as np
import pandas as pd

from src.infrastructure.embedder import Embedder
from src.pipeline import PipelineBuilder, RuleChunker


def test_rule_chunker_splits_each_heading(tmp_path) -> None:
    rule_file = tmp_path / "demo_cards.md"
    rule_file.write_text(
        "### Demo Card - Income Requirement\nIncome must exceed a threshold.\n\n"
        "### Demo Card - Age Requirement\nApplicants must be adults.\n",
        encoding="utf-8",
    )

    chunks = RuleChunker().chunk_file(rule_file)

    assert len(chunks) == 2
    assert chunks[0].bank == "DEMO"
    assert chunks[0].card == "Demo Card"
    assert "Income must exceed" in chunks[0].text
    assert "Applicants must be adults" in chunks[1].text


def test_embedder_returns_normalized_float32(monkeypatch) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(
            self,
            texts: list[str],
            convert_to_numpy: bool,
            normalize_embeddings: bool,
            show_progress_bar: bool,
        ) -> np.ndarray:
            assert convert_to_numpy is True
            assert normalize_embeddings is True
            assert show_progress_bar is False
            vectors = np.zeros((len(texts), 384), dtype=np.float64)
            vectors[:, 0] = 3.0
            vectors[:, 1] = 4.0
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors

    monkeypatch.setattr("src.infrastructure.embedder.SentenceTransformer", FakeSentenceTransformer)
    embedder = Embedder("test-model", allow_fallback=False)

    vectors = embedder.encode(["first", "second"])

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-6)


def test_pipeline_persists_faiss_and_chunk_table(tmp_path, unit_vector_embedder) -> None:
    rule_docs = tmp_path / "rule_docs"
    rule_docs.mkdir()
    (rule_docs / "demo_cards.md").write_text(
        "### Demo Card - Income Requirement\nMinimum income is required.\n\n"
        "### Demo Card - Age Requirement\nMinimum age is required.\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "artifacts" / "faiss.index"
    chunks_path = tmp_path / "artifacts" / "chunks.parquet"
    builder = PipelineBuilder(
        rule_docs_dir=rule_docs,
        faiss_index_path=index_path,
        chunks_parquet_path=chunks_path,
        embedder=unit_vector_embedder,
        bank_by_filename={"demo_cards.md": "DEMO"},
    )

    builder.run()

    index = faiss.read_index(str(index_path))
    chunks = pd.read_parquet(chunks_path)
    assert index.d == 384
    assert index.ntotal == 2
    assert chunks["chunk_id"].tolist() == ["demo_cards#0", "demo_cards#1"]
    assert chunks["bank"].tolist() == ["DEMO", "DEMO"]
    vectors = np.stack([index.reconstruct(index_id) for index_id in range(index.ntotal)])
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-6)