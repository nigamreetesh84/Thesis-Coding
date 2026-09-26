"""Offline ETL pipeline: chunk the rule documents, embed them, and persist a
FAISS index plus a parquet chunk table for the online retriever to load.

Run as a script: `python -m src.pipeline`
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.domain.eligibility import load_eligibility_table
from src.infrastructure.embedder import Embedder

RULE_HEADING_RE = re.compile(r"^### (?P<card>.+?) [–—-] (?P<rule>.+)$", re.MULTILINE)

DEFAULT_RULE_DOCS_DIR = Path("data/rule_docs")
DEFAULT_FAISS_INDEX_PATH = Path("artifacts/faiss.index")
DEFAULT_CHUNKS_PARQUET_PATH = Path("artifacts/chunks.parquet")
DEFAULT_ELIGIBILITY_TABLE_PATH = Path("data/eligibility_table.csv")


@dataclass(frozen=True)
class RawChunk:
    """One self-contained eligibility rule extracted from a rule document."""
    chunk_id: str
    bank: str
    card: str
    text: str


class RuleChunker:
    """Splits markdown rule documents into one chunk per H3 heading.

    Each heading is expected in the form '### <CARD> — <RULE NAME>'; the
    heading text plus the prose that follows (up to the next heading) forms
    one self-contained chunk describing a single eligibility rule.
    """

    def __init__(self, bank_by_filename: dict[str, str] | None = None) -> None:
        self.bank_by_filename = bank_by_filename or {}

    def chunk_file(self, path: Path) -> list[RawChunk]:
        """Chunk a single markdown rule document into one chunk per H3 rule."""
        content = path.read_text(encoding="utf-8")
        bank = self.bank_by_filename.get(path.name, path.stem.split("_")[0].upper())

        headings = list(RULE_HEADING_RE.finditer(content))
        chunks: list[RawChunk] = []
        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            block = content[start:end].strip()
            card = match.group("card").strip()
            chunk_id = f"{path.stem}#{i}"
            chunks.append(RawChunk(chunk_id=chunk_id, bank=bank, card=card, text=block))
        return chunks

    def chunk_directory(self, directory: Path) -> list[RawChunk]:
        """Chunk every `*.md` rule document in `directory`, in filename order."""
        chunks: list[RawChunk] = []
        for path in sorted(directory.glob("*.md")):
            chunks.extend(self.chunk_file(path))
        return chunks



class Vectorizer:
    """Wraps `Embedder` to turn `RawChunk` text into normalized embeddings."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()

    def embed_chunks(self, chunks: list[RawChunk]) -> np.ndarray:
        """Return a (n_chunks, 384) normalized float32 embedding matrix."""
        return self.embedder.encode([c.text for c in chunks])



class PipelineBuilder:
    """Orchestrates load -> chunk -> embed -> persist for the offline ETL."""

    def __init__(
        self,
        rule_docs_dir: Path = DEFAULT_RULE_DOCS_DIR,
        faiss_index_path: Path = DEFAULT_FAISS_INDEX_PATH,
        chunks_parquet_path: Path = DEFAULT_CHUNKS_PARQUET_PATH,
        embedder: Embedder | None = None,
        bank_by_filename: dict[str, str] | None = None,
    ) -> None:
        self.rule_docs_dir = rule_docs_dir
        self.faiss_index_path = faiss_index_path
        self.chunks_parquet_path = chunks_parquet_path
        self.chunker = RuleChunker(bank_by_filename or {"sbi_cards.md": "SBI", "hdfc_cards.md": "HDFC"})
        self.vectorizer = Vectorizer(embedder)

    def run(self) -> None:
        """Chunk the rule docs, embed them, persist artefacts."""
        chunks = self.chunker.chunk_directory(self.rule_docs_dir)
        if not chunks:
            raise RuntimeError(f"No rule chunks found under {self.rule_docs_dir}")

        embeddings = self.vectorizer.embed_chunks(chunks)

        base_index = faiss.IndexFlatIP(embeddings.shape[1])
        index = faiss.IndexIDMap2(base_index)
        ids = np.arange(len(chunks), dtype=np.int64)
        index.add_with_ids(embeddings, ids)

        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.faiss_index_path))

        chunks_df = pd.DataFrame(
            {
                "chunk_id": [c.chunk_id for c in chunks],
                "bank": [c.bank for c in chunks],
                "card": [c.card for c in chunks],
                "text": [c.text for c in chunks],
            }
        )
        self.chunks_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_df.to_parquet(self.chunks_parquet_path, index=False)



def main() -> None:
    """CLI entry point: `python -m src.pipeline`."""
    parser = argparse.ArgumentParser(description="Build the FAISS index and chunk store.")
    parser.add_argument("--rule-docs-dir", type=Path, default=DEFAULT_RULE_DOCS_DIR)
    parser.add_argument("--faiss-index-path", type=Path, default=DEFAULT_FAISS_INDEX_PATH)
    parser.add_argument("--chunks-parquet-path", type=Path, default=DEFAULT_CHUNKS_PARQUET_PATH)
    parser.add_argument("--eligibility-table-path", type=Path, default=DEFAULT_ELIGIBILITY_TABLE_PATH)
    args = parser.parse_args()

    # Sanity-check the (physically and logically separate) eligibility table
    # exists and is well-formed, even though this ETL never uses its
    # contents for chunking or embedding -- it only touches rule_docs.
    load_eligibility_table(args.eligibility_table_path)

    builder = PipelineBuilder(
        rule_docs_dir=args.rule_docs_dir,
        faiss_index_path=args.faiss_index_path,
        chunks_parquet_path=args.chunks_parquet_path,
    )
    builder.run()
    print(f"Wrote {args.faiss_index_path} and {args.chunks_parquet_path}")


if __name__ == "__main__":
    main()