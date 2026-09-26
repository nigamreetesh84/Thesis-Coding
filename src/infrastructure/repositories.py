"""Repository interfaces and concrete adapters.

Repositories are the infrastructure-layer boundary between the raw data
files on disk (the eligibility CSV, the persisted FAISS/parquet chunk
store) and the rest of the application. They depend on `domain` (allowed:
infrastructure -> domain) but never the reverse.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from src.domain.eligibility import DEFAULT_ELIGIBILITY_TABLE_PATH, load_eligibility_table


class EligibilityRepository(Protocol):
    """Read-only access to the ground-truth eligibility table."""

    def all_rules(self) -> pd.DataFrame:
        """Return the full, cleaned eligibility table."""
        ...


class ChunkRepository(Protocol):
    """Read-only access to the persisted, embedded rule-document chunks."""

    def all_chunks(self) -> pd.DataFrame:
        """Return the full chunk table (chunk_id, bank, card, text)."""
        ...


class CsvEligibilityRepository:
    """Concrete `EligibilityRepository` backed by the eligibility CSV file."""

    def __init__(self, table_path: str | Path = DEFAULT_ELIGIBILITY_TABLE_PATH) -> None:
        self.table_path = Path(table_path)

    def all_rules(self) -> pd.DataFrame:
        return load_eligibility_table(self.table_path)


class ParquetChunkRepository:
    """Concrete `ChunkRepository` backed by the ETL-produced chunks.parquet."""

    def __init__(self, chunks_path: str | Path) -> None:
        self.chunks_path = Path(chunks_path)

    def all_chunks(self) -> pd.DataFrame:
        return pd.read_parquet(self.chunks_path)