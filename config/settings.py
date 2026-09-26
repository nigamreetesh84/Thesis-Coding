"""Application configuration loaded from environment variables.

Uses pydantic-settings so every value can be overridden via the process
environment or a `.env` file without touching source code. All paths are
resolved relative to the current working directory unless an absolute path
is supplied via the environment.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised, typed application configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    top_k: int = 4
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    judge_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    rule_docs_dir: Path = Path("data/rule_docs")
    eligibility_table_path: Path = Path("data/eligibility_table.csv")
    gold_test_set_path: Path = Path("data/gold_test_set.jsonl")

    faiss_index_path: Path = Path("artifacts/faiss.index")
    chunks_parquet_path: Path = Path("artifacts/chunks.parquet")

    use_stub_llm: bool = True
    embedding_dim: int = 384

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    telemetry_maxlen: int = 500


settings = Settings()
