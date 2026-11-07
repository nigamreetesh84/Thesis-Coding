"""FastAPI service exposing the RAG credit-card recommender.

Endpoints: POST /recommend, GET /status, GET /telemetry, GET /history.
There is deliberately NO runtime validation/correction gate placed between
the LLM's grounded output and the HTTP response -- see src/model.py.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from src.application.recommend_service import RecommendService
from src.domain.entities import UserProfile
from src.infrastructure.embedder import Embedder
from src.infrastructure.llm_client import LocalLLMClient, StubLLM
from src.infrastructure.vector_store import VectorStore
from src.model import GroundedPromptBuilder, RAGRecommender, Retriever

EmploymentType = Literal["salaried", "self-employed", "business owner"]


class ProfileIn(BaseModel):
    """Inbound applicant profile payload for POST /recommend."""

    annual_income: float = Field(ge=0)
    age: int = Field(ge=18, le=100)
    cibil: int = Field(ge=300, le=900)
    employment: EmploymentType
    existing_relationship: bool = False
    banks: list[str] = Field(default_factory=lambda: ["SBI", "HDFC"])


class DecisionOut(BaseModel):
    """A single card decision returned to the client."""

    bank: str
    card: str
    recommended: bool
    reason: str
    cited_chunk_id: str | None = None
    needs_review: bool = False


class RecommendOut(BaseModel):
    """Response payload for POST /recommend."""

    decisions: list[DecisionOut]
    latency_ms: float
    retrieved_k: int


class StatusOut(BaseModel):
    """Response payload for GET /status."""

    status: str
    use_stub_llm: bool
    top_k: int


class TelemetryOut(BaseModel):
    """Response payload for GET /telemetry."""

    count: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


_telemetry: deque[float] = deque(maxlen=settings.telemetry_maxlen)
_history: deque[RecommendOut] = deque(maxlen=settings.telemetry_maxlen)
_state: dict[str, RecommendService] = {}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))
    return ordered[idx]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Wire the recommender (embedder, FAISS index, LLM client) on startup."""
    embedder = Embedder(settings.model_name)
    embedder.encode(["startup warmup"])
    vector_store = VectorStore(settings.faiss_index_path, settings.chunks_parquet_path).load()
    retriever = Retriever(embedder, vector_store, top_k=settings.top_k)
    llm_client = StubLLM() if settings.use_stub_llm else LocalLLMClient(settings.llm_model_name)
    recommender = RAGRecommender(retriever, llm_client, GroundedPromptBuilder())
    _state["service"] = RecommendService(recommender)
    yield
    _state.clear()


app = FastAPI(title="Eligibility-Aware Credit Card Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/recommend", response_model=RecommendOut)
def recommend(profile_in: ProfileIn) -> RecommendOut:
    """Retrieve, prompt, generate and return card decisions for a profile."""
    service = _state["service"]
    profile = UserProfile(
        annual_income=profile_in.annual_income,
        age=profile_in.age,
        cibil=profile_in.cibil,
        employment=profile_in.employment,
        existing_relationship=profile_in.existing_relationship,
        banks=profile_in.banks,
    )
    start = time.perf_counter()
    decisions, retrieved_k = service.recommend(profile)
    latency_ms = (time.perf_counter() - start) * 1000

    _telemetry.append(latency_ms)
    result = RecommendOut(
        decisions=[DecisionOut(**vars(d)) for d in decisions],
        latency_ms=latency_ms,
        retrieved_k=retrieved_k,
    )
    _history.append(result)
    return result


@app.get("/status", response_model=StatusOut)
def status() -> StatusOut:
    """Report basic liveness and configuration information."""
    return StatusOut(status="ok", use_stub_llm=settings.use_stub_llm, top_k=settings.top_k)


@app.get("/telemetry", response_model=TelemetryOut)
def telemetry() -> TelemetryOut:
    """Report latency telemetry (count, mean, p50, p95, p99) over recent requests."""
    values = list(_telemetry)
    if not values:
        return TelemetryOut(count=0, mean_latency_ms=0.0, p50_latency_ms=0.0, p95_latency_ms=0.0, p99_latency_ms=0.0)
    return TelemetryOut(
        count=len(values),
        mean_latency_ms=sum(values) / len(values),
        p50_latency_ms=_percentile(values, 0.50),
        p95_latency_ms=_percentile(values, 0.95),
        p99_latency_ms=_percentile(values, 0.99),
    )


@app.get("/history", response_model=list[RecommendOut])
def history(limit: int = 20) -> list[RecommendOut]:
    """Return the most recent `limit` recommend results."""
    return list(_history)[-limit:]