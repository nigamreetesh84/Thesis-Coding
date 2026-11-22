"""FastAPI service exposing the RAG credit-card recommender.

Endpoints: POST /recommend, POST /ask (natural language), GET /status,
GET /telemetry, GET /history.
There is deliberately NO runtime validation/correction gate placed between
the LLM's grounded output and the HTTP response -- see src/model.py.
"""
from __future__ import annotations

from fastapi.responses import FileResponse
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from src.application.recommend_service import RecommendService
from src.domain.entities import UserProfile
from src.domain.query_parsing import ParsedQuery
from src.infrastructure.embedder import Embedder
from src.infrastructure.llm_client import LocalLLMClient, StubLLM
from src.infrastructure.vector_store import VectorStore
from src.model import GroundedPromptBuilder, RAGRecommender, Retriever
from src.nl_interface import AnswerWriter, CardRanker, NLRecommender, QueryParser

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


class AskIn(BaseModel):
    """Inbound payload for POST /ask: a free-text message, plus what was understood
    from the earlier message when answering a follow-up question."""

    message: str = Field(min_length=1, max_length=2000)
    previous: dict[str, Any] | None = None


class CardAnswerOut(BaseModel):
    bank: str
    card: str
    recommended: bool
    needs_review: bool
    sentence: str
    source: str
    cited_chunk_id: str | None = None


class AskOut(BaseModel):
    """Response payload for POST /ask."""

    status: str
    understood: dict[str, Any]
    questions: list[str]
    answer_text: str
    cards: list[CardAnswerOut]
    latency_ms: float


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
_state: dict[str, Any] = {}


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
    # Dashboard form uses the same frozen final design as /ask, so both paths
    # show the system's own decisions (not the stub echoing the eligibility table).
    recommender = RAGRecommender(retriever, llm_client, GroundedPromptBuilder(), decision_mode="extract_compare")
    _state["service"] = RecommendService(recommender)
    # Natural-language interface: always the frozen extract_compare design.
    nl_recommender = RAGRecommender(retriever, llm_client, GroundedPromptBuilder(), decision_mode="extract_compare")
    _state["nl"] = NLRecommender(
        QueryParser(llm_client), nl_recommender, CardRanker(embedder, vector_store), AnswerWriter(llm_client)
    )
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

@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the dashboard from the API, so http://<host>:8000/ opens it."""
    return FileResponse("dashboard/index.html")

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


def _understood_to_dict(parsed: ParsedQuery) -> dict[str, Any]:
    return {
        "income": parsed.income,
        "age": parsed.age,
        "cibil": parsed.cibil,
        "employment": parsed.employment,
        "existing_relationship": parsed.existing_relationship,
        "banks": list(parsed.banks),
        "preferences": parsed.preferences,
        "income_text": parsed.income_text,
    }


def _understood_from_dict(data: dict[str, Any]) -> ParsedQuery:
    return ParsedQuery(
        income=data.get("income"),
        age=data.get("age"),
        cibil=data.get("cibil"),
        employment=data.get("employment"),
        existing_relationship=data.get("existing_relationship"),
        banks=tuple(data.get("banks") or ("SBI", "HDFC")),
        preferences=str(data.get("preferences") or ""),
        income_text=data.get("income_text"),
    )


@app.post("/ask", response_model=AskOut)
def ask(ask_in: AskIn) -> AskOut:
    """Natural-language recommendation: parse -> (ask for missing facts) -> decide -> explain."""
    nl: NLRecommender = _state["nl"]
    previous = _understood_from_dict(ask_in.previous) if ask_in.previous else None
    start = time.perf_counter()
    result = nl.ask(ask_in.message, previous)
    latency_ms = (time.perf_counter() - start) * 1000
    _telemetry.append(latency_ms)
    return AskOut(
        status=result.status,
        understood=_understood_to_dict(result.understood),
        questions=result.questions,
        answer_text=result.text,
        cards=[
            CardAnswerOut(
                bank=a.bank, card=a.card, recommended=a.recommended, needs_review=a.needs_review,
                sentence=a.sentence, source=a.source, cited_chunk_id=a.cited_chunk_id,
            )
            for a in result.answers
        ],
        latency_ms=latency_ms,
    )


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