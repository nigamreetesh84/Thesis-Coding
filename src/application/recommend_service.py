"""Use-case orchestrator for the recommend-a-card application flow.

Framework-free: no FastAPI, pydantic, or web imports here. Wires together
the retriever and an LLM client (via `RAGRecommender`) and exposes a single
`recommend` use case to callers such as `src/api.py`.
"""
from __future__ import annotations

from src.domain.entities import Decision, UserProfile
from src.model import RAGRecommender


class RecommendService:
    """Orchestrates a single recommend-a-card use case."""

    def __init__(self, recommender: RAGRecommender) -> None:
        self.recommender = recommender

    def recommend(self, profile: UserProfile) -> tuple[list[Decision], int]:
        """Return (decisions, retrieved_k) for the given applicant profile."""
        return self.recommender.recommend(profile)