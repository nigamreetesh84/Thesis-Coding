"""Decisions from the APPROVED knowledge base (the live portal's decision path).

Same deterministic comparison as the thesis's extract_compare design
(src/domain/requirements.py::decide), but the requirements come from the
version a bank employee approved instead of being re-read on every request.
So no LLM sits between an approved rule and a customer's decision, and each
decision names the rule version (and its hash) it used.

Per-bank relationship: if the applicant names the banks they already bank
with, a relationship requirement is checked against THAT bank only (the frozen
thesis profile had a single yes/no for all banks).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.domain.entities import Decision, UserProfile
from src.domain.requirements import decide
from src.portal.registry import RuleRegistry


@dataclass(frozen=True)
class VersionedDecision:
    decision: Decision
    version: str
    version_hash: str


class ApprovedRecommender:
    decision_mode = "approved_kb"

    def __init__(self, registry: RuleRegistry) -> None:
        self.registry = registry
        self.last_versions: dict[str, dict[str, str]] = {}
        self.existing_banks: tuple[str, ...] | None = None  # set per request by the portal service

    def recommend_versioned(self, profile: UserProfile, existing_banks: tuple[str, ...] | None = None) -> list[VersionedDecision]:
        cards = self.registry.active_cards(profile.banks)
        out: list[VersionedDecision] = []
        named = {b.upper() for b in existing_banks or ()} - {"ANY"}
        for card in cards:
            p = profile
            if named and card.requirements.relationship_required:
                p = replace(profile, existing_relationship=card.requirements.bank.upper() in named)
            out.append(VersionedDecision(decide(p, card.requirements), card.version, card.version_hash))
        self.last_versions = {c.requirements.bank: {"version": c.version, "hash": c.version_hash} for c in cards}
        return out

    def recommend(self, profile: UserProfile) -> tuple[list[Decision], int]:
        """RAGRecommender-compatible signature, so NLRecommender can sit on top."""
        versioned = self.recommend_versioned(profile, self.existing_banks)
        return [v.decision for v in versioned], len(versioned)
