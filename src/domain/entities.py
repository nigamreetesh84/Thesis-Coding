"""Framework-free domain entities shared across the application.

These dataclasses have no dependency on pandas, FAISS, FastAPI or any other
infrastructure concern, per the Clean Architecture rule that `domain/`
depends on nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserProfile:
    """A prospective credit-card applicant's stated attributes."""

    annual_income: float
    age: int
    cibil: int
    employment: str
    existing_relationship: bool
    banks: list[str] = field(default_factory=lambda: ["SBI", "HDFC"])

    def to_query(self) -> str:
        """Serialise the profile into a natural-language retrieval query.

        The wording deliberately mirrors how a customer might describe
        themselves, so it can be embedded and matched against the rule
        corpus using semantic similarity rather than exact keys.
        """
        relationship_clause = (
            "and already holds an account with the bank"
            if self.existing_relationship
            else "and does not currently hold an account with the bank"
        )
        banks_clause = " or ".join(self.banks)
        return (
            f"A {self.age}-year-old {self.employment} applicant with an annual "
            f"income of {self.annual_income:.0f} rupees and a CIBIL score of "
            f"{self.cibil}, {relationship_clause}, is asking which credit cards "
            f"from {banks_clause} they are eligible for."
        )


@dataclass(frozen=True)
class Decision:
    """The recommendation outcome for a single bank/card pair."""

    bank: str
    card: str
    recommended: bool
    reason: str
    cited_chunk_id: str | None = None
    # True when the system could not establish the card's requirements reliably
    # (extract_compare guard); such a decision is never a recommendation.
    needs_review: bool = False