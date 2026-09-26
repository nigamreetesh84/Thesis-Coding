"""Deterministic eligibility oracle.

CRITICAL ISOLATION RULE: this module reads ONLY the eligibility
ground-truth table ( `data/eligibility_table.csv` by default). It must
NEVER import, open, or otherwise reference anything under
`data/rule_docs/` -- that corpus exists for the retriever only. This
oracle serves two purposes: the rule-based baseline recommender (ablation
arm B0), and the source of "expected" answers used throughout evaluation
(gold-set generation, McNemar comparisons, etc). If you ever find this
module importing `rule_docs`, that is a bug -- refactor immediately.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.domain.entities import Decision, UserProfile

REQUIRED_COLUMNS = [
    "bank",
    "card",
    "min_income",
    "min_age",
    "min_cibil",
    "employment",
    "relationship_required",
]

DEFAULT_ELIGIBILITY_TABLE_PATH = Path("data/eligibility_table.csv")


@dataclass(frozen=True)
class _CardRule:
    """One row of the eligibility table, parsed into typed fields."""

    bank: str
    card: str
    min_income: float
    min_age: int
    min_cibil: int
    employment: tuple[str, ...]
    relationship_required: bool


def _parse_employment(raw: str) -> tuple[str, ...]:
    raw = (raw or "any").strip().lower()
    if raw == "any":
        return ("any",)
    return tuple(part.strip() for part in raw.split(","))


def _coerce_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("true", "1", "yes")


def load_eligibility_table(path: str | Path = DEFAULT_ELIGIBILITY_TABLE_PATH) -> pd.DataFrame:
    """Load and conservatively clean the eligibility ground-truth table.

    Missing/blank values are filled with conservative defaults so that a
    gap in the data never silently makes a card easier to qualify for:
    min_income -> inf, min_age -> 18, min_cibil -> 300, employment -> 'any',
    relationship_required -> False.

    Args:
        path: path to the eligibility CSV.

    Returns:
        A cleaned DataFrame with the required columns coerced to sane types.

    Raises:
        ValueError: if any required column is missing from the CSV.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Eligibility table at {path} is missing required columns: {missing}")

    df["min_income"] = pd.to_numeric(df["min_income"], errors="coerce").fillna(float("inf"))
    df["min_age"] = pd.to_numeric(df["min_age"], errors="coerce").fillna(18).astype(int)
    df["min_cibil"] = pd.to_numeric(df["min_cibil"], errors="coerce").fillna(300).astype(int)
    df["employment"] = df["employment"].fillna("any")
    df["relationship_required"] = df["relationship_required"].apply(_coerce_bool)
    return df


class EligibilityOracle:
    """Pure, deterministic eligibility rule engine (no ML, no retrieval).

    Serves two purposes in this system:
      1. The rule-based baseline recommender (ablation arm B0).
      2. The ground-truth "expected" answer generator used to build and
         score the gold evaluation set.
    """

    def __init__(self, table_path: str | Path = DEFAULT_ELIGIBILITY_TABLE_PATH) -> None:
        df = load_eligibility_table(table_path)
        self._rules: list[_CardRule] = [
            _CardRule(
                bank=str(row.bank),
                card=str(row.card),
                min_income=float(row.min_income),
                min_age=int(row.min_age),
                min_cibil=int(row.min_cibil),
                employment=_parse_employment(str(row.employment)),
                relationship_required=bool(row.relationship_required),
            )
            for row in df.itertuples(index=False)
        ]

    def cards_for_banks(self, banks: list[str]) -> list[tuple[str, str]]:
        """Return (bank, card) pairs offered by any of the requested banks."""
        banks_norm = {b.strip().lower() for b in banks}
        return [(rule.bank, rule.card) for rule in self._rules if rule.bank.strip().lower() in banks_norm]

    def evaluate(self, profile: UserProfile) -> list[Decision]:
        """Return a Decision for every card offered by the requested banks."""
        banks_norm = {b.strip().lower() for b in profile.banks}
        return [
            self._evaluate_rule(rule, profile) for rule in self._rules if rule.bank.strip().lower() in banks_norm
        ]

    def _evaluate_rule(self, rule: _CardRule, profile: UserProfile) -> Decision:
        failures: list[str] = []

        if profile.annual_income < rule.min_income:
            failures.append(
                f"annual income of {profile.annual_income:.0f} is below the "
                f"required minimum of {rule.min_income:.0f}"
            )

        if profile.age < rule.min_age:
            failures.append(f"age {profile.age} is below the required minimum of {rule.min_age}")

        if profile.cibil < rule.min_cibil:
            failures.append(
                f"CIBIL score of {profile.cibil} is below the required minimum of {rule.min_cibil}"
            )

        if "any" not in rule.employment and profile.employment.strip().lower() not in rule.employment:
            allowed = ", ".join(rule.employment)
            failures.append(
                f"employment type '{profile.employment}' is not one of the accepted types ({allowed})"
            )

        if rule.relationship_required and not profile.existing_relationship:
            failures.append("an existing account relationship with the bank is required but not held")

        if failures:
            reason = "Not eligible because " + "; ".join(failures) + "."
            return Decision(bank=rule.bank, card=rule.card, recommended=False, reason=reason)

        reason = "Eligible: meets all stated minimums for income, age, CIBIL score, employment and relationship."
        return Decision(bank=rule.bank, card=rule.card, recommended=True, reason=reason)