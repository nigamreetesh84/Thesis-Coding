"""Tests for counterfactual "what needs to change" explanations (round 9; CPU, no model downloads)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.run_counterfactual_eval import requirements_from_table, undo_one
from src.domain.counterfactual import apply_changes, closest_declined, summary, what_needs_to_change
from src.domain.eligibility import EligibilityOracle
from src.domain.entities import UserProfile
from src.domain.query_parsing import ParsedQuery
from src.domain.requirements import CardRequirements, decide, decide_guarded
from src.nl_interface import NLRecommender, QueryParser

REGALIA = CardRequirements("HDFC", "HDFC Regalia", min_income=1_200_000, min_age=21, min_cibil=750,
                           allowed_employment=("salaried", "self-employed"), relationship_required=False,
                           sources={"income": "hdfc_cards#7", "cibil": "hdfc_cards#8"})
ELITE = CardRequirements("SBI", "SBI Elite", min_income=2_000_000, min_age=21, min_cibil=780,
                         allowed_employment=("business owner", "salaried", "self-employed"), relationship_required=True)
P = UserProfile(annual_income=900_000, age=32, cibil=735, employment="salaried", existing_relationship=False)


def test_income_and_cibil_gaps_are_exact_and_minimal() -> None:
    changes = what_needs_to_change(P, REGALIA)
    assert [c.criterion for c in changes] == ["income", "cibil"]
    assert changes[0].gap == 300_000 and changes[1].gap == 15
    assert "₹3,00,000" in changes[0].text and "₹25,000 a month" in changes[0].text
    assert all(c.kind == "actionable" for c in changes)
    assert decide(apply_changes(P, changes), REGALIA).recommended
    assert not decide(replace(apply_changes(P, changes), cibil=749), REGALIA).recommended


def test_eligible_card_has_no_changes() -> None:
    rich = replace(P, annual_income=1_500_000, cibil=800)
    d = decide(rich, REGALIA)
    assert d.recommended and d.changes == () and summary(d.changes) == ""


def test_relationship_age_and_employment_kinds() -> None:
    young_owner = UserProfile(annual_income=500_000, age=19, cibil=760, employment="business owner",
                              existing_relationship=False)
    kinds = {c.criterion: c.kind for c in what_needs_to_change(young_owner, ELITE)}
    assert kinds == {"income": "actionable", "cibil": "actionable", "relationship": "actionable", "age": "time"}
    fixed = what_needs_to_change(young_owner, REGALIA)
    assert any(c.kind == "fixed" for c in fixed)
    assert summary(fixed).startswith("Not reachable by changing income or score")
    assert "salaried or self-employed" in summary(fixed)


def test_unstated_requirement_gives_no_change() -> None:
    req = replace(REGALIA, min_cibil=None)
    assert [c.criterion for c in what_needs_to_change(P, req)] == ["income"]


def test_counterfactual_does_not_change_verdict_reason_or_citation() -> None:
    d = decide(P, REGALIA)
    assert not d.recommended
    assert d.reason.startswith("Not eligible because annual income of ₹9,00,000 is below the ₹12,00,000 minimum")
    assert d.cited_chunk_id == "hdfc_cards#7"
    assert len(d.changes) == 2


def test_needs_review_decision_gets_no_counterfactual() -> None:
    unreadable = replace(REGALIA, min_income=None)
    d = decide_guarded(P, unreadable, stated={"income"})
    assert d.needs_review and d.changes == ()


def test_all_gold_declines_are_valid_minimal_and_sparse() -> None:
    """Every counterfactual on the 120 gold profiles x 8 cards, checked against the independent oracle."""
    oracle = EligibilityOracle()
    reqs = requirements_from_table(oracle)
    n = 0
    for line in Path("data/gold_test_set.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(line)["profile"]
        profile = UserProfile(p["annual_income"], p["age"], p["cibil"], p["employment"], p["existing_relationship"])
        truth = {d.card: d.recommended for d in oracle.evaluate(profile)}
        for (_bank, card), req in reqs.items():
            d = decide(profile, req)
            assert d.recommended == truth[card]
            if d.recommended:
                continue
            n += 1
            after = apply_changes(profile, d.changes)
            assert {x.card: x.recommended for x in oracle.evaluate(after)}[card], (card, p)
            for c in d.changes:
                undone = undo_one(after, profile, c)
                assert not {x.card: x.recommended for x in oracle.evaluate(undone)}[card], (card, c.criterion)
    assert n == 738


def test_closest_card_prefers_fewest_actionable_changes() -> None:
    decisions = [decide(P, REGALIA), decide(P, ELITE)]
    near = closest_declined(decisions, P)
    assert near is not None and near.card == "HDFC Regalia"


class _FakeRecommender:
    decision_mode = "extract_compare"

    def recommend(self, profile: UserProfile):
        return [decide(profile, REGALIA), decide(profile, ELITE)], 0


def test_nl_answer_shows_what_needs_to_change() -> None:
    nl = NLRecommender(QueryParser(None), _FakeRecommender())  # type: ignore[arg-type]
    parsed = ParsedQuery(income=900_000, age=32, cibil=735, employment="salaried", existing_relationship=False,
                         banks=("SBI", "HDFC"))
    res = nl.decide(parsed)
    regalia = next(a for a in res.answers if a.card == "HDFC Regalia")
    assert regalia.what_to_change.startswith("To qualify: raise annual income by ₹3,00,000")
    assert regalia.sentence == regalia.engine_reason  # writer off: engine reason shown, unchanged
    assert "→ To qualify" in res.text
    assert res.closest and res.closest.startswith("Closest card to reach: HDFC Regalia")


def test_api_decision_out_carries_changes() -> None:
    pytest.importorskip("fastapi")
    from src.api import _decision_out

    out = _decision_out(decide(P, REGALIA))
    assert out.what_to_change.startswith("To qualify:")
    assert [c.criterion for c in out.changes] == ["income", "cibil"]
    assert out.changes[0].gap == 300_000
