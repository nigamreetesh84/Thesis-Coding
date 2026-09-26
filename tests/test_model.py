from __future__ import annotations

import json

from src.domain.eligibility import EligibilityOracle
from src.domain.entities import Decision
from evaluation.run_ablation import citation_is_valid
from src.infrastructure.llm_client import StubLLM
from src.model import GroundedPromptBuilder, parse_llm_response


def test_oracle_applies_expected_card_rules(example_profile) -> None:
    decisions = EligibilityOracle().evaluate(example_profile)
    outcomes = {(decision.bank, decision.card): decision.recommended for decision in decisions}

    assert outcomes[("SBI", "SBI SimplySAVE")] is True
    assert outcomes[("SBI", "SBI Prime")] is False
    assert outcomes[("HDFC", "HDFC MoneyBack+")] is True
    assert outcomes[("HDFC", "HDFC Diners Club Black")] is False


def test_prompt_includes_retrieved_rule_and_profile(example_profile) -> None:
    prompt = GroundedPromptBuilder().build(
        example_profile,
        [{"chunk_id": "sbi_cards#4", "bank": "SBI", "card": "SBI SimplySAVE", "text": "Minimum income rule."}],
    )

    assert "chunk_id=sbi_cards#4 bank=SBI card=SBI SimplySAVE" in prompt
    assert "Minimum income rule." in prompt
    assert "annual_income=800000" in prompt
    assert "### APPLICANT PROFILE" in prompt
    assert "cited_chunk_id" in prompt


def test_parse_llm_response_parses_decisions_and_malformed_output() -> None:
    payload = json.dumps(
        [{"bank": "SBI", "card": "SBI SimplySAVE", "recommended": True, "reason": "Eligible", "cited_chunk_id": "sbi_cards#4"}]
    )

    decisions = parse_llm_response(payload)

    assert len(decisions) == 1
    assert decisions[0].recommended is True
    assert decisions[0].cited_chunk_id == "sbi_cards#4"
    assert parse_llm_response("not valid JSON") == []
    assert parse_llm_response('[{"bank":"SBI"}]') == []


def test_stub_llm_returns_decisions_for_retrieved_cards(example_profile) -> None:
    prompt = GroundedPromptBuilder().build(
        example_profile,
        [{"chunk_id": "sbi_cards#4", "bank": "SBI", "card": "SBI SimplySAVE", "text": "Minimum income rule."}],
    )

    decisions = parse_llm_response(StubLLM().generate(prompt))

    assert decisions
    assert all(decision.bank in example_profile.banks for decision in decisions)
    assert any(decision.cited_chunk_id == "sbi_cards#4" for decision in decisions)


def test_citation_validity_requires_retrieved_matching_card() -> None:
    retrieved = [{"chunk_id": "sbi_cards#4", "bank": "SBI", "card": "SBI SimplySAVE"}]
    valid_decision = Decision("SBI", "SBI SimplySAVE", True, "Eligible", "sbi_cards#4")
    unknown_chunk = Decision("SBI", "SBI SimplySAVE", True, "Eligible", "missing#1")
    wrong_card = Decision("SBI", "SBI Prime", True, "Eligible", "sbi_cards#4")
    no_citation = Decision("SBI", "SBI SimplySAVE", True, "Eligible")

    assert citation_is_valid(valid_decision, retrieved)
    assert not citation_is_valid(unknown_chunk, retrieved)
    assert not citation_is_valid(wrong_card, retrieved)
    assert not citation_is_valid(no_citation, retrieved)