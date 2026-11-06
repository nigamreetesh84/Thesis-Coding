from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from evaluation.run_ablation import citation_is_valid
from src.domain.eligibility import EligibilityOracle
from src.domain.entities import Decision, UserProfile
from src.infrastructure.llm_client import LocalLLMClient, StubLLM
from src.model import GroundedPromptBuilder, Retriever, parse_llm_response


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
    assert "### CARDS TO EVALUATE" in prompt
    assert "Return exactly 1 decision object(s)" in prompt
    assert "bank=SBI; card=SBI SimplySAVE" in prompt
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


def test_local_llm_formats_mistral_instruction_prompt() -> None:
    client = LocalLLMClient("mistralai/Mistral-7B-Instruct-v0.2")
    client._pipeline = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None))

    assert client._format_prompt("Return JSON.") == "<s>[INST] Return JSON. [/INST]"


def test_local_llm_reuses_pipeline_for_same_model(monkeypatch) -> None:
    shared_pipeline = object()
    monkeypatch.setattr(LocalLLMClient, "_shared_model_name", "test-model")
    monkeypatch.setattr(LocalLLMClient, "_shared_pipeline", shared_pipeline)
    client = LocalLLMClient("test-model")

    client._load()

    assert client._pipeline is shared_pipeline


def test_retrieve_for_profile_covers_each_card_in_requested_banks() -> None:
    chunks = [
        {"chunk_id": "sbi_a#0", "bank": "SBI", "card": "Card A", "text": "Income rule."},
        {"chunk_id": "sbi_a#1", "bank": "SBI", "card": "Card A", "text": "Age rule."},
        {"chunk_id": "sbi_b#0", "bank": "SBI", "card": "Card B", "text": "Income rule."},
        {"chunk_id": "hdfc_c#0", "bank": "HDFC", "card": "Card C", "text": "Income rule."},
    ]

    class FakeEmbedder:
        def encode(self, queries):
            return queries

    class FakeVectorStore:
        def __init__(self):
            self.chunks = pd.DataFrame(chunks)

        def search(self, query_vector, top_k):
            return chunks[:top_k]

    profile = UserProfile(annual_income=400_000, age=30, cibil=750, employment="salaried", existing_relationship=False, banks=["SBI"])
    retriever = Retriever(FakeEmbedder(), FakeVectorStore(), top_k=2)

    retrieved = retriever.retrieve_for_profile(profile)

    assert {chunk["card"] for chunk in retrieved} == {"Card A", "Card B"}
    assert len(retrieved) == 3


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