"""Regression tests for the round-5 changes (Nov 2026).

Each test pins down a failure seen in the 14 Oct evaluation outputs or a
guarantee the dissertation relies on.
"""

from __future__ import annotations

import json

import pytest

from evaluation.metrics import extraction_field_matches, wilson_interval
from evaluation.run_ablation import chunk_criteria, failing_criteria, select_records
from src.domain.eligibility import EligibilityOracle
from src.domain.entities import Decision, UserProfile
from src.domain.requirements import CardRequirements, decide, normalise_employment
from src.infrastructure.llm_client import StubLLM, UngroundedStubLLM, to_ungrounded_prompt
from src.model import (
    ExtractionPromptBuilder,
    RAGRecommender,
    coerce_recommended,
    extract_json_items,
    parse_llm_response,
)
from src.prompts import EXTRACTION_TASK_MARKER, UNGROUNDED_EXTRACTION_SYSTEM_PROMPT

# ---------------------------------------------------------------- parser


def test_parser_accepts_single_object_instead_of_array() -> None:
    raw = '{"bank": "HDFC", "card": "HDFC Regalia", "reason": "ok", "recommended": true, "cited_chunk_id": "hdfc_cards#6"}'
    decisions = parse_llm_response(raw)
    assert len(decisions) == 1 and decisions[0].recommended is True


def test_parser_handles_code_fence_prose_and_trailing_comma() -> None:
    raw = (
        "Here is my answer:\n```json\n[\n"
        '{"bank": "SBI", "card": "SBI Prime", "reason": "income [15 lakh] met", "recommended": "Yes", "cited_chunk_id": null},\n'
        "]\n```\nHope this helps."
    )
    decisions = parse_llm_response(raw)
    assert [(d.card, d.recommended, d.cited_chunk_id) for d in decisions] == [("SBI Prime", True, None)]


def test_parser_handles_wrapper_object() -> None:
    raw = json.dumps({"decisions": [{"bank": "SBI", "card": "SBI Elite", "reason": "x", "recommended": False}]})
    assert parse_llm_response(raw)[0].card == "SBI Elite"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), ("false", False), ("No", False), ("Eligible", True), ("not eligible", False), ("maybe", None), (None, None)],
)
def test_verdict_coercion_never_turns_unknown_into_true(value: object, expected: bool | None) -> None:
    assert coerce_recommended(value) is expected


def test_unreadable_verdict_is_skipped_not_guessed() -> None:
    raw = '[{"bank": "SBI", "card": "SBI Prime", "reason": "x", "recommended": "maybe"}]'
    assert parse_llm_response(raw) == []


def test_extract_json_items_returns_empty_for_garbage() -> None:
    assert extract_json_items("I cannot answer that.") == []
    assert extract_json_items("") == []


# ---------------------------------------------------------------- per-card name recovery


class _FixedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str) -> str:
        return self.reply


class _OneCardRetriever:
    def retrieve_for_profile(self, profile: UserProfile) -> list[dict[str, str]]:
        return [{"chunk_id": "hdfc_cards#0", "bank": "HDFC", "card": "HDFC MoneyBack+",
                 "text": "### HDFC MoneyBack+ – Income Requirement\nAbout ₹2,50,000."}]


def test_per_card_reply_with_different_card_spelling_is_kept(example_profile: UserProfile) -> None:
    reply = '[{"bank": "HDFC Bank", "card": "MoneyBack Plus", "reason": "meets all", "recommended": true}]'
    rec = RAGRecommender(_OneCardRetriever(), _FixedLLM(reply), per_card=True, decision_mode="llm")  # type: ignore[arg-type]
    decisions, _ = rec.recommend(example_profile)
    assert [(d.bank, d.card, d.recommended) for d in decisions] == [("HDFC", "HDFC MoneyBack+", True)]
    assert rec.last_name_recoveries == 1


# ---------------------------------------------------------------- deterministic comparison


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Salaried", "salaried"), ("self employed professionals", "self-employed"),
     ("small business owners", "business owner"), ("astronaut", None)],
)
def test_employment_normalisation(text: str, expected: str | None) -> None:
    assert normalise_employment(text) == expected


def test_requirements_accept_lakh_and_indian_grouping() -> None:
    req = CardRequirements.from_llm_dict("SBI", "SBI SimplyCLICK", {"requirements": {
        "min_annual_income_inr": "₹6,00,000", "min_age": "21 years", "min_cibil": 750,
        "allowed_employment": ["Salaried"], "existing_relationship_required": "no"}})
    assert (req.min_income, req.min_age, req.min_cibil, req.allowed_employment, req.relationship_required) == (
        600000.0, 21, 750, ("salaried",), False)
    assert CardRequirements.from_llm_dict("X", "Y", {"requirements": {"min_annual_income_inr": "6 lakh"}}).min_income == 600000.0


def test_decide_gets_the_comparisons_right_that_the_llm_got_wrong() -> None:
    """Real 14 Oct failure: 'income of Rs 28,04,342 is below Rs 6,00,000' and 'age 56 below 21'."""
    profile = UserProfile(annual_income=2804342.14, age=56, cibil=806, employment="salaried",
                          existing_relationship=False, banks=["SBI"])
    req = CardRequirements("SBI", "SBI SimplyCLICK", 600000, 21, 750, ("salaried",), False,
                           sources={"income": "sbi_cards#0", "age": "sbi_cards#1"})
    decision = decide(profile, req)
    assert decision.recommended is True
    assert "₹28,04,342" in decision.reason and "₹6,00,000" in decision.reason


def test_decide_cites_the_chunk_of_the_failing_criterion() -> None:
    profile = UserProfile(annual_income=900000, age=30, cibil=700, employment="salaried",
                          existing_relationship=False, banks=["HDFC"])
    req = CardRequirements("HDFC", "HDFC Regalia", 1200000, 21, 750, ("salaried", "self-employed"), False,
                           sources={"income": "hdfc_cards#6", "cibil": "hdfc_cards#7"})
    decision = decide(profile, req)
    assert decision.recommended is False
    assert decision.cited_chunk_id == "hdfc_cards#6"
    assert "below the ₹12,00,000 minimum" in decision.reason and "CIBIL score 700" in decision.reason


def test_unstated_requirement_is_not_applied() -> None:
    profile = UserProfile(annual_income=100, age=18, cibil=300, employment="business owner",
                          existing_relationship=False, banks=["HDFC"])
    assert decide(profile, CardRequirements("HDFC", "Card")).recommended is True


# ---------------------------------------------------------------- extraction prompts and stub


def _chunks() -> list[dict[str, str]]:
    return [
        {"chunk_id": "hdfc_cards#7", "bank": "HDFC", "card": "HDFC Regalia",
         "text": "### HDFC Regalia – Age, Credit Score and Employment\n21+, ~750, salaried or self-employed."},
        {"chunk_id": "hdfc_cards#6", "bank": "HDFC", "card": "HDFC Regalia",
         "text": "### HDFC Regalia – Income Requirement\nAt least twelve lakh rupees (₹12,00,000)."},
    ]


def test_extraction_prompt_has_no_profile_and_is_order_independent() -> None:
    builder = ExtractionPromptBuilder()
    prompt = builder.build(_chunks())
    assert EXTRACTION_TASK_MARKER in prompt
    assert "APPLICANT PROFILE" not in prompt
    assert prompt == builder.build(list(reversed(_chunks())))
    assert prompt.index("hdfc_cards#6") < prompt.index("hdfc_cards#7")


def test_ungrounded_extraction_prompt_drops_rules_and_swaps_system_prompt() -> None:
    ungrounded = to_ungrounded_prompt(ExtractionPromptBuilder().build(_chunks()))
    assert "twelve lakh" not in ungrounded and "hdfc_cards#6" not in ungrounded
    assert UNGROUNDED_EXTRACTION_SYSTEM_PROMPT in ungrounded
    assert "- bank=HDFC; card=HDFC Regalia" in ungrounded


def test_stub_extraction_round_trip_matches_oracle() -> None:
    profile = UserProfile(annual_income=1_300_000, age=30, cibil=760, employment="self-employed",
                          existing_relationship=False, banks=["HDFC"])

    class _R:
        def retrieve_for_profile(self, p: UserProfile) -> list[dict[str, str]]:
            return _chunks()

    rec = RAGRecommender(_R(), StubLLM(), decision_mode="extract_compare")  # type: ignore[arg-type]
    decisions, _ = rec.recommend(profile)
    oracle = {(d.bank, d.card): d.recommended for d in EligibilityOracle().evaluate(profile)}
    assert [(d.card, d.recommended) for d in decisions] == [("HDFC Regalia", oracle[("HDFC", "HDFC Regalia")])]
    assert rec.last_requirements[0].sources["income"] == "hdfc_cards#6"

    rec_u = RAGRecommender(_R(), UngroundedStubLLM(), decision_mode="extract_compare")  # type: ignore[arg-type]
    rec_u.recommend(profile)
    assert rec_u.last_requirements[0].min_income == 500000  # crude guess, not the documented 12 lakh


def test_invalid_decision_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        RAGRecommender(_OneCardRetriever(), StubLLM(), decision_mode="vibes")  # type: ignore[arg-type]


# ---------------------------------------------------------------- evaluation helpers


def test_wilson_interval_behaves_at_the_edges() -> None:
    assert wilson_interval(0, 40)[0] == 0.0
    lo, hi = wilson_interval(16, 40)
    assert 0.26 < lo < 0.27 and 0.55 < hi < 0.56


def test_extraction_scoring_treats_all_three_types_as_any() -> None:
    truth = {"min_income": 2000000, "min_age": 21, "min_cibil": 780,
             "employment": ("salaried", "self-employed", "business owner"), "relationship_required": True}
    extracted = {"min_income": 2000000, "min_age": 21, "min_cibil": 780, "allowed_employment": "any",
                 "relationship_required": None}
    m = extraction_field_matches(extracted, truth)
    assert m["allowed_employment"] and not m["relationship_required"] and m["min_income"]


def test_chunk_criteria_and_failing_criteria() -> None:
    assert chunk_criteria("### HDFC MoneyBack+ – Age and Credit Score\ntext") == {"age", "cibil"}
    assert chunk_criteria("### SBI SimplyCLICK – Credit History Requirement\ntext") == {"cibil"}
    rule = {r.card: r for r in EligibilityOracle().rules}["HDFC Regalia"]
    profile = UserProfile(annual_income=900000, age=30, cibil=760, employment="business owner",
                          existing_relationship=False, banks=["HDFC"])
    assert failing_criteria(profile, rule) == {"income", "employment"}


def test_holdout_split_is_disjoint_from_every_evaluated_split() -> None:
    records = [Decision(bank="b", card=str(i), recommended=False, reason="") for i in range(120)]
    test = {r.card for r in select_records(records, "test", 0)}  # type: ignore[arg-type]
    dev40 = {r.card for r in select_records(records, "dev", 40)}  # type: ignore[arg-type]
    assert dev40 == {r.card for r in select_records(records, "dev40", 0)}  # type: ignore[arg-type]
    holdout = {r.card for r in select_records(records, "holdout", 0)}  # type: ignore[arg-type]
    assert len(holdout) == 40 and not holdout & (test | dev40)


def test_sources_labelled_by_field_name_are_mapped_to_criteria() -> None:
    """6 Nov run: Qwen labelled sources 'min_annual_income_inr', 'min_age', ... so declines had no citation."""
    req = CardRequirements.from_llm_dict("HDFC", "HDFC Regalia", {
        "requirements": {"min_annual_income_inr": 1200000, "min_age": 21, "min_cibil": 750},
        "sources": {"min_annual_income_inr": "hdfc_cards#6", "min_age": "hdfc_cards#7",
                    "min_cibil": "hdfc_cards#7", "existing_relationship_required": "hdfc_cards#8"}})
    assert req.sources == {"income": "hdfc_cards#6", "age": "hdfc_cards#7", "cibil": "hdfc_cards#7",
                           "relationship": "hdfc_cards#8"}
    profile = UserProfile(annual_income=900000, age=30, cibil=760, employment="salaried",
                          existing_relationship=False, banks=["HDFC"])
    assert decide(profile, req).cited_chunk_id == "hdfc_cards#6"


def test_card_name_variants_match() -> None:
    from src.model import _same_card

    assert _same_card("HDFC", "HDFC MoneyBack+", "HDFC Bank", "HDFC MoneyBack Plus")
    assert _same_card("HDFC", "HDFC MoneyBack+", "HDFC", "MoneyBack")
    assert not _same_card("HDFC", "HDFC Regalia", "HDFC", "HDFC Millennia")


def test_python_style_literals_are_read_but_strings_untouched() -> None:
    """6 Nov cache: the ungrounded extraction for MoneyBack+ used Python's False."""
    raw = ('[{"bank":"HDFC","card":"HDFC MoneyBack+","requirements":{"min_annual_income_inr":300000,'
           '"existing_relationship_required":False},"sources":None,"note":"None of the True rules"}]')
    items = extract_json_items(raw)
    assert items[0]["requirements"]["existing_relationship_required"] is False
    assert items[0]["sources"] is None and items[0]["note"] == "None of the True rules"


def test_exact_mcnemar_p_matches_binomial() -> None:
    from evaluation.make_report import exact_mcnemar_p

    assert exact_mcnemar_p(16, 0) == 2 / 2**16
    assert abs(exact_mcnemar_p(9, 0) - 0.00390625) < 1e-12
    assert exact_mcnemar_p(0, 0) == 1.0 and exact_mcnemar_p(3, 3) == 1.0


# ---------------------------------------------------------------- fail-closed guard


def test_guard_refuses_to_recommend_when_a_stated_threshold_was_not_read() -> None:
    """Without the guard a missed income rule means 'no income requirement' -> false recommendation."""
    from src.domain.requirements import decide_guarded, stated_criteria

    profile = UserProfile(annual_income=400000, age=30, cibil=780, employment="salaried",
                          existing_relationship=False, banks=["HDFC"])
    missed_income = CardRequirements("HDFC", "HDFC Regalia", None, 21, 750, ("salaried", "self-employed"), None,
                                     sources={"age": "hdfc_cards#7"})
    stated = stated_criteria([c["text"] for c in _chunks()])
    assert stated >= {"income", "age", "cibil", "employment"}
    assert decide(profile, missed_income).recommended is True  # the unguarded failure mode
    guarded = decide_guarded(profile, missed_income, stated)
    assert guarded.recommended is False and guarded.needs_review is True
    assert "income requirement stated in the rules could not be read" in guarded.reason


def test_guard_rejects_implausible_values_but_passes_good_ones() -> None:
    from src.domain.requirements import decide_guarded

    profile = UserProfile(annual_income=1_300_000, age=30, cibil=780, employment="salaried",
                          existing_relationship=False, banks=["HDFC"])
    stated = {"income", "age", "cibil"}
    lakh_slip = CardRequirements("HDFC", "HDFC Regalia", 12, 21, 750)  # "12" lakh read as Rs 12
    assert decide_guarded(profile, lakh_slip, stated).needs_review is True
    good = CardRequirements("HDFC", "HDFC Regalia", 1_200_000, 21, 750)
    d = decide_guarded(profile, good, stated)
    assert d.recommended is True and d.needs_review is False


def test_guard_does_not_fire_on_legitimately_null_relationship_or_employment() -> None:
    from src.domain.requirements import decide_guarded

    profile = UserProfile(annual_income=500000, age=25, cibil=720, employment="business owner",
                          existing_relationship=False, banks=["HDFC"])
    req = CardRequirements("HDFC", "HDFC MoneyBack+", 250000, 18, 650, None, None)
    d = decide_guarded(profile, req, {"income", "age", "cibil", "employment", "relationship"})
    assert d.recommended is True and not d.needs_review


def test_recommender_applies_guard_and_it_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = json.dumps([{"bank": "HDFC", "card": "HDFC Regalia", "requirements": {
        "min_annual_income_inr": None, "min_age": 21, "min_cibil": 750,
        "allowed_employment": ["salaried", "self-employed"], "existing_relationship_required": None}, "sources": {}}])

    class _R:
        def retrieve_for_profile(self, p: UserProfile) -> list[dict[str, str]]:
            return _chunks()

    profile = UserProfile(annual_income=400000, age=30, cibil=780, employment="salaried",
                          existing_relationship=False, banks=["HDFC"])
    guarded = RAGRecommender(_R(), _FixedLLM(reply), decision_mode="extract_compare")  # type: ignore[arg-type]
    assert guarded.recommend(profile)[0][0].needs_review is True
    monkeypatch.setenv("RAG_GUARD", "0")
    unguarded = RAGRecommender(_R(), _FixedLLM(reply), decision_mode="extract_compare")  # type: ignore[arg-type]
    assert unguarded.recommend(profile)[0][0].recommended is True


def test_hard_rule_documents_keep_the_same_cards_and_sections() -> None:
    """A8 changes only the wording, so the gold labels (from the table) stay valid."""
    from pathlib import Path

    from src.pipeline import RuleChunker

    banks = {"sbi_cards.md": "SBI", "hdfc_cards.md": "HDFC"}

    def outline(folder: str) -> list[tuple[str, str, str]]:
        return [(c.bank, c.card, c.text.splitlines()[0]) for c in RuleChunker(banks).chunk_directory(Path(folder))]

    assert outline("data/rule_docs_hard") == outline("data/rule_docs")


def test_multiple_source_ids_keep_the_first() -> None:
    """A8 run: Qwen cited 'sbi_cards#7,sbi_cards#8' for SBI Prime's income."""
    req = CardRequirements.from_llm_dict("SBI", "SBI Prime", {
        "requirements": {"min_annual_income_inr": 1500000},
        "sources": {"income": "sbi_cards#7,sbi_cards#8", "age": ["sbi_cards#8"]}})
    assert req.sources == {"income": "sbi_cards#7", "age": "sbi_cards#8"}
