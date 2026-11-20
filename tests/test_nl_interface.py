"""Tests for the natural-language interface (stub LLM; no GPU)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from config.settings import settings
from evaluation.run_ablation import build_retriever, load_gold_set, profile_correct
from src.domain.answer_guard import check_sentence
from src.domain.eligibility import EligibilityOracle
from src.domain.entities import Decision, UserProfile
from src.domain.query_parsing import ParsedQuery, normalise_income, rule_based_fields
from src.infrastructure.llm_client import StubLLM
from src.model import RAGRecommender
from src.nl_interface import AnswerWriter, CardRanker, NLRecommender, QueryParser
from src.prompts import ANSWER_WRITER_TASK_MARKER, QUERY_PARSE_TASK_MARKER


# ---------------------------------------------------------------------------
# Income normalisation (Python, never the LLM, converts units)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "rupees"),
    [
        ("9 lakh a year", 900_000),
        ("Rs 75,000 per month", 900_000),
        ("12 LPA", 1_200_000),
        ("about 28.0 lakh rupees a year", 2_800_000),
        ("251531 rupees", 251_531),
        ("Rs 28,04,342 per annum", 2_804_342),
        ("₹1.2 crore", 12_000_000),
        ("60k per month", 720_000),
        ("nine lakh", 900_000),
        ("Rs 50,000/month", 600_000),
        ("4.5 lakhs annually", 450_000),
        ("an annual income of 731330 rupees", 731_330),
    ],
)
def test_normalise_income(text: str, rupees: int) -> None:
    assert normalise_income(text) == pytest.approx(rupees)


def test_parsed_query_validates_and_asks_for_missing_fields() -> None:
    q = ParsedQuery.from_llm_dict({"income_text": "9 lakh", "age": 32, "cibil": 1200, "employment": "freelancer",
                                   "existing_banks": None, "preferred_banks": ["hdfc"], "preferences": "travel"})
    assert q.income == 900_000 and q.age == 32 and q.employment == "self-employed"
    assert q.cibil is None and "CIBIL 1200" in q.notes[0]  # out of range -> asked, never guessed
    assert q.missing == ["cibil", "existing_relationship"] and q.banks == ("HDFC",)
    follow = ParsedQuery.from_llm_dict({"cibil": "760", "existing_banks": []})
    merged = q.merged_with(follow)
    assert merged.missing == [] and merged.cibil == 760 and merged.existing_relationship is False
    assert merged.income == 900_000  # earlier answers are kept


def test_unnamed_bank_relationship_counts_as_existing() -> None:
    assert ParsedQuery.from_llm_dict({"existing_banks": ["ANY"]}).existing_relationship is True
    assert ParsedQuery.from_llm_dict({"existing_banks": ["the bank"]}).existing_relationship is True


def test_regex_baseline_reads_the_gold_texts() -> None:
    """Sanity check of the baseline (and StubLLM) on the gold set's own texts."""
    records = load_gold_set()
    ok = 0
    for r in records:
        q = ParsedQuery.from_llm_dict(rule_based_fields(r.profile_text))
        stated = round(r.profile.annual_income / 1e5, 1) * 1e5 if r.variant == "reworded" else round(r.profile.annual_income)
        ok += (q.income is not None and abs(q.income - stated) <= 1 and q.age == r.profile.age
               and q.cibil == r.profile.cibil and q.employment == r.profile.employment
               and q.existing_relationship == r.profile.existing_relationship)
    assert ok / len(records) >= 0.95


# ---------------------------------------------------------------------------
# Answer guard
# ---------------------------------------------------------------------------
PROFILE = UserProfile(annual_income=900_000, age=32, cibil=760, employment="salaried", existing_relationship=True)
DECLINE = Decision("HDFC", "HDFC Regalia", False,
                   "Not eligible because annual income of ₹9,00,000 is below the ₹12,00,000 minimum.", "hdfc_cards#7")
ACCEPT = Decision("SBI", "SBI SimplyCLICK", True, "Eligible: annual income of ₹9,00,000 is at or above the ₹6,00,000 minimum.")
REVIEW = Decision("SBI", "SBI Elite", False, "Eligibility could not be confirmed because ...", None, needs_review=True)


def test_guard_accepts_faithful_sentences() -> None:
    assert check_sentence("You don't qualify: your income of Rs 9 lakh is below the Rs 12 lakh minimum.", DECLINE, PROFILE)[0]
    assert check_sentence("Good news, you qualify for SimplyCLICK.", ACCEPT, PROFILE)[0]
    assert check_sentence("We could not confirm your eligibility; please check SBI's criteria.", REVIEW, PROFILE)[0]


@pytest.mark.parametrize(
    ("sentence", "decision"),
    [
        ("You qualify for Regalia.", DECLINE),                                   # wrong verdict
        ("You do not qualify: income below the Rs 15 lakh minimum.", DECLINE),  # invented threshold
        ("Unfortunately you are not eligible.", ACCEPT),                         # wrong verdict
        ("You qualify for Elite.", REVIEW),                                      # review shown as a verdict
        (None, ACCEPT),
    ],
)
def test_guard_rejects_unfaithful_sentences(sentence: Any, decision: Decision) -> None:
    assert not check_sentence(sentence, decision, PROFILE)[0]


class _WriterLLM:
    """Returns one faithful and one hallucinated sentence."""

    def generate(self, prompt: str) -> str:
        assert ANSWER_WRITER_TASK_MARKER in prompt
        return json.dumps({"SBI SimplyCLICK": "You qualify for SimplyCLICK.",
                           "HDFC Regalia": "You qualify for Regalia with your Rs 20 lakh income."})


def test_writer_falls_back_to_engine_reason_when_guard_rejects() -> None:
    answers = AnswerWriter(_WriterLLM()).write([ACCEPT, DECLINE], PROFILE)
    assert answers[0].source == "llm" and answers[0].sentence == "You qualify for SimplyCLICK."
    assert answers[1].source == "engine_reason" and answers[1].sentence == DECLINE.reason
    assert [a.recommended for a in answers] == [True, False]  # verdicts never change


# ---------------------------------------------------------------------------
# End to end through the frozen extract_compare recommender
# ---------------------------------------------------------------------------
@pytest.fixture
def nl() -> NLRecommender:
    if not settings.faiss_index_path.exists():
        pytest.skip("index not built")
    retriever = build_retriever(settings.top_k)
    llm = StubLLM()
    rec = RAGRecommender(retriever, llm, decision_mode="extract_compare")
    return NLRecommender(QueryParser(llm), rec, CardRanker(retriever.embedder, retriever.vector_store), AnswerWriter(llm))


def test_nl_recommender_matches_the_oracle(nl: NLRecommender) -> None:
    res = nl.ask("I'm 32 years old, a salaried employee earning 9 lakh a year, CIBIL score 760. "
                 "I already have a savings account with SBI. I want a travel card.")
    assert res.status == "ok"
    profile = res.understood.to_profile()
    assert profile.annual_income == 900_000 and profile.existing_relationship is True
    expected = EligibilityOracle(settings.eligibility_table_path).evaluate(profile)
    got = [Decision(a.bank, a.card, a.recommended, a.sentence) for a in res.answers]
    assert profile_correct(got, expected)
    order = [a.recommended for a in res.answers]
    assert order == sorted(order, reverse=True)  # eligible cards listed first
    assert "You qualify for" in res.text


def test_nl_recommender_asks_then_answers(nl: NLRecommender) -> None:
    first = nl.ask("I'm 40 and run my own business, income Rs 1,80,000 per month.")
    assert first.status == "need_more_info"
    assert set(first.understood.missing) == {"cibil", "existing_relationship"}
    second = nl.ask("My CIBIL is 790 and I'm not an existing customer of either bank.", previous=first.understood)
    assert second.status == "ok" and second.understood.income == 2_160_000 and len(second.answers) == 8


def test_nl_recommender_rejects_llm_decides_mode() -> None:
    if not settings.faiss_index_path.exists():
        pytest.skip("index not built")
    rec = RAGRecommender(build_retriever(4), StubLLM(), decision_mode="llm")
    with pytest.raises(ValueError):
        NLRecommender(QueryParser(StubLLM()), rec)


def test_stub_handles_new_markers_without_touching_old_paths() -> None:
    stub = StubLLM()
    out = json.loads(stub.generate(QueryParser.build_prompt("I'm 25, salaried, 6 lakh a year, CIBIL 720.")))
    assert out["age"] == 25 and out["cibil"] == 720
    assert QUERY_PARSE_TASK_MARKER not in AnswerWriter.build_prompt([ACCEPT])
    assert stub.generate(AnswerWriter.build_prompt([ACCEPT])) == "{}"


def test_ask_endpoint() -> None:
    if not settings.faiss_index_path.exists():
        pytest.skip("index not built")
    from fastapi.testclient import TestClient

    from src.api import app

    with TestClient(app) as client:
        r = client.post("/ask", json={"message": "I'm 30, salaried, 7 lakh a year."})
        body = r.json()
        assert r.status_code == 200 and body["status"] == "need_more_info" and len(body["questions"]) == 2
        r2 = client.post("/ask", json={"message": "CIBIL 770, and I already bank with HDFC.",
                                       "previous": body["understood"]})
        body2 = r2.json()
        assert body2["status"] == "ok" and len(body2["cards"]) == 8 and body2["answer_text"]


def test_nl_eval_smoke() -> None:
    if not settings.faiss_index_path.exists():
        pytest.skip("index not built")
    from evaluation.run_ablation import select_records
    from evaluation.run_nl_eval import evaluate

    records = select_records(load_gold_set(), "holdout", 8)
    llm = StubLLM()
    out = evaluate(records, llm, RAGRecommender(build_retriever(4), llm, decision_mode="extract_compare"),
                   with_writer=True)
    res = out["results"]
    assert res["gold_text"]["llm_parser"]["fields_correct"]["n"] == 40
    assert res["user_style"]["answer_writer"]["final_answer_matches_engine_verdict"]["rate"] == 1.0
    assert len(out["rows"]) == 16


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I already have an SBI savings account.", True),
        ("I have a salary account with HDFC.", True),
        ("I'm already a customer of SBI.", True),
        ("I bank with HDFC.", True),
        ("I don't have any account with SBI or HDFC.", False),
        ("I'm new to both banks.", False),
        ("I'm 30 and salaried.", None),
    ],
)
def test_regex_baseline_reads_bank_relationship(message: str, expected: bool | None) -> None:
    assert ParsedQuery.from_llm_dict(rule_based_fields(message)).existing_relationship is expected


def test_short_yes_answers_the_relationship_question(nl: NLRecommender) -> None:
    first = nl.ask("I'm 32, salaried, earning 9 lakh a year, CIBIL 760.")
    assert first.status == "need_more_info" and first.understood.missing == ["existing_relationship"]
    assert nl.ask("yes", previous=first.understood).understood.existing_relationship is True
    assert nl.ask("No.", previous=first.understood).understood.existing_relationship is False
