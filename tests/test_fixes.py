"""Regression tests for the end-to-end fixes (parser, A3, RQ2, A6, LLM cache)."""
from __future__ import annotations

import json

import pytest

from evaluation.run_ablation import _FreeTextPromptBuilder, load_gold_set
from src.domain.entities import UserProfile
from src.infrastructure.llm_client import LocalLLMClient
from src.model import ANTI_HALLUCINATION_CLAUSE, SYSTEM_PROMPT, GroundedPromptBuilder, parse_llm_response

PROFILE = UserProfile(
    annual_income=731330.18, age=30, cibil=768, employment="salaried", existing_relationship=True, banks=["SBI"]
)
CHUNK = {"chunk_id": "sbi_cards#0", "bank": "SBI", "card": "SBI Prime", "text": "### SBI Prime - Income\nrule"}


def test_parser_keeps_good_items_and_handles_null_reason() -> None:
    raw = json.dumps(
        [
            {"bank": "SBI", "card": "A", "reason": None, "recommended": True, "cited_chunk_id": None},
            {"bank": "SBI", "card": "B"},  # malformed: no 'recommended'
            {"bank": "SBI", "card": "C", "reason": "x", "recommended": "false", "cited_chunk_id": "c#1"},
        ]
    )
    decisions = parse_llm_response(raw)
    assert [(d.card, d.recommended, d.reason) for d in decisions] == [("A", True, ""), ("C", False, "x")]
    assert parse_llm_response("not json") == []


def test_prompt_lists_income_in_lakh_and_keeps_structured_field() -> None:
    prompt = GroundedPromptBuilder().build(PROFILE, [CHUNK])
    assert "annual_income=731330" in prompt
    assert "annual_income_in_lakh=7.31" in prompt


def test_a3_clause_is_present_in_system_prompt() -> None:
    assert ANTI_HALLUCINATION_CLAUSE in SYSTEM_PROMPT
    assert ANTI_HALLUCINATION_CLAUSE in GroundedPromptBuilder().build(PROFILE, [CHUNK])


def test_free_text_builder_replaces_only_the_profile_block() -> None:
    builder = _FreeTextPromptBuilder()
    builder.text = "I'm 30, salaried, earning about 7.3 lakh rupees."
    prompt = builder.build(PROFILE, [CHUNK])
    assert "I'm 30, salaried, earning about 7.3 lakh rupees." in prompt
    assert "annual_income=" not in prompt
    assert "### OUTPUT INSTRUCTIONS" in prompt and "### CARDS TO EVALUATE" in prompt


def test_gold_records_carry_profile_text() -> None:
    records = load_gold_set("data/gold_test_set.jsonl")
    assert all(r.profile_text for r in records)


def test_llm_cache_generates_once_and_persists(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "cache.jsonl"))
    monkeypatch.setenv("LLM_CACHE", "1")
    monkeypatch.setattr(LocalLLMClient, "_cache", None)
    monkeypatch.setattr(LocalLLMClient, "_cache_path", None)

    calls: list[str] = []

    class FakeTokenizer:
        chat_template = None

    class FakePipeline:
        tokenizer = FakeTokenizer()

        def __call__(self, text: str, **_: object) -> list[dict[str, str]]:
            calls.append(text)
            return [{"generated_text": "OUT:" + text}]

    client = LocalLLMClient("fake-model")
    client._pipeline = FakePipeline()
    assert client.generate("hello") == "OUT:hello"
    assert client.generate("hello") == "OUT:hello"
    assert len(calls) == 1

    # a fresh process (empty in-memory cache) must reuse the on-disk cache
    monkeypatch.setattr(LocalLLMClient, "_cache", None)
    monkeypatch.setattr(LocalLLMClient, "_cache_path", None)
    client2 = LocalLLMClient("fake-model")
    client2._pipeline = FakePipeline()
    assert client2.generate("hello") == "OUT:hello"
    assert len(calls) == 1


def test_per_card_prompting_makes_one_prompt_per_card() -> None:
    from src.model import RAGRecommender

    chunks = [
        {"chunk_id": f"{b}#{i}", "bank": b, "card": c, "text": f"### {c}\nrule {i}"}
        for i, (b, c) in enumerate([("SBI", "A"), ("SBI", "A"), ("SBI", "B"), ("HDFC", "C")])
    ]

    class FakeRetriever:
        def retrieve_for_profile(self, profile):  # noqa: ANN001
            return chunks

    class FakeLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            card = prompt.split("### CARDS TO EVALUATE\n")[1].split("\n")[0]
            bank = card.split("bank=")[1].split(";")[0]
            name = card.split("card=")[1]
            return json.dumps([
                {"bank": bank, "card": name, "reason": "r", "recommended": True, "cited_chunk_id": "x"},
                {"bank": "OTHER", "card": "extra", "reason": "r", "recommended": True, "cited_chunk_id": "x"},
            ])

    llm = FakeLLM()
    decisions, k = RAGRecommender(FakeRetriever(), llm, per_card=True).recommend(PROFILE)
    assert len(llm.prompts) == 3 and k == 4
    assert [(d.bank, d.card) for d in decisions] == [("SBI", "A"), ("SBI", "B"), ("HDFC", "C")]  # extras dropped
    assert all(p.count("- bank=") == 1 for p in llm.prompts)

    llm2 = FakeLLM()
    RAGRecommender(FakeRetriever(), llm2, per_card=False).recommend(PROFILE)
    assert len(llm2.prompts) == 1


def test_a6_balanced_sample_has_both_variants() -> None:
    from evaluation.run_ablation import a6_balanced_sample

    records = load_gold_set("data/gold_test_set.jsonl")
    sample = a6_balanced_sample(records, 10)
    assert sum(r.variant == "reworded" for r in sample) == 10
    assert sum(r.variant == "normal" for r in sample) == 10
    assert len(a6_balanced_sample(records, 0)) == 2 * sum(r.variant == "reworded" for r in records)


def test_rq4_goes_through_the_per_card_recommender(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.run_ablation as ra

    if not (ra.settings.artifacts_dir / "faiss.index").exists():
        pytest.skip("index not built")
    calls: list[str] = []

    class CountingLLM:
        def generate(self, prompt: str) -> str:
            calls.append(prompt)
            return "[]"

    monkeypatch.setattr(ra, "make_llm_clients", lambda use_stub: (CountingLLM(), CountingLLM()))
    records = ra.load_gold_set("data/gold_test_set.jsonl")[:2]
    ra.rq4_traceability_metrics(records, top_k=4, use_stub=False)
    assert len(calls) == 2 * 8  # 8 cards x 2 profiles, one prompt per card


class _BatchPipeline:
    """Mimics a transformers text-generation pipeline (single and batched calls)."""

    class tokenizer:  # noqa: N801
        chat_template = None
        pad_token_id = 0
        eos_token = "</s>"
        padding_side = "right"

    def __init__(self, fail_batch: bool = False) -> None:
        self.calls: list[object] = []
        self.fail_batch = fail_batch

    def __call__(self, text, **kwargs):  # noqa: ANN001
        self.calls.append(text)
        if isinstance(text, list):
            if self.fail_batch:
                raise RuntimeError("CUDA out of memory")
            return [[{"generated_text": "OUT:" + t}] for t in text]
        return [{"generated_text": "OUT:" + text}]


def _fresh_client(monkeypatch: pytest.MonkeyPatch, tmp_path, pipe: _BatchPipeline) -> LocalLLMClient:
    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "c.jsonl"))
    monkeypatch.setenv("LLM_CACHE", "1")
    monkeypatch.setattr(LocalLLMClient, "_cache", None)
    monkeypatch.setattr(LocalLLMClient, "_cache_path", None)
    client = LocalLLMClient("fake-model")
    client._pipeline = pipe
    return client


def test_batch_generation_is_batched_cached_and_ordered(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _BatchPipeline()
    client = _fresh_client(monkeypatch, tmp_path, pipe)
    assert client.generate_batch(["a", "b", "c"]) == ["OUT:a", "OUT:b", "OUT:c"]
    assert len(pipe.calls) == 1 and isinstance(pipe.calls[0], list)
    # 'b' is cached now: only 'd' is generated, order is preserved
    assert client.generate_batch(["b", "d"]) == ["OUT:b", "OUT:d"]
    assert pipe.calls[-1] == "OUT:d" or pipe.calls[-1] == "d"


def test_batch_falls_back_to_sequential_on_error(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _BatchPipeline(fail_batch=True)
    client = _fresh_client(monkeypatch, tmp_path, pipe)
    assert client.generate_batch(["x", "y"]) == ["OUT:x", "OUT:y"]


def test_ungrounded_prompt_keeps_cards_and_profile_but_not_rules() -> None:
    from src.infrastructure.llm_client import to_ungrounded_prompt

    full = GroundedPromptBuilder().build(PROFILE, [CHUNK])
    out = to_ungrounded_prompt(full)
    assert "### CARDS TO EVALUATE\n- bank=SBI; card=SBI Prime" in out
    assert "annual_income=731330" in out
    assert "chunk_id=sbi_cards#0" not in out and "### RETRIEVED RULES\n" not in out
    assert "No rule documents are provided" in out and "Use ONLY the rules" not in out


def test_ungrounded_client_prepares_the_ungrounded_prompt(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.infrastructure.llm_client import UngroundedLLMClient

    pipe = _BatchPipeline()
    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "u.jsonl"))
    monkeypatch.setattr(LocalLLMClient, "_cache", None)
    monkeypatch.setattr(LocalLLMClient, "_cache_path", None)
    client = UngroundedLLMClient("fake-model")
    client._pipeline = pipe
    out = client.generate(GroundedPromptBuilder().build(PROFILE, [CHUNK]))
    assert "chunk_id=sbi_cards#0" not in out and "card=SBI Prime" in out


def test_select_records_dev_and_test_are_disjoint_and_cover_all() -> None:
    from evaluation.run_ablation import select_records

    records = load_gold_set("data/gold_test_set.jsonl")
    test, dev = select_records(records, "test", 0), select_records(records, "dev", 0)
    assert len(test) == 40 and len(dev) == 80
    assert not ({id(r) for r in test} & {id(r) for r in dev})
    assert [id(r) for r in select_records(records, "all", 40)] == [id(records[int(i * 3)]) for i in range(40)]
    assert [id(r) for r in select_records(records, "test", 0)] == [id(r) for r in select_records(records, "all", 40)]
