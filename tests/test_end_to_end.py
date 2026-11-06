"""End-to-end and integrity tests (stub LLM; no GPU, no downloads beyond the hashing fallback)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

import evaluation.run_ablation as ra
from config.settings import settings
from evaluation import verify_setup


def test_integrity_checks_pass_on_shipped_data() -> None:
    assert verify_setup.check_docs_match_table() == []
    assert verify_setup.check_gold_matches_oracle(settings.gold_test_set_path) == []
    assert verify_setup.check_prompts() == []


def test_integrity_check_detects_doc_table_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = tmp_path / "rule_docs"
    shutil.copytree(settings.rule_docs_dir, docs)
    sbi = docs / "sbi_cards.md"
    sbi.write_bytes(sbi.read_bytes().replace(b"6,00,000", b"7,00,000"))  # SimplyCLICK income drift
    monkeypatch.setattr(settings, "rule_docs_dir", docs)
    problems = verify_setup.check_docs_match_table()
    assert any("SimplyCLICK" in p and "min_income" in p for p in problems)


def test_integrity_check_detects_gold_label_drift(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in settings.gold_test_set_path.read_text(encoding="utf-8").splitlines()[:3]]
    rows[0]["expected"][0]["recommended"] = not rows[0]["expected"][0]["recommended"]
    bad = tmp_path / "gold.jsonl"
    bad.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert len(verify_setup.check_gold_matches_oracle(bad)) == 1


def test_full_ablation_cli_writes_complete_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not settings.faiss_index_path.exists():
        pytest.skip("index not built")
    out = tmp_path / "results"
    argv = ["run_ablation", "--use-stub-llm", "--split", "test", "--limit", "6", "--out-dir", str(out),
            "--arms", "A2", "A3", "A4", "A5", "A6"]
    monkeypatch.setattr(sys, "argv", argv)
    ra.main()

    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert {"A1", "RQ1", "RQ2", "RQ4", "A2", "A3", "A4", "A5", "A6"} <= set(data["arms"])
    a1 = data["arms"]["A1"]["result"]
    assert a1["n_profiles"] == 6 and a1["grounded_cards_returned_rate"] == 1.0 and a1["ungrounded_cards_returned_rate"] == 1.0
    assert set(data["arms"]["A2"]["result"]) == {"1", "2", "3", "4"}
    assert data["arms"]["RQ2"]["result"]["mean_context_recall"] == 1.0  # same retrieval path as the system
    rows = [json.loads(line) for line in (out / "predictions_A1_grounded.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 and set(rows[0]) >= {"expected", "actual", "reasons", "cited", "correct"}
    assert data["runs"][-1]["per_card_prompting"] is True


def test_recommender_uses_one_batched_call_per_profile_with_local_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RAGRecommender + LocalLLMClient: 8 per-card prompts go through ONE batched pipeline call."""
    import re

    from src.infrastructure.llm_client import LocalLLMClient
    from src.model import RAGRecommender

    if not settings.faiss_index_path.exists():
        pytest.skip("index not built")

    class Tok:
        chat_template = None
        pad_token_id = 0
        eos_token = "</s>"
        padding_side = "right"

    class Pipe:
        tokenizer = Tok()
        calls: list[int] = []

        def __call__(self, text, **kwargs):  # noqa: ANN001
            texts = text if isinstance(text, list) else [text]
            self.calls.append(len(texts))
            outs = []
            for t in texts:
                m = re.search(r"### CARDS TO EVALUATE\n- bank=(.+?); card=(.+)\n", t)
                payload = [{"bank": m.group(1), "card": m.group(2), "reason": "ok", "recommended": False, "cited_chunk_id": None}]
                outs.append([{"generated_text": json.dumps(payload)}])
            return outs if isinstance(text, list) else outs[0]

    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "cache.jsonl"))
    monkeypatch.setattr(LocalLLMClient, "_cache", None)
    monkeypatch.setattr(LocalLLMClient, "_cache_path", None)
    client = LocalLLMClient("fake-model")
    pipe = Pipe()
    client._pipeline = pipe

    record = ra.load_gold_set(settings.gold_test_set_path)[0]
    decisions, k = RAGRecommender(ra.build_retriever(4), client).recommend(record.profile)
    assert len(decisions) == 8 and k == 25
    assert pipe.calls == [8]  # a single batched call carrying all 8 per-card prompts
    assert {(d.bank, d.card) for d in decisions} == {(d.bank, d.card) for d in record.expected}


def test_report_is_built_from_results_and_no_core_does_not_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not settings.faiss_index_path.exists():
        pytest.skip("index not built")
    from evaluation.make_report import build_report

    out = tmp_path / "r"
    monkeypatch.setattr(sys, "argv", ["x", "--use-stub-llm", "--split", "test", "--limit", "6", "--out-dir", str(out), "--arms"])
    ra.main()
    monkeypatch.setattr(sys, "argv", ["x", "--use-stub-llm", "--split", "test", "--limit", "4", "--out-dir", str(out), "--no-core", "--arms", "A3"])
    ra.main()
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["arms"]["A1"]["result"]["n_profiles"] == 6  # not overwritten by the 4-profile --no-core run
    assert data["arms"]["A3"]["meta"]["n_profiles"] == 4
    report = build_report(out)
    assert "## A1 / RQ3" in report and "## A3" in report and "Stub run" in report and "Error analysis" in report
