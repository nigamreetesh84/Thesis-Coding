"""Round 7 bank portal: logins, upload -> review -> approve, versions, audit chain, customer flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from src.api import app
from src.portal.audit import AuditLog
from src.portal.ingest import IngestError, ingest
from src.portal.registry import diff_requirements, req_from_dict
from src.portal.rule_extractor import RuleBasedExtractor, read_age, read_income
from src.portal.samples import CARD_NAMES, build

SAMPLES = Path("data/portal_samples")


@pytest.fixture(scope="module", autouse=True)
def _samples() -> None:
    if not (SAMPLES / "axis_eligibility_2026-11.pdf").exists():
        build()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "use_stub_llm", True)
    with TestClient(app) as c:
        yield c


def login(client: TestClient, user: str, pw: str) -> dict[str, str]:
    r = client.post("/portal/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def upload_axis(client: TestClient, h: dict[str, str], name: str = "axis_eligibility_2026-11.pdf") -> dict:
    with (SAMPLES / name).open("rb") as f:
        r = client.post("/portal/admin/upload", headers=h, files={"file": (name, f, "application/pdf")},
                        data={"card_names": CARD_NAMES})
    assert r.status_code == 200, r.text
    return r.json()


# ---- units ---------------------------------------------------------------------------------
def test_income_reader_handles_monthly_words_and_superseded() -> None:
    assert read_income("### X – Income\nSalaried applicants need Rs 15,000 per month.") == 180_000
    assert read_income("### X – Income\nAt least six lakh rupees (₹6,00,000) a year.") == 600_000
    assert read_income("### X – Income\nMinimum Rs 24 lakh. The requirement was revised from Rs 18 lakh.") == 2_400_000
    assert read_income("### X – Income\nThe annual fee is Rs 500.") is None
    assert read_age("### X – Age\nBetween the age of 18 and 70 years.") == 18


def test_rule_format_document_is_kept_and_free_text_needs_card_names() -> None:
    md = Path("data/rule_docs/sbi_cards.md").read_bytes()
    result = ingest(md, "sbi_cards.md", "SBI")
    assert result.mode == "rule-format" and "SBI SimplyCLICK" in result.cards
    with pytest.raises(IngestError):
        ingest(b"Any card. Minimum income Rs 6 lakh.", "page.txt", "AXIS")


def test_diff_and_validation() -> None:
    old = [{"card": "A", "min_income": 600000, "min_age": 18, "min_cibil": None, "allowed_employment": "any",
            "relationship_required": False}]
    new = [{**old[0], "min_income": 720000}, {**old[0], "card": "B"}]
    changes = diff_requirements(old, new)
    assert {"card": "A", "change": "modified", "field": "min_income", "old": 600000.0, "new": 720000.0} in changes
    assert {"card": "B", "change": "added"} in changes
    with pytest.raises(ValueError):
        req_from_dict({"bank": "AXIS", "card": "A", "min_cibil": 1200})


def test_audit_chain_detects_tampering(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "log.jsonl")
    for i in range(3):
        log.append("event", {"i": i}, "tester")
    assert log.verify()["ok"]
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["data"]["i"] = 99
    lines[1] = json.dumps(tampered)
    (tmp_path / "log.jsonl").write_text("\n".join(lines) + "\n")
    result = log.verify()
    assert not result["ok"] and result["first_bad_index"] == 1


# ---- API flow --------------------------------------------------------------------------------
def test_login_and_role_separation(client: TestClient) -> None:
    assert client.post("/portal/login", json={"username": "admin_sbi", "password": "wrong"}).status_code == 401
    assert client.get("/portal/banks").status_code == 401
    customer = login(client, "customer", "customer@123")
    assert client.get("/portal/admin/versions", headers=customer).status_code == 403
    admin = login(client, "admin_hdfc", "hdfc@123")
    assert client.post("/portal/ask", headers=admin, json={"message": "hi", "banks": ["SBI", "HDFC"]}).status_code == 403
    versions = client.get("/portal/admin/versions", headers=admin).json()
    assert versions["bank"] == "HDFC" and versions["active_version"] == "v001"  # seeded thesis rule base


def test_axis_upload_review_approve_and_customer_decision(client: TestClient) -> None:
    axis = login(client, "admin_axis", "axis@123")
    banks = client.get("/portal/banks", headers=axis).json()
    assert {b["code"]: b["active_version"] for b in banks}["AXIS"] is None  # not live before approval

    draft = upload_axis(client, axis)
    assert draft["meta"]["status"] == "draft" and draft["meta"]["mode"] == "converted"
    by_card = {r["card"]: r for r in draft["extracted"]}
    assert by_card["Flipkart Axis Bank Credit Card"]["min_income"] == 180000
    assert by_card["Axis Bank Magnus Credit Card"]["min_income"] == 2400000  # superseded Rs 18 lakh ignored
    assert by_card["Axis Bank Neo Credit Card"]["guard_problems"]  # income stated but no amount: flagged

    # the employee corrects one value (sets a CIBIL floor for ACE) and approves
    approved = [dict(r) for r in draft["extracted"]]
    for r in approved:
        if r["card"] == "Axis Bank ACE Credit Card":
            r["min_cibil"] = 750
    out = client.post(f"/portal/admin/versions/{draft['meta']['version']}/approve", headers=axis,
                      json={"approved": approved, "note": "checked"})
    assert out.status_code == 200, out.text
    assert out.json()["meta"]["n_corrections"] == 1
    corrections = (settings.portal_dir / "banks/AXIS/corrections.jsonl").read_text().splitlines()
    assert any("min_cibil" in json.loads(line)["changed_fields"] for line in corrections)

    customer = login(client, "customer", "customer@123")
    assert client.post("/portal/ask", headers=customer, json={"message": "x", "banks": ["AXIS"]}).status_code == 400
    r = client.post("/portal/ask", headers=customer, json={
        "message": "I'm 34, salaried, earn 18 lakh a year, CIBIL 700, I bank with Axis", "banks": ["HDFC", "AXIS"]})
    assert r.status_code == 200, r.text
    data = r.json()
    cards = {c["card"]: c for c in data["cards"]}
    assert set(data["rule_versions"]) == {"HDFC", "AXIS"}
    assert not any(c["bank"] == "SBI" for c in data["cards"])
    assert cards["Axis Bank Magnus Credit Card"]["recommended"] is False  # 18 lakh < 24 lakh
    assert cards["Axis Bank ACE Credit Card"]["recommended"] is False  # CIBIL 700 < approved 750
    assert cards["Flipkart Axis Bank Credit Card"]["recommended"] is True
    assert data["audit_entry"]

    audit = client.get("/portal/admin/audit", headers=axis).json()
    assert audit["verify"]["ok"]
    assert [e["event"] for e in audit["entries"]][-1] == "customer_decision"
    assert "34" not in json.dumps(audit["entries"][-1]["data"])  # no applicant details stored


def test_new_version_shows_diff_and_supersedes(client: TestClient) -> None:
    axis = login(client, "admin_axis", "axis@123")
    v1 = upload_axis(client, axis)
    client.post(f"/portal/admin/versions/{v1['meta']['version']}/approve", headers=axis, json={"approved": v1["extracted"]})
    v2 = upload_axis(client, axis, "axis_eligibility_ILLUSTRATIVE_update.pdf")
    assert v2["changes_vs_active"] == [{"card": "Axis Bank ACE Credit Card", "change": "modified", "field": "min_income",
                                        "old": 600000.0, "new": 720000.0}]
    out = client.post(f"/portal/admin/versions/{v2['meta']['version']}/approve", headers=axis, json={"approved": v2["extracted"]})
    assert out.json()["meta"]["previous_version"] == v1["meta"]["version"]
    statuses = {v["version"]: v["status"] for v in client.get("/portal/admin/versions", headers=axis).json()["versions"]}
    assert statuses[v1["meta"]["version"]] == "superseded"


def test_per_bank_relationship_and_form_path(client: TestClient) -> None:
    customer = login(client, "customer", "customer@123")
    base = {"annual_income": 3_000_000, "age": 40, "cibil": 820, "employment": "salaried", "banks": ["SBI", "HDFC"]}
    elite = lambda data: {c["card"]: c for c in data["cards"]}["SBI Elite"]["recommended"]  # noqa: E731
    assert elite(client.post("/portal/recommend", headers=customer, json={**base, "existing_banks": ["HDFC"]}).json()) is False
    assert elite(client.post("/portal/recommend", headers=customer, json={**base, "existing_banks": ["SBI"]}).json()) is True


def test_follow_up_question_when_a_fact_is_missing(client: TestClient) -> None:
    customer = login(client, "customer", "customer@123")
    r = client.post("/portal/ask", headers=customer, json={
        "message": "I'm 30, salaried, earn 9 lakh a year, I bank with SBI", "banks": ["SBI", "HDFC"]}).json()
    assert r["status"] == "need_more_info" and any("CIBIL" in q for q in r["questions"])
    r2 = client.post("/portal/ask", headers=customer, json={"message": "760", "banks": ["SBI", "HDFC"],
                                                              "previous": r["understood"]}).json()
    assert r2["status"] == "ok" and len(r2["cards"]) == 8


def test_regex_baseline_scores_on_thesis_documents() -> None:
    """Baseline numbers quoted in PORTAL_AND_FL.md (rule-based reader vs the frozen LLM's 40/40)."""
    from evaluation.metrics import extraction_field_matches
    from src.domain.eligibility import EligibilityOracle
    from src.pipeline import RuleChunker

    truth = {(r.bank, r.card): {"min_income": r.min_income, "min_age": r.min_age, "min_cibil": r.min_cibil,
                                "employment": r.employment, "relationship_required": r.relationship_required}
             for r in EligibilityOracle().rules}
    scores = {}
    for d in ("data/rule_docs", "data/rule_docs_hard"):
        chunks = [vars(c) for c in RuleChunker({"sbi_cards.md": "SBI", "hdfc_cards.md": "HDFC"}).chunk_directory(Path(d))]
        reqs = RuleBasedExtractor().extract(chunks)
        scores[d] = sum(sum(extraction_field_matches(r.as_dict(), truth[(r.bank, r.card)]).values()) for r in reqs)
    assert scores == {"data/rule_docs": 34, "data/rule_docs_hard": 18}
