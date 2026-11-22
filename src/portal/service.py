"""Wires the portal together: registry, extractor, approved-KB recommender, NL front end."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.domain.eligibility import EligibilityOracle
from src.domain.entities import UserProfile
from src.domain.query_parsing import ParsedQuery
from src.infrastructure.vector_store import VectorStore
from src.nl_interface import AnswerWriter, CardRanker, NLRecommender, NLResult, QueryParser
from src.portal.approved_recommender import ApprovedRecommender
from src.portal.audit import canonical, sha256_text
from src.portal.auth import BANK_DISPLAY, BANKS
from src.portal.registry import RegistryError, RuleRegistry
from src.portal.rule_extractor import Extractor

SEED_DOCS = {"SBI": "sbi_cards.md", "HDFC": "hdfc_cards.md"}
MIN_BANKS = 2


def verified_from_table(table_path: str | Path) -> list[dict[str, Any]]:
    """Author-verified thresholds (eligibility table) in the registry's field format."""
    rows = []
    for r in EligibilityOracle(table_path).rules:
        rows.append({"bank": r.bank, "card": r.card, "min_income": r.min_income, "min_age": r.min_age,
                     "min_cibil": r.min_cibil,
                     "allowed_employment": "any" if "any" in r.employment else list(r.employment),
                     "relationship_required": r.relationship_required})
    return rows


class _BankScopedParser:
    """Wraps the QueryParser so the customer's bank checkboxes decide which banks are compared."""

    def __init__(self, parser: QueryParser, banks: list[str]) -> None:
        self.parser = parser
        self.banks = tuple(banks)

    def parse(self, message: str) -> ParsedQuery:
        return replace(self.parser.parse(message), banks=self.banks)


def existing_banks_from(message: str, parsed: ParsedQuery) -> tuple[str, ...] | None:
    """Banks the applicant already has a relationship with (adds Axis, which the frozen parser does not know)."""
    if not parsed.existing_relationship:
        return parsed.existing_banks
    named = {b for b in (parsed.existing_banks or ()) if b != "ANY"}
    for code in BANKS:
        if re.search(rf"\b{code}\b", message, re.IGNORECASE):
            named.add(code)
    return tuple(sorted(named)) if named else parsed.existing_banks


class PortalService:
    def __init__(self, registry: RuleRegistry, extractor: Extractor, llm_client: Any,
                 embedder: Any, salt: str = "portal", use_writer: bool = True) -> None:
        self.registry = registry
        self.extractor = extractor
        self.llm_client = llm_client  # message parser (the StubLLM falls back to the regex parser)
        self.writer_client = llm_client if use_writer else None  # no phrasing without a real LLM
        self.embedder = embedder
        self.recommender = ApprovedRecommender(registry)
        self.salt = salt

    # ---- setup -----------------------------------------------------------------
    def seed(self, rule_docs_dir: str | Path, table_path: str | Path) -> None:
        verified = verified_from_table(table_path)
        for bank, name in SEED_DOCS.items():
            self.registry.seed_bank(bank, Path(rule_docs_dir) / name, self.extractor,
                                    [v for v in verified if v["bank"] == bank])
        if not (self.registry.index_dir / "faiss.index").exists():
            self.registry.rebuild_index()

    def banks_overview(self) -> list[dict[str, Any]]:
        out = []
        for code in BANKS:
            version = self.registry.active_version(code)
            cards = [c.requirements.card for c in self.registry.active_cards([code])] if version else []
            out.append({"code": code, "name": BANK_DISPLAY[code], "active_version": version, "cards": cards})
        return out

    # ---- customer ----------------------------------------------------------------
    def _check_banks(self, banks: list[str]) -> list[str]:
        codes = list(dict.fromkeys(b.strip().upper() for b in banks if b.strip()))
        unknown = [b for b in codes if b not in BANKS]
        if unknown:
            raise RegistryError(f"Unknown bank(s): {unknown}")
        if len(codes) < MIN_BANKS:
            raise RegistryError(f"Choose at least {MIN_BANKS} banks to compare")
        not_live = [b for b in codes if not self.registry.active_version(b)]
        if not_live:
            raise RegistryError(f"No approved rules yet for: {', '.join(BANK_DISPLAY[b] for b in not_live)}")
        return codes

    def _ranker(self) -> CardRanker | None:
        idx = self.registry.index_dir
        if not (idx / "faiss.index").exists():
            return None
        return CardRanker(self.embedder, VectorStore(idx / "faiss.index", idx / "chunks.parquet").load())

    def _log_decision(self, actor: str, profile: UserProfile, result_cards: list[dict[str, Any]]) -> str:
        request_hash = sha256_text(self.salt + canonical(vars(profile)))  # no applicant data is stored
        entry = self.registry.audit.append("customer_decision", {
            "request_hash": request_hash,
            "banks": list(profile.banks),
            "rule_versions": self.recommender.last_versions,
            "n_cards": len(result_cards),
            "n_eligible": sum(1 for c in result_cards if c["recommended"] and not c["needs_review"]),
        }, actor)
        return str(entry["entry_hash"])

    def ask(self, message: str, previous: ParsedQuery | None, banks: list[str], actor: str) -> dict[str, Any]:
        codes = self._check_banks(banks)
        parser = _BankScopedParser(QueryParser(self.llm_client), codes)
        nl = NLRecommender(parser, self.recommender, self._ranker(), AnswerWriter(self.writer_client))  # type: ignore[arg-type]
        if previous is not None:
            previous = replace(previous, banks=tuple(codes))
        # the applicant's own banks are known only after parsing; parse once more cheaply for that
        probe = parser.parse(message)
        merged = previous.merged_with(probe) if previous is not None else probe
        self.recommender.existing_banks = existing_banks_from(message, merged)
        result: NLResult = nl.ask(message, previous)
        return self._result(result, actor)

    def recommend_form(self, profile: UserProfile, existing_banks: list[str], actor: str) -> dict[str, Any]:
        codes = self._check_banks(list(profile.banks))
        profile = replace(profile, banks=codes,
                          existing_relationship=profile.existing_relationship or bool(existing_banks))
        self.recommender.existing_banks = tuple(b.upper() for b in existing_banks) or None
        parsed = ParsedQuery(income=profile.annual_income, age=profile.age, cibil=profile.cibil,
                             employment=profile.employment, existing_relationship=profile.existing_relationship,
                             banks=tuple(codes), existing_banks=self.recommender.existing_banks)
        nl = NLRecommender(QueryParser(self.llm_client), self.recommender, self._ranker(), AnswerWriter(self.writer_client))  # type: ignore[arg-type]
        return self._result(nl.decide(parsed), actor)

    def _result(self, result: NLResult, actor: str) -> dict[str, Any]:
        u = result.understood
        understood = {"income": u.income, "age": u.age, "cibil": u.cibil, "employment": u.employment,
                      "existing_relationship": u.existing_relationship,
                      "existing_banks": list(self.recommender.existing_banks or u.existing_banks or []),
                      "banks": list(u.banks), "preferences": u.preferences, "income_text": u.income_text}
        cards = [{"bank": a.bank, "bank_name": BANK_DISPLAY.get(a.bank.upper(), a.bank), "card": a.card,
                  "recommended": a.recommended, "needs_review": a.needs_review, "sentence": a.sentence,
                  "source": a.source, "cited_chunk_id": a.cited_chunk_id,
                  "rule_version": self.recommender.last_versions.get(a.bank, {}).get("version")}
                 for a in result.answers]
        out: dict[str, Any] = {"status": result.status, "understood": understood, "questions": result.questions,
                               "answer_text": result.text, "cards": cards,
                               "rule_versions": self.recommender.last_versions if result.status == "ok" else {}}
        if result.status == "ok":
            out["audit_entry"] = self._log_decision(actor, u.to_profile(), cards)
        return out
