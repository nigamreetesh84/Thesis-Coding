"""Versioned, per-bank rule knowledge base with human approval.

Layout (all under ``data/portal``)::

    banks/<BANK>/versions/v001/
        source.<pdf|md|txt>     the uploaded file, unchanged
        <bank>_cards.md         rule-format Markdown (one rule per chunk)
        extracted.json          what the extractor read (value + cited chunk per field)
        approved.json           what the bank employee approved (after corrections)
        meta.json               status, uploader, approver, hashes, times
    banks/<BANK>/active.json    which approved version is live
    banks/<BANK>/corrections.jsonl   (chunk text -> corrected value): the bank's
                                     LOCAL training data for federated learning
    audit_log.jsonl             hash-chained log of every upload / approval / decision

Only APPROVED versions reach customers. The live vector index
(``artifacts/portal``) is rebuilt from the active version of every bank after
each approval.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.domain.requirements import EMPLOYMENT_TYPES, CardRequirements, guard_problems, stated_criteria
from src.pipeline import PipelineBuilder, RuleChunker
from src.portal.audit import AuditLog, canonical, sha256_text
from src.portal.auth import BANKS
from src.portal.ingest import IngestResult, ingest
from src.portal.rule_extractor import Extractor

FIELDS = ("min_income", "min_age", "min_cibil", "allowed_employment", "relationship_required")
CRITERION_OF_FIELD = {"min_income": "income", "min_age": "age", "min_cibil": "cibil",
                      "allowed_employment": "employment", "relationship_required": "relationship"}


class RegistryError(ValueError):
    """A request to the rule registry was invalid."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def bank_code(bank: str) -> str:
    code = bank.strip().upper()
    if code not in BANKS:
        raise RegistryError(f"Unknown bank {bank!r}; expected one of {', '.join(BANKS)}")
    return code


def req_to_dict(req: CardRequirements) -> dict[str, Any]:
    return {"bank": req.bank, "card": req.card, **req.as_dict()}


def req_from_dict(data: dict[str, Any]) -> CardRequirements:
    """Validate an (edited) requirement dict from the review screen."""

    def num(name: str, low: float, high: float, integer: bool) -> float | int | None:
        value = data.get(name)
        if value in (None, "", "null"):
            return None
        try:
            number = float(str(value).replace(",", ""))
        except ValueError as exc:
            raise RegistryError(f"{data.get('card')}: {name} must be a number") from exc
        if not low <= number <= high:
            raise RegistryError(f"{data.get('card')}: {name} {number:g} is outside {low:g}-{high:g}")
        return int(round(number)) if integer else number

    employment = data.get("allowed_employment")
    if employment in (None, "", "any", ["any"]) or employment == []:
        allowed = None
    else:
        items = employment if isinstance(employment, list) else re.split(r"[,/]", str(employment))
        cleaned = sorted({str(i).strip().lower() for i in items if str(i).strip()})
        bad = [i for i in cleaned if i not in EMPLOYMENT_TYPES]
        if bad:
            raise RegistryError(f"{data.get('card')}: unknown employment type(s) {bad}")
        allowed = tuple(cleaned) if set(cleaned) != set(EMPLOYMENT_TYPES) else None
    rel = data.get("relationship_required")
    relationship = None if rel in (None, "", "null") else (str(rel).lower() in {"true", "yes", "1"} if not isinstance(rel, bool) else rel)
    return CardRequirements(
        bank=str(data["bank"]), card=str(data["card"]),
        min_income=num("min_income", 10_000, 100_000_000, False),  # type: ignore[arg-type]
        min_age=num("min_age", 16, 80, True),  # type: ignore[arg-type]
        min_cibil=num("min_cibil", 300, 900, True),  # type: ignore[arg-type]
        allowed_employment=allowed, relationship_required=relationship,
        sources={str(k): str(v) for k, v in (data.get("sources") or {}).items() if v},
    )


def _comparable(req: dict[str, Any]) -> dict[str, Any]:
    emp = req.get("allowed_employment")
    emp_norm = "any" if emp in (None, "any") or (isinstance(emp, list) and set(emp) == set(EMPLOYMENT_TYPES)) else sorted(emp)
    out = {f: req.get(f) for f in FIELDS}
    out["allowed_employment"] = emp_norm
    out["relationship_required"] = bool(req.get("relationship_required"))
    if out["min_income"] is not None:
        out["min_income"] = float(out["min_income"])
    return out


def diff_requirements(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Card-by-card, field-by-field changes from one approved version to the next."""
    old_by = {r["card"]: _comparable(r) for r in old}
    new_by = {r["card"]: _comparable(r) for r in new}
    changes: list[dict[str, Any]] = []
    for card in sorted(set(old_by) | set(new_by)):
        if card not in old_by:
            changes.append({"card": card, "change": "added"})
        elif card not in new_by:
            changes.append({"card": card, "change": "removed"})
        else:
            for f in FIELDS:
                if old_by[card][f] != new_by[card][f]:
                    changes.append({"card": card, "change": "modified", "field": f,
                                    "old": old_by[card][f], "new": new_by[card][f]})
    return changes


@dataclass(frozen=True)
class ActiveCard:
    requirements: CardRequirements
    version: str
    version_hash: str


class RuleRegistry:
    def __init__(self, root: str | Path = "data/portal", index_dir: str | Path = "artifacts/portal",
                 embedder: Any | None = None) -> None:
        self.root = Path(root)
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.audit = AuditLog(self.root / "audit_log.jsonl")
        self._lock = threading.RLock()

    # ---- paths -------------------------------------------------------------
    def bank_dir(self, bank: str) -> Path:
        return self.root / "banks" / bank_code(bank)

    def version_dir(self, bank: str, version: str) -> Path:
        if not re.fullmatch(r"v\d{3}", version):
            raise RegistryError(f"Bad version id {version!r}")
        return self.bank_dir(bank) / "versions" / version

    def _read(self, path: Path, default: Any = None) -> Any:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def _write(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ---- queries -----------------------------------------------------------
    def versions(self, bank: str) -> list[dict[str, Any]]:
        vdir = self.bank_dir(bank) / "versions"
        if not vdir.exists():
            return []
        return [self._read(p / "meta.json", {}) for p in sorted(vdir.iterdir()) if (p / "meta.json").exists()]

    def active_version(self, bank: str) -> str | None:
        data = self._read(self.bank_dir(bank) / "active.json")
        return data.get("version") if data else None

    def version_detail(self, bank: str, version: str) -> dict[str, Any]:
        vdir = self.version_dir(bank, version)
        if not (vdir / "meta.json").exists():
            raise RegistryError(f"{bank} {version} does not exist")
        md_path = vdir / f"{bank_code(bank).lower()}_cards.md"
        chunks = [vars(c) for c in RuleChunker({md_path.name: bank_code(bank)}).chunk_file(md_path)]
        return {
            "meta": self._read(vdir / "meta.json"),
            "extracted": self._read(vdir / "extracted.json", []),
            "approved": self._read(vdir / "approved.json"),
            "chunks": chunks,
        }

    def active_cards(self, banks: list[str] | None = None) -> list[ActiveCard]:
        out: list[ActiveCard] = []
        for bank in BANKS if banks is None else [bank_code(b) for b in banks]:
            version = self.active_version(bank)
            if version is None:
                continue
            vdir = self.version_dir(bank, version)
            meta = self._read(vdir / "meta.json", {})
            for item in self._read(vdir / "approved.json", []):
                out.append(ActiveCard(req_from_dict(item), version, str(meta.get("approved_hash", ""))))
        return out

    def active_banks(self) -> list[str]:
        return [b for b in BANKS if self.active_version(b)]

    # ---- upload -> draft ---------------------------------------------------
    def create_draft(self, bank: str, data: bytes, filename: str, uploader: str, extractor: Extractor,
                     card_names: list[str] | None = None) -> dict[str, Any]:
        bank = bank_code(bank)
        result: IngestResult = ingest(data, filename, bank, card_names)
        with self._lock:
            existing = [m.get("version", "v000") for m in self.versions(bank)]
            version = f"v{(max(int(v[1:]) for v in existing) + 1) if existing else 1:03d}"
            vdir = self.version_dir(bank, version)
            vdir.mkdir(parents=True, exist_ok=False)
            suffix = Path(filename).suffix.lower() or ".txt"
            (vdir / f"source{suffix}").write_bytes(data)
            md_path = vdir / f"{bank.lower()}_cards.md"
            md_path.write_text(result.markdown, encoding="utf-8")
            chunks = [vars(c) for c in RuleChunker({md_path.name: bank}).chunk_file(md_path)]
            reqs = extractor.extract(chunks)
            extracted = []
            for req in reqs:
                stated = stated_criteria([c["text"] for c in chunks if c["card"] == req.card])
                extracted.append({**req_to_dict(req), "guard_problems": guard_problems(req, stated)})
            self._write(vdir / "extracted.json", extracted)
            meta = {
                "bank": bank, "version": version, "status": "draft", "filename": filename, "mode": result.mode,
                "cards": result.cards, "uploader": uploader, "uploaded_at": _now(), "extractor": extractor.name,
                "source_sha256": hashlib.sha256(data).hexdigest(), "n_chunks": len(chunks),
            }
            self._write(vdir / "meta.json", meta)
            self.audit.append("rule_upload", {"bank": bank, "version": version, "filename": filename,
                                              "source_sha256": meta["source_sha256"], "extractor": extractor.name}, uploader)
            return self.version_detail(bank, version)

    # ---- approve / reject ----------------------------------------------------
    def approve(self, bank: str, version: str, approved: list[dict[str, Any]], approver: str,
                note: str = "") -> dict[str, Any]:
        bank = bank_code(bank)
        with self._lock:
            vdir = self.version_dir(bank, version)
            meta = self._read(vdir / "meta.json")
            if not meta:
                raise RegistryError(f"{bank} {version} does not exist")
            if meta["status"] != "draft":
                raise RegistryError(f"{bank} {version} is already {meta['status']}")
            extracted = {r["card"]: r for r in self._read(vdir / "extracted.json", [])}
            cards = set(meta["cards"])
            reqs = []
            for item in approved:
                item = {**item, "bank": bank}
                if item.get("card") not in cards:
                    raise RegistryError(f"Card {item.get('card')!r} is not in this upload")
                if not item.get("sources"):
                    item["sources"] = extracted.get(item["card"], {}).get("sources", {})
                reqs.append(req_from_dict(item))
            if {r.card for r in reqs} != cards:
                raise RegistryError(f"Approve every card in the upload: {sorted(cards)}")
            approved_dicts = [req_to_dict(r) for r in reqs]
            approved_hash = sha256_text(canonical(approved_dicts))
            previous = self.active_version(bank)
            prev_approved = self._read(self.version_dir(bank, previous) / "approved.json", []) if previous else []
            changes = diff_requirements(prev_approved, approved_dicts)
            corrections = self._record_corrections(bank, version, extracted, approved_dicts, vdir)
            self._write(vdir / "approved.json", approved_dicts)
            meta.update({"status": "approved", "approver": approver, "approved_at": _now(), "note": note,
                         "approved_hash": approved_hash, "previous_version": previous, "changes": changes,
                         "n_corrections": corrections})
            self._write(vdir / "meta.json", meta)
            if previous:
                prev_meta = self._read(self.version_dir(bank, previous) / "meta.json")
                prev_meta.update({"status": "superseded", "superseded_by": version})
                self._write(self.version_dir(bank, previous) / "meta.json", prev_meta)
            self._write(self.bank_dir(bank) / "active.json", {"version": version, "since": _now()})
            self.audit.append("rule_approval", {"bank": bank, "version": version, "approved_hash": approved_hash,
                                                "previous_version": previous, "n_changes": len(changes),
                                                "n_corrections": corrections}, approver)
            self.rebuild_index()
            return {"meta": meta, "changes": changes}

    def reject(self, bank: str, version: str, actor: str, note: str = "") -> dict[str, Any]:
        bank = bank_code(bank)
        with self._lock:
            vdir = self.version_dir(bank, version)
            meta = self._read(vdir / "meta.json")
            if not meta or meta["status"] != "draft":
                raise RegistryError(f"{bank} {version} is not a draft")
            meta.update({"status": "rejected", "rejected_by": actor, "rejected_at": _now(), "note": note})
            self._write(vdir / "meta.json", meta)
            self.audit.append("rule_rejection", {"bank": bank, "version": version}, actor)
            return meta

    def _record_corrections(self, bank: str, version: str, extracted: dict[str, Any],
                            approved: list[dict[str, Any]], vdir: Path) -> int:
        """Save every field the employee changed: (rule text -> correct value) = local FL training data."""
        md_path = vdir / f"{bank.lower()}_cards.md"
        chunks = [vars(c) for c in RuleChunker({md_path.name: bank}).chunk_file(md_path)]
        n = 0
        with (self.bank_dir(bank) / "corrections.jsonl").open("a", encoding="utf-8") as f:
            for item in approved:
                old = _comparable(extracted.get(item["card"], {}))
                new = _comparable(item)
                changed = [fld for fld in FIELDS if old.get(fld) != new.get(fld)]
                example = {"bank": bank, "card": item["card"], "version": version, "changed_fields": changed,
                           "chunks": [c for c in chunks if c["card"] == item["card"]],
                           "target": {fld: item.get(fld) for fld in FIELDS}}
                f.write(json.dumps(example, ensure_ascii=False, default=str) + "\n")
                n += len(changed)
        return n

    # ---- live index ------------------------------------------------------------
    def live_docs_dir(self) -> Path:
        return self.index_dir / "live_docs"

    def rebuild_index(self) -> None:
        live = self.live_docs_dir()
        if live.exists():
            shutil.rmtree(live)
        live.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, str] = {}
        for bank in BANKS:
            version = self.active_version(bank)
            if version is None:
                continue
            name = f"{bank.lower()}_cards.md"
            shutil.copy(self.version_dir(bank, version) / name, live / name)
            mapping[name] = bank
        if not mapping:
            return
        PipelineBuilder(rule_docs_dir=live, faiss_index_path=self.index_dir / "faiss.index",
                        chunks_parquet_path=self.index_dir / "chunks.parquet", embedder=self.embedder,
                        bank_by_filename=mapping).run()

    # ---- seeding -----------------------------------------------------------------
    def seed_bank(self, bank: str, rule_doc: Path, extractor: Extractor, verified: list[dict[str, Any]],
                  approver: str = "seed:author-verified-eligibility-table") -> None:
        """Load an existing thesis rule document as version 1 if the bank has no version yet.

        The extractor's reading is recorded as-is; the approved values are the
        author-verified eligibility table, standing in for the bank employee's review.
        """
        bank = bank_code(bank)
        if self.versions(bank):
            return
        detail = self.create_draft(bank, rule_doc.read_bytes(), rule_doc.name, "seed", extractor)
        by_card = {v["card"]: v for v in verified}
        approved = []
        for item in detail["extracted"]:
            truth = by_card.get(item["card"])
            approved.append({**item, **({k: truth[k] for k in FIELDS} if truth else {})})
        self.approve(bank, detail["meta"]["version"], approved, approver, note="initial load of the thesis rule base")
