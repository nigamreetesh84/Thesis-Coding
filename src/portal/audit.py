"""Tamper-evident audit log: an append-only, hash-chained JSONL file.

Each entry stores the SHA-256 of the previous entry, so changing, deleting or
re-ordering any past entry breaks every hash after it, and ``verify()`` reports
the first broken entry. This gives the property a blockchain ledger is used for
(an append-only record whose history cannot be silently rewritten) without
running a distributed network, which a single-operator prototype does not need.

Logged events: rule uploads, approvals (with the approved rule-set hash) and
customer decisions (which rule version each bank's decision used). Customer
decisions store only a salted hash of the request -- never the applicant's details.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _last_hash(self) -> str:
        entries = self.entries()
        return str(entries[-1]["entry_hash"]) if entries else GENESIS

    def append(self, event: str, data: dict[str, Any], actor: str) -> dict[str, Any]:
        with self._lock:
            entries = self.entries()
            body = {
                "index": len(entries),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": event,
                "actor": actor,
                "data": data,
                "prev_hash": str(entries[-1]["entry_hash"]) if entries else GENESIS,
            }
            entry = {**body, "entry_hash": sha256_text(canonical(body))}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            return entry

    def verify(self) -> dict[str, Any]:
        """Recompute the chain. Returns {'ok', 'entries', 'first_bad_index', 'reason'}."""
        prev = GENESIS
        entries = self.entries()
        for i, entry in enumerate(entries):
            body = {k: v for k, v in entry.items() if k != "entry_hash"}
            if body.get("prev_hash") != prev:
                return {"ok": False, "entries": len(entries), "first_bad_index": i, "reason": "chain link broken"}
            if sha256_text(canonical(body)) != entry.get("entry_hash"):
                return {"ok": False, "entries": len(entries), "first_bad_index": i, "reason": "entry content altered"}
            prev = str(entry["entry_hash"])
        return {"ok": True, "entries": len(entries), "first_bad_index": None, "reason": ""}
