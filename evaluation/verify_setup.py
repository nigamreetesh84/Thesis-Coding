"""Integrity checks that must pass before any evaluation number is trusted.

Run: `python -m evaluation.verify_setup` (exit code 1 if any check fails).

It checks that (1) the rule documents agree with the eligibility table the gold
labels come from, (2) every gold label equals the oracle's decision, (3) the
index/chunk artefacts match the documents, and (4) the grounded and ungrounded
prompts are built correctly. These are the failure modes that previously made
results meaningless without raising any error.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import settings
from src.domain.eligibility import EligibilityOracle, load_eligibility_table
from src.domain.entities import UserProfile
from src.infrastructure.llm_client import to_ungrounded_prompt
from src.model import GroundedPromptBuilder
from src.pipeline import RuleChunker


def _numbers(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", text.replace(",", ""))}


def check_docs_match_table() -> list[str]:
    """Every threshold in the eligibility table must appear in that card's rule text."""
    problems: list[str] = []
    table = load_eligibility_table(settings.eligibility_table_path)
    chunks = RuleChunker({"sbi_cards.md": "SBI", "hdfc_cards.md": "HDFC"}).chunk_directory(settings.rule_docs_dir)
    by_card: dict[tuple[str, str], str] = {}
    for c in chunks:
        by_card[(c.bank, c.card)] = by_card.get((c.bank, c.card), "") + "\n" + c.text
    for row in table.itertuples(index=False):
        text = by_card.get((row.bank, row.card))
        if text is None:
            problems.append(f"no rule chunks for {row.bank} {row.card}")
            continue
        nums = _numbers(text)
        for label, value in (("min_income", int(row.min_income)), ("min_age", int(row.min_age)), ("min_cibil", int(row.min_cibil))):
            if value not in nums:
                problems.append(f"{row.card}: {label}={value} not found in its rule text")
        lowered = text.lower()
        if bool(row.relationship_required) != any(w in lowered for w in ("expected to already", "must already", "already hold an active", "already maintain")):
            problems.append(f"{row.card}: relationship_required={row.relationship_required} not reflected in rule text")
    return problems


def check_gold_matches_oracle(path: Path) -> list[str]:
    oracle = EligibilityOracle(settings.eligibility_table_path)
    problems: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        profile = UserProfile(**row["profile"])
        expected = {(d["bank"], d["card"]): bool(d["recommended"]) for d in row["expected"]}
        actual = {(d.bank, d.card): d.recommended for d in oracle.evaluate(profile)}
        if expected != actual:
            problems.append(f"gold profile {i}: expected labels differ from the oracle")
    return problems


def check_artifacts() -> list[str]:
    problems: list[str] = []
    if not (settings.faiss_index_path.exists() and settings.chunks_parquet_path.exists()):
        return ["artifacts/faiss.index or chunks.parquet missing: run `python -m src.pipeline`"]
    chunks = pd.read_parquet(settings.chunks_parquet_path)
    fresh = RuleChunker({"sbi_cards.md": "SBI", "hdfc_cards.md": "HDFC"}).chunk_directory(settings.rule_docs_dir)
    if len(chunks) != len(fresh) or list(chunks["text"]) != [c.text for c in fresh]:
        problems.append("artifacts/chunks.parquet is stale compared with data/rule_docs: rebuild with `python -m src.pipeline`")
    return problems


def check_prompts() -> list[str]:
    problems: list[str] = []
    profile = UserProfile(annual_income=731330.0, age=30, cibil=768, employment="salaried", existing_relationship=True)
    chunk: dict[str, Any] = {"chunk_id": "x#0", "bank": "SBI", "card": "SBI Prime", "text": "### SBI Prime - Income\nrule text"}
    grounded = GroundedPromptBuilder().build(profile, [chunk])
    ungrounded = to_ungrounded_prompt(grounded)
    if "### CARDS TO EVALUATE\n- bank=SBI; card=SBI Prime" not in ungrounded:
        problems.append("ungrounded prompt lost the card list")
    if "rule text" in ungrounded or "chunk_id=" in ungrounded:
        problems.append("ungrounded prompt still contains retrieved rules")
    if "Use ONLY the rules" in ungrounded:
        problems.append("ungrounded prompt still tells the model to use retrieved rules")
    if "rule text" not in grounded:
        problems.append("grounded prompt is missing the retrieved rules")
    return problems


def main() -> None:
    checks = {
        "rule documents agree with the eligibility table": check_docs_match_table,
        "gold labels equal the oracle": lambda: check_gold_matches_oracle(settings.gold_test_set_path),
        "index/chunk artefacts match the rule documents": check_artifacts,
        "grounded/ungrounded prompts are built correctly": check_prompts,
    }
    failed = False
    for name, fn in checks.items():
        problems = fn()
        print(("PASS  " if not problems else "FAIL  ") + name)
        for problem in problems:
            print("      -", problem)
        failed |= bool(problems)
    if failed:
        sys.exit(1)
    print("\nAll integrity checks passed.")


if __name__ == "__main__":
    main()
