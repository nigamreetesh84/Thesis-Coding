"""Read each card's requirements out of its rule chunks, with a cited chunk per value.

Two interchangeable extractors return ``CardRequirements`` (src/domain/requirements.py):

* ``LLMExtractor``  -- the thesis's frozen extraction prompt (EXTRACTION_SYSTEM_PROMPT,
  one prompt per card, applicant never in the prompt). Used with a real LLM.
* ``RuleBasedExtractor`` -- a transparent regular-expression reader. It lets the
  portal run without a GPU (stub mode) and is the non-learned BASELINE in the
  federated-learning experiment (fl/). It is deliberately simple; the
  administrator's review step exists exactly because a reader can be wrong.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Protocol

from src.domain.query_parsing import normalise_income
from src.domain.requirements import CardRequirements, normalise_employment
from src.model import ExtractionPromptBuilder, _same_card, extract_json_items
from src.portal.ingest import CRITERIA

Chunk = dict[str, Any]  # chunk_id, bank, card, text


class Extractor(Protocol):
    name: str

    def extract(self, chunks: list[Chunk]) -> list[CardRequirements]: ...


def group_by_card(chunks: Iterable[Chunk]) -> dict[tuple[str, str], list[Chunk]]:
    groups: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault((str(chunk["bank"]), str(chunk["card"])), []).append(chunk)
    return groups


def heading_criterion(text: str) -> str | None:
    """Criterion named by a chunk's heading line ('### Card – Income Requirement' -> 'income')."""
    first = text.strip().splitlines()[0].lower() if text.strip() else ""
    heading = first.split("–", 1)[-1] if "–" in first else first
    for key, title, words in CRITERIA:
        if title.lower().split()[0] in heading or any(w.strip() in heading for w in words if len(w.strip()) > 3):
            return key
    if "employment" in heading:
        return "employment"
    return None


# ---------------------------------------------------------------------------
# LLM extractor (frozen prompt)
# ---------------------------------------------------------------------------
class LLMExtractor:
    name = "llm"

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client
        self.builder = ExtractionPromptBuilder()

    def extract(self, chunks: list[Chunk]) -> list[CardRequirements]:
        groups = group_by_card(chunks)
        keys = list(groups)
        prompts = [self.builder.build(groups[k]) for k in keys]
        batch = getattr(self.llm_client, "generate_batch", None)
        raws = list(batch(prompts)) if callable(batch) else [self.llm_client.generate(p) for p in prompts]
        out: list[CardRequirements] = []
        for (bank, card), raw in zip(keys, raws, strict=True):
            items = extract_json_items(raw)
            own = [i for i in items if _same_card(bank, card, str(i.get("bank", "")), str(i.get("card", "")))]
            item = own[0] if own else (items[0] if len(items) == 1 else None)
            out.append(CardRequirements.from_llm_dict(bank, card, item) if item else CardRequirements(bank, card))
        return out


# ---------------------------------------------------------------------------
# Rule-based extractor (no LLM)
# ---------------------------------------------------------------------------
_SUPERSEDED_RE = re.compile(
    r"\b(previously|earlier|formerly|used to|was revised|revised from|old (?:threshold|limit|criteria)|no longer|"
    r"until \w+ \d{4}|before \w+ \d{4}|prior to)\b", re.IGNORECASE)
_NOT_A_CONDITION_RE = re.compile(
    r"\b(fee|fees|waiver|waived|spend|spends|spending|reward|cashback|lounge|milestone|credit limit|joining)\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(
    r"((?:₹|rs\.?|inr)\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:lakhs?|lacs?|crores?|thousand|k)\b)?"
    r"(?:\s*(?:per month|a month|monthly|/\s*month|p\.?m\.?|per annum|a year|annually|p\.?a\.?))?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:lakhs?|lacs?|crores?)\b(?:\s*(?:per month|a month|monthly|per annum|a year|annually|p\.?a\.?))?"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|twenty[- ]five|thirty|forty|fifty)\s+(?:lakhs?|crores?)\b"
    r"(?:\s*(?:rupees)?\s*(?:per month|a month|monthly))?)",
    re.IGNORECASE)
_SENT_RE = re.compile(r"(?<=[.;])\s+")
_AGE_RANGE_RE = re.compile(r"\b(\d{2})\s*(?:to|-|–|and)\s*(\d{2})\s*(?:years|yrs)", re.IGNORECASE)
_AGE_MIN_RE = re.compile(r"(?:at least|minimum(?: age)?(?: of)?|min\.?|not (?:be )?younger than|above|over)\s*(\d{2})\b"
                         r"|\b(\d{2})\s*(?:years|yrs)(?: of age| old)?(?: or (?:above|older|more))", re.IGNORECASE)
_SCORE_RE = re.compile(r"\b([3-9]\d\d)\b")
_REL_REQUIRED_RE = re.compile(
    r"(must (?:already )?(?:hold|have|maintain)|only (?:for|to) existing|existing (?:\w+ )?customers only|"
    r"is required|is mandatory|required to hold|need(?:s)? to (?:hold|have))", re.IGNORECASE)
_REL_NOT_RE = re.compile(r"(not (?:required|mandatory|necessary|insist)|does not insist|no need|need not|"
                         r"regardless of|whether or not|not need)", re.IGNORECASE)
_ANY_EMPLOYMENT_RE = re.compile(r"(any employment|all employment|any (?:type of )?occupation|regardless of (?:their )?"
                                r"(?:employment|occupation)|of any employment type)", re.IGNORECASE)
_EMPLOYMENT_NEG_RE = re.compile(r"(not (?:currently )?(?:accepted|eligible|considered)|cannot apply|excluded|"
                                r"are not|is not open)", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    body = "\n".join(text.strip().splitlines()[1:]) if text.strip().startswith("###") else text
    return [s.strip() for s in _SENT_RE.split(" ".join(body.split())) if s.strip()]


def read_income(text: str) -> float | None:
    """Smallest stated minimum income (annual rupees) in non-superseded, condition sentences."""
    values: list[float] = []
    for sentence in _sentences(text):
        if _SUPERSEDED_RE.search(sentence) or _NOT_A_CONDITION_RE.search(sentence):
            continue
        monthly_sentence = bool(re.search(r"per month|a month|monthly|/\s*month", sentence, re.IGNORECASE))
        for match in _AMOUNT_RE.finditer(sentence):
            phrase = match.group(1)
            value = normalise_income(phrase)
            if value is None:
                continue
            if monthly_sentence and not re.search(r"month|p\.?m", phrase, re.IGNORECASE) and value < 300_000:
                value *= 12  # "₹15,000 net income per month" with the unit at the end of the sentence
            if 10_000 <= value <= 100_000_000:
                values.append(value)
        if values:
            break  # first condition sentence that states an amount wins
    return min(values) if values else None


def read_age(text: str) -> int | None:
    for sentence in _sentences(text):
        if _SUPERSEDED_RE.search(sentence):
            continue
        rng = _AGE_RANGE_RE.search(sentence)
        if rng:
            return int(rng.group(1))
        m = _AGE_MIN_RE.search(sentence)
        if m:
            value = int(next(g for g in m.groups() if g))
            if 16 <= value <= 80:
                return value
    return None


def read_cibil(text: str) -> int | None:
    for sentence in _sentences(text):
        if _SUPERSEDED_RE.search(sentence) or not re.search(r"score|cibil|bureau|credit", sentence, re.IGNORECASE):
            continue
        scores = [int(s) for s in _SCORE_RE.findall(sentence) if 300 <= int(s) <= 900]
        if scores:
            return min(scores)
    return None


def read_employment(text: str) -> tuple[str, ...] | None:
    """Allowed employment types; None means any type is accepted (or none is stated)."""
    allowed: set[str] = set()
    for sentence in _sentences(text):
        if _ANY_EMPLOYMENT_RE.search(sentence):
            return None
        negative = bool(_EMPLOYMENT_NEG_RE.search(sentence))
        found = set()
        low = sentence.lower()
        for phrase in ("self-employed", "self employed", "salaried", "business owner", "business owners", "freelancer",
                       "entrepreneur", "professional"):
            if phrase in low:
                canonical = normalise_employment(phrase)
                if canonical:
                    found.add(canonical)
        if "self-employed" in found and "self-employed professional" not in low:
            pass
        if negative:
            allowed -= found
        else:
            allowed |= found
    return tuple(sorted(allowed)) if allowed else None


def read_relationship(text: str) -> bool | None:
    for sentence in _sentences(text):
        if not re.search(r"account|relationship|customer", sentence, re.IGNORECASE):
            continue
        if _REL_NOT_RE.search(sentence):
            return False
        if _REL_REQUIRED_RE.search(sentence):
            return True
    return None


class RuleBasedExtractor:
    name = "rule_based"

    def extract(self, chunks: list[Chunk]) -> list[CardRequirements]:
        out: list[CardRequirements] = []
        for (bank, card), group in group_by_card(chunks).items():
            fields: dict[str, Any] = {}
            sources: dict[str, str] = {}
            for chunk in sorted(group, key=lambda c: str(c["chunk_id"])):
                text = str(chunk["text"])
                criterion = heading_criterion(text)
                readers = {
                    "income": ("min_income", read_income),
                    "age": ("min_age", read_age),
                    "cibil": ("min_cibil", read_cibil),
                    "employment": ("allowed_employment", read_employment),
                    "relationship": ("relationship_required", read_relationship),
                }
                # read the heading's criterion first; a combined chunk ("Employment and Relationship") feeds both
                for key, (field_name, reader) in readers.items():
                    heading = text.strip().splitlines()[0].lower()
                    if key != criterion and key not in heading:
                        continue
                    value = reader(text)
                    if value is not None and field_name not in fields:
                        fields[field_name] = value
                        sources.setdefault(key, str(chunk["chunk_id"]))
                    elif value is None and key == "employment" and field_name not in fields:
                        sources.setdefault(key, str(chunk["chunk_id"]))
            out.append(CardRequirements(bank=bank, card=card, sources=sources, **fields))
        return out


def make_extractor(llm_client: Any | None, use_stub: bool) -> Extractor:
    """LLM extractor with a real model; the transparent rule-based reader in stub mode."""
    if use_stub or llm_client is None:
        return RuleBasedExtractor()
    return LLMExtractor(llm_client)
