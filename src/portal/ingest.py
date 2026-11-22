"""Turn an uploaded rule document (PDF, Markdown or text) into rule chunks.

Two cases:

1. The document already uses the system's rule format, one heading per rule:
   ``### <Card> – <Rule>``. It is kept as it is (only the dash is normalised).
2. Any other document (e.g. a bank's eligibility page saved as PDF): the
   administrator names the card(s) it covers. The text is split into sentences,
   each sentence goes to the card section it appears in and to the criterion it
   talks about (income, age, credit score, employment, relationship), and the
   result is written as one rule chunk per (card, criterion) -- the same
   one-rule-per-chunk granularity the thesis evaluated (A5). Sentences about no
   eligibility criterion (fees, rewards, documents) go to an "Other Conditions"
   chunk, which the fail-closed guard does not count as a stated criterion.

The output is Markdown that the existing ``RuleChunker`` (src/pipeline.py) reads.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from src.pipeline import RULE_HEADING_RE

CRITERIA: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # (key, heading title, keywords) -- order matters: first match wins
    ("cibil", "Credit Score Requirement", ("cibil", "credit score", "bureau", "credit history", "credit record")),
    ("age", "Age Requirement", (" age", "years of age", "years old", "aged ")),
    ("relationship", "Existing Relationship Requirement",
     ("existing customer", "relationship", "savings account", "salary account", "already hold", "existing account")),
    ("income", "Income Requirement",
     ("income", "salary", "itr", "earning", "earn ", "lakh", "per month", "per annum", "p.a", "₹", "rs.", "rs ")),
    ("employment", "Employment Requirement",
     ("salaried", "self-employed", "self employed", "business owner", "employment", "occupation", "professional")),
)
OTHER_TITLE = "Other Conditions"
_DASH_HEADING_RE = re.compile(r"^###\s*(?P<card>.+?)\s+[–—-]\s+(?P<rule>.+)$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z₹0-9•\-*(])|\n\s*[•\-*]\s+|\n{2,}")


class IngestError(ValueError):
    """The document could not be turned into rule chunks."""


@dataclass(frozen=True)
class IngestResult:
    markdown: str
    cards: list[str]
    mode: str  # "rule-format" | "converted"
    source_text: str


def read_text(data: bytes, filename: str) -> str:
    """Extract plain text from PDF / Markdown / text bytes."""
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency listed in requirements.txt
            raise IngestError("PDF support needs the 'pypdf' package") from exc
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith((".md", ".txt", ".markdown")):
        text = data.decode("utf-8", errors="replace")
    else:
        raise IngestError("Upload a .pdf, .md or .txt file")
    text = text.replace("\r\n", "\n").replace(" ", " ")
    if not text.strip():
        raise IngestError("No text could be read from the file (a scanned PDF needs OCR first)")
    return text


_NOT_A_CONDITION_RE = re.compile(
    r"\b(fee|fees|waiver|waived|reward|rewards|cashback|lounge|milestone|joining|documents?|photograph)\b", re.IGNORECASE)


def _criterion_of(sentence: str) -> str | None:
    if _NOT_A_CONDITION_RE.search(sentence):
        return None  # fees, rewards, documents: not an eligibility condition
    text = f" {sentence.lower()} "
    for key, _title, words in CRITERIA:
        if any(w in text for w in words):
            return key
    return None


def _title(key: str) -> str:
    return next(t for k, t, _ in CRITERIA if k == key)


def _split_sections(text: str, cards: list[str]) -> dict[str, str]:
    """Assign text to cards: a line naming a card starts that card's section."""
    if len(cards) == 1:
        return {cards[0]: text}
    patterns = [(card, re.compile(re.escape(card), re.IGNORECASE)) for card in cards]
    sections: dict[str, list[str]] = {card: [] for card in cards}
    current: str | None = None
    for line in text.splitlines():
        hits = [card for card, pat in patterns if pat.search(line)]
        if hits:
            current = max(hits, key=len)  # "Axis Magnus Burgundy" beats "Axis Magnus"
            if line.strip().lower() == current.lower():
                continue  # a heading line holding only the card name is not a rule
        if current is not None:
            sections[current].append(line)
    return {card: "\n".join(lines) for card, lines in sections.items() if "".join(lines).strip()}


def convert_to_rule_markdown(text: str, cards: list[str], bank: str) -> str:
    """Free text + card names -> one '### <Card> – <Criterion>' chunk per stated criterion."""
    if not cards:
        raise IngestError("This document is not in rule format: enter the card name(s) it covers")
    sections = _split_sections(text, cards)
    if not sections:
        raise IngestError("None of the card names entered appears in the document")
    out = [f"{bank} credit-card eligibility rules (converted from an uploaded document).", ""]
    for card in cards:
        body = sections.get(card)
        if not body:
            continue
        grouped: dict[str, list[str]] = {}
        for raw in _SENTENCE_SPLIT_RE.split(body):
            sentence = " ".join(raw.split())
            if len(sentence) < 3 or sentence.lower() == card.lower():
                continue
            grouped.setdefault(_criterion_of(sentence) or "other", []).append(sentence)
        for key, _title_text, _ in CRITERIA:
            if key in grouped:
                out += [f"### {card} – {_title(key)}", " ".join(grouped[key]), ""]
        if "other" in grouped:
            out += [f"### {card} – {OTHER_TITLE}", " ".join(grouped["other"]), ""]
    return "\n".join(out).strip() + "\n"


def ingest(data: bytes, filename: str, bank: str, card_names: list[str] | None = None) -> IngestResult:
    """Read an uploaded document and return rule-format Markdown."""
    text = read_text(data, filename)
    normalised = _DASH_HEADING_RE.sub(lambda m: f"### {m.group('card').strip()} – {m.group('rule').strip()}", text)
    headings = list(RULE_HEADING_RE.finditer(normalised))
    if headings:
        cards = list(dict.fromkeys(m.group("card").strip() for m in headings))
        return IngestResult(markdown=normalised, cards=cards, mode="rule-format", source_text=text)
    cards = [c.strip() for c in (card_names or []) if c.strip()]
    markdown = convert_to_rule_markdown(text, cards, bank)
    found = list(dict.fromkeys(m.group("card").strip() for m in RULE_HEADING_RE.finditer(markdown)))
    if not found:
        raise IngestError("No eligibility rules were found for the card name(s) entered")
    return IngestResult(markdown=markdown, cards=found, mode="converted", source_text=text)
