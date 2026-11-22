"""Turn an applicant's free-text message into a UserProfile (natural-language interface).

Same division of labour as `extract_compare` (src/domain/requirements.py):

* the LLM READS the message and copies the values it finds -- the income
  phrase verbatim ("about 9 lakh a year", "Rs 75,000 per month") -- because
  understanding everyday wording is the language task;
* this module COMPUTES: it converts the income phrase to annual rupees (lakh,
  crore, thousand/k, per month, LPA) and maps employment wording onto the three
  canonical types. The 14 Oct error analysis showed the LLM confusing rupees and
  lakh when it handled numbers itself, so it never converts units here.

A field the message does not state stays None, and `missing` lists it: the
interface then asks the applicant instead of guessing (fail-closed, as in the
extract_compare guard). `rule_based_fields` is a regex baseline used by the
GPU-free StubLLM and as the comparison arm in evaluation/run_nl_eval.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.domain.entities import UserProfile
from src.domain.requirements import normalise_employment

KNOWN_BANKS = ("SBI", "HDFC")
REQUIRED_FIELDS = ("income", "age", "cibil", "employment", "existing_relationship")

FOLLOW_UP_QUESTIONS: dict[str, str] = {
    "income": "What is your income (annual, or your monthly salary)?",
    "age": "How old are you?",
    "cibil": "What is your CIBIL credit score (300-900)?",
    "employment": "Are you salaried, self-employed, or a business owner?",
    "existing_relationship": "Do you already have an account or relationship with SBI or HDFC?",
}

# ---------------------------------------------------------------------------
# Income phrase -> annual rupees
# ---------------------------------------------------------------------------
_UNIT_MULTIPLIERS: tuple[tuple[str, float], ...] = (
    (r"crores?|cr", 1e7),
    (r"lakhs?|lacs?|lpa|l", 1e5),
    (r"millions?|mn", 1e6),
    (r"thousands?|k", 1e3),
)
_MONTHLY_RE = re.compile(
    r"(per\s*month|a\s*month|each\s*month|every\s*month|monthly|/\s*month|/\s*mo\b|\bp\.?\s*m\.?(?=\s|$|[,.;)])|\bpm\b)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_NUMBERS: dict[str, int] = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_WORD_NUMBERS.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
                      "seventy": 70, "eighty": 80, "ninety": 90})


def _words_to_number(text: str) -> float | None:
    """'nine' -> 9, 'twenty five' -> 25, 'one hundred' -> 100 (leading filler words skipped)."""
    total, seen = 0.0, False
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _WORD_NUMBERS:
            total += _WORD_NUMBERS[word]
            seen = True
        elif word == "hundred" and seen:
            total *= 100
        elif word == "and" or not seen:
            continue
        else:
            break
    return total if seen else None


def normalise_income(text: str | None) -> float | None:
    """Convert an income phrase to ANNUAL rupees.

    '9 lakh a year' -> 900000; 'Rs 75,000 per month' -> 900000; '12 LPA' -> 1200000;
    'about 28.0 lakh rupees a year' -> 2800000; '251531 rupees' -> 251531;
    '₹1.2 crore' -> 12000000; '60k per month' -> 720000; 'nine lakh' -> 900000.
    """
    if not text:
        return None
    t = str(text).lower().replace("₹", " rs ")
    match = _NUMBER_RE.search(t)
    if match:
        number = float(match.group(0).replace(",", ""))
        after = t[match.end():]
    else:
        number_or_none = _words_to_number(t)
        if number_or_none is None:
            return None
        number, after = number_or_none, t
    multiplier = 1.0
    # the unit word right after the number wins ("6 lakh"), else any unit word in the phrase
    for scope in (after[:12], t):
        for pattern, mult in _UNIT_MULTIPLIERS:
            if re.search(rf"(?<![a-z])({pattern})(?![a-z])", scope):
                multiplier = mult
                break
        if multiplier != 1.0:
            break
    value = number * multiplier
    if _MONTHLY_RE.search(t):
        value *= 12
    return value


# ---------------------------------------------------------------------------
# Employment wording -> canonical type
# ---------------------------------------------------------------------------
# Extra everyday wording handled HERE, so the frozen normalise_employment()
# used by the comparison engine (src/domain/requirements.py) is not changed.
_EXTRA_EMPLOYMENT: tuple[tuple[str, str], ...] = (
    ("freelanc", "self-employed"),
    ("consultant", "self-employed"),
    ("independent", "self-employed"),
    ("own practice", "self-employed"),
    ("proprietor", "business owner"),
    ("own company", "business owner"),
    ("own business", "business owner"),
    ("own shop", "business owner"),
    ("regular job", "salaried"),
    ("monthly salary", "salaried"),
    ("government job", "salaried"),
    ("full-time", "salaried"),
)


def parse_employment(value: str) -> str | None:
    text = value.lower()
    canonical = normalise_employment(text)
    if canonical is not None:
        return canonical
    return next((c for needle, c in _EXTRA_EMPLOYMENT if needle in text), None)


# ---------------------------------------------------------------------------
# Parsed query
# ---------------------------------------------------------------------------
def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    match = _NUMBER_RE.search(str(value))
    return int(round(float(match.group(0).replace(",", "")))) if match else None


def _banks(value: Any) -> list[str] | None:
    """Normalise a bank list; ['ANY'] means 'already banks with them', unnamed."""
    if value is None:
        return None
    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in items:
        text = str(item).upper()
        if text in {"", "NONE", "NULL", "NO"}:
            continue
        named = [b for b in KNOWN_BANKS if b in text]
        out.extend(b for b in named if b not in out)
        if not named and "ANY" not in out:
            out.append("ANY")
    return out


@dataclass(frozen=True)
class ParsedQuery:
    """What the interface understood from one message (None = not stated)."""

    income: float | None = None
    age: int | None = None
    cibil: int | None = None
    employment: str | None = None
    existing_relationship: bool | None = None
    banks: tuple[str, ...] = KNOWN_BANKS
    preferences: str = ""
    income_text: str | None = None
    existing_banks: tuple[str, ...] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def missing(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if getattr(self, f) is None]

    def follow_up_questions(self) -> list[str]:
        return [FOLLOW_UP_QUESTIONS[f] for f in self.missing]

    def merged_with(self, later: ParsedQuery) -> ParsedQuery:
        """Fill this query's missing fields from a follow-up reply; keep what was already known."""
        values = {f: getattr(self, f) if getattr(self, f) is not None else getattr(later, f) for f in REQUIRED_FIELDS}
        return ParsedQuery(
            **values,
            banks=self.banks,
            preferences=self.preferences or later.preferences,
            income_text=self.income_text or later.income_text,
            existing_banks=self.existing_banks if self.existing_banks is not None else later.existing_banks,
            notes=self.notes + later.notes,
        )

    def to_profile(self) -> UserProfile:
        if self.missing:
            raise ValueError(f"cannot build a profile, missing: {self.missing}")
        return UserProfile(
            annual_income=float(self.income),  # type: ignore[arg-type]
            age=int(self.age),  # type: ignore[arg-type]
            cibil=int(self.cibil),  # type: ignore[arg-type]
            employment=str(self.employment),
            existing_relationship=bool(self.existing_relationship),
            banks=list(self.banks),
        )

    @classmethod
    def from_llm_dict(cls, data: dict[str, Any]) -> ParsedQuery:
        """Build from the parser's JSON; implausible values become None (then asked for)."""
        notes: list[str] = []
        income_text = data.get("income_text")
        income = normalise_income(str(income_text)) if income_text not in (None, "", "null") else None
        if income is not None and not 10_000 <= income <= 100_000_000:
            notes.append(f"income {income:g} implausible, ignored")
            income = None
        age = _to_int(data.get("age"))
        if age is not None and not 16 <= age <= 100:
            notes.append(f"age {age} implausible, ignored")
            age = None
        cibil = _to_int(data.get("cibil"))
        if cibil is not None and not 300 <= cibil <= 900:
            notes.append(f"CIBIL {cibil} outside 300-900, ignored")
            cibil = None
        employment_raw = data.get("employment")
        employment = parse_employment(str(employment_raw)) if employment_raw not in (None, "", "null") else None
        existing = _banks(data.get("existing_banks"))
        preferred = [b for b in (_banks(data.get("preferred_banks")) or []) if b in KNOWN_BANKS]
        return cls(
            income=income,
            age=age,
            cibil=cibil,
            employment=employment,
            existing_relationship=None if existing is None else bool(existing),
            banks=tuple(preferred) if preferred else KNOWN_BANKS,
            preferences=str(data.get("preferences") or "").strip(),
            income_text=None if income_text in (None, "", "null") else str(income_text),
            existing_banks=None if existing is None else tuple(existing),
            notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# Regex baseline (no LLM). Used by StubLLM and as the evaluation baseline arm.
# ---------------------------------------------------------------------------
_INCOME_PHRASE_RE = re.compile(
    r"((?:rs\.?|inr|₹)?\s*\d[\d,]*(?:\.\d+)?\s*(?:lakhs?|lacs?|lpa|crores?|cr|k|thousand)?\s*(?:rupees|inr)?"
    r"\s*(?:a year|per year|per annum|p\.?a\.?|annually|yearly|a month|per month|monthly|/\s*month|pm)?)",
    re.IGNORECASE,
)
_INCOME_CONTEXT_RE = re.compile(r"(income|earn|salary|make|making|drawing|ctc|package)[^.\d₹]{0,40}", re.IGNORECASE)
_HAS_REL_RE = re.compile(
    r"(already (?:hold|holds|have|has|bank|banks)\b|already a customer|existing customer(?! of no)|"
    r"(?:have|hold|has|holds) an? (?:\w+ ){0,2}account|banked with|bank with (?:sbi|hdfc|axis)|"
    r"(?:sbi|hdfc|axis) (?:savings|salary|current|bank)? ?account|customer of (?:sbi|hdfc|axis))",
    re.IGNORECASE,
)
_NO_REL_RE = re.compile(
    r"(does not currently hold|do not (?:have|hold)|don't (?:have|hold)|not an existing customer|no (?:existing )?"
    r"(?:account|relationship|bank account)|new to (?:both|the bank|sbi|hdfc)|not a customer)",
    re.IGNORECASE,
)


_YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|y|i do|sure|correct)\b", re.IGNORECASE)
_NO_RE = re.compile(r"^\s*(no|nope|nah|n|i don't|i do not|none)\b", re.IGNORECASE)


def short_yes_no(message: str) -> bool | None:
    """'yes' / 'no' (and close variants) at the start of a reply, else None."""
    if _NO_RE.search(message):
        return False
    if _YES_RE.search(message):
        return True
    return None


def rule_based_fields(message: str) -> dict[str, Any]:
    """Regex reading of a message, returned in the same JSON shape as the LLM parser."""
    text = message.replace("\n", " ")
    income_text = None
    ctx = _INCOME_CONTEXT_RE.search(text)
    if ctx:
        m = _INCOME_PHRASE_RE.search(text, ctx.end() - 1)
        if m and _NUMBER_RE.search(m.group(1)):
            income_text = m.group(1).strip()
    if income_text is None:  # no cue word: take an amount that carries an income unit
        unit = re.search(r"((?:rs\.?|inr|₹)\s*\d[\d,]*(?:\.\d+)?[^.;]{0,25}|\d[\d,]*(?:\.\d+)?\s*"
                         r"(?:lakhs?|lacs?|lpa|crores?)[^.;,]{0,15}|\d[\d,]*\s*(?:k|rupees)\s*(?:a|per)\s*(?:year|month|annum))",
                         text, re.IGNORECASE)
        if unit:
            income_text = unit.group(1).strip()
    age_m = re.search(r"(\d{2})\s*[- ]?(?:years?|yrs?)[- ]?old|\bage[d]?\s*(?:is\s*)?(\d{2})\b|(?:i'?m|i am)\s*(\d{2})\b", text, re.I)
    cibil_m = re.search(r"(?:cibil|credit|bureau)[^\d]{0,25}(\d{3})", text, re.I)
    employment = None
    for phrase in ("business owner", "runs their own company", "run my own business", "own a", "entrepreneur",
                   "self-employed", "self employed", "freelancer", "working independently",
                   "salaried", "monthly salary", "regular job", "employee"):
        if phrase in text.lower():
            employment = {"runs their own company": "business owner", "run my own business": "business owner",
                          "own a": "business owner", "entrepreneur": "business owner",
                          "freelancer": "self-employed", "working independently": "self-employed",
                          "self employed": "self-employed", "monthly salary": "salaried",
                          "regular job": "salaried", "employee": "salaried"}.get(phrase, phrase)
            break
    if _NO_REL_RE.search(text):
        existing: list[str] | None = []
    elif _HAS_REL_RE.search(text):
        named = [b for b in KNOWN_BANKS if re.search(rf"\b{b}\b", text)]
        existing = named or ["ANY"]
    else:
        existing = None
    return {
        "income_text": income_text,
        "age": next((int(g) for g in age_m.groups() if g), None) if age_m else None,
        "cibil": int(cibil_m.group(1)) if cibil_m else None,
        "employment": employment,
        "existing_banks": existing,
        "preferred_banks": [],
        "preferences": " ".join(w for w in ("travel", "cashback", "shopping", "lounge", "rewards", "fuel")
                                if w in text.lower()),
    }
