"""Deterministic comparison of an applicant against card requirements read by the LLM.

Why this module exists (diagnosed from the 14 Oct evaluation outputs):
88% (test) and 96% (dev) of the grounded system's false negatives were the
LLM stating that a criterion the applicant PASSES had failed, e.g. "income of
Rs 28,04,342 is below the required Rs 6,00,000" or "age 56 is below the
minimum of 21". Retrieval was not the problem: the right rules were in the
prompt. The model mis-compared numbers.

In the `extract_compare` decision mode the work is therefore split:

* the LLM reads the retrieved rule text and EXTRACTS each card's requirements
  (minimum income, age, CIBIL, allowed employment types, relationship) with the
  chunk it read each one from -- the language task that a hand-coded rule
  engine cannot do on heterogeneous, inconsistently worded documents;
* this module COMPARES the applicant with those extracted requirements -- the
  arithmetic that a small LLM gets wrong -- and writes a specific,
  counterfactual reason that cites the chunk of the failing criterion.

Nothing here reads `data/eligibility_table.csv`: every threshold comes from
what the LLM extracted from the retrieved documents, so a wrong extraction
still produces a wrong decision and is still measured as an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.domain.entities import Decision, UserProfile

EMPLOYMENT_TYPES = ("salaried", "self-employed", "business owner")

_EMPLOYMENT_SYNONYMS: dict[str, str] = {
    "salaried": "salaried",
    "salary": "salaried",
    "salaried employee": "salaried",
    "salaried professional": "salaried",
    "salaried staff": "salaried",
    "employee": "salaried",
    "self employed": "self-employed",
    "self-employed": "self-employed",
    "selfemployed": "self-employed",
    "self-employed professional": "self-employed",
    "self employed professional": "self-employed",
    "professional": "self-employed",
    "business owner": "business owner",
    "business": "business owner",
    "businessman": "business owner",
    "business person": "business owner",
    "small business owner": "business owner",
    "established business owner": "business owner",
    "entrepreneur": "business owner",
}

_ANY_EMPLOYMENT = {"any", "all", "any employment", "all employment types", "no restriction"}


def normalise_employment(value: str) -> str | None:
    """Map free-text employment wording onto one of EMPLOYMENT_TYPES (or None)."""
    text = re.sub(r"[^a-z\- ]", " ", value.lower()).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"s$", "", text)  # 'business owners' -> 'business owner'
    if text in _EMPLOYMENT_SYNONYMS:
        return _EMPLOYMENT_SYNONYMS[text]
    for key, canonical in _EMPLOYMENT_SYNONYMS.items():
        if key in text:
            return canonical
    return None


# The model may label its sources by criterion ("income") or by the field it
# filled ("min_annual_income_inr"); Qwen2.5-7B used the field names in the
# 6 Nov run, which left declines without a citation. Both are accepted.
_SOURCE_KEY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("income", "income"),
    ("age", "age"),
    ("cibil", "cibil"),
    ("credit", "cibil"),
    ("score", "cibil"),
    ("employ", "employment"),
    ("relationship", "relationship"),
    ("account", "relationship"),
)


def criterion_for_key(key: str) -> str | None:
    """Map a source label such as 'min_annual_income_inr' or 'Income' to a criterion name."""
    text = key.lower()
    for needle, criterion in _SOURCE_KEY_PATTERNS:
        if needle in text:
            return criterion
    return None


def _to_number(value: Any) -> float | None:
    """Parse 600000, '6,00,000', 'Rs 6 lakh', '6 lakh', 6.0 ... into a float (rupees)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).lower().replace("₹", " ").replace("rs.", " ").replace("rs", " ").replace("inr", " ")
    multiplier = 1.0
    if "crore" in text:
        multiplier = 1e7
    elif "lakh" in text or "lac" in text:
        multiplier = 1e5
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0).replace(",", ""))
    # "6 lakh" -> 600000; but "6,00,000 (six lakh)" is already in rupees
    if multiplier > 1 and number < 1000:
        number *= multiplier
    return number


def _to_int(value: Any) -> int | None:
    number = _to_number(value)
    return int(round(number)) if number is not None else None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "required", "y", "1"}:
        return True
    if text in {"false", "no", "not required", "n", "0", "none"}:
        return False
    return None


@dataclass(frozen=True)
class CardRequirements:
    """Requirements for one card as EXTRACTED by the LLM from retrieved rules.

    A field left as None means the retrieved rules did not state it, so it is
    not applied (the grounded prompt's rule: never reject on an unstated
    condition). Extraction quality is measured separately against the
    eligibility table (evaluation, not inference).
    """

    bank: str
    card: str
    min_income: float | None = None
    min_age: int | None = None
    min_cibil: int | None = None
    allowed_employment: tuple[str, ...] | None = None  # None = any
    relationship_required: bool | None = None
    sources: dict[str, str] = field(default_factory=dict)  # criterion -> chunk_id

    @classmethod
    def from_llm_dict(cls, bank: str, card: str, item: dict[str, Any]) -> CardRequirements:
        """Build from the model's JSON, tolerating common formatting variations."""
        req = item.get("requirements", item)
        if not isinstance(req, dict):
            req = {}

        def pick(*names: str) -> Any:
            for name in names:
                if name in req:
                    return req[name]
            return None

        employment_raw = pick("allowed_employment", "employment", "employment_types", "allowed_employment_types")
        allowed: tuple[str, ...] | None
        if employment_raw is None:
            allowed = None
        else:
            values = employment_raw if isinstance(employment_raw, list) else re.split(r",|/| or | and ", str(employment_raw))
            texts = [str(v).strip() for v in values if str(v).strip()]
            if not texts or any(t.lower() in _ANY_EMPLOYMENT for t in texts):
                allowed = None
            else:
                mapped = {normalise_employment(t) for t in texts}
                mapped.discard(None)
                allowed = tuple(sorted(mapped)) if mapped else None  # type: ignore[arg-type]

        sources_raw = item.get("sources") or item.get("cited_chunk_ids") or {}
        sources: dict[str, str] = {}
        if isinstance(sources_raw, dict):
            for key, value in sources_raw.items():
                if value in (None, "", "null"):
                    continue
                criterion = criterion_for_key(str(key))
                sources.setdefault(criterion or str(key), str(value))
        cited = item.get("cited_chunk_id")
        if cited and "default" not in sources:
            sources["default"] = str(cited)

        return cls(
            bank=bank,
            card=card,
            min_income=_to_number(pick("min_annual_income_inr", "min_income", "min_annual_income", "minimum_income")),
            min_age=_to_int(pick("min_age", "minimum_age")),
            min_cibil=_to_int(pick("min_cibil", "min_credit_score", "minimum_cibil", "min_cibil_score")),
            allowed_employment=allowed,
            relationship_required=_to_bool(
                pick("existing_relationship_required", "relationship_required", "requires_existing_relationship")
            ),
            sources=sources,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_income": self.min_income,
            "min_age": self.min_age,
            "min_cibil": self.min_cibil,
            "allowed_employment": list(self.allowed_employment) if self.allowed_employment else "any",
            "relationship_required": self.relationship_required,
            "sources": dict(self.sources),
        }


def _rupees(value: float) -> str:
    """Format rupees with Indian digit grouping, e.g. 2804342 -> 'Rs 28,04,342'."""
    digits = str(int(round(value)))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    return f"₹{digits}"


@dataclass(frozen=True)
class CriterionResult:
    criterion: str  # income | age | cibil | employment | relationship
    passed: bool
    detail: str


def check_requirements(profile: UserProfile, req: CardRequirements) -> list[CriterionResult]:
    """Compare the applicant with every stated requirement (unstated ones are skipped)."""
    results: list[CriterionResult] = []
    if req.min_income is not None:
        ok = profile.annual_income >= req.min_income
        results.append(
            CriterionResult(
                "income",
                ok,
                f"annual income of {_rupees(profile.annual_income)} is "
                f"{'at or above' if ok else 'below'} the {_rupees(req.min_income)} minimum",
            )
        )
    if req.min_age is not None:
        ok = profile.age >= req.min_age
        results.append(
            CriterionResult("age", ok, f"age {profile.age} is {'at or above' if ok else 'below'} the minimum of {req.min_age}")
        )
    if req.min_cibil is not None:
        ok = profile.cibil >= req.min_cibil
        results.append(
            CriterionResult(
                "cibil", ok, f"CIBIL score {profile.cibil} is {'at or above' if ok else 'below'} the minimum of {req.min_cibil}"
            )
        )
    if req.allowed_employment is not None:
        applicant = normalise_employment(profile.employment) or profile.employment.lower()
        ok = applicant in req.allowed_employment
        results.append(
            CriterionResult(
                "employment",
                ok,
                f"employment type '{profile.employment}' is {'' if ok else 'not '}one of the accepted types "
                f"({', '.join(req.allowed_employment)})",
            )
        )
    if req.relationship_required:
        ok = profile.existing_relationship
        results.append(
            CriterionResult(
                "relationship",
                ok,
                f"an existing {req.bank} relationship is required and the applicant "
                f"{'has one' if ok else 'does not have one'}",
            )
        )
    return results


def decide(profile: UserProfile, req: CardRequirements) -> Decision:
    """Turn extracted requirements into a decision with a specific, cited reason."""
    results = check_requirements(profile, req)
    failed = [r for r in results if not r.passed]
    if failed:
        reason = "Not eligible because " + "; ".join(r.detail for r in failed) + "."
        cited = next((req.sources[r.criterion] for r in failed if r.criterion in req.sources), None) or req.sources.get(
            "default"
        )
    else:
        reason = "Eligible: " + "; ".join(r.detail for r in results) + "." if results else (
            "Eligible: the retrieved rules state no requirement that this applicant fails."
        )
        cited = req.sources.get("income") or req.sources.get("default") or next(iter(req.sources.values()), None)
    return Decision(bank=req.bank, card=req.card, recommended=not failed, reason=reason, cited_chunk_id=cited)
